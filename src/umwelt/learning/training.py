"""
Training runner — continuous Hebbian learning with adaptive pacing.

The fractal stack learns H from live sensor data. This module manages
WHEN and HOW MUCH to learn — training intensity adapts to the field's
state, like a slime mold exploring its environment.

Three modes:
    1. Historical batch: replay saved data at max speed (experiments)
    2. Live continuous: learn from each sensor event as it arrives
    3. Periodic retrain: replay recent history on a schedule

The training runner is itself adaptive:
    - High surprise → train harder (larger lr, more frequent updates)
    - Low surprise → cruise (smaller lr, back off)
    - All training parameters are on the fiber (learnable)

After each training epoch, phase alignment ensures learned rotations
match the physical direction of the data.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from umwelt.learning.adaptive_sampling import AdaptiveSampler, register_sampler
from umwelt.substrate.params import ParameterBundle

logger = logging.getLogger(__name__)

# Global registry for training runners (mirrors sampler registry)
_RUNNERS: dict[str, "TrainingRunner"] = {}


def all_runner_stats() -> dict[str, dict]:
    return {name: r.snapshot() for name, r in _RUNNERS.items()}


@dataclass
class TrainingConfig:
    """Configuration for training runner. All are initial values —
    the runner's ParameterBundle makes them learnable."""

    enabled: bool = True
    # Base learning rate for Hebbian gradient
    lr: float = 0.5
    lr_lo: float = 0.01     # minimum lr (exploitation)
    lr_hi: float = 2.0      # maximum lr (exploration)
    # Phase alignment after every N epochs
    phase_align_interval: int = 1
    # Training epoch: process this many steps before phase alignment
    epoch_steps: int = 0     # 0 = continuous (no epochs, always learning)


@dataclass
class TrainingBurstProfile:
    name: str = "demo_window"
    lr_multiplier: float = 1.75
    surprise_multiplier: float = 1.5
    phase_align_interval: int | None = 1
    started_at: float | None = None
    expires_at: float | None = None

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "lr_multiplier": round(self.lr_multiplier, 4),
            "surprise_multiplier": round(self.surprise_multiplier, 4),
            "phase_align_interval": self.phase_align_interval,
            "started_at": self.started_at,
            "expires_at": self.expires_at,
        }


class TrainingRunner:
    """Manages Hebbian learning rate and training lifecycle.

    The lr adapts with surprise: high surprise → explore aggressively
    (large lr, more variance), low surprise → fine-tune (small lr,
    stable). This is the training analogue of Thompson Sampling —
    uncertainty drives exploration of the learning rate space.

    The runner wraps the fractal stack's hebbian_update with adaptive
    pacing. It doesn't own the update logic — that stays in FractalScale.
    """

    def __init__(self, config: TrainingConfig | None = None):
        config = config or TrainingConfig()
        self.config = config

        # Learnable training parameters
        self.params = ParameterBundle.from_dict({
            "lr": (config.lr, config.lr * 0.3, config.lr_lo, config.lr_hi),
            "momentum": (0.0, 0.1, 0.0, 0.95),
        })

        # Training state
        self._epoch = 0
        self._total_steps = 0
        self._surprise_ema = 0.0
        self._surprise_alpha = 0.1
        self._grad_norm_ema = 0.0
        self._last_train_time = 0.0
        self._burst_profile: TrainingBurstProfile | None = None

    @property
    def effective_lr(self) -> float:
        """Current learning rate (point estimate)."""
        return self.params.get("lr")

    @property
    def sampled_lr(self) -> float:
        """Thompson-sampled lr for exploration.

        Early (high sigma): lr varies wildly → explores learning dynamics.
        Late (low sigma): lr is stable → consistent fine-tuning.
        """
        sampled = self.params.get("lr", explore=True)
        return sampled * self._effective_lr_multiplier()

    @property
    def burst_profile(self) -> TrainingBurstProfile | None:
        self._expire_burst_if_needed()
        return self._burst_profile

    def _effective_lr_multiplier(self) -> float:
        burst = self.burst_profile
        return burst.lr_multiplier if burst is not None else 1.0

    def _effective_surprise_multiplier(self) -> float:
        burst = self.burst_profile
        return burst.surprise_multiplier if burst is not None else 1.0

    def effective_phase_align_interval(self) -> int:
        burst = self.burst_profile
        if burst is None or burst.phase_align_interval is None:
            return self.config.phase_align_interval
        return max(1, int(burst.phase_align_interval))

    def start_demo_burst(self, profile: TrainingBurstProfile) -> None:
        self._burst_profile = profile

    def stop_demo_burst(self) -> None:
        self._burst_profile = None

    def _expire_burst_if_needed(self) -> None:
        burst = self._burst_profile
        if burst is None or burst.expires_at is None:
            return
        if time.time() >= burst.expires_at:
            self._burst_profile = None

    def adapt_lr(self, surprise: float):
        """Adapt learning rate from surprise changes.

        Rising surprise → increase lr (something new, learn harder).
        Falling surprise → decrease lr (converging, fine-tune).

        Uses the DERIVATIVE of surprise, not absolute level. Absolute
        surprise has an irreducible floor from sensor injection noise.
        The derivative tells us whether the field is getting better or
        worse — which is what actually matters for learning rate.
        """
        surprise = float(surprise) * self._effective_surprise_multiplier()
        prev_surprise = self._surprise_ema
        self._surprise_ema = (
            self._surprise_alpha * surprise
            + (1 - self._surprise_alpha) * self._surprise_ema
        )

        # Derivative: positive = getting worse, negative = getting better
        d_surprise = self._surprise_ema - prev_surprise
        if abs(prev_surprise) < 1e-10:
            return

        # Scale lr by 1 + 0.1 * sign(d_surprise)
        # Getting worse → lr *= 1.1 (learn harder)
        # Getting better → lr *= 0.9 (coast)
        if d_surprise > 0:
            ratio = 1.05
        else:
            ratio = 0.95
        observed_lr = self.effective_lr * ratio
        self.params.update("lr", observed_lr, obs_sigma=0.05)

    def apply_to_stack(self, fractal_stack):
        """Set the fractal stack's learning rate from the runner.

        Called each step — propagates the adapted lr to all scales.
        """
        if fractal_stack is None:
            return

        lr = self.sampled_lr
        for scale in fractal_stack.scales:
            scale._lr = lr

    def step(self, fractal_stack, production_residuals=None):
        """One training step: adapt lr, track metrics.

        The actual Hebbian update happens inside fractal_stack.step() —
        we just control the pacing.
        """
        if fractal_stack is None or not self.config.enabled:
            return

        self._total_steps += 1
        t0 = time.time()

        # Adapt lr from the stack's surprise
        if fractal_stack.scales:
            surprise = fractal_stack.scales[0]._surprise_ema
            self.adapt_lr(surprise)

        # Apply adapted lr to the stack
        self.apply_to_stack(fractal_stack)

        # Track gradient norms for diagnostics
        if production_residuals:
            gnorm = np.mean([
                np.linalg.norm(r) for r in production_residuals.values()
            ])
            self._grad_norm_ema = (
                0.1 * gnorm + 0.9 * self._grad_norm_ema
            )

        self._last_train_time = time.time() - t0

        # Phase alignment at epoch boundaries
        if (self.config.epoch_steps > 0
                and self._total_steps % self.config.epoch_steps == 0):
            self._epoch += 1
            if self._epoch % self.effective_phase_align_interval() == 0:
                self._phase_align(fractal_stack)

    def _phase_align(self, fractal_stack):
        """Run phase alignment on scale 0."""
        if fractal_stack.scales:
            fractal_stack.scales[0].phase_align(
                fractal_stack.production_field.clusters,
            )
            logger.info("Training epoch %d: phase aligned", self._epoch)

    def force_phase_align(self, fractal_stack):
        """Manual phase alignment (callable from API)."""
        self._phase_align(fractal_stack)

    def snapshot(self) -> dict:
        burst = self.burst_profile
        boosted_lr = self.effective_lr * self._effective_lr_multiplier()
        return {
            "enabled": self.config.enabled,
            "epoch": self._epoch,
            "total_steps": self._total_steps,
            "lr": round(self.effective_lr, 6),
            "effective_lr": round(boosted_lr, 6),
            "lr_sigma": round(self.params.get_param("lr").sigma, 6),
            "surprise_ema": round(self._surprise_ema, 6),
            "grad_norm_ema": round(self._grad_norm_ema, 6),
            "demo_window_active": burst is not None,
            "burst_profile": burst.snapshot() if burst is not None else None,
            "burst_started_at": burst.started_at if burst is not None else None,
            "burst_expires_at": burst.expires_at if burst is not None else None,
            "effective_stats": {
                "lr_multiplier": round(self._effective_lr_multiplier(), 4),
                "surprise_multiplier": round(self._effective_surprise_multiplier(), 4),
                "phase_align_interval": self.effective_phase_align_interval(),
            },
            "baseline_stats": {
                "lr": round(self.effective_lr, 6),
                "phase_align_interval": self.config.phase_align_interval,
            },
            "params": self.params.snapshot(),
        }


def training_interest(runner: TrainingRunner) -> Callable[[], float]:
    """Interest function for adaptive training scheduling.

    Returns [0, 1]: 0 = system is well-learned (train less),
    1 = system is confused (train more).
    """
    def _interest() -> float:
        s = runner._surprise_ema
        # Map surprise to [0, 1] via sigmoid centered at 0.005
        return min(1.0, s / 0.005) if s > 0 else 0.0
    return _interest
