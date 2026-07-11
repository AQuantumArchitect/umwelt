"""RewardChannel + ReceptorProfile — the neuromodulator descriptors.

A RewardChannel is a DESCRIPTOR (+ a thin runtime handle), NOT a control loop: the existing learners
already fire on their own timescales and call `bundle.update`; the channel just names them and partitions
the fiber. ReceptorProfile is deliberately a dict of channel→weight so Phase-1 (one channel, weight 1.0)
is a special case of the future multi-receptor form — the later phases only relax the invariant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class ReceptorProfile:
    """Which reward channels modulate a param, with what sensitivity.

    PHASE-1 INVARIANT: exactly one entry, weight 1.0 — e.g. {"surprise": 1.0}.
    Future multi-receptor: {"surprise": 0.7, "outcome:sleep": 0.3}.
    Future learnable: these weights become qubit-backed (a meta-graph layer)."""
    channels: dict[str, float] = field(default_factory=dict)

    @property
    def owner(self) -> str:
        """The dominant channel (Phase-1: the only one). 'unlearned' if no receptor."""
        return max(self.channels, key=self.channels.get) if self.channels else "unlearned"

    def responds_to(self, channel_name: str) -> bool:
        return channel_name in self.channels


@dataclass
class RewardChannel:
    """A neuromodulator: a named reward field with its own expression manifold + timescale + tone.

    `release_level` is the global broadcast tone ∈ [0,1] — applied as an obs_sigma MULTIPLIER at the
    collapse seam (lower tone → wider obs_sigma → smaller Kalman α → gentler collapse). At 1.0 (the
    Phase-1 default for every channel) it is a NO-OP, so the learning math is byte-identical to today.
    `target_fn` is the OPTIONAL expression-manifold hook the FUTURE outcome channel uses — None for the
    three existing channels (their learner IS the target_fn). `fiber_cluster` is the ProductQubitCluster
    this channel's params live on (the sector)."""
    name: str
    fiber_cluster: str
    timescale: str = "per_tick"      # PER_TICK | PHI_STRIDE | GENERATIONAL | EVENT | NONE (documentary)
    release_level: float = 1.0
    target_fn: Optional[Callable] = None   # (param, ctx) -> (observation, obs_sigma); future outcome hook
