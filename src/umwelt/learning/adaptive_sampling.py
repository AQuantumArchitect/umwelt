"""
Adaptive sampling — graph-computed sensor polling rates.

The sampling interval for each sensor feed is itself a parameter
driven by the quantum field. When a cluster is active (low purity,
high surprise), its sensors sample faster. When settled, back off.

This is fractal sampling: resolution follows information content.
The field decides how much attention each part of the world needs.

Every tunable number lives on the parameter fiber — lo, hi, alpha,
interest weights, surprise sensitivity. No magic constants.

Usage:
    sampler = AdaptiveSampler.create(
        name="system_metrics",
        base=30.0, lo=5.0, hi=120.0,
        interest_fn=composite_interest(reservoir, "rdk"),
    )

    while True:
        readings = read_sensors()
        reservoir.ingest(sensor_readings=readings)
        await asyncio.sleep(sampler.tick())

Built-in interest functions read from the live field:
    purity_interest   — 1 - cluster purity (mixed state = interesting)
    surprise_interest — fractal stack surprise EMA (unexpected = interesting)
    composite         — learned blend of both

All return float in [0, 1]: 0 = boring (sample slow), 1 = fascinating (sample fast).
"""
from __future__ import annotations
from umwelt._util import clamp01

import math
from typing import Callable

from umwelt.substrate.params import ParameterBundle

InterestFn = Callable[[], float]

# Global registry — feed loops register their samplers here,
# the API reads from it. Simple dict, no locking needed (GIL).
_SAMPLERS: dict[str, "AdaptiveSampler"] = {}


def register_sampler(name: str, sampler: "AdaptiveSampler"):
    _SAMPLERS[name] = sampler


def all_sampler_stats() -> dict[str, dict]:
    return {name: s.snapshot() for name, s in _SAMPLERS.items()}


class AdaptiveSampler:
    """Adapts a polling interval from a graph-computed interest signal.

    The interval moves in log-space between lo and hi:
        interest=1.0 → interval=lo  (maximum attention)
        interest=0.0 → interval=hi  (minimum attention)

    All parameters (lo, hi, alpha) live on a ParameterBundle and
    are continuous, learnable values — no hardcoded buckets.
    """

    def __init__(
        self,
        params: ParameterBundle,
        interest_fn: InterestFn,
    ):
        self.params = params
        self.interest_fn = interest_fn
        self._interval = params.get("base")
        self._interest_ema = 0.0

    @classmethod
    def create(
        cls,
        name: str,
        base: float,
        lo: float,
        hi: float,
        interest_fn: InterestFn,
        alpha: float = 0.15,
    ) -> "AdaptiveSampler":
        """Convenience constructor: builds the ParameterBundle from initial values."""
        params = ParameterBundle.from_dict(
            {
                # (value, sigma, clip_lo, clip_hi)
                "base": (base, base * 0.2, 1.0, 7200.0),
                "lo": (lo, lo * 0.3, 1.0, hi),
                "hi": (hi, hi * 0.2, lo, 7200.0),
                "alpha": (alpha, 0.05, 0.01, 0.5),
            },
            frozen_keys=set(),  # all learnable
        )
        sampler = cls(params=params, interest_fn=interest_fn)
        register_sampler(name, sampler)
        return sampler

    @property
    def interval(self) -> float:
        return self._interval

    @property
    def interest(self) -> float:
        return self._interest_ema

    def tick(self) -> float:
        """Recompute interval from current interest. Returns seconds to sleep."""
        raw = clamp01(self.interest_fn())

        alpha = self.params.get("alpha")
        lo = self.params.get("lo")
        hi = max(lo + 1.0, self.params.get("hi"))

        # EMA smooth the interest signal
        self._interest_ema += alpha * (raw - self._interest_ema)

        # Interpolate in log-space: interest=1 → lo, interest=0 → hi
        log_lo = math.log(lo)
        log_hi = math.log(hi)
        target = math.exp(log_hi - self._interest_ema * (log_hi - log_lo))
        self._interval += alpha * (target - self._interval)
        return self._interval

    def snapshot(self) -> dict:
        return {
            "interval": round(self._interval, 2),
            "interest_ema": round(self._interest_ema, 4),
            "lo": round(self.params.get("lo"), 2),
            "hi": round(self.params.get("hi"), 2),
            "alpha": round(self.params.get("alpha"), 4),
            "base": round(self.params.get("base"), 2),
        }


# ================================================================
# Built-in interest functions — closures over the reservoir
# ================================================================

def purity_interest(reservoir, cluster_name: str) -> InterestFn:
    """Interest = 1 - purity. Mixed state = something happening.

    Continuous on [0, 1]. Pure |0⟩ → 0, maximally mixed → 1-1/dim.
    """
    def fn() -> float:
        cluster = reservoir.field.clusters.get(cluster_name)
        if cluster is None:
            return 0.0
        return max(0.0, 1.0 - cluster.purity)
    return fn


def surprise_interest(reservoir, sensitivity: float = 100.0) -> InterestFn:
    """Interest from fractal stack surprise, soft-saturated through tanh.

    sensitivity controls how quickly surprise maps to full interest.
    Continuous — tanh is smooth, no thresholds or buckets.
    """
    def fn() -> float:
        fs = reservoir.fractal_stack
        if fs is None or not fs.scales:
            return 0.0
        surprise = fs.scales[0]._surprise_ema
        return math.tanh(surprise * sensitivity)
    return fn


def composite_interest(
    reservoir,
    cluster_name: str,
    purity_weight: float = 0.6,
    surprise_sensitivity: float = 100.0,
) -> InterestFn:
    """Continuous blend of purity departure and fractal surprise.

    Weights are initial values — both are continuous, no categories.
    The blend itself is a weighted sum clamped to [0, 1].
    """
    p_fn = purity_interest(reservoir, cluster_name)
    s_fn = surprise_interest(reservoir, sensitivity=surprise_sensitivity)
    surprise_weight = 1.0 - purity_weight

    def fn() -> float:
        return min(1.0, purity_weight * p_fn() + surprise_weight * s_fn())
    return fn
