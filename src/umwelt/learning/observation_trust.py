"""Observation trust — the DISSOLUTION of the hand-set collapse_alpha into a learned coordinate.

`sensors/bridge.py` collapses a belief toward a reading at `collapse_alpha` (0.95 real sensors, 0.15
forecast) — load-bearing (experiments/strip_lowpass.py proved decisions chatter 15× without it) but
NOT learned, so (Luke) not earned. This earns it: each sensor leaf learns its own collapse rate from the
CONSISTENCY of its readings.

The signal is the INNOVATION — |observation − current belief|. A sensor whose readings keep DISAGREEING
with the settled belief is noisy → it should earn a LOW alpha (smooth / distrust); a reliable sensor's
readings sit on the belief → low innovation → HIGH alpha (snap / trust). That is exactly adaptive Kalman
gain: the innovation variance IS the learned observation noise R. A one-off real change spikes the
innovation briefly then the belief catches up and it decays, so a SLOW EMA distinguishes a noisy sensor
(persistently high) from a clean one with occasional real changes (low on average) — the pole-robust
noise-vs-signal split, for free.

  alpha = 1 / (1 + (k · innov_ema)²) ,  clipped to [alpha_min, alpha_max]
    innov_ema small (≈0.05, reliable)  → alpha ≈ 0.95  (matches the hand-tuned sensor value)
    innov_ema large (≈0.5,  noisy)     → alpha ≈ 0.16  (matches the hand-tuned forecast value)

So the 0.95/0.15 magic numbers become the two ends of one learned curve the brain rides from evidence.
Gated UMWELT_LEARN_COLLAPSE (default off = the hand-set values, exact parity). The learned per-leaf
innov_ema is brain state (pickled with the reservoir); reset-tolerant (starts at init_innov).
"""
from __future__ import annotations

import os


class ObservationTrust:
    def __init__(self, *, alpha_min: float = 0.10, alpha_max: float = 0.97, ema: float = 0.05,
                 noise_k: float = 4.5, init_innov: float = 0.20):
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.ema = float(ema)
        self.noise_k = float(noise_k)
        self.init_innov = float(init_innov)
        self.innov: dict = {}          # leaf (node, role) -> EMA of |obs − belief|

    def _alpha_from_innov(self, e: float) -> float:
        a = 1.0 / (1.0 + (self.noise_k * e) ** 2)
        return self.alpha_min if a < self.alpha_min else self.alpha_max if a > self.alpha_max else a

    def learned_alpha(self, leaf, obs_z: float, belief_z: float) -> float:
        """Update this leaf's innovation EMA from the new reading and return its learned collapse alpha.
        Confidence still rides on top (the bridge multiplies conf_brake): this learns the BASE rate."""
        e = self.innov.get(leaf, self.init_innov)
        e = (1.0 - self.ema) * e + self.ema * abs(float(obs_z) - float(belief_z))
        self.innov[leaf] = e
        return self._alpha_from_innov(e)

    def snapshot(self) -> dict:
        """Per-leaf learned rate, for the gauge / console (the earned coordinate, made visible)."""
        return {f"{n}.{r}": {"innov_ema": round(e, 4), "alpha": round(self._alpha_from_innov(e), 4)}
                for (n, r), e in sorted(self.innov.items())}


LEARN_COLLAPSE = os.environ.get("UMWELT_LEARN_COLLAPSE", "0") == "1"
