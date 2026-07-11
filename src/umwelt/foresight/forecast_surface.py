"""
ForecastSurface — the forecast brain's readout: every comprehended leaf learns
to anticipate its OWN future Bloch-z, at fractal (φ-separated) timescales.

This is the third brain in the braid (forecast → live → hind). It runs the
`LeafForecaster` machinery as a *bank* over a curated forecast allowlist
(`DEFAULT_FORECAST_LEAVES`, empty by default — a domain supplies its leaves) × a
ladder of Fibonacci/golden-ratio horizons. Each leaf reads its current Bloch-z
from the field and predicts that same z forward H minutes; the "fractal
timescales" are simply the same leaf run at several H at once.

The product is a *post-comprehended forecast stream*: per leaf, a predicted
future z and a `confidence = forecast skill × belief purity`. That stream flows
downstream into the live brain as a confidence-gated observation (the shipped
confidence contract: `observe_qubit(target, alpha × confidence)`), so a low-skill
forecast (skill≈0 → confidence≈0) is a provable no-op and a bad forecast can
never hijack the live belief — it can only gently nudge.

The comprehension core never references this surface; it attaches as
`engine.forecast_surface`.

Forecast emission is a foresight concern (distinct from the action-router egress
surface): the allowlist of forecast leaves lives HERE, self-contained.
"""
from __future__ import annotations
from umwelt._util import clamp01

import logging
from datetime import datetime

from umwelt.foresight.leaf_forecast import LeafForecaster

logger = logging.getLogger(__name__)

# The curated forecast allowlist — the high-utility (node, role) leaves scored
# against observed Bloch-z at expiry. Empty by default (domain-free); a domain (or
# the app) passes its own leaves in, or the origin deployment's example registers
# them. Kept HERE so forecast emission stays self-contained — it does NOT belong on
# the action-router egress surface.
DEFAULT_FORECAST_LEAVES: tuple[tuple[str, str], ...] = ()

# Fractal timescales: Fibonacci minutes (consecutive ratios → φ). The same leaf
# anticipated at several horizons at once is the "fractal" forecast — near
# horizons are actionable anticipation, far ones carry slow structure.
DEFAULT_HORIZONS_MIN: tuple[float, ...] = (13.0, 21.0, 34.0, 55.0)

# The leaf whose near-horizon prediction is published for the live brain to
# consume is the SHORTEST horizon (the most actionable anticipation). The full
# ladder rides the snapshot for observability + any future deeper consumer.


class ForecastSurface:
    """A bank of per-leaf forward models over the field's comprehension leaves.

    bank[(node, role, horizon)] = LeafForecaster predicting that leaf's Bloch-z
    H minutes ahead. Leaves come from the curated `DEFAULT_FORECAST_LEAVES`
    allowlist (the compute/memory throttle — all-leaves × all-horizons is the cliff).
    """

    def __init__(
        self,
        leaves: tuple[tuple[str, str], ...] | None = None,
        horizons_min: tuple[float, ...] = DEFAULT_HORIZONS_MIN,
        lr: float = 0.03,
        l2: float = 0.005,
    ):
        self.leaves = tuple(leaves) if leaves is not None else DEFAULT_FORECAST_LEAVES
        self.horizons = tuple(float(h) for h in horizons_min)
        self.bank: dict[tuple[str, str, float], LeafForecaster] = {}
        for node, role in self.leaves:
            for h in self.horizons:
                # Forecast in raw Bloch-z space: continuous, centred at 0, unit
                # scale (z ∈ [-1, +1]). _denorm returns the value directly, so the
                # prediction IS a Bloch-z we can publish and re-observe.
                self.bank[(node, role, h)] = LeafForecaster(
                    leaf_id=f"{node}_{role}@{int(h)}m",
                    sensor_id=f"forecast_{node}_{role}",
                    node=node, role=role, binary=False,
                    center=0.0, scale=1.0, horizon_minutes=h, lr=lr, l2=l2,
                )
        # Last belief purity read per (node, role) — the OUTPUT-side confidence,
        # multiplied with each leaf's forecast skill (INPUT-side confidence).
        self._purity: dict[tuple[str, str], float] = {}

    # -- live step: read each leaf's Bloch-z, predict it forward ----------------
    def step(self, now: datetime, field) -> None:
        """Read each comprehension leaf's current Bloch-z from the field and feed
        it to every horizon's forecaster (online delayed-label learning)."""
        for node, role in self.leaves:
            cluster = field.clusters.get(node)
            if cluster is None:
                continue
            idx = cluster.role_index.get(role)
            if idx is None:
                continue
            try:
                z = float(cluster.qubit_bloch(idx)[2])
                self._purity[(node, role)] = float(cluster.purity)
            except Exception:
                continue
            for h in self.horizons:
                self.bank[(node, role, h)].step(now, z)

    def _confidence(self, fc: LeafForecaster) -> float:
        """confidence = forecast skill × belief purity, clamped to [0, 1]. A
        low-skill OR low-purity forecast self-mutes downstream (alpha → 0)."""
        skill = clamp01(float(fc.fc.skill))
        purity = self._purity.get((fc.node, fc.role), 1.0)
        return clamp01(skill * purity)

    # -- the full fractal ladder (observability) --------------------------------
    def predictions(self) -> dict[tuple[str, str, float], dict]:
        """Every (node, role, horizon) with a live prediction → {z_pred, confidence,
        skill, horizon_min, prediction_for}."""
        out: dict[tuple[str, str, float], dict] = {}
        for key, fc in self.bank.items():
            if fc.prediction is None:
                continue
            out[key] = {
                "z_pred": round(float(fc.prediction), 5),
                "confidence": round(self._confidence(fc), 5),
                "skill": round(float(fc.fc.skill), 5),
                "horizon_min": fc.horizon_min,
                "prediction_for": fc.prediction_for.isoformat() if fc.prediction_for else None,
            }
        return out

    # -- the consumable near-horizon stream (the braid feed) --------------------
    def consume_targets(self) -> dict[tuple[str, str], dict]:
        """Per (node, role), the SHORTEST-horizon prediction — the actionable
        anticipation the live brain consumes. One value per leaf so the downstream
        consume binding stays a single sensor_id per leaf."""
        h0 = min(self.horizons) if self.horizons else None
        out: dict[tuple[str, str], dict] = {}
        if h0 is None:
            return out
        for node, role in self.leaves:
            fc = self.bank.get((node, role, h0))
            if fc is None or fc.prediction is None:
                continue
            out[(node, role)] = {
                "z_pred": round(float(fc.prediction), 5),
                "confidence": round(self._confidence(fc), 5),
                "horizon_min": fc.horizon_min,
            }
        return out

    # -- offline warm-start (optional) ------------------------------------------
    def pre_train(self, z_series_by_leaf: dict[tuple[str, str], list[tuple[datetime, float]]],
                  epochs: int = 2) -> int:
        """Warm-start from per-leaf Bloch-z series `{(node, role): [(dt, z)]}`. Each
        horizon's forecaster pre-trains on the same series at its own H. Returns the
        total (feature → label) pairs fit. Optional — the online step() learns too."""
        total = 0
        for (node, role), series in z_series_by_leaf.items():
            for h in self.horizons:
                fc = self.bank.get((node, role, h))
                if fc is None:
                    continue
                total += fc.pre_train(series, epochs=epochs)
        return total

    def snapshot(self) -> dict:
        return {
            "leaves": len(self.leaves),
            "horizons_min": [round(h) for h in self.horizons],
            "predictions": {
                f"{n}.{r}@{int(h)}m": p
                for (n, r, h), p in self.predictions().items()
            },
        }
