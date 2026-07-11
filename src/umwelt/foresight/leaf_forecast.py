"""
Generic leaf forecasting — every leaf learns to anticipate itself.

The DRY generalization of a per-signal forward model: ONE capability that every
(node, role) leaf inherits. Each leaf runs a tiny online forward model that
predicts its own future value one (learned, per-leaf) horizon ahead, from
time-as-phase features + a persistence anchor + optional cross-leaf context
(regions predicting each other), and reports a `skill`. Comprehension becomes a
skill per leaf — visible, universal.

Reuses the ONE online regressor (learning.regressor.OnlineRegressor). Continuous
leaves predict a value; binary leaves (on/off) predict P(on). Pre-trainable from
logged history so a leaf boots already skilled where a pattern exists;
constant/patternless leaves simply show ~0 skill (self-pruning).

Periodic-driver features (deterministic cycles a leaf should anticipate against)
are supplied through the register_leaf_feature() registry — empty by default. A
domain registers its own drivers; the origin deployment registered a periodic
astronomical builder, left behind in its example. The universal wall-clock cyclic
features stay built in.
"""
from __future__ import annotations
from typing import Callable
from umwelt._util import clamp01

import bisect
import math
from collections import deque
from datetime import datetime, timedelta

import numpy as np

from umwelt._util import round_or_none
from umwelt.learning.regressor import OnlineRegressor


# Registry of periodic-driver feature builders: name -> (datetime -> fixed-width feature list).
# Empty by default (domain-free). A domain registers the deterministic cycles its leaves should
# anticipate against; every builder must return a stable-width list so the feature geometry is
# fixed across a forecaster's lifetime. Register at import, before any forecaster runs.
_LEAF_FEATURE_BUILDERS: "dict[str, Callable[[datetime], list[float]]]" = {}


def register_leaf_feature(name: str, fn: "Callable[[datetime], list[float]]") -> None:
    """Register a periodic-driver feature builder (datetime -> fixed-width feature list)."""
    _LEAF_FEATURE_BUILDERS[name] = fn


def clear_leaf_features() -> None:
    """Drop all registered driver-feature builders (test/reset hook)."""
    _LEAF_FEATURE_BUILDERS.clear()


def _driver_features(dt: datetime) -> list[float]:
    """Concatenate every registered periodic-driver feature builder's output for `dt`."""
    feats: list[float] = []
    for fn in _LEAF_FEATURE_BUILDERS.values():
        feats.extend(fn(dt))
    return feats


def time_features(dt: datetime) -> list[float]:
    """Cyclic clock features a periodic driver can't carry: hour-of-day + day-of-week.
    Routines are clock+calendar driven (a morning peak, weekday != weekend) — the one
    encoding pure driver-phase features miss."""
    h = dt.hour + dt.minute / 60.0
    dow = dt.weekday()
    return [math.cos(2 * math.pi * h / 24.0), math.sin(2 * math.pi * h / 24.0),
            math.cos(2 * math.pi * dow / 7.0), math.sin(2 * math.pi * dow / 7.0)]


class LeafForecaster:
    """One leaf's intrinsic forward model: features(t) → its own value at t+H.

    feature = [ *driver_features(t+H), hour/dow-cyclic(t+H)(4),
                anchor=norm(current), <context...>, bias ]
    """

    def __init__(
        self,
        leaf_id: str,
        sensor_id: str,
        node: str,
        role: str,
        binary: bool = False,
        center: float = 0.0,
        scale: float = 1.0,
        horizon_minutes: float = 60.0,
        lr: float = 0.02,
        l2: float = 0.005,
        n_context: int = 0,
    ):
        self.leaf_id = leaf_id
        self.sensor_id = sensor_id
        self.node, self.role = node, role
        self.binary = binary
        self.center, self.scale = float(center), float(scale or 1.0)
        self.horizon = timedelta(minutes=float(horizon_minutes))
        self.n_context = int(n_context)
        self.fc = OnlineRegressor(1, [leaf_id], lr=lr, l2=l2, error_mode="mean_abs")
        self.buffer: deque = deque()       # (due_dt, feature) awaiting their label
        self.prediction: float | None = None
        self.prediction_for: datetime | None = None
        self.last_value: float | None = None
        self.pretrained_pairs = 0

    # -- normalization (continuous: physical; binary: 0/1 -> -1/+1) ----------
    def _norm(self, v: float) -> float:
        return (v - self.center) / self.scale

    def _denorm(self, z: float) -> float:
        v = z * self.scale + self.center
        return float(clamp01(v)) if self.binary else float(v)

    def _feature(self, future: datetime, anchor_v: float, context: list[float] | None) -> np.ndarray:
        parts = [*_driver_features(future), *time_features(future),
                 self._norm(anchor_v)]
        if self.n_context:
            ctx = list(context or [])[: self.n_context]
            ctx += [0.0] * (self.n_context - len(ctx))
            parts += ctx
        parts.append(1.0)
        return np.asarray(parts, dtype=float)

    def set_horizon_minutes(self, m: float) -> None:
        self.horizon = timedelta(minutes=float(max(1.0, m)))

    @property
    def horizon_min(self) -> float:
        return self.horizon.total_seconds() / 60.0

    # -- offline pre-training from history -----------------------------------
    def pre_train(self, history: list[tuple[datetime, float]], tol_minutes: float = 30.0,
                  epochs: int = 2, context_at=None) -> int:
        """history: ordered [(datetime, value)]. Builds (feature(t)->value(t+H))
        pairs by matching each reading to the one nearest t+horizon (within tol).
        context_at(dt) -> list[float] supplies historical cross-leaf context."""
        pts = sorted([(dt, float(v)) for dt, v in history if v is not None], key=lambda p: p[0])
        if len(pts) < 20:
            return 0
        dts = [p[0] for p in pts]
        tol = timedelta(minutes=tol_minutes)
        pairs = []
        for dt, v in pts:
            tgt = dt + self.horizon
            k = bisect.bisect_left(dts, tgt)
            best = None
            for cand in (k - 1, k):
                if 0 <= cand < len(pts) and abs(pts[cand][0] - tgt) <= tol:
                    if best is None or abs(pts[cand][0] - tgt) < abs(pts[best][0] - tgt):
                        best = cand
            if best is not None:
                ctx = context_at(dt) if (self.n_context and context_at) else None
                pairs.append((self._feature(pts[best][0], v, ctx), pts[best][1]))
        for _ in range(max(1, epochs)):
            for f, label in pairs:
                self.fc.update(f, np.array([self._norm(label)]))
        self.pretrained_pairs = len(pairs)
        return len(pairs)

    # -- live online step (delayed-label) ------------------------------------
    def step(self, now: datetime, current_value: float | None, context: list[float] | None = None) -> None:
        if current_value is not None:
            self.last_value = float(current_value)
        anchor = self.last_value if self.last_value is not None else self.center
        feat = self._feature(now + self.horizon, anchor, context)
        p = self.fc.predict(feat)
        if p is not None:
            self.prediction = self._denorm(float(p[0]))
            self.prediction_for = now + self.horizon
        if current_value is not None:
            self.buffer.append((now + self.horizon, feat))
            y = np.array([self._norm(current_value)])
            while self.buffer and self.buffer[0][0] <= now:
                _, past_feat = self.buffer.popleft()
                self.fc.update(past_feat, y)
            stale = now - self.horizon
            while self.buffer and self.buffer[0][0] < stale:
                self.buffer.popleft()

    def snapshot(self) -> dict:
        return {
            "leaf": self.leaf_id, "node": self.node, "role": self.role,
            "binary": self.binary, "skill": round(self.fc.skill, 4),
            "n_updates": self.fc.n_updates, "pretrained_pairs": self.pretrained_pairs,
            "horizon_min": round(self.horizon_min),
            "prediction": round_or_none(self.prediction, 3),
            "prediction_for": self.prediction_for.isoformat() if self.prediction_for else None,
            "last_value": round_or_none(self.last_value, 3),
        }
