"""The surprise→forecast context wire + persistence-relative skill.

Pins: (a) n_context=0 is byte-identical to the context-free surface (regression-safe default);
(b) n_context>0 threads the cross-leaf displacement so predictions actually change; (c) the spec
flag turns the channel on through build_engine; (d) skill_vs_persistence is signed, 0 without a
baseline, and ~0 for a frozen belief (the persistence mirage, killed).
"""
from datetime import datetime, timedelta

import numpy as np

from umwelt.boot import build_engine
from umwelt.foresight.forecast_surface import ForecastSurface
from umwelt.foresight.leaf_forecast import LeafForecaster
from umwelt.learning.regressor import OnlineRegressor
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec


def _spec(context: bool):
    return DomainSpec(
        name="ctxtest",
        nodes=(NodeSpec("root", parent=None, kind="root", roles=()),
               NodeSpec("a", parent="root", roles=("s",)),
               NodeSpec("b", parent="root", roles=("s",))),
        bindings=(BindingSpec("a_s", zone="a", role="s",
                              normalizer={"type": "range", "lo": -1.0, "hi": 1.0},
                              force_observe=True),
                  BindingSpec("b_s", zone="b", role="s",
                              normalizer={"type": "range", "lo": -1.0, "hi": 1.0},
                              force_observe=True)),
        forecast_leaves=(("a", "s"), ("b", "s")),
        forecast_horizons_min=(8.0, 16.0),
        forecast_context_surprise=context,
    )


def test_spec_flag_sets_n_context():
    """forecast_context_surprise=True → the higher-order monomial context is on; False → 0.
    2 leaves → 4 atoms (z+trace each); order-2 monomials = C(4,1)+C(4,2) = 4+6 = 10."""
    on = build_engine(spec=_spec(True), population=False)
    off = build_engine(spec=_spec(False), population=False)
    assert on.forecast_surface.n_context == 10
    assert on.forecast_surface.interaction_order == 2
    assert off.forecast_surface.n_context == 0


def test_no_context_is_byte_identical():
    """n_context=0 predictions match a surface built the old way (default), tick for tick."""
    leaves = (("a", "s"), ("b", "s"))
    default = ForecastSurface(leaves=leaves, horizons_min=(8.0, 16.0))          # no n_context arg
    explicit0 = ForecastSurface(leaves=leaves, horizons_min=(8.0, 16.0), n_context=0)
    field = _FakeField()
    t0 = datetime(2026, 7, 22, 9, 0, 0)
    for k in range(200):
        now = t0 + timedelta(minutes=2.0 * k)
        field.set(k)
        default.step(now, field)
        explicit0.step(now, field)
    assert default.predictions() == explicit0.predictions()


def test_context_changes_predictions():
    """Same field trajectory, n_context=0 vs n_context=2 → the context channel changes the forecast."""
    leaves = (("a", "s"), ("b", "s"))
    clock = ForecastSurface(leaves=leaves, horizons_min=(8.0, 16.0), n_context=0)
    surp = ForecastSurface(leaves=leaves, horizons_min=(8.0, 16.0), n_context=2)
    field = _FakeField()
    t0 = datetime(2026, 7, 22, 9, 0, 0)
    for k in range(400):
        now = t0 + timedelta(minutes=2.0 * k)
        field.set(k)
        clock.step(now, field)
        surp.step(now, field)
    pc = clock.predictions()
    ps = surp.predictions()
    # at least one leaf/horizon prediction differs — the context is not inert
    assert any(abs(pc[key]["z_pred"] - ps[key]["z_pred"]) > 1e-6 for key in pc if key in ps)


def test_skill_vs_persistence_signed_and_zero_without_baseline():
    r = OnlineRegressor(1, ["t"], lr=0.05)
    for _ in range(50):
        r.update(np.array([1.0, 0.5, 1.0]), np.array([0.3]))       # no baseline_pred
    assert r.skill_vs_persistence == 0.0                            # baseline never fed → 0
    r2 = OnlineRegressor(1, ["t"], lr=0.05)
    for _ in range(50):
        r2.update(np.array([1.0, 0.5, 1.0]), np.array([0.3]), baseline_pred=np.array([0.9]))
    assert r2.skill_vs_persistence != 0.0                          # baseline fed → a real number
    assert r2.baseline_error_ema is not None


def test_frozen_belief_has_zero_skill_vs_persistence():
    """A belief that never moves: absolute skill inflates to ~1, skill_vs_persistence stays ~0."""
    fc = LeafForecaster("t", "s", "n", "r", center=0.0, scale=1.0, horizon_minutes=8.0)
    t0 = datetime(2026, 7, 22, 9, 0, 0)
    for k in range(600):
        fc.step(t0 + timedelta(minutes=2.0 * k), 0.02)             # constant
    snap = fc.snapshot()
    assert snap["skill"] > 0.9                                     # the mirage: looks near-perfect
    assert abs(snap["skill_vs_persistence"]) < 0.05               # but honestly: no foresight to add


class _FakeField:
    """Minimal field: two 1-qubit clusters whose z follows a scripted, leaf-coupled signal."""
    def __init__(self):
        self.clusters = {"a": _FakeCluster(), "b": _FakeCluster()}

    def set(self, k):
        # a is a slow ramp; b flips sign whenever a crosses zero (b depends on a's motion)
        za = float(np.sin(2 * np.pi * k / 40.0))
        zb = 0.8 if (k // 20) % 2 == 0 else -0.8
        self.clusters["a"].z = za
        self.clusters["b"].z = zb


class _FakeCluster:
    def __init__(self):
        self.z = 0.0
        self.role_index = {"s": 0}
        self.purity = 0.95

    def qubit_bloch(self, idx):
        return np.array([0.0, 0.0, self.z])
