"""Compute demand — the womb's metabolism, pinned on canonical states.

Pins the honest emitter: struggle (PROXY, source-labelled), the backend-aware cost ESTIMATE, the
field deficit sign, the surprise-driven rate policy, and the DENIED channel kept honest with numbers
(dispatched:0, 0 bytes leave the process). Duck-typed cumulant/dense fixtures in the test_higher_order
idiom — nothing here touches a live field or the integrator.
"""
from __future__ import annotations

import numpy as np

from umwelt.substrate import compute_demand as CD


class _Cum:
    """Minimal cumulant cluster: e1 (n,3) + e2, plus a purity() the real CumulantCluster exposes."""

    def __init__(self, roles, e1=None):
        n = len(roles)
        self.qubit_roles = list(roles)
        self.n_qubits = n
        self.e1 = np.zeros((n, 3)) if e1 is None else np.asarray(e1, float).reshape(n, 3)
        self.e2 = np.zeros((n, n, 3, 3))

    def purity(self) -> float:
        r2 = np.clip(np.sum(self.e1 * self.e1, axis=1), 0.0, 1.0)
        return float(np.mean((1.0 + r2) / 2.0))


def _sharp(roles):
    """Every belief certain (|r|=1 along +z) → mixedness 0."""
    e1 = np.zeros((len(roles), 3))
    e1[:, 2] = 1.0
    return _Cum(roles, e1=e1)


def _mixed(roles):
    """Every belief maximally uncertain (r=0) → mixedness 1."""
    return _Cum(roles)


class _Dense:
    """Minimal dense cluster: n qubits, a joint entropy in bits, no e1/e2 → 'dense' backend + entropy
    mixedness fallback."""

    def __init__(self, roles, entropy):
        self.qubit_roles = list(roles)
        self.n_qubits = len(roles)
        self._entropy = float(entropy)

    @property
    def entropy(self) -> float:
        return self._entropy


def _field(clusters: dict):
    return type("F", (), {"clusters": clusters})()


# ── struggle: soft-OR(mixedness, surprise), honestly sourced ──────────────────────

def test_mixed_surprised_high_struggle():
    """A maximally-mixed, still-erring cluster is learning hard → high struggle, source='surprise'."""
    got = CD.cluster_struggle(_mixed(["a", "b"]), role_surprise={"a": 0.9, "b": 0.9})
    assert got["struggle"] > 0.8
    assert got["source"] == "surprise" and got["surprise"] is not None
    assert abs(got["mixedness"] - 1.0) < 1e-9


def test_settled_low_struggle():
    """A sharp, un-surprised cluster is settled → struggle ≈ 0 (soft-OR of zero mixedness and zero surprise)."""
    got = CD.cluster_struggle(_sharp(["a", "b"]), role_surprise={"a": 0.0, "b": 0.0})
    assert got["struggle"] < 1e-6
    assert got["source"] == "surprise"


def test_struggle_handles_none_surprise():
    """No live surprise → mixedness stand-in, and the source names the blind spot so nothing overclaims."""
    got = CD.cluster_struggle(_mixed(["a", "b"]), role_surprise=None)
    assert abs(got["struggle"] - 1.0) < 1e-9        # mixedness alone
    assert got["surprise"] is None and got["source"] == "purity_proxy"


def test_struggle_monotone_in_surprise():
    """At a SHARP belief (mixedness 0), surprise is the sole driver — the soft-OR lets a certain-but-
    erring cluster still earn cadence, which is exactly where the womb must spend ticks. Monotone."""
    lo = CD.cluster_struggle(_sharp(["a"]), role_surprise=0.1)["struggle"]
    hi = CD.cluster_struggle(_sharp(["a"]), role_surprise=0.9)["struggle"]
    assert abs(lo - 0.1) < 1e-9 and abs(hi - 0.9) < 1e-9        # struggle == surprise at zero mixedness
    assert hi > lo


# ── cost: the tier economics, backend-aware ───────────────────────────────────────

def test_cost_backend_aware():
    """Dense (2^n)^2 dwarfs cumulant n^3 at equal n — the whole reason the closure exists."""
    dense = CD.compute_cost_estimate(_Dense(list("abcde"), 3.0))
    cum = CD.compute_cost_estimate(_mixed(list("abcde")))
    assert dense["backend"] == "dense" and cum["backend"] == "cumulant"
    assert dense["cost"] == float((2 ** 5) ** 2) and cum["cost"] == float(5 ** 3)
    assert dense["cost"] > cum["cost"]


# ── field demand + deficit ─────────────────────────────────────────────────────────

def test_deficit_math_and_sign():
    """Demand concentrates on the struggler; against a small budget the deficit is positive and its
    ratio sits in [0,1] — the honest 'this CPU cannot serve it' signal."""
    field = _field({"hot": _mixed(["a", "b"]), "cold": _sharp(["c", "d"])})
    surprise = {"hot": 0.9, "cold": 0.0}
    dem = CD.field_demand(field, role_surprise_fn=lambda name, c: surprise[name], budget=1.0)
    assert dem["per_cluster"]["hot"]["demand"] > dem["per_cluster"]["cold"]["demand"]
    assert dem["per_cluster"]["cold"]["demand"] < 1e-9      # settled → no demand
    # hot struggle 1 × cost 2^3=8 = 8; cold 0. total 8, budget 1 → deficit 7, ratio 7/8 (pins the formula,
    # not just the clamp range).
    assert abs(dem["field_demand"] - 8.0) < 1e-9 and abs(dem["deficit"] - 7.0) < 1e-9
    assert abs(dem["deficit_ratio"] - 0.875) < 1e-9


def test_field_demand_guards_broken_cluster():
    """A cluster that raises on read is skipped, never crashes the scan (field_higher_order_scan idiom)."""
    class _Bad:
        qubit_roles = ["x", "y"]
        @property
        def entropy(self):
            raise RuntimeError("boom")
        def purity(self):
            raise RuntimeError("boom")
    field = _field({"ok": _mixed(["a"]), "bad": _Bad()})
    dem = CD.field_demand(field, budget=None)
    assert "ok" in dem["per_cluster"] and "bad" not in dem["per_cluster"]


# ── rate plan: surprise-driven cadence, budget-conserving ─────────────────────────

def test_rate_plan_prefers_strugglers_and_conserves_budget():
    """More demand → at least as many ticks (monotone); every value in [floor,cap]; Σ ≤ budget."""
    demand = {"per_cluster": {"hot": {"demand": 10.0}, "warm": {"demand": 3.0}, "cold": {"demand": 0.0}}}
    plan = CD.rate_plan(demand, floor=1, cap=16, budget=20)
    assert plan["hot"] >= plan["warm"] >= plan["cold"]
    assert plan["cold"] == 1                       # floor for the settled
    assert all(1 <= v <= 16 for v in plan.values())
    assert sum(plan.values()) <= 20                # budget-conserving


def test_rate_plan_unbudgeted_scales_to_cap():
    """With no budget the top struggler rides to the cap, the floor stays the floor — pure need shape."""
    demand = {"hot": 10.0, "cold": 0.0}
    plan = CD.rate_plan(demand, floor=1, cap=8)
    assert plan["hot"] == 8 and plan["cold"] == 1


# ── the DENIED channel, kept honest with numbers ──────────────────────────────────

def test_channel_is_denied():
    """The emitter names what it WOULD offload and the flops saved — then dispatches 0, 0 bytes leave.
    The remote channel stays a fork; the demand for it is real and computed."""
    field = _field({"hot": _mixed(["a", "b", "c"]), "cold": _sharp(["d"])})
    dem = CD.field_demand(field, role_surprise_fn=lambda n, c: 0.9 if n == "hot" else 0.0)
    man = CD.channel_offload_manifest(dem)
    assert man["status"] == "DENIED"
    assert man["dispatched"] == 0 and man["bytes_left_process"] == 0
    assert man["would_offload"] == ["hot"]          # the struggler
    assert man["est_saving_flops"] == 27.0          # exactly hot's cost 3^3 (pins the aggregation)
