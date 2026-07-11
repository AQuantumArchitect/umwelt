"""Learning router — world-vs-actuator gating by the observe/actuate regime.

Exogenous (clean) → learn the world; downstream (self-caused) → discount the world,
learn the actuator. The actuation-echo signal bites without H maturity. Pure +
shadow-measurable. See meerkat/brain/learning_router.py.
"""
from __future__ import annotations

from umwelt.learning.learning_router import LearningRouter, LearningGate


def test_gate_splits_world_and_actuator():
    r = LearningRouter()
    clean = r.gate(0.0)
    assert clean.world == 1.0 and clean.actuator == 0.0 and clean.regime == "exogenous"
    caused = r.gate(1.0)
    assert caused.world == 0.0 and caused.actuator == 1.0 and caused.regime == "downstream"
    mid = r.gate(0.5)
    assert mid.world == 0.5 and mid.actuator == 0.5 and mid.regime == "mixed"


def test_world_floor_keeps_a_trickle():
    r = LearningRouter(world_floor=0.2)
    g = r.gate(1.0)
    assert g.world == 0.2          # never fully blind on an always-actuated leaf
    assert g.actuator == 1.0


def test_route_clusters_combines_forecast_and_echo_by_max():
    r = LearningRouter()
    gates = r.route_clusters(
        attribution={"house": 0.1, "exterior": 0.0},
        echoes={"house": 0.9, "kasa_mirror_light": 0.7},  # echo dominates house; echo-only leaf
    )
    assert gates["house"].downstream == 0.9          # max(0.1, 0.9)
    assert gates["exterior"].downstream == 0.0       # weather: never self-caused
    assert "kasa_mirror_light" in gates              # echo-only cluster still gated
    assert abs(gates["kasa_mirror_light"].world - 0.3) < 1e-9   # 1 - 0.7


def test_echo_likelihood_fresh_and_close_is_confounded():
    # we dispatched 0.50; device reports 0.50, 1s ago, tol 0.1, recent window 30s
    e = LearningRouter.echo_likelihood(0.50, 0.50, age_s=1.0, tol=0.1, recent_s=30.0)
    assert e > 0.9                                   # near-exact echo, fresh → confounded


def test_echo_likelihood_stale_or_far_is_independent():
    assert LearningRouter.echo_likelihood(0.5, 0.5, age_s=60.0, tol=0.1, recent_s=30.0) == 0.0  # stale
    assert LearningRouter.echo_likelihood(0.5, 0.9, age_s=1.0, tol=0.1, recent_s=30.0) == 0.0   # far (Δ≥tol)
    assert LearningRouter.echo_likelihood(0.5, None, age_s=1.0, tol=0.1, recent_s=30.0) == 0.0  # no dispatch


def test_shadow_summary_reports_confounded_fraction():
    r = LearningRouter()
    gates = r.route_clusters({"house": 0.8, "exterior": 0.0, "studio": 0.4})
    s = LearningRouter.shadow_summary(gates)
    assert s["n"] == 3
    # confounded = 1-world = downstream here: (0.8 + 0.0 + 0.4)/3 = 0.4
    assert abs(s["confounded_fraction"] - 0.4) < 1e-9
    assert s["downstream"][0][0] == "house"          # most confounded first
    assert ("exterior" not in [d[0] for d in s["downstream"]])  # 0 downstream filtered out


def test_shadow_summary_empty():
    assert LearningRouter.shadow_summary({})["confounded_fraction"] == 0.0


def test_actuate_level_gates_confound_attribution():
    # the agency read: while LISTENING (|act⟩→0) nothing is downstream-of-us → world-learning full,
    # even on a cluster the forecast called fully self-caused.
    r = LearningRouter()
    full_act = r.route_clusters({"house": 1.0}, {"house": 1.0}, actuate_level=1.0)
    assert full_act["house"].world == 0.0 and full_act["house"].downstream == 1.0   # acting → confounded
    listening = r.route_clusters({"house": 1.0}, {"house": 1.0}, actuate_level=0.0)
    assert listening["house"].world == 1.0 and listening["house"].downstream == 0.0  # listening → clean
    half = r.route_clusters({"house": 1.0}, {}, actuate_level=0.5)
    assert abs(half["house"].downstream - 0.5) < 1e-9                                # scales linearly


# ── tendril dispatch-echo (the actuation-echo source) ────────────────────────
# The cluster/role a dispatch confounds is now GRAPH-derived (confounding.actuator_confounding
# keyed by the tendril's graph_node); the tendril only reports ITS OWN dispatch recency. See
# tests/brain/test_confounding.py for the graph-derivation + the decay-curve test.

def test_tendril_base_no_dispatch_echo():
    """A tendril with no graph node + no dispatch tracked contributes no confounding echo."""
    from umwelt.membranes.tendril import Tendril
    t = Tendril()
    assert t.graph_node is None
    assert t.last_dispatch_ts() is None
    assert t.dispatch_echo(now_ts=1000.0) is None


# (The origin's concrete-actuator dispatch-echo test referenced a domain actuator (ACActuator) that
# doesn't exist in umwelt; the generic decay curve is covered by the Tendril subclass test in
# test_confounding.py::test_tendril_dispatch_echo_decays_over_window_and_is_none_when_idle.)
