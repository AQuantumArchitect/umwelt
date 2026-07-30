"""cognifold_trace — the belief-field as a renderable Bloch atlas (a per-register field export).

Pins the contract SpaceWheat's 3D cognifold field renderer consumes: a flat register list whose
geometry is a pure Bloch-Cartesian→spherical transform through the repo's own `substrate.bloch`
atlas (round-trips to float precision), a unified gauge that passes reliability/forecast_skill
through from `host.api.beliefs()` (null when absent), and coupling edges that index valid registers.
Additive: no physics touched. See umwelt/projection/cognifold_trace.py.
"""
from __future__ import annotations

import math

from datetime import datetime, timedelta

from umwelt.host import GameHost
from umwelt.projection import cognifold_trace as C
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec, OutputSpec
from tests.test_spec_to_field import tiny_grid_spec


# ── (a) the geometric transform round-trips against bloch.py ──────────────────

# A spread of Bloch vectors: poles, equatorial axes, mixed sub-unit states, the
# maximally-mixed origin, and a pure generic direction.
_KNOWN_BLOCH = [
    (0.0, 0.0, 1.0),      # north pole (z up)
    (0.0, 0.0, -1.0),     # south pole
    (1.0, 0.0, 0.0),      # +x equator
    (0.0, 1.0, 0.0),      # +y equator
    (-1.0, 0.0, 0.0),     # -x equator
    (0.3, -0.4, 0.5),     # generic sub-unit
    (-0.2, 0.1, -0.7),    # generic sub-unit, other octant
    (0.1, 0.2, 0.3),      # low-radius mixed
    (0.0, 0.0, 0.0),      # maximally mixed (origin)
    (0.57735, 0.57735, 0.57735),  # pure body diagonal (|r|≈1)
]


def test_geometry_round_trips_against_bloch_atlas():
    """For each known (x,y,z), the emitted {theta,phi,r_bloch,r_xy,p0,p1,purity} reconstruct
    (x,y,z) within 1e-9 — the geometry is an invertible chart, not a lossy summary."""
    for (x, y, z) in _KNOWN_BLOCH:
        g = C.register_geometry(x, y, z)

        # radius/purity agree with the atlas exactly
        from umwelt.substrate.bloch import bloch_radius, qubit_purity
        assert abs(g["r_bloch"] - bloch_radius(x, y, z)) < 1e-12
        assert abs(g["purity"] - float(qubit_purity((x, y, z)))) < 1e-12
        # purity is the PER-REGISTER single-qubit form (1+|r|²)/2
        assert abs(g["purity"] - 0.5 * (1.0 + g["r_bloch"] ** 2)) < 1e-12

        # spherical → cartesian reconstruction
        x_rec = g["r_xy"] * math.cos(g["phi"])
        y_rec = g["r_xy"] * math.sin(g["phi"])
        z_rec = g["r_bloch"] * math.cos(g["theta"])
        assert abs(x_rec - x) < 1e-9, (x, y, z, x_rec)
        assert abs(y_rec - y) < 1e-9, (x, y, z, y_rec)
        assert abs(z_rec - z) < 1e-9, (x, y, z, z_rec)

        # value/probabilities round-trip z too
        assert abs((2.0 * g["p0"] - 1.0) - z) < 1e-9
        assert abs((g["p0"] + g["p1"]) - 1.0) < 1e-12


# ── a seeded, observed world (the export's real substrate) ────────────────────

def _grid_host() -> GameHost:
    h = GameHost()
    h.register_world(tiny_grid_spec(), population=False)
    # drive a couple of cells off the ground state so beliefs (and reliability) are live
    t = datetime(2026, 7, 20, 9, 0)
    for i in range(8):
        h.engine.ingest(
            sensor_readings={"sight_cell_0_0": 1.0, "sight_cell_0_1": 0.0},
            now=t + timedelta(minutes=5 * i),
        )
    return h


def _trace():
    return C.cognifold_trace(_grid_host())


def test_envelope_shape_and_format():
    tr = _trace()
    assert tr["format"] == "cognifold_trace_v1"
    assert tr["world"] == "grid"
    assert tr["n_registers"] == len(tr["registers"]) > 0
    for r in tr["registers"]:
        assert set(("id", "node", "role", "p0", "p1", "r_xy", "r_bloch", "phi", "theta",
                    "purity", "value", "confidence", "reliability", "surprise", "berry_phase",
                    "forecast_skill", "forecast")).issubset(r)
        assert r["value"] == r["p0"] and r["confidence"] == r["r_bloch"]
        # Bloch coords carry the engine's evolution FP drift; match the repo's ±1e-6 bound
        # convention (see tests/test_spec_to_field.py: `abs(v) <= 1.0 + 1e-6`). The export is a
        # faithful pure transform — it does NOT clamp, so the test tolerates the same drift.
        assert -1e-6 <= r["p0"] <= 1.0 + 1e-6 and 0.0 <= r["r_bloch"] <= 1.0 + 1e-6
        assert isinstance(r["forecast"], list)


def test_register_ids_are_dense_and_ordered():
    tr = _trace()
    assert [r["id"] for r in tr["registers"]] == list(range(tr["n_registers"]))


# ── (b) edges index valid registers ──────────────────────────────────────────

def test_edges_index_valid_registers():
    tr = _trace()
    n = tr["n_registers"]
    assert tr["edges"], "the gridworld's bridges should yield at least one edge"
    kinds = {e["kind"] for e in tr["edges"]}
    assert kinds <= {"bridge", "zz", "xy", "mi"}
    for e in tr["edges"]:
        assert 0 <= e["i"] < n and 0 <= e["j"] < n
        assert e["i"] != e["j"]
        assert isinstance(e["weight"], (int, float)) and e["weight"] >= 0.0
    # bridges couple registers on DIFFERENT nodes (cross-cluster)
    node_of = {r["id"]: r["node"] for r in tr["registers"]}
    for e in tr["edges"]:
        if e["kind"] == "bridge":
            assert node_of[e["i"]] != node_of[e["j"]]
        else:  # zz / xy are intra-cluster
            assert node_of[e["i"]] == node_of[e["j"]]


def test_intra_cluster_zz_xy_edges_map_to_registers(monkeypatch):
    """The gridworld's dense clusters carry no learned zz/xy, so real data yields bridge edges
    only. Prove the intra-cluster path anyway: feed a synthetic transparency snapshot with zz/xy
    between two real co-located roles and assert the export emits kind 'zz'/'xy' edges that index
    the correct (same-node) registers."""
    h = _grid_host()
    base = C.cognifold_trace(h)                 # to learn the real register ids
    ids = {(r["node"], r["role"]): r["id"] for r in base["registers"]}
    a, b = ("cell_0_0", "agent_near"), ("cell_0_0", "resource")
    assert a in ids and b in ids

    fake = {"clusters": [{
        "name": "cell_0_0",
        "zz": [{"a": "agent_near", "b": "resource", "j": -0.42}],
        "xy": [{"a": "agent_near", "b": "resource", "kxx": 0.3, "kyy": -0.4}],
    }]}
    monkeypatch.setattr(C._transparency, "model_snapshot", lambda eng: fake)

    tr = C.cognifold_trace(h)
    zz = [e for e in tr["edges"] if e["kind"] == "zz"]
    xy = [e for e in tr["edges"] if e["kind"] == "xy"]
    assert len(zz) == 1 and len(xy) == 1
    assert {zz[0]["i"], zz[0]["j"]} == {ids[a], ids[b]}
    assert abs(zz[0]["weight"] - 0.42) < 1e-9      # |J|
    assert {xy[0]["i"], xy[0]["j"]} == {ids[a], ids[b]}
    assert abs(xy[0]["weight"] - 0.5) < 1e-9        # hypot(0.3, 0.4)


# ── (c) reliability / forecast_skill pass through (null when absent) ──────────

def _gauge_spec() -> DomainSpec:
    return DomainSpec(
        name="d",
        nodes=(NodeSpec("root", parent=None, kind="root", roles=("level",)),
               NodeSpec("a", parent="root", roles=("level",))),
        bindings=(BindingSpec("a_in", zone="a", role="level", normalizer="binary",
                              force_observe=True, collapse_alpha=0.5),),
        outputs=(OutputSpec("a_out", node="a", role="level"),))


def test_reliability_passes_through_and_forecast_skill_null_without_surface():
    h = GameHost()
    h.register_world(_gauge_spec(), population=False)
    t = datetime(2026, 7, 20, 9, 0)
    for i in range(6):
        h.engine.ingest(sensor_readings={"a_in": 1.0}, now=t + timedelta(minutes=5 * i))

    # cross-check the export against the canonical beliefs() gauge it reuses
    belief = h.beliefs()["a.level"]
    tr = C.cognifold_trace(h)
    reg = next(r for r in tr["registers"] if r["node"] == "a" and r["role"] == "level")
    assert reg["reliability"] == belief.reliability
    assert reg["reliability"] is not None            # observed → learned reliability present
    # no forecast surface attached → forecast_skill is cleanly null, forecast list empty
    assert reg["forecast_skill"] is None and belief.forecast_skill is None
    assert reg["forecast"] == []


def test_forecast_ladder_populates_when_surface_attached_and_stepped():
    """The positive counterpart to the null case: attach a ForecastSurface over the leaf and
    step it each ingest with the sim clock, and the trace's per-register `forecast` ladder
    populates with {horizon_min, z_pred, skill} rungs plus a non-null forecast_skill. Pins the
    otherwise-DORMANT forecast path (nothing in build_engine/ingest attaches or steps a surface;
    DEFAULT_FORECAST_LEAVES is empty) so it can't silently rot — this is the exact wiring
    proofs/forecast_walk.py drives to make SpaceWheat's cognifold self-forecast overlay real."""
    from umwelt.foresight.forecast_surface import ForecastSurface

    h = GameHost()
    h.register_world(_gauge_spec(), population=False)
    # shortest single horizon so the delayed label matures quickly within the test's sim span
    h.engine.forecast_surface = ForecastSurface(leaves=(("a", "level"),), horizons_min=(13.0,))
    t = datetime(2026, 7, 20, 9, 0)
    for i in range(14):                                  # 14 * 5min = 70min > 13min horizon
        now = t + timedelta(minutes=5 * i)
        h.engine.ingest(sensor_readings={"a_in": 1.0}, now=now)
        h.engine.forecast_surface.step(now, h.engine.field)

    tr = C.cognifold_trace(h)
    reg = next(r for r in tr["registers"] if r["node"] == "a" and r["role"] == "level")
    assert reg["forecast"], "the shortest horizon should have matured into a prediction"
    for e in reg["forecast"]:
        assert set(("horizon_min", "z_pred", "skill")).issubset(e)
        assert isinstance(e["z_pred"], (int, float))
        assert 0.0 <= float(e["skill"]) <= 1.0
    assert reg["forecast_skill"] is not None            # shortest-horizon skill flows to the gauge


def test_actions_section_wires_decisions_to_driving_registers():
    """The decision layer ("why did it act"): each output tendril surfaces in `actions`, wired by
    `driving_register` to the belief register (node,role) that drives it, with the honest
    shadow/gated fields. It must be a PURE read — building the trace must not step/mutate the
    tendrils. This is the producer half of SpaceWheat's belief→action overlay."""
    h = GameHost()
    h.register_world(_gauge_spec(), population=False)   # _gauge_spec declares OutputSpec("a_out", a, level)
    t = datetime(2026, 7, 20, 9, 0)
    for i in range(6):
        h.engine.ingest(sensor_readings={"a_in": 1.0}, now=t + timedelta(minutes=5 * i))

    tr = C.cognifold_trace(h)
    assert isinstance(tr.get("actions"), list)
    act = next((a for a in tr["actions"] if a["id"] == "a_out"), None)
    assert act is not None, "the a_out output tendril should surface as an action"
    assert act["node"] == "a" and act["role"] == "level"
    dr = act["driving_register"]
    assert dr is not None and tr["registers"][dr]["node"] == "a" and tr["registers"][dr]["role"] == "level"
    # gauge/gridworld outputs are shadow by default → honest "decided, but dispatched nothing"
    assert act["shadow"] is True and "shadow" in act["gates_held"] and act["gated"] is True
    for k in ("value", "committed_level", "confidence", "decode", "kind"):
        assert k in act
    # pure read: a second trace is identical (no tendril mutation between calls)
    assert C.cognifold_trace(h)["actions"] == tr["actions"]


def test_mi_edges_surface_real_correlation():
    """The MI-sensor edges: a cumulant cluster carrying real connected correlation yields a
    kind='mi' edge wired to the right registers; a product cluster yields none (gated, ~free)."""
    import numpy as np

    class _FakeCumulant:
        qubit_roles = ["a", "b"]

        def __init__(self, correlate):
            self.e1 = np.zeros((2, 3))
            self.e2 = np.zeros((2, 2, 3, 3))
            if correlate:
                self.e2[0, 1, 2, 2] = 0.6      # connected <zz> between a and b
                self.e2[1, 0, 2, 2] = 0.6

    class _FakeEng:
        def __init__(self, correlate):
            self.field = type("F", (), {"clusters": {"n": _FakeCumulant(correlate)}})()

    reg_id = {("n", "a"): 0, ("n", "b"): 1}
    edges = C._mi_edges(_FakeEng(correlate=True), reg_id)
    assert len(edges) == 1 and edges[0]["kind"] == "mi"
    assert {edges[0]["i"], edges[0]["j"]} == {0, 1} and edges[0]["weight"] > 0.0
    assert C._mi_edges(_FakeEng(correlate=False), reg_id) == []   # product → gated, no edge


def test_berry_phase_is_a_float_after_the_engine_has_ticked():
    """Every register on a ticked engine carries a float `berry_phase`: engine.ingest advances
    engine.bloch_berry per qubit (keyed "cluster:qubit_index"), and the trace looks each register
    up through the cluster's role_index. γ may legitimately be 0.0 (a purely polar trajectory
    encloses no solid angle) — the pin is presence + type, and agreement with the tape itself."""
    h = _grid_host()
    tr = C.cognifold_trace(h)
    tape = h.engine.bloch_berry.phases
    assert tape, "8 ingests should have sampled the Berry tape"
    for r in tr["registers"]:
        assert isinstance(r["berry_phase"], float), (r["node"], r["role"], r["berry_phase"])
    # spot-check one register against the tape directly (role_index → "cluster:idx" key)
    reg = tr["registers"][0]
    cluster = h.engine.field.clusters[reg["node"]]
    key = f"{reg['node']}:{cluster.role_index[reg['role']]}"
    assert reg["berry_phase"] == tape[key]


def test_berry_phase_null_when_tape_absent():
    """A foreign/berry-less engine (bloch_berry = None) degrades to null berry_phase on every
    register — the lookup guard, not an exception."""
    h = _grid_host()
    h.engine.bloch_berry = None
    tr = C.cognifold_trace(h)
    assert tr["n_registers"] > 0
    assert all(r["berry_phase"] is None for r in tr["registers"])


def test_surprise_is_innov_ema_on_observed_leaf_and_null_when_never_observed():
    """`surprise` is the leaf's learned innovation EMA (reliability's un-clipped sibling — the
    same observation-trust snapshot whose alpha IS reliability): a float on an observed leaf,
    matching `_obs_trust.snapshot()` verbatim, and null everywhere on a never-observed engine
    (lazy `_obs_trust` is None — same guard as host.api.beliefs())."""
    h = GameHost()
    h.register_world(_gauge_spec(), population=False)
    t = datetime(2026, 7, 20, 9, 0)
    for i in range(6):
        h.engine.ingest(sensor_readings={"a_in": 1.0}, now=t + timedelta(minutes=5 * i))

    tr = C.cognifold_trace(h)
    reg = next(r for r in tr["registers"] if r["node"] == "a" and r["role"] == "level")
    assert isinstance(reg["surprise"], float)
    snap = h.engine._obs_trust.snapshot()
    assert reg["surprise"] == snap["a.level"]["innov_ema"]
    # both trust coordinates ride together: reliability (alpha) alongside surprise (innov_ema)
    assert reg["reliability"] is not None

    # never-observed engine: no trust ledger at all → surprise cleanly null on every register
    cold = GameHost()
    cold.register_world(_gauge_spec(), population=False)
    tr_cold = C.cognifold_trace(cold)
    assert tr_cold["n_registers"] > 0
    assert all(r["surprise"] is None for r in tr_cold["registers"])


def test_never_observed_leaf_has_null_reliability():
    """A leaf that was never observed carries null reliability — the gauge is honest about
    'I have no trust estimate here' rather than fabricating one."""
    h = GameHost()
    h.register_world(_gauge_spec(), population=False)   # boot, no observations
    tr = C.cognifold_trace(h)
    reg = next(r for r in tr["registers"] if r["node"] == "a" and r["role"] == "level")
    assert reg["reliability"] is None


# ── robustness: a raw engine (worker path) and a broken engine ────────────────

def test_accepts_a_raw_engine_and_matches_host_route():
    h = _grid_host()
    from_host = C.cognifold_trace(h)
    from_engine = C.cognifold_trace(h.engine, world="grid")
    assert from_engine["n_registers"] == from_host["n_registers"]
    assert [r["id"] for r in from_engine["registers"]] == [r["id"] for r in from_host["registers"]]


def test_broken_engine_degrades_to_empty_envelope():
    tr = C.cognifold_trace(object())      # not an engine, not a host
    assert tr["format"] == "cognifold_trace_v1"
    assert tr["n_registers"] == 0 and tr["registers"] == [] and tr["edges"] == []
    assert tr["manifold"] == {"clusters": []}


# ── (f) the manifold section: higher-order shape + grain + constellations ─────

def test_manifold_section_present_and_additive():
    """The manifold section rides on a real trace WITHOUT disturbing the existing contract: the edge
    allowlist is intact (it's a section, not a new edge kind), every register gains a `constellation`
    field (safe under the .issubset register-shape test), and any manifold cluster is well-formed."""
    tr = _trace()
    assert isinstance(tr["manifold"], dict) and isinstance(tr["manifold"]["clusters"], list)
    assert {e["kind"] for e in tr["edges"]} <= {"bridge", "zz", "xy", "mi"}   # no new edge kind
    for r in tr["registers"]:
        assert "constellation" in r                                          # additive register field
        assert r["constellation"] is None or isinstance(r["constellation"], int)
    for m in tr["manifold"]["clusters"]:
        assert set(m) >= {"name", "roles", "n", "total_correlation", "o_information",
                          "grain", "tier", "max_corr_norm", "constellations", "metric"}
        assert m["grain"] in {"chorus", "conspiracy", "flat"}
        assert m["tier"] in {"gated", "proxy", "exact"}
        dist, roles = m["metric"]["distance"], m["metric"]["roles"]
        assert len(dist) == len(roles) and all(len(row) == len(roles) for row in dist)


def test_manifold_section_binds_correlated_cumulant():
    """The section surfaces a cluster's higher-order structure: a cumulant with a real correlated pair
    reports total_correlation>0 at the proxy tier — o_information None (it feels the bind but cannot
    sign the grain) — and the bound roles share a constellation; a product cluster reports zero, flat."""
    import numpy as np

    class _FakeCumulant:
        def __init__(self, correlate):
            self.qubit_roles = ["a", "b", "c"]
            self.n_qubits = 3
            self.e1 = np.zeros((3, 3))
            self.e2 = np.zeros((3, 3, 3, 3))
            if correlate:
                self.e2[0, 1, 2, 2] = 0.6                # connected <zz> a~b; c stays product
                self.e2[1, 0, 2, 2] = 0.6

    class _FakeEng:
        def __init__(self, correlate):
            self.field = type("F", (), {"clusters": {"pantry": _FakeCumulant(correlate)}})()

    (m,) = C._manifold_clusters(_FakeEng(correlate=True))
    assert m["name"] == "pantry" and m["n"] == 3
    assert m["total_correlation"] > 0.0 and m["tier"] == "proxy"
    assert m["o_information"] is None and m["grain"] == "flat"
    assert ["a", "b"] in m["constellations"] and ["c"] in m["constellations"]
    dist, roles = m["metric"]["distance"], m["metric"]["roles"]
    n = len(roles)
    assert len(dist) == n and all(len(row) == n for row in dist)
    assert all(abs(dist[i][i]) < 1e-9 for i in range(n))
    assert all(abs(dist[i][j] - dist[j][i]) < 1e-9 for i in range(n) for j in range(n))

    (prod,) = C._manifold_clusters(_FakeEng(correlate=False))
    assert prod["total_correlation"] == 0.0 and prod["grain"] == "flat"
