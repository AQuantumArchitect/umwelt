"""Structure selection — the offspring, the variable-rate womb, and the grow↔prune organ, pinned.

Pins the mechanism on canonical candidates: candidate topologies match resolve_zz_pairs; unitary
gestation grows binding and more ticks build more; composition is TC-additive with NO fabricated
cross-binding and reads proxy/flat at scale (the fractal grain law); prune mirrors grow in the
dream-loop's own edge-set currency; and one evolve rung is shadow-only (auto_apply False, the channel
DENIED). The ZZ-conserves-populations limit ships as a passing assertion (EVAL-OWED).
"""
from __future__ import annotations

import numpy as np

from umwelt.substrate import structure_selection as SS
from umwelt.substrate.hamiltonian import resolve_zz_pairs
from umwelt.substrate.higher_order import cluster_higher_order, total_correlation_proxy
from umwelt.substrate.information_geometry import constellations


_R = ["a", "b", "c"]


def _maxc(c) -> float:
    return total_correlation_proxy(c)[1]


def _tc(c) -> float:
    return total_correlation_proxy(c)[0]


# ── genotypes speak resolve_zz_pairs ──────────────────────────────────────────────

def test_candidate_edges_match_resolve_zz_pairs():
    """Each family's connectivity spec resolves to the intended edge-set through the same seam the
    live H-basis uses — so a candidate's couplings and its H-tower stay consistent."""
    g = SS.candidate_topologies(_R)
    assert resolve_zz_pairs(g["dense"], 3, _R) == [(0, 1), (0, 2), (1, 2)]
    assert resolve_zz_pairs(g["chain"], 3, _R) == [(0, 1), (1, 2)]
    assert resolve_zz_pairs(g["star"], 3, _R) == [(0, 1), (0, 2)]
    assert resolve_zz_pairs(g["ring"], 3, _R) == [(0, 1), (0, 2), (1, 2)]     # ring==dense at n=3
    assert resolve_zz_pairs(g["empty"], 3, _R) == []


# ── gestation: unitary development grows binding, variable-rate builds more ────────

def test_develop_grows_binding_and_is_variable_rate():
    """Gestation binds: a coupled candidate's ‖C‖ grows from ~0; and MORE ticks build MORE — the
    variable-rate lever (learn hard and fast where cadence is spent)."""
    seed = SS.default_seed(2)
    slow = SS.build_candidates(["a", "b"], {"x": None}, seed_e1=seed)["x"]
    fast = SS.build_candidates(["a", "b"], {"x": None}, seed_e1=seed)["x"]
    assert _maxc(slow) < 1e-9                                  # a fresh coupled candidate is product
    SS.develop(slow, 20)
    SS.develop(fast, 80)
    assert _maxc(slow) > 1e-3                                  # gestation built binding
    assert _maxc(fast) > _maxc(slow)                           # more ticks → more development


def test_zz_only_conserves_populations():
    """EVAL-OWED limit, as a passing assertion: ZZ commutes with σz, so a ZZ-only candidate conserves
    populations exactly — real z-differentiation needs exchange (xy) or the coupling learner."""
    seed = SS.default_seed(2)
    zz_only = SS.build_candidates(["a", "b"], {"x": None}, xy=0.0, seed_e1=seed)["x"]
    z0 = zz_only.e1[:, 2].copy()
    SS.develop(zz_only, 80)
    assert float(np.max(np.abs(zz_only.e1[:, 2] - z0))) < 1e-12


def test_topologies_differentiate():
    """Denser genotypes develop more binding at equal cadence — topology is a real phenotype axis:
    dense > chain/star > empty(=0)."""
    seed = SS.default_seed(3)
    cands = SS.build_candidates(_R, SS.candidate_topologies(_R), seed_e1=seed)
    for c in cands.values():
        SS.develop(c, 60)
    assert _tc(cands["dense"]) > _tc(cands["chain"]) > _tc(cands["empty"])
    assert _tc(cands["empty"]) == 0.0                          # no edges → no binding, ever


# ── offspring: composition is honest at scale ─────────────────────────────────────

def _two_developed_winners():
    seed = SS.default_seed(2)
    w1 = SS.build_candidates(["a", "b"], {"x": None}, seed_e1=seed)["x"]
    w2 = SS.build_candidates(["a", "b"], {"x": None}, zz=0.12, xy=0.09, seed_e1=seed)["x"]
    SS.develop(w1, 60)
    SS.develop(w2, 60)
    return w1, w2


def test_offspring_tc_additivity():
    """Composition fabricates NO cross-binding: composite TC = Σ winners' intra-TC (cross-blocks are
    the product e1⊗e1 → 0 added). No free lunch — the offspring is exactly its parents welded."""
    w1, w2 = _two_developed_winners()
    tc1, tc2 = _tc(w1), _tc(w2)
    off = SS.compose_offspring({"w1": w1, "w2": w2})
    assert tc1 > 0.0 and tc2 > 0.0                             # non-vacuous
    assert abs(_tc(off) - (tc1 + tc2)) < 1e-9


def test_offspring_is_proxy_and_flat():
    """The fractal grain law: the composed offspring is a cumulant → proxy tier, grain='flat'. Small
    clusters can name their grain; the big offspring runs full and is grain-blind (L136 at scale)."""
    w1, w2 = _two_developed_winners()
    off = SS.compose_offspring({"w1": w1, "w2": w2})
    ho = cluster_higher_order(off)
    assert ho["tier"] == "proxy" and ho["grain"] == "flat"


def test_offspring_constellations_preserved():
    """Each winner stays its own bound community in the offspring — global role names, no cross-merge."""
    w1, w2 = _two_developed_winners()
    off = SS.compose_offspring({"w1": w1, "w2": w2})
    assert off.qubit_roles == ["w1\x1fa", "w1\x1fb", "w2\x1fa", "w2\x1fb"]
    comps = constellations(off)
    assert comps == [["w1\x1fa", "w1\x1fb"], ["w2\x1fa", "w2\x1fb"]]


# ── the grow↔prune mirror (the elevation of the dream-loop) ───────────────────────

def test_prune_mirrors_grow():
    """prune_to_constellations reads a developed manifold and returns the sparse edge-set of pairs that
    actually bind — the dream-loop's growth, mirrored: two independent pairs prune to exactly their two
    edges (the cross pairs dropped); a fully-bound cluster keeps every edge."""
    two_pair = SS.build_candidates(["a", "b", "c", "d"], {"x": [(0, 1), (2, 3)]},
                                   seed_e1=SS.default_seed(4))["x"]
    SS.develop(two_pair, 60)
    assert SS.prune_to_constellations(two_pair) == [(0, 1), (2, 3)]
    all_bound = SS.build_candidates(["a", "b", "c", "d"], {"x": None}, seed_e1=SS.default_seed(4))["x"]
    SS.develop(all_bound, 60)
    assert SS.prune_to_constellations(all_bound) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


# ── the one rung: shadow-only ──────────────────────────────────────────────────────

def test_evolve_generation_is_shadow_only():
    """One courtship rung: variable-rate cadence, the father picks the bound densest genotype, the
    offspring re-scores at scale, and the channel is DENIED — all shadow-only, NEVER auto-applied."""
    res = SS.evolve_generation(
        _R, demand_fn=lambda n, c: {"dense": 0.9, "chain": 0.5, "star": 0.4, "ring": 0.3}.get(n, 0.05),
        rate_budget=40, k=1)
    assert res.rate_plan["dense"] > res.rate_plan["empty"]        # strugglers get more cadence
    assert res.winners == ["dense"]                              # bound + most-developed wins
    assert res.candidates["dense"]["total_correlation"] > res.candidates["empty"]["total_correlation"]
    assert res.offspring_score["tier"] == "proxy" and res.offspring_score["grain"] == "flat"
    assert res.audit["auto_apply"] is False                     # the warden clamp — a live graft is a fork
    assert res.channel["status"] == "DENIED" and res.channel["dispatched"] == 0


def test_degenerate_rung_does_not_crash():
    """A degenerate rung (no winner survives — every candidate n<2) composes to None and scores None,
    not an opaque max()-of-empty crash. compose_offspring of an empty set is None."""
    assert SS.compose_offspring({}) is None
    res = SS.evolve_generation(["solo"])                         # 1 role → no cluster has n≥2
    assert res.offspring is None and res.offspring_score is None
    assert res.audit["auto_apply"] is False
