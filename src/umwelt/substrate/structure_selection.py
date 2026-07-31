"""Structure selection — the offspring, the variable-rate womb, and the grow↔prune developmental organ.

This closes the courtship-and-gestation braid, all OFFLINE on candidate cumulant clusters — it never
touches field.step or engine.ingest. It is shadow-only: it returns a promotion RECOMMENDATION and an
audit, and mutates no live field. It learns the way umwelt learns, and elevates it:

  GESTATION (develop)  — CumulantCluster.free_run at a demand-driven cadence: a struggling candidate
      gets more cheap Python ticks, a settled one the floor. "Learn hard and fast" as running code,
      the integrator untouched. This is the membrane's surprise-cadence, turned inward on candidates.
  COURTSHIP (select)   — the father ranks the developed structures by the mother-math fitness under a
      compute penalty; SHADOW-first, never-auto-apply, the warden's clamp — value is a rook fork.
  OFFSPRING (compose)  — winners welded via field_unify.pack→to_manifold into ONE larger connected
      cluster, re-scored by the SAME sensor at the larger scale (fractal self-similarity). Composition
      fabricates NO cross-binding: cross-cluster e2 is the product, so composite TC = Σ winners' TC.
      And the offspring is cumulant/proxy → grain='flat': small clusters can name their grain, the big
      offspring runs full and is grain-blind — the DENIED, DENSE-ONLY law reborn as a law of SCALE.

The elevation over umwelt's dream-loop (which only GROWS couplings where surprise demands) is the
mirror: `prune_to_constellations` reads a developed manifold and hands the sparse edge-set of pairs
that actually bind back through the SAME `resolve_zz_pairs` seam the dream-loop grows into. Grow-where-
surprised and prune-where-independent are one organ seen from two directions.

Honest limit (EVAL-OWED, pinned as a passing assertion): ZZ couplings commute with σz and CONSERVE
populations, so a blank ZZ-only candidate does not differentiate under free_run — real differentiation
needs exchange (xy) couplings or the coupling learner. This pass develops candidates that carry xy
exchange from a seeded coherent state; fitting couplings to a real predictive contrast (events.db) is
the next rung.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field

import numpy as np

from umwelt.substrate.cumulant_cluster import CumulantCluster
from umwelt.substrate import field_unify
from umwelt.substrate.mutual_information import DEFAULT_FLOOR, DEFAULT_STRONG, _pair_stats, mi_pair_bits
from umwelt.substrate.compute_demand import (
    cluster_struggle, compute_cost_estimate, rate_plan, field_demand, channel_offload_manifest,
)
from umwelt.substrate.manifold_fitness import structure_fitness, select, DEFAULT_WEIGHTS


# ── genotypes: candidate topologies ───────────────────────────────────────────────

def candidate_topologies(roles, *, families=("dense", "chain", "star", "ring", "empty")) -> dict:
    """Enumerate connectivity genotypes: family → a spec `resolve_zz_pairs` understands (None=dense
    all-pairs, 'chain', an explicit edge-set, or [] = no couplings). Deterministic."""
    n = len(roles)
    out: dict[str, object] = {}
    for fam in families:
        if fam == "dense":
            spec: object = None
        elif fam == "chain":
            spec = "chain"
        elif fam == "star":
            spec = [(0, i) for i in range(1, n)]
        elif fam == "ring":
            spec = [(i, (i + 1) % n) for i in range(n)] if n > 2 else [(i, i + 1) for i in range(n - 1)]
        elif fam == "empty":
            spec = []
        else:
            continue
        out[fam] = spec
    return out


def default_seed(n: int, *, tilt: float = 0.4) -> np.ndarray:
    """A pure, coherent seed state (each belief tilted into x) so exchange couplings have something to
    act on under free_run — otherwise a ground-state ZZ candidate would conserve populations and stand
    still. |r|=1 (pure), coherence in the x-component."""
    e1 = np.zeros((n, 3))
    e1[:, 0] = tilt
    e1[:, 2] = float(np.sqrt(max(0.0, 1.0 - tilt * tilt)))
    return e1


def build_candidates(roles, genotypes: dict, *, gamma: float = 0.0, gamma_diss: float = 0.0,
                     dt: float = 0.1, zz: float = 0.2, xy: float = 0.15, seed_e1=None) -> dict:
    """One CumulantCluster per genotype, its allowed edges carrying (zz, xy) couplings — the phenotype
    capacity the genotype develops into. Optional `seed_e1` (n,3) installs the initial coherent state
    (else ground state). Gestation is UNITARY by default (gamma=gamma_diss=0): offline vessels develop
    under their own H so the topology's couplings actually build correlation — a dissipative candidate
    would relax to the mixed origin before any binding formed. Nothing here is a live field."""
    out: dict[str, CumulantCluster] = {}
    for name, spec in genotypes.items():
        c = CumulantCluster(name, list(roles), gamma=gamma, dt=dt, gamma_diss=gamma_diss, connectivity=spec)
        if c._zz_pairs:
            c.set_couplings(zz={p: zz for p in c._zz_pairs}, xy={p: xy for p in c._zz_pairs})
        if seed_e1 is not None:
            c.e1[:] = np.asarray(seed_e1, float).reshape(c.n_qubits, 3)
            c._sync_e2_product()
        out[name] = c
    return out


# ── gestation: variable-rate offline development ──────────────────────────────────

def develop(cluster, n_steps, *, dt_scale: float = 1.0) -> None:
    """Gestate one candidate: free_run at n_steps (from rate_plan) × dt_scale. Variable-rate — the
    strugglers get more cheap ticks. Offline on the candidate itself; the live integrator is untouched."""
    cluster.free_run(int(max(0, n_steps)), dt_scale=float(dt_scale))


# ── the grow↔prune mirror (the elevation of umwelt's dream-loop) ──────────────────

def prune_to_constellations(cluster, *, bind_floor: float = DEFAULT_FLOOR,
                            strong: float = DEFAULT_STRONG) -> list:
    """Read a developed manifold and return the sparse connectivity of pairs that actually BIND — the
    mirror of the dream-loop's growth, in the same currency (a `resolve_zz_pairs` edge-set). Pairs
    below the MI floor are dropped (independent → decoupled). Feed the result back as `connectivity`
    to build a cheaper cluster that keeps only what matters."""
    roles = list(getattr(cluster, "qubit_roles", ()) or ())
    n = len(roles)
    kept = []
    for i in range(n):
        for j in range(i + 1, n):
            e1i, e1j, e2ij = _pair_stats(cluster, i, j)
            mi, _cn, _tier = mi_pair_bits(e1i, e1j, e2ij, floor=bind_floor, strong=strong)
            if mi > 0.0:
                kept.append((i, j))
    return kept


# ── offspring: composition, re-scored at the larger scale ─────────────────────────

def compose_offspring(winners, *, node: str = "offspring") -> CumulantCluster:
    """Weld the winning structures into ONE connected cumulant cluster over their shared qubits (the
    '1-matrix' offspring), re-scoreable by the SAME sensors at the larger scale. `winners` is a
    {name: cluster} dict (or a list, keyed by zone_name). Cross-cluster e2 is the product → no
    fabricated cross-binding; the composite is proxy tier → grain-blind by construction."""
    if not isinstance(winners, dict):
        winners = {getattr(c, "zone_name", f"w{k}"): c for k, c in enumerate(winners)}
    if not winners:
        return None                       # nothing to compose (degenerate rung: no winner survived)
    canon = field_unify.pack(winners)
    return field_unify.to_manifold(canon, node=node)


# ── the one rung ───────────────────────────────────────────────────────────────────

@dataclass
class SelectionResult:
    """The audit of one courtship-and-gestation rung. Shadow-only — `offspring` is a recommendation,
    never grafted into a live field here."""
    roles: list
    rate_plan: dict
    candidates: dict                      # name -> fitness scorecard (+ rate, struggle)
    winners: list
    offspring: object                     # CumulantCluster
    offspring_score: dict
    field_demand: dict
    channel: dict
    audit: dict = _dc_field(default_factory=dict)


def evolve_generation(roles, *, families=("dense", "chain", "star", "ring", "empty"),
                      weights=None, demand_fn=None, rate_budget=None, seed_e1=None,
                      cost_weight: float = 0.0, k: int = 1, gamma: float = 0.0, gamma_diss: float = 0.0,
                      dt: float = 0.1, zz: float = 0.2, xy: float = 0.15, floor: float = DEFAULT_FLOOR,
                      strong: float = DEFAULT_STRONG, rate_floor: int = 1, rate_cap: int = 16) -> SelectionResult:
    """Build candidate topologies → develop them at a surprise-driven variable rate → score by the
    mother-math fitness → the father selects k winners → compose the offspring → re-score at scale.
    Entirely offline and shadow-only: no live field.step / engine.ingest, no live graft. `demand_fn(name,
    cluster)` supplies per-candidate surprise (else struggle rides mixedness alone → uniform cadence)."""
    weights = weights or DEFAULT_WEIGHTS
    roles = list(roles)
    if seed_e1 is None:
        seed_e1 = default_seed(len(roles))

    genotypes = candidate_topologies(roles, families=families)
    candidates = build_candidates(roles, genotypes, gamma=gamma, gamma_diss=gamma_diss,
                                  dt=dt, zz=zz, xy=xy, seed_e1=seed_e1)

    # struggle → demand → cadence: umwelt's surprise currency, allocating development effort.
    demand_map: dict[str, dict] = {}
    for name, c in candidates.items():
        rs = demand_fn(name, c) if demand_fn is not None else None
        struggle = cluster_struggle(c, role_surprise=rs)["struggle"]
        cost = compute_cost_estimate(c)["cost"]
        demand_map[name] = {"demand": struggle * cost, "struggle": struggle, "cost": cost}
    rates = rate_plan(demand_map, floor=rate_floor, cap=rate_cap, budget=rate_budget)

    # gestation (offline) — strugglers develop harder/faster.
    for name, c in candidates.items():
        develop(c, rates.get(name, rate_floor))

    # the father: rank the developed structures by pure fitness, priced by the cost_weight scarcity knob.
    cost_of = lambda c: compute_cost_estimate(c)["cost"]
    sel = select(candidates, cost_weight=cost_weight, cost_of=cost_of, weights=weights,
                 demand_of=None, floor=floor, strong=strong, k=k)
    winners = sel["chosen"]

    # Fitness is PURE structure quality — struggle drove the development RATE, and compute scarcity
    # enters only through select()'s cost_weight (raw FLOPs), so it isn't double-counted or swamped by
    # a FLOP-scale penalty inside the [0,1] structural terms.
    scored: dict[str, dict] = {}
    for name, c in candidates.items():
        fit = structure_fitness(c, weights=weights, demand=0.0, floor=floor, strong=strong)
        if fit is not None:
            scored[name] = {**fit, "rate": rates.get(name, rate_floor),
                            "struggle": demand_map[name]["struggle"]}

    # offspring: compose winners, re-score at the larger scale (fractal self-similarity). A degenerate
    # rung (no winner survived — e.g. every candidate n<2) composes to None and scores None, not a crash.
    winner_clusters = {name: candidates[name] for name in winners}
    offspring = compose_offspring(winner_clusters, node="offspring")
    offspring_score = (structure_fitness(offspring, weights=weights, floor=floor, strong=strong)
                       if offspring is not None else None)

    # field-level demand + the DENIED channel signal (post-develop).
    fake_field = type("F", (), {"clusters": candidates})()
    fd = field_demand(fake_field, role_surprise_fn=demand_fn, budget=rate_budget)
    channel = channel_offload_manifest(fd)

    best_cand_fit = max((scored[n]["fitness"] for n in scored), default=0.0)
    audit = {
        "families": list(genotypes),
        "tiers_seen": sel["tiers_seen"],
        "grain_used": sel["grain_used"],
        # A weak shadow suggestion; NEVER auto-applied. A live graft (rest-window handoff) is a rook
        # fork mirroring the warden's never-auto-apply clamp — value/promotion is policy, not math.
        "promote_suggested": bool(offspring_score and offspring_score["fitness"] > best_cand_fit),
        "auto_apply": False,
        "note": "shadow-only; offspring reads proxy/flat (grain is a small-dense luxury); live graft owed",
    }
    return SelectionResult(roles=roles, rate_plan=rates, candidates=scored, winners=winners,
                           offspring=offspring, offspring_score=offspring_score,
                           field_demand=fd, channel=channel, audit=audit)
