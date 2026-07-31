"""Manifold fitness — where the mother's math becomes a selection pressure, and the father chooses.

The higher-order sensors (higher_order.py + information_geometry.py) already read the SHAPE of a belief
cluster: how bound it is (total correlation), the character of the binding (the O-information grain —
chorus vs conspiracy), which beliefs form a concept (MI constellations), and the metric between them
(variation of information). That shape is the MOTHER-MATH — the field's own structure, untouched here.

THE KISS is one line — `structure_fitness` folds that shape into a single scalar:

    fitness = w_bind·(TC/n) + w_concept·integration + w_chorus·Ω + w_vi·compactness − w_cost·demand

the exact place a sensor reading BECOMES a selection pressure. THE FATHER is `select`: he computes no
physics; he ranks the structures the mother scored, top-k by score under a compute-cost penalty. His
only free will — the value DIRECTION `w_chorus` (does he prefer redundancy/chorus or synergy/conspiracy?)
— defaults to NEUTRAL (0.0) and is a rook policy knob, because value is policy, not math.

This rhymes with how umwelt learns. umwelt's dream-loop searches structure to MINIMIZE surprise; here
the manifold's binding + integration are the cheap proxy for "a structure worth keeping", and the
`demand` term (surprise × cost, from compute_demand) is subtracted so the father is honest about what a
structure costs to run — scarce compute (high cost_weight) picks the cheap tight small cluster, abundant
compute picks the big high-fitness one.

Tier honesty (inherited verbatim from the sensors):
  - On a PROXY (cumulant) candidate the grain is unnameable: `grain_known=False`, the w_chorus term is 0,
    and the fitness is a declared BINDING LOWER BOUND — never a grain claim (rides the DENIED, DENSE-ONLY
    limit). `select` refuses to let grain decide a cross-tier contest (`grain_used=False`).
  - integration comes from the pairwise MI graph, so a pure-synergy (XOR) structure earns binding-density
    credit but ZERO integration credit on any backend — the pairwise sensor is blind to parity.
  - vi_compactness is clamped to [0,1] so an entangled pair (VI=−2, the separable-regime limit) cannot
    spuriously inflate fitness.
  - n<2 → None (rides cluster_higher_order); Ω is None for n<2 so grain is never named for a pair.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from umwelt.substrate.mutual_information import DEFAULT_FLOOR, DEFAULT_STRONG
from umwelt.substrate.higher_order import cluster_higher_order
from umwelt.substrate.information_geometry import constellations, vi_distance_matrix


@dataclass(frozen=True)
class FitnessWeights:
    """The father's values. w_chorus and w_vi default NEUTRAL — direction/geometry preferences are rook
    policy, not math. w_bind rewards binding density (TC/n), w_concept rewards integration into one
    concept, w_cost prices the compute demand a structure carries."""
    w_bind: float = 1.0
    w_concept: float = 0.5
    w_chorus: float = 0.0        # >0 prefer chorus (redundancy), <0 prefer conspiracy (synergy); 0 neutral
    w_vi: float = 0.0            # geometric compactness reward; neutral by default
    w_cost: float = 1.0          # price on the per-structure compute demand (subtracted)


DEFAULT_WEIGHTS = FitnessWeights()


def _roles(cluster) -> list:
    return list(getattr(cluster, "qubit_roles", ()) or ())


def _integration(cluster, *, floor: float, strong: float) -> float:
    """Fraction of the maximal MI-graph connectivity: (n − n_components)/(n − 1) ∈ [0,1]. 0 when every
    belief is its own island (product / pure synergy), 1 when all beliefs bind into one concept. Read
    from the shipped `constellations` (the MI union-find) — blind to parity, honestly."""
    roles = _roles(cluster)
    n = len(roles)
    if n < 2:
        return 0.0
    comps = constellations(cluster, bind_floor=floor, strong=strong)
    return (n - len(comps)) / max(1, n - 1)


def _vi_compactness(cluster, *, floor: float, strong: float) -> float:
    """Mean pairwise closeness from the VI metric, clamped to [0,1]. per-pair closeness = clip(1 − VI/2)
    so a tight (small-VI) manifold reads high, and an entangled pair (VI<0, the separable-regime limit)
    saturates at 1 rather than inflating. w_vi is 0 by default — this rides along as a transparent number."""
    roles = _roles(cluster)
    n = len(roles)
    if n < 2:
        return 0.0
    _r, D = vi_distance_matrix(cluster, floor=floor, strong=strong)
    vals = [float(np.clip(1.0 - D[i, j] / 2.0, 0.0, 1.0))
            for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals)) if vals else 0.0


def _fitness_from(ho: dict, integration: float, vi_compactness: float,
                  weights: FitnessWeights, demand: float) -> tuple[float, bool]:
    """Fold the sensor scorecard + geometry into the scalar. Returns (fitness, grain_known). The grain
    term (w_chorus·Ω) is applied ONLY when the tier is exact AND Ω is named (n≥3, dense) — else it is 0
    and grain_known is False (a binding lower bound, never a grain claim)."""
    n = ho["n"]
    tc = float(ho["total_correlation"])
    binding_density = tc / n if n else 0.0
    omega = ho["o_information"]
    grain_known = (ho["tier"] == "exact" and omega is not None)
    grain_term = weights.w_chorus * float(omega) if grain_known else 0.0
    fitness = (weights.w_bind * binding_density
               + weights.w_concept * integration
               + grain_term
               + weights.w_vi * vi_compactness
               - weights.w_cost * float(demand))
    return fitness, grain_known


def structure_fitness(cluster, *, weights: FitnessWeights = DEFAULT_WEIGHTS, demand: float = 0.0,
                      floor: float = DEFAULT_FLOOR, strong: float = DEFAULT_STRONG) -> dict | None:
    """The KISS: collapse one cluster's higher-order structure into a fitness scalar + a scorecard.
    Reads ONLY the shipped sensors. Returns None when n<2 (nothing to bind). Scorecard keys:
        {n, tier, total_correlation, binding_density, integration, o_information, grain, grain_known,
         vi_compactness, demand, fitness}"""
    ho = cluster_higher_order(cluster, floor=floor, strong=strong)
    if ho is None:
        return None
    n = ho["n"]
    integration = _integration(cluster, floor=floor, strong=strong)
    vi_compactness = _vi_compactness(cluster, floor=floor, strong=strong)
    fitness, grain_known = _fitness_from(ho, integration, vi_compactness, weights, demand)
    return {"n": n, "tier": ho["tier"], "total_correlation": ho["total_correlation"],
            "binding_density": ho["total_correlation"] / n if n else 0.0,
            "integration": integration, "o_information": ho["o_information"],
            "grain": ho["grain"], "grain_known": grain_known,
            "vi_compactness": vi_compactness, "demand": float(demand), "fitness": fitness}


def select(candidates: dict, *, cost_weight: float = 0.0, cost_of=None,
           weights: FitnessWeights = DEFAULT_WEIGHTS, demand_of=None,
           floor: float = DEFAULT_FLOOR, strong: float = DEFAULT_STRONG, k: int = 1) -> dict:
    """FATHER-SELECTION: rank scored candidate structures, pick the top k. `candidates` is {name: cluster}.
    score = fitness − cost_weight·cost, where cost = `cost_of(cluster)` (else 0) and each structure's
    intrinsic compute demand comes from `demand_of(cluster)` (else 0). A CROSS-TIER contest (exact vs
    proxy) drops the grain term — a proxy candidate structurally cannot have one, so letting it decide
    would be unfair — and reports grain_used=False. Deterministic tie-break by name. Returns
        {chosen:[names], ranking:[{name, fitness, cost, score, tier}], grain_used, tiers_seen}."""
    scored = []
    for name in candidates:
        c = candidates[name]
        demand = float(demand_of(c)) if demand_of is not None else 0.0
        fit = structure_fitness(c, weights=weights, demand=demand, floor=floor, strong=strong)
        if fit is None:
            continue
        scored.append((name, c, fit))

    tiers_seen = sorted({fit["tier"] for _n, _c, fit in scored})
    cross_tier = len(tiers_seen) > 1
    eff_weights = weights
    if cross_tier and weights.w_chorus != 0.0:
        # Neutralize the grain term so a proxy candidate isn't judged on a bonus it cannot earn.
        eff_weights = FitnessWeights(w_bind=weights.w_bind, w_concept=weights.w_concept,
                                     w_chorus=0.0, w_vi=weights.w_vi, w_cost=weights.w_cost)

    ranking = []
    grain_decided = False
    for name, c, fit in scored:
        if eff_weights is not weights:
            demand = fit["demand"]
            fit = structure_fitness(c, weights=eff_weights, demand=demand, floor=floor, strong=strong)
        cost = float(cost_of(c)) if cost_of is not None else 0.0
        score = fit["fitness"] - cost_weight * cost
        if fit["grain_known"] and eff_weights.w_chorus != 0.0:
            grain_decided = True
        ranking.append({"name": name, "fitness": fit["fitness"], "cost": cost,
                        "score": score, "tier": fit["tier"]})

    ranking.sort(key=lambda r: (-r["score"], r["name"]))
    chosen = [r["name"] for r in ranking[:max(1, int(k))]]
    return {"chosen": chosen, "ranking": ranking,
            "grain_used": grain_decided and not cross_tier, "tiers_seen": tiers_seen}


def field_fitness_scan(field, *, cumulant_only: bool = False, weights: FitnessWeights = DEFAULT_WEIGHTS,
                       floor: float = DEFAULT_FLOOR, strong: float = DEFAULT_STRONG) -> dict[str, dict]:
    """Self-similar sweep — the SAME fitness functional at every cluster size. {name: scorecard} for
    clusters with n≥2. `cumulant_only` mirrors field_higher_order_scan. Guarded per-cluster."""
    out: dict[str, dict] = {}
    for name, c in getattr(field, "clusters", {}).items():
        if str(name).startswith("_") or len(_roles(c)) < 2:
            continue
        if cumulant_only and not (hasattr(c, "e1") and hasattr(c, "e2")):
            continue
        try:
            fit = structure_fitness(c, weights=weights, floor=floor, strong=strong)
        except Exception:
            continue
        if fit is not None:
            out[name] = fit
    return out
