"""Compute demand — the womb's metabolism: where is development owed, and can this CPU serve it.

This is the HONEST EMITTER half of the courtship-and-gestation braid. It answers, per belief
cluster, one question the way umwelt already answers it for evidence — *where is learning owed?* —
and turns that into a development cadence and a compute-demand signal. It emits and advises only:
it consumes nothing, meters no wall clock, and steps no cluster.

It rhymes with how umwelt learns. umwelt's dream-loop spends its coupling-search where the field is
SURPRISED; the membrane spends ingest cadence where evidence is live. Here the same currency —

    struggle(cluster) = soft-OR( mixedness , mean|surprise| ) = 1−(1−mixedness)(1−surprise)

drives the DEVELOPMENTAL cadence: a candidate that is uncertain (mixed marginals) OR still erring
(surprised) is "learning hard", and `rate_plan` gives it more cheap Python ticks; a settled candidate
(certain AND un-surprised) gets the floor. Uncertain OR erring = still learning is the competence law
(learnedness × prediction_skill) resolved to one cluster and inverted — so a sharp-but-surprised belief
still owes development, which is exactly where the womb must spend ticks.

Tiers, named out loud (the honesty spine):
  struggle  = PROXY. With a real `role_surprise` it rides measured prediction-error (source='surprise');
              with none it degrades to a mixedness stand-in (source='purity_proxy') that CANNOT tell
              "unknown because unlearned" from "unknown at equilibrium"; source='neutral' when nothing
              is readable. Never a calibrated probability.
  cost      = ESTIMATE, backend-aware FLOP proxy — dense ~ (2^n)^2 (the un-scalable exact oracle),
              cumulant ~ n^3 (the cheap scalable path, calibrated to the measured O(n^3) cumulant step).
              Explicitly NOT a measurement; the measured per-cluster wall meter edits the shared hot
              loop and is a review-gated fork.
  demand    = struggle × cost, a field-level aggregate with an honest deficit sign against a budget —
              the first-class signal a remote/virtual compute channel WOULD consume.
  channel   = DENIED. `channel_offload_manifest` computes what it WOULD offload and the flops it would
              save, then dispatches 0 and lets 0 bytes leave the process. The emitter with the channel
              amputated — the demand that would someday summon remote compute, built without the channel.
"""
from __future__ import annotations

import numpy as np

from umwelt._util import clamp01 as _clamp01


def _roles(cluster) -> list:
    return list(getattr(cluster, "qubit_roles", ()) or ())


def _mixedness(cluster) -> tuple[float, str]:
    """Marginal uncertainty of a cluster's beliefs in [0,1] — 0 = every belief sharp, 1 = maximally
    mixed. Read the cheapest faithful way the backend allows: a cumulant's own `purity()`, else the
    mean single-qubit purity from `e1` ((1+|r|^2)/2 per qubit), else normalized joint entropy, else
    neutral. Returns (mixedness, source) so the caller can name how the number was obtained."""
    purity_fn = getattr(cluster, "purity", None)
    if callable(purity_fn):
        try:
            return _clamp01(2.0 * (1.0 - float(purity_fn()))), "purity"
        except Exception:
            pass
    e1 = getattr(cluster, "e1", None)
    if e1 is not None:
        e1 = np.asarray(e1, dtype=float)
        if e1.ndim == 2 and e1.shape[0] > 0:
            r2 = np.clip(np.sum(e1 * e1, axis=1), 0.0, 1.0)      # |r_i|^2 per qubit
            return _clamp01(float(np.mean(1.0 - r2))), "bloch"    # 1-|r|^2 = 2·(1-purity_i)
    ent = getattr(cluster, "entropy", None)
    n = max(1, len(_roles(cluster)))
    if ent is not None:
        try:
            return _clamp01(float(ent) / n), "entropy"           # normalized joint entropy (bits/qubit)
        except Exception:
            pass
    return 0.5, "neutral"


def cluster_struggle(cluster, *, role_surprise=None) -> dict:
    """How hard this cluster is still learning — the developmental pressure, in umwelt's own currency.

    struggle = soft-OR(mixedness, mean|surprise|) = 1−(1−mixedness)(1−surprise): a belief cluster earns
    cadence when it is uncertain OR still erring — the competence law (learnedness × skill) inverted into
    "still learning", so a sharp-but-surprised belief still owes development. `role_surprise` is an
    optional {role: prediction_error} (or a flat sequence / scalar); None degrades to mixedness alone.
    Returns
        {struggle, mixedness, surprise|None, source, n}
    with `source` naming the honesty tier of the surprise ('surprise' measured, 'purity_proxy'/'neutral'
    otherwise) — never a claim the number cannot back."""
    roles = _roles(cluster)
    n = len(roles)
    mixedness, mx_src = _mixedness(cluster)

    s = None
    if role_surprise is not None:
        if isinstance(role_surprise, dict):
            vals = [abs(float(role_surprise[r])) for r in roles if r in role_surprise]
            if not vals:
                vals = [abs(float(v)) for v in role_surprise.values()]
        elif np.isscalar(role_surprise):
            vals = [abs(float(role_surprise))]
        else:
            vals = [abs(float(v)) for v in np.asarray(role_surprise, dtype=float).ravel()]
        if vals:
            s = float(np.mean(vals))

    if s is None:
        # No live surprise: the mixedness stand-in, honestly flagged. It cannot distinguish
        # "unknown because unlearned" from "unknown at equilibrium" — the emitter's blind spot.
        struggle = mixedness
        source = "purity_proxy" if mx_src != "neutral" else "neutral"
    else:
        # soft-OR: still-learning = uncertain OR erring (the competence law learnedness×skill, inverted:
        # 1−(1−mixedness)(1−surprise)). A sharp-but-surprised belief still owes development — surprise
        # drives cadence even at zero mixedness, which is exactly where the womb must spend ticks.
        m, sv = _clamp01(mixedness), _clamp01(s)
        struggle = m + sv - m * sv
        source = "surprise"

    return {"struggle": _clamp01(struggle), "mixedness": mixedness,
            "surprise": s, "source": source, "n": n}


def compute_cost_estimate(cluster) -> dict:
    """Backend-aware per-step FLOP proxy: the tier economics in one number. dense ~ (2^n)^2 = 4^n
    (the exact oracle that does not scale), cumulant ~ n^3 (the cheap path, calibrated to the measured
    O(n^3) cumulant step). An ESTIMATE, never a wall measurement. Returns {cost, backend, n}."""
    n = len(_roles(cluster))
    is_cumulant = hasattr(cluster, "e1") and hasattr(cluster, "e2")
    if is_cumulant:
        return {"cost": float(max(1, n) ** 3), "backend": "cumulant", "n": n}
    return {"cost": float((2 ** n) ** 2), "backend": "dense", "n": n}


def field_demand(field, *, role_surprise_fn=None, wall_ms_map=None, budget=None) -> dict:
    """Where development is owed across a whole field, and whether this CPU can serve it in-budget.

    For each cluster: struggle (PROXY, surprise from `role_surprise_fn(name, cluster)` if given) × cost
    (a measured wall from `wall_ms_map[name]` if the caller has one, else the ESTIMATE). `budget` is the
    tick/flop envelope this CPU can serve; deficit = field_demand − budget is the honest shortfall — a
    positive deficit is the signal a remote channel would drink. Guarded per-cluster like
    field_higher_order_scan: a broken cluster is skipped, never raises. Returns
        {per_cluster: {name: {struggle, cost, demand}}, field_demand, budget, deficit, deficit_ratio}."""
    per: dict[str, dict] = {}
    total = 0.0
    for name, c in getattr(field, "clusters", {}).items():
        if str(name).startswith("_") or len(_roles(c)) < 1:
            continue
        try:
            rs = role_surprise_fn(name, c) if role_surprise_fn is not None else None
            struggle = cluster_struggle(c, role_surprise=rs)["struggle"]
            if wall_ms_map is not None and name in wall_ms_map:
                cost = float(wall_ms_map[name])
            else:
                cost = compute_cost_estimate(c)["cost"]
        except Exception:
            continue
        demand = struggle * cost
        per[name] = {"struggle": struggle, "cost": cost, "demand": demand}
        total += demand

    deficit = None
    deficit_ratio = 0.0
    if budget is not None:
        deficit = total - float(budget)
        deficit_ratio = _clamp01(deficit / total) if total > 0.0 else 0.0
    return {"per_cluster": per, "field_demand": total, "budget": budget,
            "deficit": deficit, "deficit_ratio": deficit_ratio}


def _demand_values(demand_map) -> dict:
    """Accept either field_demand()['per_cluster'] ({name:{demand}}) or a bare {name: float}."""
    out = {}
    src = demand_map.get("per_cluster", demand_map) if isinstance(demand_map, dict) else {}
    for name, v in src.items():
        out[name] = float(v["demand"] if isinstance(v, dict) else v)
    return out


def rate_plan(demand_map, *, floor: int = 1, cap: int = 16, budget=None) -> dict[str, int]:
    """Turn demand into a per-cluster development cadence — the variable-rate womb's schedule.

    Higher demand → more cheap Python ticks (learn hard and fast where struggle is real); a settled
    cluster gets `floor`. Monotone in demand and, when a `budget` (total ticks to spend) is given,
    budget-conserving (Σ steps ≤ budget) WHEN budget ≥ floor·N; below that the per-cluster `floor` is a
    hard minimum and wins (Σ steps = floor·N > budget). Consumed OFFLINE by structure_selection.develop
    — it steps no live field. Returns {name: n_steps} with every value in [floor, cap]."""
    vals = _demand_values(demand_map)
    names = sorted(vals)
    if not names:
        return {}
    dmax = max(vals.values())
    if dmax <= 0.0:
        return {name: int(floor) for name in names}
    total = sum(vals.values())

    plan: dict[str, int] = {}
    if budget is not None:
        extra_budget = max(0, int(budget) - int(floor) * len(names))
        for name in names:
            share = vals[name] / total if total > 0 else 0.0
            steps = int(floor) + int(share * extra_budget)
            plan[name] = int(min(cap, max(floor, steps)))
    else:
        for name in names:
            steps = int(floor) + int(round((cap - floor) * (vals[name] / dmax)))
            plan[name] = int(min(cap, max(floor, steps)))
    return plan


def channel_offload_manifest(demand) -> dict:
    """The emitter with the channel amputated. Given a field_demand() result, name the clusters whose
    demand exceeds the field's mean (the strugglers a remote/virtual channel WOULD serve) and the flops
    it would save — then dispatch 0, and let 0 bytes leave the process. The remote channel Luke keeps
    flirting with is a deployment/trust fork; this ships only the honest demand for it.
        {status:'DENIED', would_offload:[names], est_saving_flops, dispatched:0, bytes_left_process:0}"""
    per = demand.get("per_cluster", {}) if isinstance(demand, dict) else {}
    if per:
        mean_demand = sum(v["demand"] for v in per.values()) / len(per)
        would = sorted(name for name, v in per.items() if v["demand"] > mean_demand)
        est_saving = float(sum(per[name]["cost"] for name in would))
    else:
        would, est_saving = [], 0.0
    return {"status": "DENIED", "would_offload": would, "est_saving_flops": est_saving,
            "dispatched": 0, "bytes_left_process": 0}
