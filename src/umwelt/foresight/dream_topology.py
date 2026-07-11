"""dream_topology — the DREAM's topology-mutation organ: grow the field's couplings from real data, at rest.

Dreaming already mutates cassettes (experience); this makes it also mutate TOPOLOGY (structure). The genetic
graph-mutation idea and the predictive coupling-learner meet here, as downtime compute. One pass:

  1. SCAN   — for every role pair in a cluster, measure the predictive contrast D* from the real events.db
              (lagged co-occurrence; forecast/synthetic rows excluded). |D*| ranks candidate EDGES and names
              the leader→follower direction — data-seeded mutation proposals, not random.
  2. PRUNE  — |D*| below a floor = no predictive structure = no edge.
  3. VALIDATE — k-fold time-CV on HELD-OUT real data: keep an edge ONLY IF the learned coupling reduces
              forecast surprise on unseen folds (majority of folds + positive mean). NO synthetic shadows —
              those can be under-specified in ways that matter; only real held-out observations judge.
  4. CONSOLIDATE (opt-in) — install the surviving J on the cluster's _xy exchange slot. It persists through
              the reservoir pickle, so the brain wakes with the grown edge.

Selection is PURELY forecast-surprise reduction on real data. Gated default-OFF; consolidation is a separate
opt-in (shadow = propose-only by default). See coupling_learn for the learner + the k-fold validator.
"""
from __future__ import annotations

import itertools

from umwelt.learning import coupling_learn as cl

# field role → real event stream, as (event_type, device). Empty by default (domain-free): a domain
# registers its role→stream map, or callers pass grow_couplings(resolve=...) directly (the FAST path and
# the ported tests both supply an explicit resolver). The origin deployment's role→stream table lives in
# its example.
_ROLE_STREAM: dict[str, tuple[str, str]] = {}


def register_role_stream(role: str, event_type: str, device: str) -> None:
    """Register a role → (event_type, device) stream mapping consulted by default_resolve."""
    _ROLE_STREAM[role] = (event_type, device)


def default_resolve(role: str):
    """role → (event_type, device) or None, from the registered role→stream map (empty by default)."""
    return _ROLE_STREAM.get(role)


def facet_pairs(field, node, *, lead_facet: str, follow_facets: tuple[str, ...] = ()):
    """Curated candidate edges for the FAST preset: within-node `{prefix}_{lead_facet}` → `{prefix}_{f}`
    pairs (a region's driver facet coupling to its dependent facets) — the highest-value web candidates.
    Bounds a growth pass from the full C(n,2) sweep to a handful → ~seconds, not minutes, on the A55.
    Domain supplies the facet names; empty follow_facets → no pairs."""
    c = field.clusters.get(node)
    roles = set(getattr(c, "role_index", {})) if c is not None else set()
    out = []
    for r in sorted(roles):
        prefix, _, facet = r.rpartition("_")
        if facet != lead_facet:
            continue
        for other in follow_facets:
            q = f"{prefix}_{other}"
            if q in roles:
                out.append((r, q))
    return out


def grow_couplings(field, db_path: str, *, nodes=None, pairs=None, resolve=default_resolve, source=None,
                   consolidate=False, folds=5, lag_s=60.0, tol_s=120.0, shrink=0.5, margin=0.02, edge_floor=0.05,
                   probe_budget=12, since=None, row_cap=8000, log=None) -> dict:
    """Scan a cluster's role pairs, k-fold-validate each candidate on held-out real data, and (if consolidate)
    install the surviving couplings on _xy. Returns a report dict {node: {scanned, proposals, applied}}.
    `nodes` limits which clusters to sweep (default: every cluster with ≥2 resolvable roles).
    `pairs` (the FAST path) restricts the scan to an explicit curated [(role_a, role_b), ...] allow-list instead
    of the full C(n,2) combinatorial sweep — bounding a pass to ~seconds on the A55. See facet_pairs.
    `source` (event_type, device, since, limit) -> (ts, vals) is the stream READ path; default = the raw
    events.db pull. Pass stream_tape's bounded reader to learn from the gauge-pruned fiber history instead of
    scanning the 8.3GB firehose (the ~ms path) — then `since` should be epoch seconds, not an ISO string."""
    def _log(m):
        if log:
            log(m)

    report: dict = {}
    cluster_names = nodes if nodes is not None else list(field.clusters)
    for node in cluster_names:
        c = field.clusters.get(node)
        if c is None:
            continue
        role_set = getattr(c, "role_index", {})
        if pairs is not None:
            # FAST path: only the curated edges whose BOTH roles are present + resolvable on this cluster
            node_pairs = [(a, b) for (a, b) in pairs
                          if a in role_set and b in role_set and resolve(a) and resolve(b)]
            roles = sorted({r for pr in node_pairs for r in pr})
        else:
            roles = [r for r in role_set if resolve(r)]
            node_pairs = list(itertools.combinations(roles, 2))
        if len(roles) < 2:
            continue

        # cache each unique stream once
        cache = {}
        for r in roles:
            ev, dev = resolve(r)
            s = source(ev, dev, since, row_cap) if source else cl.pull_stream(db_path, ev, dev, since, limit=row_cap)
            cache[r] = s if (s is not None and s[0] is not None and len(s[0]) > 0) else (None, None)

        # SCAN: rank candidate edges by |D*| (full window), pick the stronger direction
        candidates = []
        for r1, r2 in node_pairs:
            best = None
            for lead, foll in ((r1, r2), (r2, r1)):
                ta, va = cache[lead]
                tb, vb = cache[foll]
                if ta is None or tb is None:
                    continue
                m = cl.contrast_from_arrays(ta, va, tb, vb, lag_s=lag_s, tol_s=tol_s)
                if m is not None and (best is None or abs(m["contrast"]) > abs(best[2])):
                    best = (lead, foll, m["contrast"])
            if best is not None and abs(best[2]) >= edge_floor:
                candidates.append(best)
        candidates.sort(key=lambda x: -abs(x[2]))

        # VALIDATE on held-out real data (k-fold); CONSOLIDATE the robust survivors
        proposals, applied, probed = [], [], 0
        for lead, foll, Dstar in candidates:
            if probed >= probe_budget:
                break
            probed += 1
            ta, va = cache[lead]
            tb, vb = cache[foll]
            v = cl.kfold_validate_edge(field, node, lead, foll, ta, va, tb, vb,
                                       folds=folds, lag_s=lag_s, tol_s=tol_s, shrink=shrink, margin=margin)
            if not v.get("robust"):
                continue
            edge = dict(node=node, leader=lead, follower=foll, J=v["J"], family=v["family"],
                        surprise_reduction=v["mean_reduction"], frac_pos=v.get("frac_pos"),
                        worst_fold=v.get("worst"), n_folds=v["n_folds"])
            proposals.append(edge)
            if consolidate and cl.apply_coupling(field, node, lead, foll, v["J"], v["family"]):
                applied.append(edge)
                _log(f"dream: grew {node}:{lead}→{foll} J={v['J']:+.2f} ({v['family']}) "
                     f"held-out surprise −{v['mean_reduction']:.2f} over {v['n_folds']} folds")

        if candidates or proposals:
            report[node] = dict(scanned=len(candidates), probed=probed,
                                proposals=proposals, applied=applied)
    return report
