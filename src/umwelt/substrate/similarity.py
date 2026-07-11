"""Similarity-driven grouping — the self-organizing core of the fractal-web.

Phase 2 (archetypes) shares meta-parameters across leaves grouped by hand-coded
`kind`. Phase 5 discovers the groups instead: cluster the field's clusters by
their BEHAVIORAL fingerprint and let similar-behaving objects find each other —
"the lights cluster together, the motion sensors do, and regions with similar
mixed usage do too." Discovered groups then share dynamics meta-parameters via
the same shared-ScalarParam mechanism as Phase 2 (param_bundles), so a group is
tuned AS A GROUP — the modular-reservoir idea (sub-reservoirs sharing per-module
hyperparameters), here in the quantum graph. See project_fractal_web.

Density without 2^n: grouping shares SCALARS (and, for same-signature groups,
H coefficients) — it never merges Hilbert spaces. The clusters stay small; only
their tuning is pooled.

This module is DISCOVERY + (optional) PARAMETER SHARING. It does not reparent
the graph (that is Phase 4's add_subdomain, a heavier structural change). The
fingerprint is cheap (a 3-vector per cluster from the fractal signature), so
re-discovery can run on a slow φ-clock as the field's behavior drifts.
"""
from __future__ import annotations

import numpy as np

from umwelt.substrate.fractal import fractal_signature


def behavioral_fingerprint(cluster) -> np.ndarray:
    """A scale-invariant behavioral signature for one cluster: the NORMALIZED
    distribution of correlation energy across fractal levels 1/2/3.

    A purely local cluster concentrates energy at level 1; a richly-correlated
    one spreads it to levels 2/3. The SHAPE (not magnitude) characterizes how
    the cluster behaves, independent of how excited it currently is — so two
    regions with the same dynamics but different occupancy still look alike.
    """
    try:
        levels = cluster.features_by_level(max_level=3)
    except TypeError:
        # Older (pre-pruning) cluster.py has no max_level kwarg; it returns all
        # levels uncapped. We only read 1/2/3 below, so extra keys are inert.
        levels = cluster.features_by_level()
    energy = np.array([fractal_signature(levels).get(lvl, 0.0) for lvl in (1, 2, 3)],
                      dtype=float)
    total = float(energy.sum())
    if total < 1e-12:
        # Ground state / silent cluster: all "energy" nominally at level 1.
        return np.array([1.0, 0.0, 0.0])
    return energy / total


def role_signature(cluster) -> tuple:
    """Structural signature: the sorted role set. Two clusters can share H
    coefficients only if this matches (same operator basis); scalar dynamics
    params can be shared across differing signatures."""
    return tuple(sorted(cluster.qubit_roles))


def similarity_groups(
    field,
    threshold: float = 0.12,
    *,
    require_same_roles: bool = True,
    min_group: int = 2,
    exclude: "set[str] | None" = None,
) -> list[list[str]]:
    """Discover behaviorally-similar cluster groups in a field.

    Union-find over clusters: two clusters join when their behavioral
    fingerprints are within `threshold` (Euclidean on the normalized
    energy-shape vector) — and, when `require_same_roles`, share a role
    signature (so a discovered group is also valid for shared H coefficients,
    not just scalars). Returns groups of ≥ `min_group` member names, each
    sorted; singletons are dropped. `exclude` skips clusters by name (e.g. the
    synthetic _params/_clock fibers, or a root you don't want pooled).
    """
    exclude = exclude or set()
    names = [n for n in field.clusters if n not in exclude]
    fp = {n: behavioral_fingerprint(field.clusters[n]) for n in names}
    rs = {n: role_signature(field.clusters[n]) for n in names}

    parent = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if require_same_roles and rs[a] != rs[b]:
                continue
            if float(np.linalg.norm(fp[a] - fp[b])) <= threshold:
                union(a, b)

    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return [sorted(g) for g in groups.values() if len(g) >= min_group]


def apply_discovered_sharing(
    graph,
    groups: list[list[str]],
    shared_keys: "tuple[str, ...]" = ("gamma",),
) -> dict[str, list[str]]:
    """Make each discovered group share dynamics ScalarParam objects — the
    self-organizing extension of Phase 2's archetype sharing. For each group,
    the first member that owns a given key donates its ScalarParam object; the
    rest re-point at it, so the group learns one shared value for that key (more
    data per param, faster convergence). Keys absent on a member are skipped.

    Returns {group_representative: members} for the groups that shared at least
    one key. Reuses the exact shared-reference idiom from
    param_bundles._attach_archetypes; the brain pickle's _snapshot_param_fiber
    already dedups shared objects, so this also shrinks the pickle.
    """
    applied: dict[str, list[str]] = {}
    for group in groups:
        nodes = [graph.find(n) for n in group]
        nodes = [n for n in nodes if n is not None and n.param_bundle is not None]
        if len(nodes) < 2:
            continue
        shared_any = False
        for key in shared_keys:
            donor = next((n for n in nodes if n.param_bundle.get_param(key) is not None), None)
            if donor is None:
                continue
            shared_param = donor.param_bundle.get_param(key)
            for n in nodes:
                if n is donor:
                    continue
                if n.param_bundle.get_param(key) is not None:
                    n.param_bundle.params[key] = shared_param  # shared reference
                    shared_any = True
        if shared_any:
            applied[group[0]] = group
    return applied
