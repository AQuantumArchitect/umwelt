"""Information geometry — the manifold's METRIC and CONSTELLATIONS, from mutual information.

The MI sensor says how much two beliefs share; this module turns that into SHAPE. A belief-field
stops being a bag of registers and becomes a manifold with a real distance:

  variation of information   VI(i,j) = H(i) + H(j) − 2·MI(i,j) = H(i|j) + H(j|i)

VI is a genuine metric (symmetric, zero on identity, triangle inequality) on the separable /
classically-correlated regime the dissipative belief-field lives in — beliefs that share more
information sit closer. And the MI graph has COMMUNITIES: thresholding the pairwise sharing splits
the roles into CONSTELLATIONS — groups of beliefs that bind, each a candidate "concept."

Honest caveat (ledgered): VI = 2S(AB) − S(A) − S(B) uses the quantum conditional entropies S(A|B),
which go NEGATIVE on a genuinely entangled sub-state — a Bell pair gives H=1, MI=2, VI=−2. So VI is
a metric on the separable regime, and degenerates on entanglement. Rides the SAME pairwise MI the
sensor computes (which has a faithful cumulant proxy), so shape + constellations work on EVERY
backend — even where the O-information grain does not. Intra-cluster only: there is no stored joint
state between clusters (bridges are mean-field), so cross-cluster geometry stays the graph's bridges.
"""
from __future__ import annotations

import numpy as np

from umwelt.substrate.mutual_information import (
    DEFAULT_FLOOR,
    DEFAULT_STRONG,
    _pair_stats,
    mi_pair_bits,
)


def _roles(cluster) -> list:
    return list(getattr(cluster, "qubit_roles", ()) or ())


def _binary_entropy_bits(p: float) -> float:
    """Shannon entropy of a bit with P(0)=p, in bits (0·log0 ≡ 0)."""
    out = 0.0
    for x in (float(p), 1.0 - float(p)):
        if x > 1e-15:
            out -= x * np.log2(x)
    return float(out)


def qubit_entropy_bits(bloch) -> float:
    """Single-qubit von Neumann entropy H(i) (bits) from its Bloch vector: λ = (1±|r|)/2."""
    r = min(float(np.linalg.norm(np.asarray(bloch, dtype=float))), 1.0)
    return _binary_entropy_bits((1.0 + r) / 2.0)


def variation_of_information(cluster, i: int, j: int, *, floor: float = DEFAULT_FLOOR,
                            strong: float = DEFAULT_STRONG) -> float:
    """VI(i,j) = H(i) + H(j) − 2·MI(i,j), the belief-manifold's metric between two roles. Reuses the
    MI sensor (`_pair_stats` + `mi_pair_bits`) — free for a cumulant pair, one RDM for a dense pair.
    A true metric on the separable regime; can go negative on a genuinely entangled pair (see module)."""
    e1i, e1j, e2ij = _pair_stats(cluster, i, j)
    mi, _cn, _tier = mi_pair_bits(e1i, e1j, e2ij, floor=floor, strong=strong)
    return qubit_entropy_bits(e1i) + qubit_entropy_bits(e1j) - 2.0 * mi


def vi_distance_matrix(cluster, *, floor: float = DEFAULT_FLOOR,
                       strong: float = DEFAULT_STRONG):
    """(roles, D): the symmetric, zero-diagonal VI distance matrix over the cluster's roles."""
    roles = _roles(cluster)
    n = len(roles)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = variation_of_information(cluster, i, j, floor=floor, strong=strong)
            D[i, j] = d
            D[j, i] = d
    return roles, D


def constellations(cluster, *, bind_floor: float = DEFAULT_FLOOR,
                   strong: float = DEFAULT_STRONG) -> list[list[str]]:
    """Communities of the intra-cluster MI graph: union-find over roles, joining i~j when they share
    any information above the floor (the MI sensor's gate). Connected components = beliefs that bind;
    singletons are kept so every role has a home. Components and members are ordered by role position
    (deterministic). Mirrors the union-find idiom in substrate.similarity."""
    roles = _roles(cluster)
    n = len(roles)
    idx = {r: k for k, r in enumerate(roles)}
    parent = {r: r for r in roles}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            e1i, e1j, e2ij = _pair_stats(cluster, i, j)
            mi, _cn, _tier = mi_pair_bits(e1i, e1j, e2ij, floor=bind_floor, strong=strong)
            if mi > 0.0:                        # cleared the ‖C‖ gate → the two beliefs bind
                union(roles[i], roles[j])

    groups: dict[str, list[str]] = {}
    for r in roles:
        groups.setdefault(find(r), []).append(r)
    comps = [sorted(g, key=lambda r: idx[r]) for g in groups.values()]
    comps.sort(key=lambda g: idx[g[0]])
    return comps
