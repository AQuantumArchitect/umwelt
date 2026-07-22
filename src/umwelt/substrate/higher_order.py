"""Higher-order information — the manifold's binding and grain, past pairwise MI.

The MI sensor reads the depth-2 shadow: how much two beliefs share. But a cluster of beliefs
binds as a WHOLE, and a binding has a CHARACTER. Two measures name it — both from the same
cumulants the MI sensor already touches (`e1`, `e2`), so nothing new is tracked:

  total correlation  TC(X) = Σ_i S(ρ_i) − S(ρ_X)             — how bound (bits); 0 iff product
  O-information       Ω(X) = (n−2)S(X) + Σ_i[S({i}) − S(−i)]  — the GRAIN (bits); the SIGN is the character
      Ω > 0  redundancy  → "chorus"      (the beliefs echo a common cause — many say the same)
      Ω < 0  synergy     → "conspiracy"  (the whole knows what no part does — e.g. parity/XOR)

The honest limit (measured, ledgered): the SECOND-ORDER O-information is identically zero
(Ω/TC → 0 as ‖C‖ → 0: 3.1e-2 → 1.3e-3 over eps 0.08→0.01), and the cumulant closure zeros the
3-body cumulant that carries Ω's sign. So a cumulant cluster can measure TC (a faithful pairwise
proxy, <1% error while ‖C_ij‖≲0.1) but CANNOT sign the grain — it reports grain="flat": it feels
the bind, it cannot name its character. The sign lives only on the dense backend, which keeps the
full joint ρ. This is the exact discipline of the MI sensor's `cumulant_only`: report only what the
backend can faithfully measure.

Tiers — by BACKEND CAPABILITY (the limiter is whether a joint ρ exists at all), not by strength:
  gated  max_{i<j}‖C_ij‖ < floor  → TC=0, Ω=0, grain="flat"            (~free: outer products only)
  proxy  cumulant backend          → TC=κ·Σ‖C_ij‖², Ω=None, grain="flat"
  exact  dense backend             → full TC and Ω          (Ω=None when n<3, undefined there)

Verified against exact (n=3): product 0/0 flat; classical-GHZ TC=2 Ω=+1 chorus;
XOR/even-parity TC=1 Ω=−1 conspiracy; pure GHZ TC=3 Ω=0 flat (entanglement ≠ redundancy).
Reuses substrate.mutual_information — same κ, same connected correlation, same tiered honesty.
"""
from __future__ import annotations

import numpy as np

from umwelt.substrate.mutual_information import (
    DEFAULT_FLOOR,
    DEFAULT_STRONG,
    MI_BITS_COEFF,
    _pair_stats,
    _vn_bits,
    connected_correlation,
    corr_norm,
)

# Below this |Ω| the grain is unresolved (numerical dust, or Ω is None/gated).
_GRAIN_EPS: float = 1e-6


def _roles(cluster) -> list:
    return list(getattr(cluster, "qubit_roles", ()) or ())


def _is_cumulant(cluster) -> bool:
    """The cumulant backend tracks e1/e2 directly (no joint ρ) — same probe the MI sensor uses."""
    return hasattr(cluster, "e1") and hasattr(cluster, "e2")


def grain(omega) -> str:
    """Name the binding's character from the O-information sign. None/|Ω|≤eps → 'flat'."""
    if omega is None:
        return "flat"
    if omega > _GRAIN_EPS:
        return "chorus"        # redundancy — beliefs echo a common cause
    if omega < -_GRAIN_EPS:
        return "conspiracy"    # synergy — the whole knows what no part does
    return "flat"


# ── proxy tier: from the pairwise cumulants alone (the free path) ─────────────────

def total_correlation_proxy(cluster) -> tuple[float, float]:
    """Second-order TC surrogate from `e1`/`e2` only: κ·Σ_{i<j}‖C_ij‖² — the same ‖C‖² the MI
    proxy rides, summed over pairs (to leading order TC ≈ Σ pairwise MI). Returns
    (TC_proxy, max‖C_ij‖). Faithful to <1% while ‖C_ij‖≲0.1; over-estimates in the strong regime,
    where the dense/exact path should be used instead. There is deliberately NO Ω proxy: the
    second-order O-information is identically zero and the closure zeros the cumulant that signs it."""
    roles = _roles(cluster)
    n = len(roles)
    tc = 0.0
    max_cn = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            C = connected_correlation(cluster.e1[i], cluster.e1[j], cluster.e2[i, j])
            cn = float(np.linalg.norm(C))
            if cn > max_cn:
                max_cn = cn
            tc += MI_BITS_COEFF * cn * cn
    return tc, max_cn


# ── exact tier: a real joint ρ (dense backend) ────────────────────────────────────

def _subset_entropy_bits(cluster, keep) -> float:
    """von Neumann entropy S (bits) of the subsystem keeping qubit indices `keep`, via one partial
    trace through the dense cluster's `subsystem_rdm` — the same read the MI sensor's dense path uses."""
    rho = np.asarray(cluster.subsystem_rdm(list(keep)), dtype=complex)
    return _vn_bits(rho)


def total_correlation_exact(cluster) -> float:
    """TC = Σ_i S(ρ_i) − S(ρ_X) in bits, from real reduced states (dense backend only)."""
    n = len(_roles(cluster))
    s_marginals = sum(_subset_entropy_bits(cluster, [i]) for i in range(n))
    s_joint = float(cluster.entropy)                      # the real joint von Neumann entropy (bits)
    return s_marginals - s_joint


def o_information_exact(cluster) -> float:
    """Ω = (n−2)·S(X) + Σ_i [ S({i}) − S(all-but-i) ] in bits. Sign is the grain. Needs n≥3
    (Ω is undefined for a pair — pairwise sharing is total correlation, not redundancy vs synergy)."""
    n = len(_roles(cluster))
    s_joint = float(cluster.entropy)
    total = (n - 2) * s_joint
    allq = list(range(n))
    for i in range(n):
        s_i = _subset_entropy_bits(cluster, [i])
        s_not_i = _subset_entropy_bits(cluster, [q for q in allq if q != i])
        total += s_i - s_not_i
    return total


def _max_corr_norm(cluster) -> float:
    """max_{i<j} ‖C_ij‖ — the cheap gate quantity, backend-agnostic (reuses the MI sensor's
    `_pair_stats`: free for cumulant, one RDM per pair for dense)."""
    n = len(_roles(cluster))
    m = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            e1i, e1j, e2ij = _pair_stats(cluster, i, j)
            cn = corr_norm(e1i, e1j, e2ij)
            if cn > m:
                m = cn
    return m


# ── the sensor ────────────────────────────────────────────────────────────────────

def cluster_higher_order(cluster, *, floor: float = DEFAULT_FLOOR,
                         strong: float = DEFAULT_STRONG) -> dict | None:
    """Higher-order structure of one cluster's beliefs:
        {n, total_correlation, o_information, grain, tier, max_corr_norm}
    Returns None if n<2 (no binding to speak of). `o_information` is dense-exact-only and needs
    n≥3, else None; a cumulant cluster reports `total_correlation` with grain='flat' (it feels the
    bind, it cannot name its character). `strong` is accepted for signature symmetry with the MI
    sensor; the higher-order tiers are chosen by backend capability, not correlation strength."""
    roles = _roles(cluster)
    n = len(roles)
    if n < 2:
        return None

    if _is_cumulant(cluster):
        # The cumulant sees only pairwise structure, so a pairwise ‖C‖ gate is sound here: at
        # product it is genuinely blind and honestly reports flat.
        tc, max_cn = total_correlation_proxy(cluster)
        if max_cn < floor:
            return {"n": n, "total_correlation": 0.0, "o_information": 0.0,
                    "grain": "flat", "tier": "gated", "max_corr_norm": max_cn}
        return {"n": n, "total_correlation": tc, "o_information": None,
                "grain": "flat", "tier": "proxy", "max_corr_norm": max_cn}

    # Dense backend: a real joint ρ → ALWAYS compute exact. There is NO sound cheap pre-gate: pure
    # synergy (e.g. XOR/parity) carries zero pairwise correlation yet nonzero TC and Ω, so a pairwise
    # ‖C‖ gate would make the exact tier blind to the very synergy it exists to see. The max ‖C_ij‖ is
    # still reported (a product reads TC≈0/flat; XOR reads TC>0 with max‖C‖≈0 — "all binding is higher-order").
    max_cn = _max_corr_norm(cluster)
    tc = total_correlation_exact(cluster)
    omega = o_information_exact(cluster) if n >= 3 else None
    return {"n": n, "total_correlation": tc, "o_information": omega,
            "grain": grain(omega), "tier": "exact", "max_corr_norm": max_cn}


def field_higher_order_scan(field, *, floor: float = DEFAULT_FLOOR, strong: float = DEFAULT_STRONG,
                            cumulant_only: bool = False) -> dict[str, dict]:
    """Scan the whole belief-field for per-cluster higher-order structure — {cluster_name: {...}}
    for every cluster with n≥2. `cumulant_only` mirrors `field_mi_scan`: skip dense clusters when
    set (stay on the free path). Product clusters cost only their gates; binding surfaces where
    beliefs genuinely bind. Guarded per-cluster — a broken cluster is skipped, never raises."""
    out: dict[str, dict] = {}
    for name, c in getattr(field, "clusters", {}).items():
        if len(_roles(c)) < 2:
            continue
        if cumulant_only and not _is_cumulant(c):
            continue
        try:
            ho = cluster_higher_order(c, floor=floor, strong=strong)
        except Exception:
            continue
        if ho is not None:
            out[name] = ho
    return out
