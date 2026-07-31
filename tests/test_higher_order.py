"""Higher-order information — the manifold's binding (total correlation) and grain (O-information).

Pins the math the sensor rides on, against canonical states with known values, and pins the two
honest limits as passing assertions: (a) the cumulant backend cannot sign the grain (its 2nd-order
O-information is identically zero and its closure zeros the 3-body cumulant), and (b) it can feel a
binding (total correlation) yet not name its character. Sibling of test_mutual_information.py.
"""
from __future__ import annotations

import numpy as np

from umwelt.substrate import higher_order as HO
from umwelt.substrate.cluster import partial_trace_keep
from umwelt.substrate.mutual_information import _pair_stats_from_rho, _rho2, _vn_bits


# ── fixtures: the read interfaces higher_order.py consumes, and canonical states ──

class _Dense:
    """A minimal dense cluster: exposes exactly the exact-tier read interface (a real joint ρ via
    `subsystem_rdm` + joint `entropy`). No e1/e2 → routes to the exact tier."""

    def __init__(self, rho, roles):
        self._rho = np.asarray(rho, dtype=complex)
        self.qubit_roles = list(roles)
        self.n_qubits = len(roles)
        self.role_index = {r: i for i, r in enumerate(roles)}

    @property
    def entropy(self) -> float:
        return _vn_bits(self._rho)

    def subsystem_rdm(self, keep):
        return partial_trace_keep(self._rho, list(keep), self.n_qubits)


class _Cumulant:
    """A minimal cumulant cluster: e1 (n,3) + e2 (n,n,3,3), the pairwise state. Has e1/e2 → proxy tier."""

    def __init__(self, roles):
        n = len(roles)
        self.qubit_roles = list(roles)
        self.n_qubits = n
        self.e1 = np.zeros((n, 3))
        self.e2 = np.zeros((n, n, 3, 3))


def _cumulant_from_rho(rho, roles):
    """Read the pairwise cumulants (e1, e2) off a dense ρ — exactly what the closure would track.
    For the symmetric states used here this is a faithful 'same state, two backends' comparison."""
    c = _Cumulant(roles)
    d = _Dense(rho, roles)
    n = len(roles)
    for i in range(n):
        for j in range(i + 1, n):
            rij = np.asarray(d.subsystem_rdm([i, j]), dtype=complex)
            e1i, e1j, e2ij = _pair_stats_from_rho(rij)
            c.e1[i], c.e1[j] = e1i, e1j
            c.e2[i, j], c.e2[j, i] = e2ij, e2ij.T
    return c


def _ket(bits: str) -> np.ndarray:
    v = np.zeros(2 ** len(bits), dtype=complex)
    v[int(bits, 2)] = 1.0
    return v


def _classical_ghz() -> np.ndarray:
    """(|000><000| + |111><111|)/2 — a classical common cause → redundancy (chorus), Ω=+1."""
    return 0.5 * (np.outer(_ket("000"), _ket("000")) + np.outer(_ket("111"), _ket("111")))


def _xor_parity() -> np.ndarray:
    """Uniform mixture of the even-parity strings — pure synergy (conspiracy), Ω=−1, zero pairwise C."""
    even = ["000", "011", "101", "110"]
    return sum(np.outer(_ket(b), _ket(b)) for b in even) / 4.0


def _pure_ghz() -> np.ndarray:
    """(|000> + |111>)/√2 — maximally entangled: TC=3 but Ω=0 (entanglement ≠ redundancy)."""
    psi = (_ket("000") + _ket("111")) / np.sqrt(2.0)
    return np.outer(psi, psi.conj())


_ROLES = ["a", "b", "c"]


# ── exact tier: the canonical states, known values ────────────────────────────────

def test_product_state_is_flat():
    """A product cumulant gates for free: no correlation ⇒ TC=0, Ω=0, flat, without touching a ρ."""
    ho = HO.cluster_higher_order(_Cumulant(_ROLES))          # all-zero e1/e2 = maximally mixed product
    assert ho["total_correlation"] == 0.0 and ho["o_information"] == 0.0
    assert ho["grain"] == "flat" and ho["tier"] == "gated"


def test_classical_ghz_is_redundancy_exact():
    """A classical common cause is a CHORUS: Ω>0. TC=2 (three bound bits over one shared bit)."""
    ho = HO.cluster_higher_order(_Dense(_classical_ghz(), _ROLES))
    assert abs(ho["total_correlation"] - 2.0) < 1e-9
    assert abs(ho["o_information"] - 1.0) < 1e-9
    assert ho["grain"] == "chorus" and ho["tier"] == "exact"


def test_xor_is_synergy_exact():
    """Parity is a CONSPIRACY: Ω<0. TC=1 lives ENTIRELY in the higher-order structure — the pairwise
    correlation (max‖C‖) is exactly zero, so no pairwise sensor could ever see this binding."""
    ho = HO.cluster_higher_order(_Dense(_xor_parity(), _ROLES))
    assert abs(ho["total_correlation"] - 1.0) < 1e-9
    assert abs(ho["o_information"] + 1.0) < 1e-9
    assert ho["grain"] == "conspiracy" and ho["tier"] == "exact"
    assert ho["max_corr_norm"] < 1e-9                        # all binding is higher-order


def test_pure_ghz_is_balanced():
    """The honesty pin: a pure GHZ is strongly bound (TC=3) but Ω=0 — entanglement is NOT redundancy.
    (For a pure tripartite state S({i}) = S(all-but-i) by Schmidt, so the redundancy cancels.)"""
    ho = HO.cluster_higher_order(_Dense(_pure_ghz(), _ROLES))
    assert abs(ho["total_correlation"] - 3.0) < 1e-9
    assert abs(ho["o_information"]) < 1e-9
    assert ho["grain"] == "flat" and ho["tier"] == "exact"


# ── the two honest limits, as passing assertions (the DENIED-tier rows) ──────────

def test_cumulant_sees_binding_but_cannot_sign_grain():
    """The cumulant FEELS the bind but cannot NAME its character: on classical-GHZ it reports a real
    total correlation (proxy tier) yet o_information is None and grain is 'flat' — where the dense
    backend, holding the full ρ, reads 'chorus'."""
    rho = _classical_ghz()
    cum = HO.cluster_higher_order(_cumulant_from_rho(rho, _ROLES))
    dense = HO.cluster_higher_order(_Dense(rho, _ROLES))
    assert cum["tier"] == "proxy" and cum["total_correlation"] > 0.5    # it feels the bind
    assert cum["o_information"] is None and cum["grain"] == "flat"      # it cannot name it
    assert dense["grain"] == "chorus"                                  # the dense backend can


def test_cumulant_is_blind_to_pure_synergy():
    """DENIED, with numbers: on a pure-synergy (XOR) state the cumulant reads TC≈0 and grain='flat'
    — utterly featureless — while the dense backend reads TC=1, Ω=−1, 'conspiracy'. The pairwise
    cumulants the closure keeps carry none of the parity structure."""
    rho = _xor_parity()
    cum = HO.cluster_higher_order(_cumulant_from_rho(rho, _ROLES))
    dense = HO.cluster_higher_order(_Dense(rho, _ROLES))
    assert cum["total_correlation"] < 1e-6 and cum["grain"] == "flat"
    assert dense["o_information"] < -0.5 and dense["grain"] == "conspiracy"


def test_tc_proxy_matches_exact_in_weak_regime():
    """With maximally-mixed marginals and a small connected zz, the pairwise TC proxy κ·Σ‖C‖²
    matches the exact TC (= the pair's mutual information at n=2) to a fraction of a percent."""
    e2 = np.zeros((3, 3))
    e2[2, 2] = 0.05
    rho = _rho2(np.zeros(3), np.zeros(3), e2)
    tc_exact = HO.total_correlation_exact(_Dense(rho, ["a", "b"]))
    tc_proxy, _cn = HO.total_correlation_proxy(_cumulant_from_rho(rho, ["a", "b"]))
    assert tc_exact > 0.0
    assert abs(tc_proxy - tc_exact) < 0.01 * tc_exact


def test_o_information_vanishes_to_second_order():
    """The proof of WHY there is no cumulant grain: for a 2-body-only state scaled by eps, the
    O-information falls away faster than the total correlation — |Ω|/TC strictly shrinks toward 0.
    So a second-order (pairwise) surrogate for Ω has no leading term; the sign is not recoverable."""
    Z, I2 = np.diag([1.0, -1.0]), np.eye(2)
    zzi = np.kron(np.kron(Z, Z), I2)
    izz = np.kron(np.kron(I2, Z), Z)
    ziz = np.kron(np.kron(Z, I2), Z)
    ratios = []
    for eps in (0.08, 0.04, 0.02, 0.01):
        rho = (np.eye(8) + eps * (zzi + izz + ziz)) / 8.0     # only 2-body zz correlations; PSD for eps≤1
        ho = HO.cluster_higher_order(_Dense(rho, _ROLES))
        ratios.append(abs(ho["o_information"]) / ho["total_correlation"])
    assert all(ratios[k + 1] < ratios[k] for k in range(len(ratios) - 1))   # strictly shrinking
    assert ratios[-1] < 0.01                                                # → 0 (measured 6.6e-3)


def test_field_scan_finds_grain_only_in_the_bound_cluster():
    """field_higher_order_scan reports a non-flat grain only where a dense cluster genuinely binds;
    a product cluster stays flat. cumulant_only mirrors field_mi_scan (skips dense clusters)."""
    field = type("F", (), {"clusters": {
        "bound": _Dense(_classical_ghz(), _ROLES),
        "loose": _Dense(np.eye(8) / 8.0, _ROLES),
        "cum": _Cumulant(["p", "q"]),
    }})()
    got = HO.field_higher_order_scan(field)
    assert got["bound"]["grain"] == "chorus"
    assert got["loose"]["grain"] == "flat"
    # cumulant_only skips the dense clusters entirely
    only = HO.field_higher_order_scan(field, cumulant_only=True)
    assert set(only) == {"cum"}


def test_single_role_cluster_has_no_higher_order():
    """A one-belief cluster has nothing to bind — None, not a fabricated zero-structure."""
    assert HO.cluster_higher_order(_Cumulant(["solo"])) is None
