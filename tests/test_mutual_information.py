"""The fast MI sensor: ||connected-correlation|| gates the exact von Neumann MI, with
MI(bits) ≈ (1/2ln2)||C||^2 in the weak regime. Pins the math the sensor rides on."""
from __future__ import annotations

import numpy as np

from umwelt.substrate import mutual_information as MI


def test_product_state_gates_to_zero():
    """C = 0 (joint factorizes) ⇒ the gate returns exactly 0 without touching a density matrix."""
    e1i = np.array([0.0, 0.0, 0.5])
    e1j = np.array([0.2, 0.0, 0.3])
    e2 = np.outer(e1i, e1j)                       # product by construction
    mi, cn, tier = MI.mi_pair_bits(e1i, e1j, e2)
    assert cn < 1e-12 and mi == 0.0 and tier == "gated"


def test_bell_state_is_two_bits():
    """A maximally-entangled pair |Φ+> has I(A:B) = 2 bits — the exact-tier path must find it."""
    e1i = np.zeros(3)
    e1j = np.zeros(3)
    e2 = np.diag([1.0, -1.0, 1.0])               # <xx>=1, <yy>=-1, <zz>=1  ⇒ |Φ+>
    mi = MI.mi_exact_bits(e1i, e1j, e2)
    assert abs(mi - 2.0) < 1e-6
    # and it routes to the exact tier (||C|| = sqrt(3) is well above `strong`)
    _mi, cn, tier = MI.mi_pair_bits(e1i, e1j, e2)
    assert tier == "exact" and abs(cn - np.sqrt(3.0)) < 1e-9


def test_proxy_matches_exact_in_weak_regime():
    """With maximally-mixed marginals and a small connected zz, the leading-order proxy
    MI ≈ 0.7213||C||^2 matches the full von Neumann MI to a fraction of a percent."""
    e1i = np.zeros(3)
    e1j = np.zeros(3)
    e2 = np.zeros((3, 3))
    e2[2, 2] = 0.05                              # a little <zz> connected correlation
    cn = MI.corr_norm(e1i, e1j, e2)
    exact = MI.mi_exact_bits(e1i, e1j, e2)
    proxy = MI.mi_proxy_bits(cn)
    assert cn > 0.0 and exact > 0.0
    assert abs(proxy - exact) < 0.01 * exact     # agree to < 1% in the weak regime
    assert abs(MI.MI_BITS_COEFF - 1.0 / (2.0 * np.log(2.0))) < 1e-12


def test_cluster_scan_finds_only_the_correlated_pair():
    """The gated scan surfaces the one correlated pair and skips the product pairs (cheap gate)."""
    class _FakeCumulant:
        qubit_roles = ["a", "b", "c"]

        def __init__(self):
            self.e1 = np.zeros((3, 3))          # all maximally mixed
            self.e2 = np.zeros((3, 3, 3, 3))    # all product (outer of zeros)
            self.e2[0, 1, 2, 2] = 0.5           # correlate a~b via connected <zz>
            self.e2[1, 0, 2, 2] = 0.5

    pairs = MI.cluster_mi_scan(_FakeCumulant(), floor=1e-4)
    got = {(p["role_i"], p["role_j"]) for p in pairs}
    assert got == {("a", "b")}                   # only the correlated pair clears the gate
    ab = pairs[0]
    assert ab["mi_bits"] > 0.0 and ab["corr_norm"] > 0.4
