"""Information geometry — the belief-manifold's VI metric and its constellations.

Pins that variation of information is a true metric on the separable/classical regime the field
lives in, documents the entangled degeneracy (VI=−2 on a Bell pair) so the caveat can't rot, and
pins that MI-graph communities split the roles into constellations. Sibling of test_higher_order.py.
"""
from __future__ import annotations

import itertools

import numpy as np

from umwelt.substrate import information_geometry as IG
from umwelt.substrate.cluster import partial_trace_keep
from umwelt.substrate.mutual_information import _vn_bits


class _Dense:
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
    def __init__(self, roles, corr=()):
        n = len(roles)
        self.qubit_roles = list(roles)
        self.n_qubits = n
        self.e1 = np.zeros((n, 3))
        self.e2 = np.zeros((n, n, 3, 3))
        for i, j, v in corr:                       # connected <zz> between roles i and j
            self.e2[i, j, 2, 2] = v
            self.e2[j, i, 2, 2] = v


def _diag_rho(p: dict) -> np.ndarray:
    """A classical joint distribution over 3 bits → a diagonal (classical) 8×8 density matrix."""
    r = np.zeros((8, 8), dtype=complex)
    for bits, pr in p.items():
        r[int(bits, 2), int(bits, 2)] = pr
    return r


# a spread of classical states: independent, perfectly-correlated, a chain, and an asymmetric one
_CLASSICAL = {
    "indep": {f"{a}{b}{c}": 1 / 8 for a in "01" for b in "01" for c in "01"},
    "ghz_diag": {"000": 0.5, "111": 0.5},
    "chain": {"000": 0.25, "001": 0.25, "110": 0.25, "111": 0.25},   # b1 = b0, b2 independent
    "skew": {"000": 0.4, "011": 0.1, "101": 0.1, "110": 0.1, "111": 0.3},
}
_ROLES = ["a", "b", "c"]


def test_vi_is_a_metric():
    """On the separable/classical regime: VI is symmetric, zero on the diagonal, non-negative, and
    obeys the triangle inequality — a genuine metric on the belief-manifold."""
    for p in _CLASSICAL.values():
        _roles, D = IG.vi_distance_matrix(_Dense(_diag_rho(p), _ROLES))
        assert np.allclose(D, D.T)                          # symmetric
        assert np.allclose(np.diag(D), 0.0)                 # zero on the diagonal
        assert (D >= -1e-9).all()                           # non-negative
        for i, j, k in itertools.permutations(range(3)):
            if len({i, j, k}) == 3:
                assert D[i, k] <= D[i, j] + D[j, k] + 1e-9  # triangle inequality


def test_vi_breaks_on_bell():
    """The documented degeneracy: VI = 2S(AB) − S(A) − S(B) uses quantum conditional entropies that
    go negative on a genuinely entangled pair — a Bell pair gives VI = −2. VI is a metric only on the
    separable regime; this pin keeps the caveat honest."""
    phi = np.zeros(4, dtype=complex)
    phi[0] = phi[3] = 1.0 / np.sqrt(2.0)                    # |Φ+> = (|00> + |11>)/√2
    bell = np.outer(phi, phi.conj())
    vi = IG.variation_of_information(_Dense(bell, ["a", "b"]), 0, 1)
    assert abs(vi + 2.0) < 1e-9


def test_constellations_group_binding_roles():
    """The MI graph's communities: a~b share information, c is on its own → two constellations."""
    cl = _Cumulant(["a", "b", "c"], corr=[(0, 1, 0.6)])
    assert IG.constellations(cl) == [["a", "b"], ["c"]]


def test_constellations_split_disjoint_pairs_and_keep_singletons():
    """Two disjoint bound pairs stay two constellations; an all-product cluster is all singletons
    (every belief keeps a home — the field never drops a role)."""
    two = _Cumulant(["a", "b", "c", "d"], corr=[(0, 1, 0.6), (2, 3, 0.5)])
    assert IG.constellations(two) == [["a", "b"], ["c", "d"]]
    loose = _Cumulant(["a", "b", "c"])                      # all product
    assert IG.constellations(loose) == [["a"], ["b"], ["c"]]
