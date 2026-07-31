"""Manifold fitness — the kiss (structure_fitness) and the father (select), pinned on canonical states.

Pins the real numbers the functional gives, and the two inherited DENIED limits as PASSING assertions:
the cumulant cannot earn a grain bonus (L136, DENSE-ONLY) and an entangled pair cannot inflate fitness
through VI (L137, SEPARABLE-REGIME-ONLY). Reuses the canonical builders from test_higher_order.py.
"""
from __future__ import annotations

import numpy as np

from umwelt.substrate import manifold_fitness as MF
from umwelt.substrate.cluster import partial_trace_keep
from umwelt.substrate.mutual_information import _pair_stats_from_rho, _vn_bits


# ── fixtures (the same read interfaces + canonical states as test_higher_order.py) ──

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
    def __init__(self, roles):
        n = len(roles)
        self.qubit_roles = list(roles)
        self.n_qubits = n
        self.e1 = np.zeros((n, 3))
        self.e2 = np.zeros((n, n, 3, 3))


def _cumulant_from_rho(rho, roles):
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


def _classical_ghz():
    return 0.5 * (np.outer(_ket("000"), _ket("000")) + np.outer(_ket("111"), _ket("111")))


def _xor_parity():
    return sum(np.outer(_ket(b), _ket(b)) for b in ["000", "011", "101", "110"]) / 4.0


def _pure_ghz():
    psi = (_ket("000") + _ket("111")) / np.sqrt(2.0)
    return np.outer(psi, psi.conj())


def _bell():
    psi = (_ket("00") + _ket("11")) / np.sqrt(2.0)
    return np.outer(psi, psi.conj())


def _classical_pair():
    return 0.5 * (np.outer(_ket("00"), _ket("00")) + np.outer(_ket("11"), _ket("11")))


def _pair_then_free():
    """a,b classically correlated ⊗ c maximally mixed — a BIGGER but LOOSER cluster (TC=1 over 3 qubits)."""
    return np.kron(_classical_pair(), np.eye(2) / 2.0)


_R = ["a", "b", "c"]


# ── the kiss: fitness ordering + honest components ────────────────────────────────

def test_fitness_ordering():
    """Neutral weights: product(0) < XOR(binding-only) < classical-GHZ < pure-GHZ (most bound)."""
    f_prod = MF.structure_fitness(_Dense(np.eye(8) / 8.0, _R))["fitness"]
    f_xor = MF.structure_fitness(_Dense(_xor_parity(), _R))["fitness"]
    f_cghz = MF.structure_fitness(_Dense(_classical_ghz(), _R))["fitness"]
    f_pghz = MF.structure_fitness(_Dense(_pure_ghz(), _R))["fitness"]
    assert abs(f_prod - 0.0) < 1e-9
    assert abs(f_xor - 1.0 / 3.0) < 1e-9
    assert abs(f_cghz - (2.0 / 3.0 + 0.5)) < 1e-9        # bd 2/3 + integration·0.5
    assert abs(f_pghz - 1.5) < 1e-9                       # bd 1 + integration·0.5
    assert f_prod < f_xor < f_cghz < f_pghz


def test_xor_earns_binding_not_integration():
    """The parity honesty: XOR binds (binding_density>0) but earns ZERO integration credit — the MI
    graph is blind to synergy, so no concept forms. classical-GHZ integrates fully."""
    xor = MF.structure_fitness(_Dense(_xor_parity(), _R))
    cghz = MF.structure_fitness(_Dense(_classical_ghz(), _R))
    assert xor["binding_density"] > 0.3 and xor["integration"] == 0.0
    assert cghz["integration"] == 1.0


def test_chorus_preference_is_tunable():
    """The father's value direction: at neutral pure-GHZ (most bound) wins; once w_chorus>0 rewards
    named redundancy, the classical chorus overtakes it — a value judgment, not math."""
    cghz, pghz = _Dense(_classical_ghz(), _R), _Dense(_pure_ghz(), _R)
    neutral = MF.select({"cghz": cghz, "pghz": pghz})
    assert neutral["chosen"] == ["pghz"]
    chorus = MF.select({"cghz": cghz, "pghz": pghz}, weights=MF.FitnessWeights(w_chorus=1.0))
    assert chorus["chosen"] == ["cghz"] and chorus["grain_used"] is True


def test_bigger_but_looser_loses():
    """Size normalization guards the 'biggest union always wins' degeneracy: a tight n=2 cluster beats
    a bigger n=3 cluster carrying the same total binding spread thinner."""
    tight = MF.structure_fitness(_Dense(_classical_pair(), ["a", "b"]))
    loose = MF.structure_fitness(_Dense(_pair_then_free(), _R))
    assert tight["binding_density"] > loose["binding_density"]
    assert tight["fitness"] > loose["fitness"]


# ── the two inherited DENIED limits, as passing assertions ────────────────────────

def test_grain_blind_on_cumulant():
    """DENIED, DENSE-ONLY (L136): a cumulant candidate cannot earn a grain bonus — its fitness is
    invariant to w_chorus (grain_known False), a binding LOWER BOUND. The dense same-state can."""
    cum = _cumulant_from_rho(_classical_ghz(), _R)
    f0 = MF.structure_fitness(cum, weights=MF.FitnessWeights(w_chorus=0.0))
    f5 = MF.structure_fitness(cum, weights=MF.FitnessWeights(w_chorus=5.0))
    assert f0["grain_known"] is False
    assert abs(f0["fitness"] - f5["fitness"]) < 1e-12          # grain never enters on proxy
    dense = _Dense(_classical_ghz(), _R)
    d0 = MF.structure_fitness(dense, weights=MF.FitnessWeights(w_chorus=0.0))
    d5 = MF.structure_fitness(dense, weights=MF.FitnessWeights(w_chorus=5.0))
    assert d0["grain_known"] is True
    assert abs((d5["fitness"] - d0["fitness"]) - 5.0) < 1e-9   # + w_chorus·Ω, Ω=+1


def test_select_drops_grain_cross_tier():
    """A cross-tier contest (exact vs proxy) cannot be decided by a grain a proxy can't have —
    grain_used is False even with w_chorus set."""
    res = MF.select({"dense": _Dense(_classical_ghz(), _R), "cum": _cumulant_from_rho(_classical_ghz(), _R)},
                    weights=MF.FitnessWeights(w_chorus=1.0))
    assert res["grain_used"] is False
    assert set(res["tiers_seen"]) == {"exact", "proxy"}
    # The mechanism must actually bite: with grain neutralized, the proxy (higher TC-proxy) wins — NOT
    # the dense candidate riding a +w_chorus·Ω bonus a proxy could never earn. (Removing the drop flips
    # this to 'dense'.)
    assert res["chosen"] == ["cum"]


def test_vi_component_clamped():
    """DENIED, SEPARABLE-REGIME-ONLY (L137): a Bell pair gives VI=−2, but vi_compactness clamps to
    [0,1] so it cannot inflate fitness; the scorecard stays finite."""
    bell = MF.structure_fitness(_Dense(_bell(), ["a", "b"]))
    assert 0.0 <= bell["vi_compactness"] <= 1.0
    assert np.isfinite(bell["fitness"])
    assert bell["grain_known"] is False                       # Ω undefined for a pair


# ── the father: cost-weighted ranking ──────────────────────────────────────────────

def test_select_ranks_by_fitness_and_cost_weight():
    """Compute scarcity is operational: at low cost_weight the big high-fitness dense candidate wins;
    crank the cost penalty and the cheap cumulant (n^3 vs 4^n) takes it — the deficit made a choice."""
    from umwelt.substrate.compute_demand import compute_cost_estimate
    cands = {"big_dense": _Dense(_pure_ghz(), _R), "cheap_cum": _cumulant_from_rho(_classical_ghz(), _R)}
    cost_of = lambda c: compute_cost_estimate(c)["cost"]
    cheap = MF.select(cands, cost_weight=0.0, cost_of=cost_of)
    dear = MF.select(cands, cost_weight=0.02, cost_of=cost_of)
    assert cheap["chosen"] == ["big_dense"]                   # fitness alone → the bound dense one
    assert dear["chosen"] == ["cheap_cum"]                    # cost penalty → the scalable cheap one


def test_single_role_is_none():
    assert MF.structure_fitness(_Cumulant(["solo"])) is None


def test_field_fitness_scan_filters():
    """Self-similar sweep; cumulant_only skips dense clusters (mirrors field_higher_order_scan)."""
    field = type("F", (), {"clusters": {
        "bound": _Dense(_classical_ghz(), _R),
        "cum": _cumulant_from_rho(_classical_ghz(), _R),
        "solo": _Cumulant(["x"]),
    }})()
    allc = MF.field_fitness_scan(field)
    assert set(allc) == {"bound", "cum"} and "solo" not in allc
    only = MF.field_fitness_scan(field, cumulant_only=True)
    assert set(only) == {"cum"}
