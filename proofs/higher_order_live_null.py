"""proofs/higher_order_live_null.py — WHY higher-order binding is (mostly) decorative on the LIVE field.

The higher-order sensors (grain/TC/constellations) are pinned true on hand-built dense states, but a
live belief-field takes evidence through LOCAL single-qubit observations that FACTORIZE the joint state.
This proof pins, with numbers, the architectural limit that R1 first surfaced on the real tape — so the
honest bound can't rot into an overclaim:

  1. the sensor is CORRECT — an installed one-hot mixture reads TC>1, grain='conspiracy';
  2. LOCAL observations cannot BUILD binding — driving every qubit one-hot from a product start → TC≈0;
  3. a diagonal (ZZ/number) Hamiltonian is POWERLESS on a diagonal state (it commutes) — seeding it at
     two strengths gives byte-identical evolution;
  4. EXCHANGE (xy) builds TC only in ISOLATION (unobserved) — and live observation crushes it to ≈0;
  5. THE ONE CRACK — a LATENT (unobserved) concept-qubit exchange-coupled to observed beliefs sustains a
     small live boundary correlation (a live constellation). TC/constellations are marginally live via
     latent concepts; grain stays at the numerical floor (still decorative live).

Run: cd /home/primearchitect/ws/umwelt && PYTHONPATH=. python proofs/higher_order_live_null.py
"""
from __future__ import annotations

import numpy as np

from umwelt.substrate.cluster import QubitCluster
from umwelt.substrate.higher_order import cluster_higher_order
from umwelt.substrate.mutual_information import connected_correlation

_X = np.array([[0, 1], [1, 0]], complex)
_Y = np.array([[0, -1j], [1j, 0]], complex)
_Z = np.array([[1, 0], [0, -1]], complex)
_I = np.eye(2, dtype=complex)


def _op_at(op, q, n):
    m = np.array([[1.0]], complex)
    for k in range(n):
        m = np.kron(m, op if k == q else _I)
    return m


def _unitary_cluster(n, gamma=0.05, gamma_diss=0.0):
    roles = [f"q{i}" for i in range(n)]
    return QubitCluster("t", roles, gamma=gamma, gamma_diss=gamma_diss,
                        role_modes={r: "unitary" for r in roles})


def _one_hot_mixture(n):
    dim = 2 ** n
    rho = np.zeros((dim, dim), complex)
    for i in range(n):
        k = 1 << (n - 1 - i)
        rho[k, k] = 1.0 / n
    return rho


def _exchange_H(pairs, n, J):
    dim = 2 ** n
    H = np.zeros((dim, dim), complex)
    for i, j in pairs:
        H += J * (_op_at(_X, i, n) @ _op_at(_X, j, n) + _op_at(_Y, i, n) @ _op_at(_Y, j, n))
    return H


def _zz_H(n, J):
    dim = 2 ** n
    H = np.zeros((dim, dim), complex)
    for i in range(n):
        for j in range(i + 1, n):
            H += J * (_op_at(_Z, i, n) @ _op_at(_Z, j, n))
    return H


def _tc(c):
    return cluster_higher_order(c)["total_correlation"]


def _boundary_corr(c, a, b):
    """‖connected correlation‖ between qubits a and b (the live constellation bond)."""
    def bloch(q):
        rdm = np.asarray(c.subsystem_rdm([q]), complex)
        return np.array([np.real(np.trace(rdm @ P)) for P in (_X, _Y, _Z)])

    rdm2 = np.asarray(c.subsystem_rdm([a, b]), complex)
    e2 = np.zeros((3, 3))
    P = (_X, _Y, _Z)
    for i in range(3):
        for j in range(3):
            e2[i, j] = np.real(np.trace(rdm2 @ np.kron(P[i], P[j])))
    return float(np.linalg.norm(connected_correlation(bloch(a), bloch(b), e2)))


# ── the five pins ──────────────────────────────────────────────────────────────────

def test_sensor_reads_installed_binding():
    """The sensor is CORRECT: an installed one-hot mixture is real binding — TC>1, grain='conspiracy'."""
    c = _unitary_cluster(4)
    c.rho = _one_hot_mixture(4)
    ho = cluster_higher_order(c)
    assert ho["total_correlation"] > 1.0
    assert ho["grain"] == "conspiracy"


def test_local_observation_cannot_build_binding():
    """LOCAL observation factorizes: driving every qubit one-hot from a product start never binds → TC≈0."""
    c = _unitary_cluster(4, gamma=0.05)
    for tick in range(60):
        occ = tick % 4
        for q in range(4):
            c.observe_qubit(q, (0.0, 0.0, -1.0 if q == occ else 1.0), alpha=0.3, confidence=0.7)
        c.step()
    assert _tc(c) < 0.01                     # measured ~0.0000; the joint stays product


def test_diagonal_hamiltonian_is_powerless_on_diagonal_state():
    """A diagonal ZZ commutes with the diagonal one-hot mixture → it does NOTHING; two strengths evolve
    identically. So no ZZ web can create or sustain classical binding."""
    def evolve(J):
        c = _unitary_cluster(4, gamma=0.05)
        c.set_hamiltonian(_zz_H(4, J))
        c.rho = _one_hot_mixture(4)
        for _ in range(40):
            c.step()
        return _tc(c)
    assert abs(evolve(0.5) - evolve(2.0)) < 1e-9


def test_exchange_builds_binding_only_in_isolation():
    """EXCHANGE delocalizes a single excitation into a correlated state (TC>0.5) when UNOBSERVED — but
    live observation crushes it to ≈0. Binding is unreachable under realistic evidence."""
    # isolated: single excitation, no dissipation, exchange only
    n = 4
    c = _unitary_cluster(n, gamma=0.0, gamma_diss=0.0)
    c.set_hamiltonian(_exchange_H([(i, j) for i in range(n) for j in range(i + 1, n)], n, 0.3))
    v = np.zeros(2 ** n, complex); v[1 << (n - 1)] = 1.0
    c.rho = np.outer(v, v.conj())
    for _ in range(40):
        c.step()
    tc_isolated = _tc(c)
    # observed: same exchange, but drive one-hot every tick
    c2 = _unitary_cluster(n, gamma=0.05)
    c2.set_hamiltonian(_exchange_H([(i, j) for i in range(n) for j in range(i + 1, n)], n, 0.3))
    for tick in range(60):
        occ = tick % n
        for q in range(n):
            c2.observe_qubit(q, (0.0, 0.0, -1.0 if q == occ else 1.0), alpha=0.3, confidence=0.7)
        c2.step()
    tc_observed = _tc(c2)
    assert tc_isolated > 0.5                  # exchange binds when unobserved (measured ~1.9)
    assert tc_observed < 0.05                 # observation crushes it (measured ~0.0001)


def test_latent_concept_sustains_small_live_binding():
    """THE CRACK: an UNOBSERVED concept qubit exchange-coupled to observed beliefs sustains a small live
    boundary correlation (a live constellation) — TC/constellations are marginally live via latent
    concepts, even though grain stays at the floor. The one live-helpful use of the higher-order sensor."""
    n = 4                                     # q0,q1,q2 observed; q3 = LATENT concept
    latent = 3
    c = _unitary_cluster(n, gamma=0.02)
    c.set_hamiltonian(_exchange_H([(latent, 0), (latent, 1), (latent, 2)], n, 0.4))
    for tick in range(40):
        z = 0.8 if (tick // 4) % 2 == 0 else -0.8      # common oscillating cause
        for q in (0, 1, 2):
            c.observe_qubit(q, (0.0, 0.0, z), alpha=0.3, confidence=0.6)
        for _ in range(3):
            c.step()
    # the latent concept has bound to the observed beliefs (sustained, well above the MI floor 1e-4)
    assert _boundary_corr(c, latent, 0) > 0.01
    assert _tc(c) > 1e-3


def main(argv=None) -> int:
    for n, fn in [(1, test_sensor_reads_installed_binding),
                  (2, test_local_observation_cannot_build_binding),
                  (3, test_diagonal_hamiltonian_is_powerless_on_diagonal_state),
                  (4, test_exchange_builds_binding_only_in_isolation),
                  (5, test_latent_concept_sustains_small_live_binding)]:
        fn()
        print(f"[higher_order_live_null] pin {n} ok: {fn.__name__}")
    print("[higher_order_live_null] all pins green — the architectural bound holds")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
