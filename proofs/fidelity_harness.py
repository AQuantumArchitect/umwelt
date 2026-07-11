"""Cumulant-closure fidelity harness — full-ρ vs cumulant on the SAME stream.

Ported from the origin deployment's safety rail (meerkat experiments/
fidelity_harness.py): drive the exact 2ⁿ QubitCluster and the O(n²) CumulantCluster
side by side — same initial state, same Hamiltonian (fields + couplings), same
dissipation, same sparse observation stream through the production observe path — and
report what the closure loses where a consumer can feel it:

    z divergence      per-leaf |z_dense − z_cumulant| (mean / max over the tape)
    purity divergence |purity_dense − purity_cumulant| (the confidence gauge)
    decision parity   sign(z) agreement rate (the collapse engine reads sign + |z|)

PORTING DECISION: the origin harness also accepted its recorded 24-day cassette; the
tape stays with the origin. Here the stream is the seeded sparse random walk over the
gridworld adjacency (proofs/ladder_walk.gridworld_stream) — the comparison machinery
is substrate-only and identical; what this file MEASURES is measured here, in-repo,
on that synthetic stream (see the tests below), and no origin number is re-claimed.

VERDICT OF RECORD (the origin deployment, 2026-07-04 — real 24-day cassette, 6908
bins, 4 nodes; quoted as the decision it produced, not re-measurable here):

    ZZ couplings (the production shape):     z divergence mean 0.0006, max 0.0104;
                                             purity divergence mean 0.0018, max 0.0100;
                                             decision parity 100.00%
    exchange couplings (the stress case):    z divergence mean 0.0278, max 0.7034;
                                             decision parity 97.21%

  On the coupling class production actually used (ZZ), the closure never moved a
  decision over 24 real days; under strong sector-mixing exchange the truncation
  honestly shows — the closure is the tractability trade, not a free lunch: grow
  exchange couplings from zero and validate here before trusting large ones.
"""
from __future__ import annotations

import numpy as np
import pytest

from proofs.ladder_walk import DT_SCALE, NODES, PAIRS, gridworld_stream
from umwelt.substrate.bloch import qubit_purity
from umwelt.substrate.cluster import QubitCluster
from umwelt.substrate.cumulant_cluster import CumulantCluster
from umwelt.substrate.density_matrix import pauli_x, pauli_y, pauli_z

ALPHA = 0.6
J = 0.6
H_FIELDS = (0.05, 0.0, 0.02)      # small common single-qubit drive (x tips, z bias)


def _dense_H(n: int, exchange: bool) -> np.ndarray:
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i in range(n):
        hx, hy, hz = H_FIELDS
        H += hx * pauli_x(i, n) + hy * pauli_y(i, n) + hz * pauli_z(i, n)
    for a, b in PAIRS:
        i, j = NODES.index(a), NODES.index(b)
        if exchange:
            H += J * (pauli_x(i, n) @ pauli_x(j, n) + pauli_y(i, n) @ pauli_y(j, n))
        else:
            H += J * (pauli_z(i, n) @ pauli_z(j, n))
    return H


def _clusters(exchange: bool):
    n = len(NODES)
    dense = QubitCluster("fid", list(NODES), gamma=0.02, dt=0.01,
                         role_modes={r: "unitary" for r in NODES})
    dense.set_hamiltonian(_dense_H(n, exchange))
    cum = CumulantCluster("fid", list(NODES), gamma=0.02, dt=0.01,
                          role_modes={r: "unitary" for r in NODES},
                          connectivity=[(a, b) for a, b in PAIRS])
    pairs_idx = {(NODES.index(a), NODES.index(b)): J for a, b in PAIRS}
    if exchange:
        cum.set_couplings(h_fields=[H_FIELDS] * n,
                          xy={p: (J, J) for p in pairs_idx})
    else:
        cum.set_couplings(h_fields=[H_FIELDS] * n, zz=pairs_idx)
    return dense, cum


def fidelity(reports: np.ndarray | None = None, exchange: bool = False,
             max_bins: int | None = None) -> dict:
    """Drive both substrates over the sparse report stream (default: the deterministic
    gridworld walk) and report the closure's divergence + decision parity."""
    if reports is None:
        reports, _ = gridworld_stream()
    if max_bins:
        reports = reports[:max_bins]
    T, R = reports.shape
    dense, cum = _clusters(exchange)
    dz = np.zeros((T, R))
    dp = np.zeros(T)
    signs = np.zeros((T, R), dtype=bool)
    for t in range(T):
        dense.step(dt_scale=DT_SCALE)
        cum.step(dt_scale=DT_SCALE)
        for k in range(R):
            z = reports[t, k]
            if np.isnan(z):
                continue
            # the PRODUCTION rule (the origin's ladder verdict: the α-blend stays)
            dense.observe_qubit(k, (0.0, 0.0, float(z)), alpha=ALPHA)
            cum.observe_qubit(k, (0.0, 0.0, float(z)), alpha=ALPHA)
        bd = np.array([dense.qubit_bloch(k) for k in range(R)])
        zd, zc = bd[:, 2], cum.e1[:, 2]
        dz[t] = np.abs(zd - zc)
        # per-LEAF purity (the gauge a consumer reads), same (1+|r|²)/2 form both
        # sides — the joint Tr(ρ²) and the cumulant's per-qubit mean are different
        # quantities and comparing them would report a fake divergence.
        dp[t] = float(np.max(np.abs(qubit_purity(bd) - qubit_purity(cum.e1))))
        signs[t] = np.sign(zd) == np.sign(zc)
    print(f"fidelity on synthetic gridworld stream "
          f"({T} bins, {'exchange' if exchange else 'ZZ'} couplings)")
    print(f"  z divergence:      mean {dz.mean():.4f}, max {dz.max():.4f}")
    print(f"  purity divergence: mean {dp.mean():.4f}, max {dp.max():.4f}")
    print(f"  decision parity:   {100.0 * signs.mean():.2f}% sign agreement")
    return {"dz_mean": float(dz.mean()), "dz_max": float(dz.max()),
            "dp_mean": float(dp.mean()), "dp_max": float(dp.max()),
            "sign_agreement": float(signs.mean())}


# ── the pytest face: what the closure loses is MEASURED HERE on the synthetic walk ──

@pytest.mark.slow
def test_closure_tracks_full_rho_on_zz_couplings():
    """The production coupling class: the O(n²) closure must track the exact 2ⁿ state
    to reading precision and never move a decision — measured on this repo's stream."""
    m = fidelity(exchange=False)
    assert m["sign_agreement"] >= 0.99, m
    assert m["dz_mean"] < 0.02, m
    assert m["dp_mean"] < 0.05, m


@pytest.mark.slow
def test_closure_divergence_shows_honestly_under_exchange_stress():
    """The stress case: sector-mixing exchange is where the truncation is allowed to
    show. The harness must still run, stay finite, and keep majority decision parity —
    and the divergence must be VISIBLE (a stress case that reads as clean as ZZ would
    mean the dial isn't actually stressing anything)."""
    m = fidelity(exchange=True)
    assert all(np.isfinite(v) for v in m.values()), m
    assert m["sign_agreement"] >= 0.8, m
    assert m["dz_max"] > 0.01, "exchange stress showed no divergence at all — dead dial?"
