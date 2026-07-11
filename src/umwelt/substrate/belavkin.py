"""Belavkin weak-measurement update — the principled sibling of observe_qubit.

The α-blend in observe_qubit is an ad-hoc convex mix (NOVELTY.md §A2 names the
simplification: "no measurement back-action noise term… dropped the stochastic
innovation"). This module implements the real thing at the single-qubit level — the
discrete Kraus form of the Belavkin/quantum-filtering update for a weak σ_z
measurement — plus the second-cumulant cross-update that lets a measurement of one
qubit move its correlated peers. See docs/QUANTUM_KALMAN.md (the estimator ladder);
this is rung L4 at cumulant order 2.

The measurement model
─────────────────────
A sensor reading on an event/discrete leaf is a noisy draw around the eigenvalues ±1
(an occupancy detector says "occupied-ish", not "the mean is 0.3"). The conditioned update
for record y with effective strength s ≥ 0 is the two-outcome Gaussian Kraus

    M(y) = diag(c₊, c₋),   c± = exp(−s·(y ∓ 1)² / 4),   ρ' = MρM† / Tr(MρM†)

which on the Bloch vector (x, y_b, z) works out to closed form with w± = c±²:

    N   = w₊(1+z) + w₋(1−z)              (2 × the normalization)
    z'  = (w₊(1+z) − w₋(1−z)) / N
    x', y'_b = (2·c₊c₋ / N) · (x, y_b)   (coherence factor ≤ 1: back-action)

Equivalently z' = tanh(atanh(z) + s·y): **the update is a shift in log-odds space by
s·y** — classical Bayesian conditioning of a two-state belief, with the coherence
damping riding along. Everything the ladder promises is visible here:

  • Wonham gain — weak limit Δz ≈ s·(1−z²)·y: the gain carries the bounded binary
    variance (1−z²); poles stop listening, the equator listens hardest.
  • Provable no-op — s = 0 ⇒ w₊ = w₋ ⇒ z' = z and the coherence factor is exactly 1.
    The confidence contract (confidence 0 → nothing happens) is a theorem of the
    formula, not a guard branch.
  • Back-action — an informative record (|y| > 0) shrinks x/y_b: pinning WHERE damps
    the phase that carries WHEN. A pure state stays pure (Kraus of one record).
  • α ↔ s mapping — at the equator, Δz = tanh(s·y) ≈ s·y while the α-blend gives
    α·y: **strength ≈ alpha** to first order at maximum uncertainty; away from the
    equator Belavkin saturates where the blend overshoots. Callers reuse their
    existing collapse_alpha (confidence pre-folded, same convention as observe_qubit).

The cumulant cross-update (the joint-Kalman payoff)
───────────────────────────────────────────────────
Exact conditioning on one qubit moves the means of correlated qubits (the full-ρ
Kraus does this automatically through the off-diagonal blocks). At cumulant order 2
we apply the linear-Gaussian conditioning:

    K_j = cov_zz(j, i) / (1 − z_i² + ε)          (regression gain onto the record)
    z_j += K_j · Δz_i                            (mean transfer)
    C(i,·) ← (1 − γ)·C(i,·),  γ = s·v/(1 + s·v)  (connected-covariance shrink, v=1−z_i²)

On a classically z-correlated pair this mean transfer is EXACT (test-pinned against
the full-ρ Kraus); the covariance shrink is the Kalman (1−K) form, approximate under
the closure. Note the old observe_qubit path only DEcorrelates on observation — the
cross-update is strictly more informative and is the reason to climb the ladder.

Gating: reservoir.ingest branches on UMWELT_BELAVKIN (default OFF → the observe
path is byte-unchanged). Substrates expose `measure_qubit(idx, record_z, strength,
confidence=None)` built on the helpers here; `confidence` is recorded as a gauge
quantity only — the caller folds it into `strength`, the same convention as
observe_qubit (see the contract note there).
"""
from __future__ import annotations

import math
import os

import numpy as np

_EPS = 1e-9


def env_belavkin_enabled() -> bool:
    """UMWELT_BELAVKIN=1 switches the reservoir's sensor-observe path from the
    α-blend to the Belavkin measurement. Unset/other → off, byte-identical."""
    return os.environ.get("UMWELT_BELAVKIN", "") == "1"


def kraus_weights(record_z: float, strength: float) -> tuple[float, float]:
    """(w₊, w₋) = squared Kraus amplitudes for record `record_z` at strength s.
    Computed in a shifted exponent so w's stay well-scaled for large s."""
    y = float(record_z)
    s = max(0.0, float(strength))
    e_plus = -s * (y - 1.0) ** 2 / 2.0
    e_minus = -s * (y + 1.0) ** 2 / 2.0
    m = max(e_plus, e_minus)
    return math.exp(e_plus - m), math.exp(e_minus - m)


def measure_bloch(
    bloch, record_z: float, strength: float
) -> tuple[np.ndarray, float, float]:
    """Apply the conditioned weak σ_z measurement to one Bloch vector.

    Returns (bloch', gain_dz, coherence_factor):
      bloch'            — the post-measurement (x, y, z)
      gain_dz           — z' − z (the innovation actually applied)
      coherence_factor  — the 2·c₊c₋/N transverse damping ∈ (0, 1]

    strength ≤ 0 is the exact no-op (returns a copy, gain 0, factor 1).
    """
    b = np.asarray(bloch, dtype=float).copy()
    s = float(strength)
    if s <= 0.0:
        return b, 0.0, 1.0
    x, yb, z = float(b[0]), float(b[1]), float(b[2])
    z = max(-1.0, min(1.0, z))
    w_plus, w_minus = kraus_weights(record_z, s)
    n = w_plus * (1.0 + z) + w_minus * (1.0 - z)
    if n <= _EPS:                      # unreachable for w's from kraus_weights; belt+braces
        return b, 0.0, 1.0
    z_new = (w_plus * (1.0 + z) - w_minus * (1.0 - z)) / n
    coh = 2.0 * math.sqrt(w_plus * w_minus) / n
    b[0], b[1], b[2] = coh * x, coh * yb, z_new
    # Numerical guard: keep the result inside the Bloch ball.
    r = float(np.linalg.norm(b))
    if r > 1.0:
        b /= r
    return b, z_new - z, coh


def kalman_shrink(strength: float, z: float) -> float:
    """γ = s·v/(1 + s·v) with v = 1 − z² — the Kalman-gain-shaped fraction by which
    the measured qubit's connected covariance shrinks (cov' = (1−γ)·cov)."""
    v = max(0.0, 1.0 - float(z) ** 2)
    sv = max(0.0, float(strength)) * v
    return sv / (1.0 + sv)


def measure_cumulant(e1: np.ndarray, e2: np.ndarray, idx: int,
                     record_z: float, strength: float) -> float:
    """In-place Belavkin measurement of qubit `idx` on a cumulant state (e1, e2).

    1. Conditioned single-qubit update on e1[idx] (measure_bloch).
    2. Mean transfer to every peer j via the regression gain K_j = cov_zz(j,i)/v_i.
    3. Connected covariance C(i,j) shrinks by (1−γ) (Kalman form) and i's
       transverse rows additionally damp by the coherence factor.

    Returns the applied Δz on the measured qubit. strength ≤ 0 touches nothing.
    """
    if strength <= 0.0:
        return 0.0
    n = e1.shape[0]
    i = int(idx)
    old_i = e1[i].copy()
    z_i = float(old_i[2])
    v_i = max(_EPS, 1.0 - z_i * z_i)

    new_i, dz, coh = measure_bloch(old_i, record_z, strength)
    gamma = kalman_shrink(strength, z_i)

    # Peers first (they read the PRE-measurement covariance), then the qubit itself.
    for j in range(n):
        if j == i:
            continue
        old_j = e1[j].copy()
        conn = e2[i, j] - np.outer(old_i, old_j)      # connected C(i,j), 3×3
        k_j = float(conn[2, 2]) / v_i                  # cov_zz(j,i)/var_i
        new_j = old_j.copy()
        new_j[2] = old_j[2] + k_j * dz
        rj = float(np.linalg.norm(new_j))
        if rj > 1.0:
            new_j /= rj
        conn_new = (1.0 - gamma) * conn
        conn_new[0, :] *= coh                          # i's transverse correlations damp
        conn_new[1, :] *= coh
        e1[j] = new_j
        e2[i, j] = conn_new + np.outer(new_i, new_j)
        e2[j, i] = e2[i, j].T
    e1[i] = new_i
    return dz


def measure_rho(rho: np.ndarray, idx: int, n_qubits: int,
                record_z: float, strength: float) -> np.ndarray:
    """Exact full-ρ Kraus update: ρ' = MρM†/Tr with M = I⊗…⊗diag(c₊,c₋)⊗…⊗I.
    The multipartite ground truth the cumulant path approximates (and the tests
    compare against). Returns a NEW matrix; strength ≤ 0 returns rho unchanged."""
    if strength <= 0.0:
        return rho
    w_plus, w_minus = kraus_weights(record_z, strength)
    c_plus, c_minus = math.sqrt(w_plus), math.sqrt(w_minus)
    m1 = np.array([[c_plus, 0.0], [0.0, c_minus]], dtype=rho.dtype)
    op = np.array([[1.0]], dtype=rho.dtype)
    for q in range(n_qubits):
        op = np.kron(op, m1 if q == idx else np.eye(2, dtype=rho.dtype))
    out = op @ rho @ op.conj().T
    tr = float(np.real(np.trace(out)))
    if tr <= _EPS:
        return rho
    return out / tr
