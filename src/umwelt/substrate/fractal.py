"""
Fractal Correlation Hierarchy.

Decomposes an n-qubit density matrix into a hierarchy of k-body
correlation tensors — the cumulant expansion of ρ:

    Level 1: Single-qubit marginals        (n terms)
    Level 2: Pairwise correlations          (n choose 2 terms)
    Level 3: Triple correlations            (n choose 3 terms)
    ...
    Level n: Full n-body correlation        (1 term)

Each level captures structure invisible to lower levels, creating a
fractal-like multi-scale representation. In a reservoir computing
context, different levels track different complexity scales:

    Level 1 → local sensor states (is a region occupied?)
    Level 2 → pairwise patterns (motion AND energy correlation)
    Level 3 → higher-order patterns (occupancy + activity + environment)

This maps to the "fractal complexity within each axial dimension"
that rolls up to the full ρ probability.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from umwelt.substrate.density_matrix import ComplexMatrix, _single_qubit_op


# Pauli matrices
_I = np.eye(2, dtype=np.complex128)
_SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_PAULIS = [_SX, _SY, _SZ]
_PAULI_LABELS = ["x", "y", "z"]


def decompose_by_level(
    rho: ComplexMatrix, n_qubits: int, max_level: int = 3
) -> dict[int, NDArray[np.floating]]:
    """
    Decompose density matrix expectations into fractal correlation levels.

    `max_level` caps which levels are COMPUTED (not just returned), so callers
    that only need single + pairwise structure (the sparse readout features)
    don't pay for the level-3 triple correlations — 27·C(n,3) triple matmuls,
    the dominant CPU cost for large clusters. Default 3 preserves the legacy
    behavior for analysis callers.

    Returns:
        dict mapping level (1..min(max_level, n_qubits)) to feature arrays:
            level 1: shape (n_qubits, 3) — single qubit ⟨σ_a⟩
            level 2: shape (n_pairs, 9) — pairwise ⟨σ_a ⊗ σ_b⟩
            level 3: shape (n_triples, 27) — triple ⟨σ_a ⊗ σ_b ⊗ σ_c⟩
    """
    result = {}

    # Level 1: single-qubit expectations
    level1 = np.zeros((n_qubits, 3))
    for q in range(n_qubits):
        for a, pauli in enumerate(_PAULIS):
            op = _single_qubit_op(pauli, q, n_qubits)
            level1[q, a] = np.real(np.trace(rho @ op))
    result[1] = level1

    # Level 2: pairwise correlations (connected part)
    if n_qubits >= 2 and max_level >= 2:
        pairs = list(combinations(range(n_qubits), 2))
        level2 = np.zeros((len(pairs), 9))
        for idx, (q1, q2) in enumerate(pairs):
            k = 0
            for a, pa in enumerate(_PAULIS):
                for b, pb in enumerate(_PAULIS):
                    op = _single_qubit_op(pa, q1, n_qubits) @ \
                         _single_qubit_op(pb, q2, n_qubits)
                    full = np.real(np.trace(rho @ op))
                    # Connected correlation: ⟨AB⟩ - ⟨A⟩⟨B⟩
                    disconnected = level1[q1, a] * level1[q2, b]
                    level2[idx, k] = full - disconnected
                    k += 1
        result[2] = level2

    # Level 3: triple correlations (connected part). Requires level 2 results,
    # which are present whenever max_level >= 3 (since 3 >= 2).
    if n_qubits >= 3 and max_level >= 3:
        triples = list(combinations(range(n_qubits), 3))
        level3 = np.zeros((len(triples), 27))
        for idx, (q1, q2, q3) in enumerate(triples):
            k = 0
            for a, pa in enumerate(_PAULIS):
                for b, pb in enumerate(_PAULIS):
                    for c, pc in enumerate(_PAULIS):
                        op = _single_qubit_op(pa, q1, n_qubits) @ \
                             _single_qubit_op(pb, q2, n_qubits) @ \
                             _single_qubit_op(pc, q3, n_qubits)
                        full = np.real(np.trace(rho @ op))
                        # Connected 3-body: subtract all lower-order contributions
                        # C3 = ⟨ABC⟩ - ⟨A⟩⟨BC⟩_c - ⟨B⟩⟨AC⟩_c - ⟨C⟩⟨AB⟩_c - ⟨A⟩⟨B⟩⟨C⟩
                        ab_idx = _pair_index(pairs, q1, q2)
                        ac_idx = _pair_index(pairs, q1, q3)
                        bc_idx = _pair_index(pairs, q2, q3)
                        c2_ab = level2[ab_idx, a * 3 + b] if ab_idx is not None else 0
                        c2_ac = level2[ac_idx, a * 3 + c] if ac_idx is not None else 0
                        c2_bc = level2[bc_idx, b * 3 + c] if bc_idx is not None else 0
                        c3 = (full
                              - level1[q1, a] * c2_bc
                              - level1[q2, b] * c2_ac
                              - level1[q3, c] * c2_ab
                              - level1[q1, a] * level1[q2, b] * level1[q3, c])
                        level3[idx, k] = c3
                        k += 1
        result[3] = level3

    return result


def fractal_signature(levels: dict[int, NDArray[np.floating]]) -> dict[int, float]:
    """
    Compute the "fractal signature" — the energy (norm) at each level.

    A system with only local correlations will have energy concentrated
    at level 1. Entangled/complex states spread energy across levels.
    This is analogous to a power spectrum across fractal scales.

    Returns:
        dict mapping level → frobenius norm of that level's features.
    """
    return {level: float(np.linalg.norm(features)) for level, features in levels.items()}


def fractal_dimension_estimate(signature: dict[int, float]) -> float:
    """
    Estimate the effective fractal dimension from the signature.

    Uses the ratio of energy across scales. A maximally fractal system
    (equal energy at all levels) → D = n_levels. A purely local system
    (all energy at level 1) → D ≈ 1.

    Returns:
        Estimated fractal dimension (1.0 to n_levels).
    """
    if not signature:
        return 1.0

    total = sum(signature.values())
    if total < 1e-15:
        return 1.0

    # Shannon entropy of the energy distribution across levels
    probs = np.array([v / total for v in signature.values()])
    probs = probs[probs > 1e-15]
    entropy = -np.sum(probs * np.log2(probs))

    # Normalize to [1, n_levels]
    n = len(signature)
    max_entropy = np.log2(n) if n > 1 else 1.0
    return 1.0 + (n - 1) * (entropy / max_entropy) if max_entropy > 0 else 1.0


def _pair_index(
    pairs: list[tuple[int, int]], q1: int, q2: int
) -> int | None:
    """Find the index of (q1, q2) in the pairs list."""
    key = (min(q1, q2), max(q1, q2))
    try:
        return pairs.index(key)
    except ValueError:
        return None
