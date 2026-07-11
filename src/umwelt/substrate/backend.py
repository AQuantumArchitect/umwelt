"""SubstrateBackend — the formal contract every belief-cluster backend satisfies.

The field (`field.py`) evolves a graph of *clusters*. Three backends already
implement the same duck-typed surface; this module promotes that de-facto contract
to an explicit `Protocol` plus a couple of predicates, so the field can treat any
cluster uniformly instead of branching on `is_cumulant` / `is_product` flags and
reaching into a dense `.rho` that only one backend has.

The three shipped backends:

    QubitCluster        (cluster.py)          dense 2ⁿ×2ⁿ density matrix; the reference.
    CumulantCluster     (cumulant_cluster.py) 1-/2-RDM cumulants, O(n²); is_cumulant=True.
    ProductQubitCluster (product_cluster.py)  N independent 2×2, O(N); is_product=True,
                                              and `.rho` *raises* to forbid materializing 2ᴺ.

This is the seam the flagship-8 ablation swaps a `ClassicalReservoirBackend` into:
same graph, same observe/step/features surface, a different substrate underneath.

The Protocol is documentation + a structural type — backends are NOT required to
inherit it (they pre-date it and stay duck-typed). `runtime_checkable` lets tests
assert conformance without an isinstance-by-inheritance constraint.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from umwelt.substrate.density_matrix import ComplexMatrix


@runtime_checkable
class SubstrateBackend(Protocol):
    """One belief cluster: a small open quantum system the field evolves and reads.

    Every method here is implemented by all three shipped backends. Methods that a
    given backend cannot honor cheaply degrade safely (e.g. the product param-fiber
    no-ops `step` and `clamp_physical`); the dense-only `.rho` matrix is deliberately
    NOT part of this contract — substrate-neutral code goes through the methods below.
    """

    # --- identity ---------------------------------------------------------
    n_qubits: int
    role_index: dict[str, int]

    # --- lifecycle / evolution -------------------------------------------
    def reset(self) -> None:
        """Reset to the ground state |0…0⟩."""
        ...

    def step(self, inputs: NDArray[np.floating] | None = None, dt_scale: float = 1.0) -> None:
        """Evolve one Lindblad timestep × dt_scale under the current Hamiltonian."""
        ...

    def set_hamiltonian(self, H: ComplexMatrix) -> None:
        """Install a (dense Hermitian) Hamiltonian; backends may decompose internally."""
        ...

    def sync_gamma_diss(self, bundle) -> None:
        """Live-read per-role dissipation rates from a parameter bundle."""
        ...

    # --- measurement / assimilation --------------------------------------
    def observe_qubit(self, qubit_idx: int, target_bloch, alpha: float = 0.5,
                      confidence: float | None = None) -> None:
        """Partial-collapse one qubit toward a target Bloch vector by strength α.
        THE contract: the caller pre-folds confidence into α; `confidence` is only
        RECORDED by the substrate as a read-only gauge quantity (uniform across all
        backends — the old cumulant-side α×conf fold double-applied it)."""
        ...

    def measure_qubit(self, qubit_idx: int, record_z: float, strength: float,
                      confidence: float | None = None) -> None:
        """Belavkin weak σ_z measurement (docs/QUANTUM_KALMAN.md, rung L4): the
        conditioned Kraus update — bounded Wonham gain, coherence back-action, and
        (where the backend carries correlations) the cross-update onto peers.
        Same folding contract as observe_qubit; strength ≤ 0 is the exact no-op.
        Behind UMWELT_BELAVKIN in reservoir.ingest; observe_qubit is its α-blend
        equator limit."""
        ...

    def nudge_toward_rdm(self, qubit_idx: int, target_rdm: NDArray, alpha: float) -> None:
        """Bridge/projection fiber connection: move one qubit's marginal toward a
        target 2×2 reduced state by α, in whatever representation the backend holds.
        Replaces the field reaching into a dense `.rho` directly."""
        ...

    # --- readout ----------------------------------------------------------
    def qubit_rdm(self, qubit_idx: int) -> ComplexMatrix:
        """The 2×2 reduced density matrix of one qubit (exact for all backends)."""
        ...

    def qubit_bloch(self, qubit_idx: int) -> NDArray[np.floating]:
        """Bloch vector (x, y, z) for one qubit."""
        ...

    def role_bloch(self, role: str) -> NDArray[np.floating]:
        """Bloch vector by semantic role name."""
        ...

    def features(self) -> NDArray[np.floating]:
        """The sparse fractal feature vector this cluster exposes to its readout."""
        ...

    # --- physical constraints --------------------------------------------
    def clamp_physical(self) -> None:
        """Re-project the state onto the physical manifold after nudging.
        Backend-specific (dense eigenvalue clamp / cumulant Bloch-radius clamp /
        no-op for the by-construction-physical product fiber)."""
        ...

    def hamiltonian_norm(self) -> float:
        """Frobenius norm ‖H‖ of the cluster's Hamiltonian, computed without materializing a
        2ⁿ matrix (the cumulant backend uses its sparse couplings) — for diagnostics."""
        ...


def cluster_kind(cluster: object) -> str:
    """The ONE place that names a backend: 'dense' | 'cumulant' | 'product' | 'classical'.
    Collapses the scattered `getattr(c, "is_cumulant"/"is_product", False)` chains (M2):
    branch on the kind ONLY where a branch is genuinely per-kind (pickle layout, gauge
    labels, telemetry); prefer the SubstrateBackend methods everywhere else."""
    if getattr(cluster, "is_cumulant", False):
        return "cumulant"
    if getattr(cluster, "is_product", False):
        return "product"
    if getattr(cluster, "is_classical", False):
        return "classical"
    return "dense"


def is_dense(cluster: object) -> bool:
    """True for the reference dense-ρ backend (QubitCluster) — the only one that owns a
    materializable joint `.rho`. Cumulant, product, and the classical-reservoir ablation backend
    all return False."""
    return (not getattr(cluster, "is_cumulant", False)
            and not getattr(cluster, "is_product", False)
            and not getattr(cluster, "is_classical", False))


def is_param_fiber(cluster: object) -> bool:
    """True for the unentangled O(N) parameter fiber (ProductQubitCluster), which the
    field skips for joint-matrix operations (no bridges, no physicality clamp)."""
    return bool(getattr(cluster, "is_product", False))
