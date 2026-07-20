"""
Qubit Cluster — a local fiber in the quantum probability field.

Each cluster manages a small density matrix (3-5 qubits) for one region
of the world graph. Qubits have semantic roles (the spec's declared roles)
and can be shared with adjacent clusters via bridge connections.

The cluster is the computational unit: it evolves its own density matrix
independently, then bridge qubits are reconciled across clusters by the
fiber bundle (field.py).

Each qubit has an input mode determined by its role:

    UNITARY:      σ_x Hamiltonian kick. Event-driven sensors (motion, contact).
    DISSIPATIVE:  Thermal Lindblad (σ_+/σ_-). Continuous sensors (temperature,
                  a periodic driver's altitude). The qubit thermalizes to the sensor reading
                  through quantum decoherence.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from umwelt.substrate.density_matrix import (
    DensityMatrixEvolver,
    ComplexMatrix,
    EVOLVE_DTYPE,
    _single_qubit_op,
)
from umwelt.spec.roles import role_input_mode


def sparse_feature_vector(
    levels: dict[int, NDArray[np.floating]], max_level: int
) -> NDArray[np.floating]:
    """Flatten a `decompose_by_level` dict into one deterministic feature vector,
    including correlation levels 1..max_level.

    Order is level-ascending, each level's array raveled in its natural
    (combination-index, pauli-index) order, so the layout is stable across calls
    for a fixed n_qubits — required for the Ridge readout's fixed input geometry.
    A cluster with no qubits (shouldn't happen) yields an empty vector.
    """
    parts: list[NDArray[np.floating]] = []
    for lvl in range(1, max(1, max_level) + 1):
        arr = levels.get(lvl)
        if arr is not None and arr.size:
            parts.append(np.ravel(arr))
    if not parts:
        return np.zeros(0, dtype=float)
    return np.concatenate(parts)


class QubitCluster:
    """
    A cluster of n qubits representing one region of the world graph.

    Each qubit has a semantic role (whatever roles the spec declares).
    The cluster wraps a DensityMatrixEvolver and provides:
      - Named qubit access (unitary vs dissipative input per role)
      - Partial trace for bridge qubit extraction
      - Fractal-level feature readout
    """

    is_cumulant = False   # the backend duck-flag (vs CumulantCluster)

    def __init__(
        self,
        zone_name: str,
        qubit_roles: list[str],
        gamma: float = 0.05,
        dt: float = 0.01,
        gamma_diss: float | dict[str, float] = 5.0,
        role_modes: dict[str, str] | None = None,
        max_feature_level: int = 2,
    ):
        self.zone_name = zone_name
        self.qubit_roles = list(qubit_roles)
        self.n_qubits = len(qubit_roles)
        self.dim = 2 ** self.n_qubits

        # Highest fractal correlation level included in the readout feature
        # vector (1 = single-qubit only, 2 = + connected pairwise, 3 = + triples).
        # This caps feature_dim at 3n + 9·C(n,2) [+ 27·C(n,3)] — dim-INDEPENDENT,
        # vs the full dim²−1 Gell-Mann basis whose (dim²−1, dim, dim) operator
        # stack spikes to gigabytes for the largest clusters at construction.
        # Depth-gated leaves can lower this per cluster. See decompose_by_level
        # in fractal.py and project_construction_oom.
        self.max_feature_level = max(1, int(max_feature_level))

        # Role → qubit index mapping
        self.role_index = {role: i for i, role in enumerate(qubit_roles)}

        # Classify each qubit: unitary (σ_x input) or dissipative (thermal)
        if role_modes is None:
            role_modes = {r: role_input_mode(r) for r in qubit_roles}
        self._role_modes = role_modes
        dissipative_indices = {
            i for i, role in enumerate(qubit_roles)
            if role_modes.get(role, role_input_mode(role)) == "dissipative"
        }
        self._dissipative_indices = dissipative_indices

        # Zero Hamiltonian — no speculative dynamics, all drive from data
        H = np.zeros((self.dim, self.dim), dtype=EVOLVE_DTYPE)

        # Input operators: σ_x only for UNITARY qubits.
        # Dissipative qubits get null operators (their input goes through
        # the thermal Lindblad channel, not the Hamiltonian).
        sx_2 = np.array([[0, 1], [1, 0]], dtype=EVOLVE_DTYPE)
        null_op = np.zeros((self.dim, self.dim), dtype=EVOLVE_DTYPE)
        input_ops = [
            null_op.copy() if q in dissipative_indices
            else _single_qubit_op(sx_2, q, self.n_qubits)
            for q in range(self.n_qubits)
        ]

        # Lindblad: amplitude damping on each qubit (σ_-).
        # For unitary qubits this provides standard decoherence.
        # For dissipative qubits the evolver replaces this with the
        # input-modulated thermal channel (σ_+/σ_- pair).
        lindblad_ops = []
        for q in range(self.n_qubits):
            sigma_minus = np.zeros((2, 2), dtype=EVOLVE_DTYPE)
            sigma_minus[0, 1] = 1.0
            L = _single_qubit_op(sigma_minus, q, self.n_qubits)
            lindblad_ops.append(L)

        self.evolver = DensityMatrixEvolver(
            n_qubits=self.n_qubits,
            hamiltonian=H,
            lindblad_ops=lindblad_ops,
            dt=dt,
            input_operators=input_ops,
            dissipative_qubit_indices=dissipative_indices,
        )
        # Set the live gamma on the evolver (can be updated later)
        self.evolver.gamma = gamma

        # Per-role gamma_diss: each dissipative qubit gets its own timescale.
        # Accepts a scalar (same rate for all) or dict {role_name: rate}.
        if isinstance(gamma_diss, dict):
            self.evolver._gamma_diss_per_qubit = {
                i: gamma_diss[role]
                for i, role in enumerate(qubit_roles)
                if i in dissipative_indices and role in gamma_diss
            }
            self.evolver._gamma_diss_default = gamma_diss.get("_default", 5.0)
        else:
            self.evolver._gamma_diss_default = float(gamma_diss)

    def sync_gamma_diss(self, bundle) -> None:
        """Live-read per-role gamma_diss from a ParameterBundle.

        Looks for gamma_diss_{role} for each dissipative role, falling
        back to gamma_diss as the node-level default.
        """
        default = bundle.get("gamma_diss", self.evolver._gamma_diss_default)
        self.evolver._gamma_diss_default = default
        for q in self._dissipative_indices:
            role = self.qubit_roles[q]
            key = f"gamma_diss_{role}"
            self.evolver._gamma_diss_per_qubit[q] = bundle.get(key, default)

    def set_hamiltonian(self, H: ComplexMatrix):
        """Replace the cluster's Hamiltonian (must be dim x dim Hermitian)."""
        assert H.shape == (self.dim, self.dim)
        self.evolver.H_base = H.astype(EVOLVE_DTYPE)

    @property
    def rho(self) -> ComplexMatrix:
        """Current density matrix."""
        return self.evolver.dm.rho

    @rho.setter
    def rho(self, value: ComplexMatrix):
        self.evolver.dm.rho = value

    @property
    def purity(self) -> float:
        return self.evolver.dm.purity

    @property
    def entropy(self) -> float:
        return self.evolver.dm.von_neumann_entropy

    def step(self, inputs: NDArray[np.floating] | None = None, dt_scale: float = 1.0):
        """Evolve one timestep × dt_scale. inputs[i] drives qubit_roles[i].

        Unitary qubits: input modulates H (σ_x kick).
        Dissipative qubits: input sets thermal target (Lindblad thermalization).
        Both go through the same Lindblad master equation — no bypass.
        dt_scale>1 = the smooth-clock catch-up after skipped calm ticks (default 1.0).
        """
        diss_targets = None
        if inputs is not None and self._dissipative_indices:
            diss_targets = {
                q: float(inputs[q])
                for q in self._dissipative_indices
                if q < len(inputs)
            }
        self.evolver.step(inputs, dissipative_targets=diss_targets, dt_scale=dt_scale)

    def reset(self):
        """Reset to |0...0⟩."""
        self.evolver.reset()

    def observe_qubit(
        self,
        qubit_idx: int,
        target_bloch: tuple[float, float, float],
        alpha: float = 0.5,
        confidence: float | None = None,
    ):
        """
        Observe a qubit: partially collapse it toward an arbitrary target
        Bloch vector. This is the SKY-update half of measurement — reality
        correcting the belief — as distinct from the read-only SKY→GROUND
        projection in collapse.py (which never touches the density matrix).

        Unlike a pole projection, the target may be ANY point on or in the
        Bloch ball:

            target_bloch = (x, y, z),  R = |r| <= 1
            ρ_target = ½ (I + x·σ_x + y·σ_y + z·σ_z)

        R sets how pure the post-observation belief is — R=1 lands on the
        sphere (a clean eigenstate), R<1 lands inside it (a partial collapse /
        weak measurement, "I saw it but I don't fully trust the reading").
        Direction sets which state. α is the collapse strength: 1 → full snap,
        <1 → a nudge that leaves space for the prior belief.

        Mechanism: ρ_new = (1-α)·ρ + α·(ρ_target_q ⊗ Tr_q[ρ]), preserving
        correlations with the other qubits in the cluster.

        `confidence` ∈ [0,1] is the edge-supplied validity of THIS observation
        (the caller has already folded it into `alpha`). We only RECORD it here as
        a read-only gauge quantity — gauge.cluster_gauge surfaces it next to purity
        (input-confidence vs output-confidence). The brain decides nothing about
        health; it just remembers what the edge supplied. Recorded even when
        alpha=0 so a null/failed read shows as obs_confidence=0 in the gauge ledger.
        """
        if confidence is not None:
            if not hasattr(self, "_obs_confidence"):
                self._obs_confidence: dict[int, float] = {}
            self._obs_confidence[qubit_idx] = float(confidence)
        if alpha <= 0:
            return
        x, y, z = (
            float(target_bloch[0]),
            float(target_bloch[1]),
            float(target_bloch[2]),
        )
        # Clamp to the Bloch ball so ρ_target stays a valid density matrix.
        r = (x * x + y * y + z * z) ** 0.5
        if r > 1.0:
            x, y, z = x / r, y / r, z / r
        sx = np.array([[0, 1], [1, 0]], dtype=EVOLVE_DTYPE)
        sy = np.array([[0, -1j], [1j, 0]], dtype=EVOLVE_DTYPE)
        sz = np.array([[1, 0], [0, -1]], dtype=EVOLVE_DTYPE)
        I2 = np.eye(2, dtype=EVOLVE_DTYPE)
        target_rdm = 0.5 * (I2 + x * sx + y * sy + z * sz)
        self._inject_target_rdm(qubit_idx, target_rdm, alpha)

    def measure_qubit(self, qubit_idx: int, record_z: float, strength: float,
                      confidence: float | None = None) -> None:
        """Belavkin weak σ_z measurement (docs/QUANTUM_KALMAN.md, rung L4) — the
        EXACT multipartite Kraus update ρ' = MρM†/Tr on this qubit. Correlated
        peers move through the off-diagonal blocks automatically (the ground
        truth the cumulant cross-update approximates). Same contract as
        observe_qubit: caller pre-folds confidence into `strength`; `confidence`
        is recorded as the gauge quantity only; strength ≤ 0 is the exact no-op."""
        from umwelt.substrate.belavkin import measure_rho
        if confidence is not None:
            if not hasattr(self, "_obs_confidence"):
                self._obs_confidence = {}
            self._obs_confidence[qubit_idx] = float(confidence)
        if strength <= 0:
            return
        self.evolver.dm.rho[:] = measure_rho(
            self.rho, qubit_idx, self.n_qubits, float(record_z), float(strength))

    def _inject_target_rdm(
        self,
        qubit_idx: int,
        target_rdm: ComplexMatrix,
        alpha: float,
    ):
        """Mix qubit `qubit_idx` toward a 2×2 target RDM with strength α,
        preserving correlations with the other qubits: ρ_new = (1-α)·ρ +
        α·(target_rdm ⊗ Tr_q[ρ])."""
        if alpha <= 0:
            return

        n = self.n_qubits
        q = qubit_idx

        if n == 1:
            self.evolver.dm.rho[:] = (1.0 - alpha) * self.rho + alpha * target_rdm
            return

        # RDM of all other qubits (preserves their mutual correlations)
        other = [i for i in range(n) if i != q]
        rest = partial_trace_keep(self.rho, other, n)     # (2^(n-1), 2^(n-1))
        rest_t = rest.reshape([2] * (2 * (n - 1)))

        # Build the full target tensor ρ_target of shape [2]*n x [2]*n.
        # ρ_target[i_0,...,i_{n-1}, j_0,...,j_{n-1}]
        #   = target_rdm[i_q, j_q] * rest_t[(other row indices), (other col indices)]
        target_t = np.zeros([2] * (2 * n), dtype=EVOLVE_DTYPE)
        for k in range(2):
            for ell in range(2):
                coeff = target_rdm[k, ell]
                if coeff == 0:
                    continue
                row_idx = [slice(None)] * n
                col_idx = [slice(None)] * n
                row_idx[q] = k
                col_idx[q] = ell
                # The remaining (n-1) row and (n-1) col axes match rest_t exactly
                target_t[tuple(row_idx) + tuple(col_idx)] = coeff * rest_t

        target_full = target_t.reshape(2 ** n, 2 ** n)
        self.evolver.dm.rho[:] = (1.0 - alpha) * self.rho + alpha * target_full

    # ================================================================
    # Single-qubit reduced density matrices
    # ================================================================

    def qubit_rdm(self, qubit_idx: int) -> ComplexMatrix:
        """
        Reduced density matrix of a single qubit (2×2).
        Traces out all other qubits.
        """
        return partial_trace_single(self.rho, qubit_idx, self.n_qubits)

    def qubit_bloch(self, qubit_idx: int) -> NDArray[np.floating]:
        """Bloch vector (x, y, z) for a single qubit — the Pauli decomposition of the 2×2 reduced
        density matrix in CLOSED FORM. tr(ρσx)/tr(ρσy)/tr(ρσz) on a 2×2 reduce to direct element reads:
          x = ρ01 + ρ10,  y = i·ρ01 − i·ρ10,  z = ρ00 − ρ11
        — the SAME terms the old `real(trace(rdm@σ))` summed, in the same order (bit-identical), but
        without building three Pauli matrices and running three matmul+trace per call. This is the field's
        hottest readout (thousands of calls per tick); the old form was ~half the tick's tiny GIL-holding
        numpy ops. See experiments/substrate_profile.py / the GIL-bottleneck science (2026-06-19)."""
        r = self.qubit_rdm(qubit_idx)
        r01, r10 = r[0, 1], r[1, 0]
        return np.array([
            np.real(r01 + r10),
            np.real(1j * r01 - 1j * r10),
            np.real(r[0, 0] - r[1, 1]),
        ])

    def role_bloch(self, role: str) -> NDArray[np.floating]:
        """Bloch vector for a qubit by its semantic role name."""
        return self.qubit_bloch(self.role_index[role])

    def all_bloch(self) -> dict[str, NDArray[np.floating]]:
        """Dict of role → Bloch vector for all qubits."""
        return {role: self.qubit_bloch(i) for role, i in self.role_index.items()}

    def role_gauge(self, role: str) -> tuple[float, float]:
        """(value, confidence) for a role in ONE read — the calibrated belief and
        how settled it is. value = (z+1)/2 ∈ [0,1]; confidence = Bloch radius |r| ∈
        [0,1] (1=pure/certain, 0=maximally mixed/unknown). This is the one accessor
        every readout should reach for instead of hand-rolling z; it makes belief
        certainty a first-class, uniform coordinate wherever a cluster is read."""
        from umwelt.substrate.bloch import bloch_radius
        x, y, z = (float(v) for v in self.role_bloch(role))
        return (z + 1.0) / 2.0, bloch_radius(x, y, z)

    # ================================================================
    # SubstrateBackend contract — the field goes through these instead of
    # reaching into the dense `.rho` directly (substrate.py).
    # ================================================================

    def nudge_toward_rdm(self, qubit_idx: int, target_rdm: NDArray, alpha: float) -> None:
        """Bridge/projection fiber connection: move one qubit's marginal toward a
        target 2×2 reduced state by α, keeping the joint ρ Hermitian + trace-1.

        Dense backend: correction = (I/2)^⊗left ⊗ (target−current) ⊗ (I/2)^⊗right,
        the maximally-mixed completion of the single-qubit delta.

        Vectorized (#338): that kron only scatters δ[i,i']/(L·R) onto the a==a', b==b' diagonal blocks of
        ρ reshaped to (L,2,R,L,2,R) — so we add it IN PLACE instead of materializing the full d×d matrix
        via two nested np.kron (the field tick's #1 call-count hotspot). L=2^left, R=2^right are powers of
        two, so the 1/(L·R) scaling is EXACT in float → bit-identical to the kron path (test_nudge_dekron)."""
        n = self.n_qubits
        delta = target_rdm - self.qubit_rdm(qubit_idx)
        L = 1 << qubit_idx                 # 2^(qubits left of the target)
        R = 1 << (n - 1 - qubit_idx)       # 2^(qubits right of the target)
        rho = self.rho.copy()
        view = rho.reshape(L, 2, R, L, 2, R)
        ai = np.arange(L)[:, None]
        bi = np.arange(R)[None, :]
        # add alpha·δ/(L·R) at [a, :, b, a, :, b] for all a,b — the maximally-mixed identity slots
        view[ai, :, bi, ai, :, bi] += (alpha / (L * R)) * delta
        rho = 0.5 * (rho + rho.conj().T)
        tr = np.trace(rho)
        if abs(tr) > 1e-15:
            rho = rho / tr
        self.rho = rho

    def hamiltonian_norm(self) -> float:
        """Frobenius norm of the cluster's Hamiltonian (dense backend: the literal matrix
        norm). The substrate-neutral way diagnostics read ‖H‖ without touching .H_base —
        the cumulant backend overrides this to avoid materializing a 2ⁿ matrix."""
        return float(np.linalg.norm(self.evolver.H_base))

    def clamp_physical(self) -> None:
        """Re-project ρ onto the physical manifold: Hermitian, PSD (clamp negative
        eigenvalues to 0), trace-1. (Moved verbatim from field._enforce_physicality's
        dense branch.)"""
        rho = self.rho
        rho = 0.5 * (rho + rho.conj().T)
        eigvals, eigvecs = np.linalg.eigh(rho)
        eigvals = np.maximum(eigvals, 0.0)
        rho = (eigvecs * eigvals) @ eigvecs.conj().T
        tr = np.trace(rho)
        if abs(tr) > 1e-15:
            rho = rho / tr
        self.rho = rho

    # ================================================================
    # Multi-qubit reduced density matrices (for bridges)
    # ================================================================

    def subsystem_rdm(self, qubit_indices: list[int]) -> ComplexMatrix:
        """
        Reduced density matrix for a subset of qubits.
        Used to extract bridge qubit states.
        """
        return partial_trace_keep(self.rho, qubit_indices, self.n_qubits)

    def roles_rdm(self, roles: list[str]) -> ComplexMatrix:
        """Reduced density matrix for qubits identified by role names."""
        indices = [self.role_index[r] for r in roles]
        return partial_trace_keep(self.rho, indices, self.n_qubits)

    # ================================================================
    # Feature extraction
    # ================================================================

    def features(self) -> NDArray[np.floating]:
        """Sparse fractal feature vector: single-qubit expectations plus
        connected correlations up to `self.max_feature_level`.

        Replaces the old full Bloch vector (`evolver.state_vector`, the dim²−1
        generalized Gell-Mann expectations), whose operator stack is
        (dim²−1, dim, dim) and spikes to ~4.3 GB for a 7-qubit cluster — the
        construction OOM. This sparse basis is dim-INDEPENDENT in length:
        3n + 9·C(n,2) [+ 27·C(n,3)], so a 7-qubit cluster yields 210 features
        at level 2 instead of 16383, with bounded peak memory. The connected
        (cumulant) correlations are exactly the physically-meaningful couplings;
        the discarded high-order Gell-Mann terms were the "maximal potential
        connectivity" we're pruning. See project_construction_oom.
        """
        levels = self.features_by_level()
        return sparse_feature_vector(levels, self.max_feature_level)

    def features_by_level(self, max_level: int | None = None) -> dict[int, NDArray[np.floating]]:
        """
        Features decomposed by correlation level (fractal hierarchy).
        Level 1: single-qubit expectations
        Level 2: pairwise correlations
        Level 3: triple correlations

        Stops at `max_level` (defaults to this cluster's `max_feature_level`) so
        the expensive high-order correlations (level 3 = 27·C(n,3) triple-matmuls)
        aren't computed when the readout doesn't use them.
        """
        from umwelt.substrate.fractal import decompose_by_level
        cap = self.max_feature_level if max_level is None else max_level
        return decompose_by_level(self.rho, self.n_qubits, cap)


# ============================================================================
# Partial trace utilities
# ============================================================================

def partial_trace_single(
    rho: ComplexMatrix, keep: int, n_qubits: int
) -> ComplexMatrix:
    """Trace out all qubits except `keep`, returning a 2×2 RDM."""
    return partial_trace_keep(rho, [keep], n_qubits)


def partial_trace_keep(
    rho: ComplexMatrix, keep: list[int], n_qubits: int
) -> ComplexMatrix:
    """
    Partial trace: keep specified qubits, trace out the rest.

    Uses reshape-and-trace approach for efficiency.
    """
    n_keep = len(keep)
    n_trace = n_qubits - n_keep
    dim_keep = 2 ** n_keep
    dim_trace = 2 ** n_trace

    trace_out = sorted(set(range(n_qubits)) - set(keep))

    # Reorder qubits: kept first, traced second
    perm = list(keep) + trace_out

    # Build permutation for rows and columns of the density matrix
    # ρ is indexed by (i_0 i_1 ... i_{n-1}, j_0 j_1 ... j_{n-1})
    rho_tensor = rho.reshape([2] * (2 * n_qubits))

    # Permute axes: first n_qubits axes are row indices, next n_qubits are col
    row_perm = perm
    col_perm = [p + n_qubits for p in perm]
    rho_tensor = rho_tensor.transpose(row_perm + col_perm)

    # Reshape to (dim_keep, dim_trace, dim_keep, dim_trace)
    rho_tensor = rho_tensor.reshape(dim_keep, dim_trace, dim_keep, dim_trace)

    # Trace over the traced-out subsystem
    rdm = np.trace(rho_tensor, axis1=1, axis2=3)

    return rdm

