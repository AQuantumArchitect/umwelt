"""
Density Matrix Evolver for Quantum-Inspired Reservoir Computing.

Evolves a density matrix ρ via the Lindblad master equation:

    dρ/dt = -i[H, ρ] + Σ_k (L_k ρ L_k† - ½{L_k† L_k, ρ})

where:
    H   = system Hamiltonian (drives unitary evolution)
    L_k = Lindblad (jump) operators (model dissipation/decoherence)
    [A,B] = AB - BA  (commutator)
    {A,B} = AB + BA  (anticommutator)

Two input channels:

    UNITARY:      Sensor value modulates H via σ_x input operators.
                  For event-driven sensors (motion, contact) — something HAPPENED.

    DISSIPATIVE:  Sensor value sets the target equilibrium via input-modulated
                  thermal Lindblad operators (σ_+ / σ_-).
                  For continuous sensors (temperature, a periodic driver's altitude) — the
                  world IS at this state.  The qubit thermalizes to the reading
                  through proper quantum decoherence, not a classical bypass.

The expectation values of observables on ρ form the readout features.
"""
from __future__ import annotations

import functools
import os

import numpy as np
from numpy.typing import NDArray

ComplexMatrix = NDArray[np.complexfloating]

# Working precision for the density-matrix evolution. complex64 (fp32 components) is the
# sweet spot on the RDK's Cortex-A55 (NEON fp32 SIMD; ~2× + half the memory bandwidth a
# bandwidth-starved A55 cares about) — fp32's ~7 digits is plenty for the dynamics, and
# step() already re-Hermitizes + renormalizes the trace every step, which absorbs the
# extra rounding drift. Set UMWELT_FP64=1 to fall back to complex128 (A/B benchmark or a
# stability escape hatch). bf16 is NOT used: the A55 (ARMv8.2) has no bf16 units, so it
# would be software-emulated — no win on this chip. See docs/FORESIGHT.md / the RDK perf notes.
_C = np.complex128 if os.environ.get("UMWELT_FP64") == "1" else np.complex64
EVOLVE_DTYPE = _C   # the canonical evolve-path complex dtype; cluster/field/hamiltonian import this
                    # so H, ρ, and every operator share ONE precision (no silent upcast back to fp128)


class DensityMatrix:
    """An n-qubit density matrix ρ living in a 2^n × 2^n Hilbert space."""

    __slots__ = ("dim", "rho")

    def __init__(self, n_qubits: int = 1, initial_state: ComplexMatrix | None = None):
        self.dim = 2 ** n_qubits
        if initial_state is not None:
            if initial_state.shape != (self.dim, self.dim):
                raise ValueError(
                    f"Expected shape ({self.dim}, {self.dim}), "
                    f"got {initial_state.shape}"
                )
            self.rho = initial_state.astype(_C)
        else:
            # Start in |0⟩⟨0| (ground state)
            self.rho = np.zeros((self.dim, self.dim), dtype=_C)
            self.rho[0, 0] = 1.0

    @property
    def purity(self) -> float:
        """Tr(ρ²) — 1.0 for pure states, 1/dim for maximally mixed (atlas: bloch.purity_from_rho)."""
        from umwelt.substrate.bloch import purity_from_rho
        return purity_from_rho(self.rho)

    @property
    def von_neumann_entropy(self) -> float:
        """S(ρ) = -Tr(ρ log ρ). Zero for pure states."""
        eigvals = np.linalg.eigvalsh(self.rho)
        eigvals = eigvals[eigvals > 1e-15]  # clip numerical zeros
        return -np.sum(eigvals * np.log2(eigvals))

    def expectation(self, observable: ComplexMatrix) -> complex:
        """⟨O⟩ = Tr(ρ O)."""
        return np.trace(self.rho @ observable)

    def bloch_vector(self) -> NDArray[np.floating]:
        """For a single qubit, return the Bloch vector (x, y, z).
        For multi-qubit systems, return expectations of all single-qubit Paulis."""
        paulis = _pauli_basis_array(self.dim)  # cached (N, dim, dim)
        # Tr(ρ O_k) = Σ_ij ρ_ij O_k_ji — vectorized over all k
        return np.real(np.einsum("ij,kji->k", self.rho, paulis))

    def reset(self):
        """Reset to |0⟩⟨0|."""
        self.rho[:] = 0
        self.rho[0, 0] = 1.0


class DensityMatrixEvolver:
    """
    Evolves a density matrix under the Lindblad master equation.

    Supports:
      - Unitary evolution via Hamiltonian H
      - Dissipation via Lindblad operators L_k
      - Input-driven Hamiltonian modulation for reservoir computing
      - RK4 integration for accuracy
    """

    def __init__(
        self,
        n_qubits: int,
        hamiltonian: ComplexMatrix | None = None,
        lindblad_ops: list[ComplexMatrix] | None = None,
        dt: float = 0.01,
        input_operators: list[ComplexMatrix] | None = None,
        dissipative_qubit_indices: set[int] | None = None,
    ):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits

        # Default: zero Hamiltonian (no speculative dynamics — all drive from data)
        if hamiltonian is not None:
            self.H_base = hamiltonian.astype(_C)
        else:
            self.H_base = np.zeros((self.dim, self.dim), dtype=_C)

        # Lindblad operators (dissipation channels for UNITARY-input qubits)
        # Stored as "bare" operators — gamma is applied as a live scalar
        # multiplier in _lindblad_rhs(), not baked into the operators.
        self.lindblad_ops = lindblad_ops or []

        # Precompute L†L for each Lindblad operator
        self._LdagL = [L.conj().T @ L for L in self.lindblad_ops]

        # Dissipation strength for unitary-input qubits (amplitude damping).
        self.gamma = 1.0

        # Input operators: each input scalar multiplies one of these
        # and adds to H. This is how sensor signals drive the reservoir.
        if input_operators is not None:
            self.input_ops = input_operators
        else:
            # Default: one Pauli-like operator per input dimension
            self.input_ops = []

        # ── Dissipative input channel ──
        # For continuous sensors: the input value sets the thermal equilibrium
        # target via input-modulated σ_+/σ_- Lindblad operators.
        # Qubit thermalizes to ⟨σ_z⟩ = target_value at rate gamma_diss.
        self._diss_qubits: set[int] = set(dissipative_qubit_indices or ())
        self._diss_down: dict[int, ComplexMatrix] = {}     # qubit -> σ_- operator
        self._diss_up: dict[int, ComplexMatrix] = {}       # qubit -> σ_+ operator
        self._diss_LdL_down: dict[int, ComplexMatrix] = {} # qubit -> σ_+σ_-
        self._diss_LdL_up: dict[int, ComplexMatrix] = {}   # qubit -> σ_-σ_+

        for q in self._diss_qubits:
            sigma_minus = np.zeros((2, 2), dtype=_C)
            sigma_minus[0, 1] = 1.0  # |0⟩⟨1|
            sigma_plus = sigma_minus.conj().T  # |1⟩⟨0|
            L_down = _single_qubit_op(sigma_minus, q, n_qubits)
            L_up = _single_qubit_op(sigma_plus, q, n_qubits)
            self._diss_down[q] = L_down
            self._diss_up[q] = L_up
            self._diss_LdL_down[q] = L_down.conj().T @ L_down
            self._diss_LdL_up[q] = L_up.conj().T @ L_up

        # Thermalization rate for dissipative-input qubits.
        # Per-qubit dict: {qubit_index: rate}. Each role can have its own
        # timescale — a smooth 24h periodic driver vs CPU load (bursty 30s).
        # Falls back to _gamma_diss_default for qubits not in the dict.
        # Accepts scalar assignment for convenience (sets the default).
        self._gamma_diss_per_qubit: dict[int, float] = {}
        self._gamma_diss_default: float = 5.0

        self.dt = dt
        self.dm = DensityMatrix(n_qubits)

    @property
    def gamma_diss(self) -> dict[int, float]:
        """Per-qubit thermalization rates. Assign a scalar to set all, or a dict for per-qubit."""
        return self._gamma_diss_per_qubit

    @gamma_diss.setter
    def gamma_diss(self, value: float | dict[int, float]):
        if isinstance(value, dict):
            self._gamma_diss_per_qubit = value
        else:
            self._gamma_diss_default = float(value)
            self._gamma_diss_per_qubit = {}

    def _lindblad_rhs(
        self,
        rho: ComplexMatrix,
        H: ComplexMatrix,
        diss_targets: dict[int, float] | None = None,
    ) -> ComplexMatrix:
        """Compute dρ/dt from the Lindblad master equation.

        Args:
            rho: Current density matrix.
            H: Effective Hamiltonian (base + unitary input modulation).
            diss_targets: {qubit_index: target_z} for dissipative-input qubits.
                Target_z in [-1, +1] sets the thermal equilibrium.
        """
        # Unitary part: -i[H, ρ]
        drho = -1j * (H @ rho - rho @ H)

        # Dissipative part for UNITARY-input qubits: amplitude damping (σ_-)
        # Dissipative-input qubits skip this — they use the thermal channel below.
        g = self.gamma
        if g > 0 and self.lindblad_ops:
            for q, (L, LdL) in enumerate(zip(self.lindblad_ops, self._LdagL)):
                if q in self._diss_qubits:
                    continue  # handled by thermal channel
                Ldag = L.conj().T
                drho += g * (L @ rho @ Ldag - 0.5 * (LdL @ rho + rho @ LdL))

        # Dissipative part for DISSIPATIVE-input qubits: thermal Lindblad
        # Two operators per qubit:
        #   L↓ (σ_-) at rate γ_diss · (1+v)/2  — decays toward |0⟩
        #   L↑ (σ_+) at rate γ_diss · (1-v)/2  — excites toward |1⟩
        # Steady state: ⟨σ_z⟩ → v at rate γ_diss.
        # Per-qubit γ_diss: each role has its own timescale.
        if self._diss_qubits and diss_targets:
            gd_map = self._gamma_diss_per_qubit
            gd_default = self._gamma_diss_default
            for q in self._diss_qubits:
                v = diss_targets.get(q, 0.0)
                gd = gd_map.get(q, gd_default)
                if gd <= 0:
                    continue
                rate_down = gd * (1.0 + v) / 2.0
                rate_up = gd * (1.0 - v) / 2.0

                if rate_down > 1e-12:
                    Ld = self._diss_down[q]
                    LdL_d = self._diss_LdL_down[q]
                    drho += rate_down * (
                        Ld @ rho @ Ld.conj().T
                        - 0.5 * (LdL_d @ rho + rho @ LdL_d)
                    )
                if rate_up > 1e-12:
                    Lu = self._diss_up[q]
                    LdL_u = self._diss_LdL_up[q]
                    drho += rate_up * (
                        Lu @ rho @ Lu.conj().T
                        - 0.5 * (LdL_u @ rho + rho @ LdL_u)
                    )

        return drho

    def step(
        self,
        inputs: NDArray[np.floating] | None = None,
        dissipative_targets: dict[int, float] | None = None,
        dt_scale: float = 1.0,
    ) -> DensityMatrix:
        """
        Advance the density matrix by one timestep dt × dt_scale.
        Uses RK4 integration. dt_scale > 1 is the smooth-clock catch-up: after skipping
        ticks during calm, the next step advances proportionally more simulated time
        (bounded by the adaptive clock so RK4 stays stable). dt_scale=1.0 = unchanged.

        Args:
            inputs: Array of scalar input values. Each value modulates
                    the corresponding input operator added to H (unitary channel).
            dissipative_targets: {qubit_index: target_z} for dissipative-input
                    qubits. The target sets the thermal equilibrium state.

        Returns:
            The evolved DensityMatrix.
        """
        # Build effective Hamiltonian: H_base + Σ_i input_i * input_op_i
        H = self.H_base.copy()
        if inputs is not None and self.input_ops:
            for val, op in zip(inputs, self.input_ops):
                H += float(val) * op

        rho = self.dm.rho
        dt = self.dt * dt_scale

        # RK4 — dissipative targets are constant within the step
        k1 = self._lindblad_rhs(rho, H, diss_targets=dissipative_targets)
        k2 = self._lindblad_rhs(rho + 0.5 * dt * k1, H, diss_targets=dissipative_targets)
        k3 = self._lindblad_rhs(rho + 0.5 * dt * k2, H, diss_targets=dissipative_targets)
        k4 = self._lindblad_rhs(rho + dt * k3, H, diss_targets=dissipative_targets)

        self.dm.rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Enforce Hermiticity (numerical drift correction)
        self.dm.rho = 0.5 * (self.dm.rho + self.dm.rho.conj().T)

        # Renormalize trace to 1
        tr = np.trace(self.dm.rho)
        if abs(tr) > 1e-15:
            self.dm.rho /= tr

        return self.dm

    def evolve(
        self,
        input_sequence: NDArray[np.floating],
        observables: list[ComplexMatrix] | None = None,
    ) -> NDArray[np.floating]:
        """
        Drive the reservoir with an input sequence and collect readout features.

        Args:
            input_sequence: Shape (T, n_inputs) — time series of input signals.
            observables: List of Hermitian operators to measure. Defaults to
                         the Pauli basis (gives Bloch-like feature vector).

        Returns:
            Feature matrix of shape (T, n_observables) — the reservoir's
            response to the input, ready for ridge regression readout.
        """
        if observables is None:
            observables = _pauli_basis(self.dim)

        T = input_sequence.shape[0]
        n_obs = len(observables)
        features = np.zeros((T, n_obs))

        for t in range(T):
            inp = input_sequence[t] if input_sequence.ndim > 1 else input_sequence[t:t+1]
            self.step(inp)
            for j, obs in enumerate(observables):
                features[t, j] = np.real(self.dm.expectation(obs))

        return features

    def reset(self):
        """Reset the density matrix to |0⟩⟨0|."""
        self.dm.reset()

    @property
    def state_vector(self) -> NDArray[np.floating]:
        """Current Bloch-like feature vector (expectations of Pauli basis)."""
        return self.dm.bloch_vector()


# ============================================================================
# Utilities
# ============================================================================

def _random_hermitian(dim: int, scale: float = 1.0) -> ComplexMatrix:
    """Generate a random Hermitian matrix (H = H†)."""
    rng = np.random.default_rng()
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    A *= scale / dim
    return 0.5 * (A + A.conj().T)


@functools.lru_cache(maxsize=16)
def _pauli_basis_array(dim: int) -> NDArray[np.complexfloating]:
    """Cached stacked array of generalized Pauli/Gell-Mann basis operators.

    Returns shape (N, dim, dim) read-only array where N = dim²-1.
    Cached by dim — only computed once per Hilbert space size.
    For dim=64 (6 qubits) this saves allocating 268 MB on every step.
    """
    result = np.array(_pauli_basis(dim), dtype=_C)
    result.flags.writeable = False
    return result


def _pauli_basis(dim: int) -> list[ComplexMatrix]:
    """
    Build a basis of traceless Hermitian operators.
    For dim=2: Pauli X, Y, Z.
    For dim>2: generalized Gell-Mann matrices.
    """
    if dim == 2:
        return [
            np.array([[0, 1], [1, 0]], dtype=_C),      # σ_x
            np.array([[0, -1j], [1j, 0]], dtype=_C),   # σ_y
            np.array([[1, 0], [0, -1]], dtype=_C),      # σ_z
        ]

    # Generalized Gell-Mann matrices for arbitrary dim
    basis = []

    # Symmetric off-diagonal
    for j in range(dim):
        for k in range(j + 1, dim):
            m = np.zeros((dim, dim), dtype=_C)
            m[j, k] = 1
            m[k, j] = 1
            basis.append(m)

    # Antisymmetric off-diagonal
    for j in range(dim):
        for k in range(j + 1, dim):
            m = np.zeros((dim, dim), dtype=_C)
            m[j, k] = -1j
            m[k, j] = 1j
            basis.append(m)

    # Diagonal
    for l in range(1, dim):
        m = np.zeros((dim, dim), dtype=_C)
        coeff = np.sqrt(2.0 / (l * (l + 1)))
        for j in range(l):
            m[j, j] = coeff
        m[l, l] = -l * coeff
        basis.append(m)

    return basis


def pauli_x(qubit: int = 0, n_qubits: int = 1) -> ComplexMatrix:
    """Pauli X on the specified qubit in an n-qubit system."""
    return _single_qubit_op(
        np.array([[0, 1], [1, 0]], dtype=_C), qubit, n_qubits
    )


def pauli_y(qubit: int = 0, n_qubits: int = 1) -> ComplexMatrix:
    """Pauli Y on the specified qubit in an n-qubit system."""
    return _single_qubit_op(
        np.array([[0, -1j], [1j, 0]], dtype=_C), qubit, n_qubits
    )


def pauli_z(qubit: int = 0, n_qubits: int = 1) -> ComplexMatrix:
    """Pauli Z on the specified qubit in an n-qubit system."""
    return _single_qubit_op(
        np.array([[1, 0], [0, -1]], dtype=_C), qubit, n_qubits
    )


_SQO_CACHE: dict = {}


def _single_qubit_op(op: ComplexMatrix, qubit: int, n_qubits: int) -> ComplexMatrix:
    """Tensor product: I ⊗ ... ⊗ op ⊗ ... ⊗ I.

    Memoized: the embedded operator is constant per (op, qubit, n_qubits) but was being
    rebuilt via kron on every call — the dominant per-tick cost lived here
    (decompose_by_level / the fractal signature embeds Paulis n×3 + pairs×9 every
    features() call). The cached matrix is returned read-only as a guard: callers use it
    read-only (trace(ρ@op), Σ coef·op), so an in-place mutation would be a bug, not silent
    cache corruption. Bit-identical to the original loop."""
    key = (op.tobytes(), op.shape, int(qubit), int(n_qubits))
    cached = _SQO_CACHE.get(key)
    if cached is not None:
        return cached
    I2 = np.eye(2, dtype=_C)
    result = np.array([[1]], dtype=_C)
    for q in range(n_qubits):
        result = np.kron(result, op if q == qubit else I2)
    result.flags.writeable = False
    _SQO_CACHE[key] = result
    return result


def create_reservoir(
    n_qubits: int = 3,
    n_inputs: int = 4,
    gamma: float = 0.05,
    dt: float = 0.01,
) -> DensityMatrixEvolver:
    """
    Factory for a quantum reservoir suitable for smart home time series.

    Args:
        n_qubits: Number of qubits (Hilbert space = 2^n_qubits).
                  3 qubits = 8×8 density matrix, 63 Bloch features.
        n_inputs: Number of input channels (e.g., motion, temp, energy, occupancy).
        gamma: Dissipation rate for Lindblad operators.
        dt: Integration timestep.

    Returns:
        A configured DensityMatrixEvolver.
    """
    dim = 2 ** n_qubits

    # Zero Hamiltonian — no speculative dynamics, all drive from data
    H = np.zeros((dim, dim), dtype=_C)

    # Input operators: σ_x on each qubit (data drives transitions directly)
    # If n_inputs > n_qubits, extra inputs wrap around to same qubits (additive)
    sx_2 = np.array([[0, 1], [1, 0]], dtype=_C)
    input_ops = [_single_qubit_op(sx_2, q % n_qubits, n_qubits) for q in range(n_inputs)]

    # Lindblad operators: bare amplitude damping on each qubit.
    # gamma is set as a live scalar on the evolver, not baked into L.
    lindblad_ops = []
    for q in range(n_qubits):
        sigma_minus = np.zeros((2, 2), dtype=_C)
        sigma_minus[0, 1] = 1.0
        L = _single_qubit_op(sigma_minus, q, n_qubits)
        lindblad_ops.append(L)

    evolver = DensityMatrixEvolver(
        n_qubits=n_qubits,
        hamiltonian=H,
        lindblad_ops=lindblad_ops,
        dt=dt,
        input_operators=input_ops,
    )
    evolver.gamma = gamma
    return evolver
