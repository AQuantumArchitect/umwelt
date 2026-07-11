"""ProductQubitCluster — the whole parameter fiber as N INDEPENDENT qubits.

Every learnable param should live on a qubit (Thompson-wriggling exploration, purity = confidence,
Berry-phase history riding the Bloch trajectory — [[project_qubit_param_pilot]],
[[feedback_wriggling_octopus]]). The pilot put a few params on one `_params` node, but that node is a
2^N JOINT density matrix, so it can't scale: ~200 params would be 2^200, unallocatable.

The unlock: parameter qubits are UNENTANGLED by design (gamma=0, no bridges/projection/input — "isolated
memory cells"). So storing them as N separate 2×2 matrices is EXACT, not an approximation:
`observe_qubit` already computes `ρ_new = (1−α)ρ + α·(ρ_q ⊗ Tr_q[ρ])`; for a product state `Tr_q[ρ]` IS
the other factors, so only the q-th 2×2 changes by the plain blend `(1−α)ρ_q + α·target`. This class
duck-types the QubitCluster surface that QubitBackedParam + field.step touch — observe_qubit,
qubit_bloch, role_bloch, role_index, step — at O(N) memory + compute instead of O(2^N).

It GROWS via `add_role` (the binding sweep adds one role per param), and its `rho` RAISES rather than
building the impossible joint kron — that's the guard that makes the .rho audit catch any joint-matrix op
that forgot to skip product clusters (check `is_product`). It lives in `field.clusters` as pure parameter
storage, NOT a graph node (no projection, no bridges, no sensors). See plan noble-sleeping-yao + #309.
"""
from __future__ import annotations

import numpy as np

_SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_I2 = np.eye(2, dtype=np.complex128)
_MIXED = 0.5 * _I2.copy()    # maximally-mixed seed (Bloch origin, purity 0) before a param binds


class _ProductEvolverStub:
    """Minimal evolver surface so field._sync_params (gamma/dt) and any code that iterates every
    cluster's evolver.H_base can duck-type without special-casing. Param qubits don't evolve (gamma=0,
    no input, no Hamiltonian) — these are inert. H_base is the empty 0×0 ("no Hamiltonian"), so
    `np.allclose(H_base, 0)` is vacuously true."""
    def __init__(self):
        self.gamma = 0.0
        self.dt = 0.0
        self.H_base = np.zeros((0, 0), dtype=np.complex128)
        self._gamma_diss_default = 0.0
        self._gamma_diss_per_qubit: dict = {}


class ProductQubitCluster:
    """N independent single-qubit density matrices. Same per-qubit API as QubitCluster, O(N) not O(2^N).

    M2 status (b9.35, MEASURED): this is NOT a duplicate substrate to dissolve — it is the
    load-bearing O(N) fast path. Conceptually it IS CumulantCluster with connectivity ∅
    (one 1-RDM per qubit, no pair blocks), but the cumulant engine still carries its
    O(n²) e2 machinery with the pairs empty: at the param fiber's size the measured gap
    is ~0.00 vs 24.4 ms/step at n=160 (x86). The named limit stays a separate class on
    the measured evidence; revisit only if the cumulant engine grows a sparse-e2 path."""

    is_product = True          # the guard flag every joint-matrix op checks (getattr(c,'is_product',False))

    def __init__(self, zone_name: str, qubit_roles: list[str] | None = None):
        self.zone_name = zone_name
        self.qubit_roles: list[str] = []
        self.role_index: dict[str, int] = {}
        self._mats: list[np.ndarray] = []
        self.evolver = _ProductEvolverStub()
        for r in (qubit_roles or []):
            self.add_role(r)

    # ── shape ───────────────────────────────────────────────────────────────
    @property
    def n_qubits(self) -> int:
        return len(self._mats)

    @property
    def dim(self) -> int:
        # NOT 2^N — these qubits never form a joint matrix. A sentinel for code that reads .dim;
        # anything that would allocate a dim×dim joint matrix must skip product clusters (is_product).
        return 2 * len(self._mats)

    def add_role(self, role: str) -> int:
        """Grow the fiber by one qubit (the binding sweep calls this per param). Idempotent."""
        if role in self.role_index:
            return self.role_index[role]
        idx = len(self._mats)
        self._mats.append(_MIXED.copy())
        self.qubit_roles.append(role)
        self.role_index[role] = idx
        return idx

    # ── the per-qubit API QubitBackedParam + field use ──────────────────────
    def observe_qubit(self, qubit_idx: int, target_bloch, alpha: float = 0.5,
                      confidence: float | None = None) -> None:
        """Partial collapse of ONE qubit toward a Bloch target — the plain 2×2 blend
        (no joint matrix). `confidence` is recorded as the gauge quantity only
        (caller pre-folds it into `alpha` — the uniform substrate contract)."""
        if confidence is not None:
            if not hasattr(self, "_obs_confidence"):
                self._obs_confidence: dict[int, float] = {}
            self._obs_confidence[qubit_idx] = float(confidence)
        if alpha <= 0:
            return
        x, y, z = (float(target_bloch[0]), float(target_bloch[1]), float(target_bloch[2]))
        r = (x * x + y * y + z * z) ** 0.5
        if r > 1.0:
            x, y, z = x / r, y / r, z / r
        target = 0.5 * (_I2 + x * _SX + y * _SY + z * _SZ)
        self._mats[qubit_idx] = (1.0 - alpha) * self._mats[qubit_idx] + alpha * target

    def measure_qubit(self, qubit_idx: int, record_z: float, strength: float,
                      confidence: float | None = None) -> None:
        """Belavkin weak σ_z measurement on one independent qubit (docs/QUANTUM_KALMAN.md).
        Product qubits have no peers by construction, so this is the pure
        single-qubit conditioned Kraus update. Same contract as observe_qubit."""
        from umwelt.substrate.belavkin import measure_bloch
        if confidence is not None:
            if not hasattr(self, "_obs_confidence"):
                self._obs_confidence = {}
            self._obs_confidence[qubit_idx] = float(confidence)
        if strength <= 0:
            return
        b, _, _ = measure_bloch(self.qubit_bloch(qubit_idx), float(record_z), float(strength))
        x, y, z = float(b[0]), float(b[1]), float(b[2])
        self._mats[qubit_idx] = 0.5 * (_I2 + x * _SX + y * _SY + z * _SZ)

    # --- SubstrateBackend contract (substrate.py) ---
    # The parameter fiber is unentangled and not bridged/projected, so these are
    # inert: there is no joint ρ to nudge and the 2×2 factors are physical by
    # construction. Present so the field can call them uniformly on any cluster.
    def nudge_toward_rdm(self, qubit_idx: int, target_rdm: np.ndarray, alpha: float) -> None:
        return  # param fiber carries no bridges/projections

    def clamp_physical(self) -> None:
        return  # 2×2 factors stay physical by construction

    def hamiltonian_norm(self) -> float:
        return 0.0  # the param fiber is inert (no Hamiltonian)

    def reset(self) -> None:
        """Reset every param qubit to the maximally-mixed seed (Bloch origin, purity 0 —
        the blank-slate floor add_role starts from). The fiber lives in field.clusters, so
        field.reset() reaches it; resetting to |0⟩ would be a meaningless pole for a param,
        unbound/mixed is the right zero."""
        for i in range(len(self._mats)):
            self._mats[i] = _MIXED.copy()

    def qubit_rdm(self, qubit_idx: int) -> np.ndarray:
        return self._mats[qubit_idx]                 # for a product state the RDM IS the factor

    def qubit_bloch(self, qubit_idx: int) -> np.ndarray:
        m = self._mats[qubit_idx]
        return np.array([float(np.real(np.trace(m @ _SX))),
                         float(np.real(np.trace(m @ _SY))),
                         float(np.real(np.trace(m @ _SZ)))])

    def role_bloch(self, role: str) -> np.ndarray:
        return self.qubit_bloch(self.role_index[role])

    def all_bloch(self) -> dict:
        return {role: self.qubit_bloch(i) for role, i in self.role_index.items()}

    # ── inert evolution + zero features (these qubits are frozen memory cells) ──
    def step(self, inputs=None, dt_scale: float = 1.0) -> None:
        return                                        # gamma=0, no input → no evolution

    def sync_gamma_diss(self, bundle) -> None:
        return                                        # duck-type for field._sync_params

    def set_hamiltonian(self, H) -> None:
        return                                        # no Hamiltonian on the inert fiber (no-op)

    def features(self) -> np.ndarray:
        return np.zeros(0)                            # not a sensor cluster → contributes nothing

    def features_by_level(self, *a, **k) -> dict:
        return {}

    # ── the joint-matrix guard ──────────────────────────────────────────────
    @property
    def rho(self) -> np.ndarray:
        raise RuntimeError(
            f"ProductQubitCluster '{self.zone_name}' has NO joint density matrix "
            f"(N={self.n_qubits} would be 2^N). Read per-qubit via qubit_bloch / state_matrices; "
            f"joint-matrix ops must skip product clusters (is_product)."
        )

    @rho.setter
    def rho(self, value) -> None:
        raise RuntimeError(f"ProductQubitCluster '{self.zone_name}' has no joint rho to set")

    # ── pickle: per-qubit 2×2 state (preserves value + purity + Bloch phase) ──
    def state_matrices(self) -> dict:
        """role → 2×2 density matrix, for the pickle's `param_fiber_qubits` block (NOT a 2^N kron)."""
        return {role: self._mats[i].copy() for role, i in self.role_index.items()}

    def load_matrices(self, mats: dict) -> int:
        """Restore the 2×2 matrices by role (the pickle's authoritative qubit state). Returns count."""
        n = 0
        for role, m in mats.items():
            i = self.role_index.get(role)
            if i is not None:
                self._mats[i] = np.asarray(m, dtype=np.complex128)
                n += 1
        return n
