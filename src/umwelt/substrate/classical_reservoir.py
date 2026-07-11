"""ClassicalReservoirCluster — the classical baseline for the existential ablation (flagship-8).

The #1 reviewer question, and the whitepaper's spine: *is the open-quantum-system formalism load-bearing,
or is this a dressed-up reservoir computer?* The honest way to answer is to swap the OQS substrate for a
genuine classical reservoir on the SAME graph + streams + decision tasks and measure. The SubstrateBackend
interface (substrate.py) was built to make exactly this swap cheap — this class implements that interface
with a leaky echo-state network instead of a density matrix.

It is a FAIR baseline, not a strawman: a real echo-state reservoir (random recurrent W at spectral radius
< 1 → the echo-state property: fading memory + a rich nonlinear feature map, the same `features()` the OQS
exposes to its Ridge readout). What it deliberately LACKS is the structure the quantum claim rests on:
  • no off-diagonal coherence — a role's belief is a single scalar z (the Bloch x,y are pinned to 0);
  • no geometric (Berry) phase — there is no Bloch trajectory to accumulate holonomy on;
  • no Hamiltonian coupling as a unitary — cross-role influence is only the reservoir's recurrent mixing.
If the OQS beats this on decision agreement / predictive log-likelihood — especially on the correlated
multi-region and geometric-phase tasks — the quantum structure is earning its keep. If it doesn't, the honest
paper is "a causally-honest edge world-model" with the substrate as a promising-but-unproven design choice.

RESEARCH ONLY: not wired into build_house; used by the ablation harness (experiments) by selecting
cluster_backend="classical". Default-off, never on the live belief path.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from umwelt.substrate.density_matrix import ComplexMatrix, EVOLVE_DTYPE


class ClassicalReservoirCluster:
    """A leaky echo-state reservoir presenting the SubstrateBackend surface. One reservoir of N units
    per cluster; the first n_roles units ARE the per-role belief units (z ∈ [-1,1]); the rest give the
    reservoir its memory + nonlinear feature richness."""

    is_cumulant = False
    is_product = False
    is_classical = True               # the ablation flag (mirrors is_cumulant/is_product)

    def __init__(self, zone_name: str, qubit_roles, *, units: int | None = None,
                 spectral_radius: float = 0.9, leak: float = 0.3, input_scale: float = 1.0,
                 seed: int = 0, **_ignored):
        self.zone_name = zone_name
        self.qubit_roles = list(qubit_roles)
        self.n_qubits = len(qubit_roles)          # "qubits" = roles, for interface parity
        self.role_index = {r: i for i, r in enumerate(qubit_roles)}
        n = self.n_qubits
        self.N = int(units) if units else max(16, 4 * n)
        self.leak = float(leak)

        # b9.53 fix: hash() is process-randomized (PYTHONHASHSEED) — seeding from it made
        # every process grow a DIFFERENT reservoir for the same (region, seed), which is
        # gate-flaky and, worse, makes the ablation baseline non-reproducible. Deterministic
        # digest instead: same region + seed = same reservoir, everywhere, forever.
        import zlib
        rng = np.random.default_rng(seed + (zlib.crc32(zone_name.encode()) & 0xFFFF))
        # Recurrent matrix, sparse, scaled to the target spectral radius (the echo-state condition).
        W = rng.standard_normal((self.N, self.N)) * (rng.random((self.N, self.N)) < 0.1)
        sr = max(1e-9, np.max(np.abs(np.linalg.eigvals(W))))
        self.W = (W * (spectral_radius / sr)).astype(float)
        # Input matrix: role r drives belief-unit r directly + a random projection into the reservoir.
        self.W_in = rng.standard_normal((self.N, n)) * input_scale
        for r in range(n):
            self.W_in[r, r] += 2.0 * input_scale   # each role's belief unit is its primary driven channel

        self.x = np.zeros(self.N)                  # reservoir state; x[:n] are the role beliefs
        self._held = np.zeros(n)                   # last observed drive per role (held between observes)

    # --- lifecycle / evolution --------------------------------------------
    def reset(self) -> None:
        self.x[:] = 0.0
        self._held[:] = 0.0

    def step(self, inputs: NDArray[np.floating] | None = None, dt_scale: float = 1.0) -> None:
        """Leaky echo-state update x ← (1-leak)x + leak·tanh(W x + W_in u) — the classical analog of the
        Lindblad evolve. dt_scale ≥ 1 runs that many sub-steps (faster internal time, like dt_factor)."""
        u = self._held if inputs is None else np.asarray(inputs, float).reshape(-1)[: self.n_qubits]
        if u.shape[0] < self.n_qubits:
            u = np.pad(u, (0, self.n_qubits - u.shape[0]))
        for _ in range(max(1, int(round(dt_scale)))):
            pre = self.W @ self.x + self.W_in @ u
            self.x = (1.0 - self.leak) * self.x + self.leak * np.tanh(pre)

    def set_hamiltonian(self, H: ComplexMatrix) -> None:
        return  # no Hamiltonian in a classical reservoir — the ablation's point

    def sync_gamma_diss(self, bundle) -> None:
        return

    # --- measurement / assimilation ---------------------------------------
    def observe_qubit(self, qubit_idx: int, target_bloch, alpha: float = 0.5,
                      confidence: float | None = None) -> None:
        """Partial assimilation: nudge the role's belief unit toward the observed z by α, and hold the
        drive so the reservoir keeps integrating it (the classical analog of weak-measurement collapse).
        `confidence` is recorded as the gauge quantity only (caller pre-folds it into `alpha` — the
        uniform substrate contract)."""
        if confidence is not None:
            if not hasattr(self, "_obs_confidence"):
                self._obs_confidence: dict[int, float] = {}
            self._obs_confidence[qubit_idx] = float(confidence)
        if alpha <= 0:
            return
        z = float(np.clip(target_bloch[2], -1.0, 1.0))
        a = min(1.0, float(alpha))
        self.x[qubit_idx] = (1.0 - a) * self.x[qubit_idx] + a * z
        self._held[qubit_idx] = z

    def measure_qubit(self, qubit_idx: int, record_z: float, strength: float,
                      confidence: float | None = None) -> None:
        """Belavkin measurement, coherence-free limit: the z-equation of measure_bloch IS the
        Wonham/log-odds update, which needs no phase — so the ablation backend runs the SAME
        measurement law as the quantum substrates, minus the back-action it has no x/y to feel.
        Same contract: caller pre-folds confidence into `strength`; strength ≤ 0 is the no-op."""
        from umwelt.substrate.belavkin import measure_bloch
        if confidence is not None:
            if not hasattr(self, "_obs_confidence"):
                self._obs_confidence = {}
            self._obs_confidence[qubit_idx] = float(confidence)
        if strength <= 0:
            return
        b, _, _ = measure_bloch((0.0, 0.0, float(self.x[qubit_idx])), float(record_z), float(strength))
        self.x[qubit_idx] = float(b[2])
        self._held[qubit_idx] = float(np.clip(record_z, -1.0, 1.0))

    def nudge_toward_rdm(self, qubit_idx: int, target_rdm: NDArray, alpha: float) -> None:
        z = float(np.real(target_rdm[0, 0] - target_rdm[1, 1]))
        self.observe_qubit(qubit_idx, (0.0, 0.0, z), alpha)

    # --- readout -----------------------------------------------------------
    def qubit_bloch(self, qubit_idx: int) -> NDArray[np.floating]:
        # z = the role belief unit; x=y=0 — NO coherence, NO phase (the structure the OQS adds).
        return np.array([0.0, 0.0, float(np.clip(self.x[qubit_idx], -1.0, 1.0))])

    def role_bloch(self, role: str) -> NDArray[np.floating]:
        return self.qubit_bloch(self.role_index[role])

    def qubit_rdm(self, qubit_idx: int) -> ComplexMatrix:
        z = float(np.clip(self.x[qubit_idx], -1.0, 1.0))
        return np.array([[(1.0 + z) / 2.0, 0.0], [0.0, (1.0 - z) / 2.0]], dtype=EVOLVE_DTYPE)

    def all_bloch(self) -> dict:
        return {r: self.qubit_bloch(i) for r, i in self.role_index.items()}

    def features(self) -> NDArray[np.floating]:
        """The full reservoir state — the classical feature map the (same) Ridge readout consumes. This is
        what makes it a real reservoir computer rather than a per-role scalar tracker."""
        return self.x.copy()

    # --- constraints / diagnostics ----------------------------------------
    def clamp_physical(self) -> None:
        np.clip(self.x, -1.0, 1.0, out=self.x)     # keep belief units in range (tanh already bounds them)

    def hamiltonian_norm(self) -> float:
        return 0.0

    @property
    def purity(self) -> float:
        # a diagonal 2-level state is the x=y=0 limit of the atlas's qubit_purity (1+|r|²)/2,
        # averaged over roles — surfaced for gauge parity
        from umwelt.substrate.bloch import qubit_purity
        if not self.n_qubits:
            return 1.0
        z = np.clip(self.x[: self.n_qubits], -1.0, 1.0)
        b = np.zeros((self.n_qubits, 3))
        b[:, 2] = z
        return float(np.mean(qubit_purity(b)))
