"""Batched Lindblad evolution — the WIDE hot path.

M2 status (b9.35): this optimization serves DENSE clusters only, and small dense clusters
are exactly what the C0→C1 cumulant cutover retires from prod (cumulant now measures
FASTER than dense even at n=2 on x86 — the old small-cluster penalty is gone since the
vectorized _deriv_vec work). Pre-cutover it stays (default ON, measured 1.30× live on the
A55); post-cutover it follows the dense path to the test-oracle side and is deleted with it.

The field evolves dozens of clusters per tick, each a small density matrix ρ
stepped through RK4 on the same Lindblad master equation (density_matrix.py).
The per-cluster Python loop pays matmul-dispatch overhead once *per cluster* —
death by a thousand small BLAS calls on a bandwidth-starved A55.

KEY OBSERVATION that makes batching exact, not approximate: for every cluster
with the same qubit count n (same Hilbert dim d=2^n), the Lindblad jump
operators are *structurally identical* — σ_-/σ_+ embedded on qubit position q
are the SAME (d,d) matrices regardless of which region the cluster models. The
only things that differ per cluster are SCALARS (the amplitude-damping γ, the
per-qubit thermal rate γ_diss, the dissipative target v) and the Hamiltonian H.

So we group clusters by dim, stack their ρ and H into (B,d,d), and replace the
per-cluster matmul loop with one native batched matmul (numpy `@` broadcasts
over the leading batch axis; torch.bmm drops in later). Per-cluster rates become
(B,) vectors broadcast over the batch. The math is bit-identical to the loop up
to fp rounding — the original's `if rate>1e-12: skip` guards just drop terms a
zero rate would contribute nothing to anyway.

Two σ_- terms in the original (amplitude damping on unitary qubits, thermal-down
on dissipative qubits) share the same operator, so they MERGE into one σ_-
channel whose per-cluster rate is whichever applies (they're mutually exclusive
per qubit). σ_+ is the thermal-up channel. That's two channels per qubit
position, each a single batched L@ρ@L† sandwich with a (B,) rate.

Backend is swappable behind `BatchedBackend`: NumpyBackend is the free default
(no new dependency on the storage-tight potato); a TorchBackend (torch.bmm /
torch.compile fusion / GPU) drops in when it earns its footprint — the
commercial direction, see project_jax_metagraph_unification for the JAX research
arc beyond that.
"""
from __future__ import annotations

import functools
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from umwelt.substrate.density_matrix import EVOLVE_DTYPE, _single_qubit_op


@functools.lru_cache(maxsize=16)
def _group_ops(n_qubits: int) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Precompute the stacked jump operators for an n-qubit Hilbert space.

    Returns (Lm, Lm_dag, LmdLm, Lp, LpdLp), each shape (n, d, d):
      Lm[q]   = σ_- on qubit q   (lowering / amplitude-damping + thermal-down)
      Lp[q]   = σ_+ on qubit q   (raising / thermal-up)  == Lm[q]†
      *dag, *dL are the precomputed Hermitian-conjugate and L†L products.
    Cached per n_qubits — every cluster of that size shares one read-only stack.
    """
    sigma_minus = np.zeros((2, 2), dtype=EVOLVE_DTYPE)
    sigma_minus[0, 1] = 1.0  # |0><1|
    sigma_plus = sigma_minus.conj().T  # |1><0|

    Lm, Lp = [], []
    for q in range(n_qubits):
        Lm.append(_single_qubit_op(sigma_minus, q, n_qubits))
        Lp.append(_single_qubit_op(sigma_plus, q, n_qubits))
    Lm = np.stack(Lm).astype(EVOLVE_DTYPE)
    Lp = np.stack(Lp).astype(EVOLVE_DTYPE)
    Lm_dag = Lm.conj().transpose(0, 2, 1).copy()
    LmdLm = (Lm_dag @ Lm).astype(EVOLVE_DTYPE)
    Lp_dag = Lp.conj().transpose(0, 2, 1).copy()
    LpdLp = (Lp_dag @ Lp).astype(EVOLVE_DTYPE)
    for a in (Lm, Lp, Lm_dag, LmdLm, LpdLp):
        a.flags.writeable = False
    return Lm, Lm_dag, LmdLm, Lp, LpdLp


class BatchedBackend(Protocol):
    """The swappable compute seam. One method: advance a same-dim group one RK4
    step. NumpyBackend is the default; a TorchBackend implements the same
    contract with torch.bmm later."""

    def rk4_step(
        self,
        rho: NDArray,        # (B, d, d) complex — stacked density matrices
        H: NDArray,          # (B, d, d) complex — stacked effective Hamiltonians
        r_minus: NDArray,    # (B, n) real — σ_- channel rate per cluster/qubit
        r_plus: NDArray,     # (B, n) real — σ_+ channel rate per cluster/qubit
        dt: NDArray,         # (B,) real — per-cluster timestep (dt * dt_scale)
        n_qubits: int,
    ) -> NDArray:            # (B, d, d) complex — evolved, Hermitized, trace-1
        ...

    def free_run(
        self,
        rho: NDArray,        # (B, d, d) complex — stacked density matrices
        H: NDArray,          # (B, d, d) complex — stacked effective Hamiltonians (held FIXED)
        r_minus: NDArray,    # (B, n) real
        r_plus: NDArray,     # (B, n) real
        dt: NDArray,         # (B,) real
        n_qubits: int,
        n_steps: int,        # advance this many RK4 steps with the operators held constant
    ) -> NDArray:            # (B, d, d) complex — the field after n_steps of its OWN dynamics
        """Advance n_steps with H/rates FIXED — the decoupled free-run between data samples.

        This is the engine: a forecast/dream rolls the field forward under its own field with no
        fresh evidence, so H and the rates don't change across the run. The default loops rk4_step;
        JaxBackend fuses the whole N-step run into ONE XLA kernel (one GIL acquire, the BPU on-ramp)."""
        ...


class NumpyBackend:
    """Native-numpy batched RK4. numpy's `@` broadcasts a (d,d) operator across
    the (B,d,d) batch, so each L@ρ@L† sandwich is one batched matmul covering
    every cluster in the group at once."""

    @staticmethod
    def _rhs(rho, H, ops, active, r_minus, r_plus):
        Lm, Lm_dag, LmdLm, Lp, LpdLp = ops
        m_active, p_active = active
        # Unitary part: -i[H, ρ] — batched commutator over the whole group.
        drho = -1j * (H @ rho - rho @ H)
        n = Lm.shape[0]
        for q in range(n):
            # Skip a channel whose rate is zero for EVERY cluster in the group
            # (e.g. σ_+ on a qubit no cluster treats as dissipative). Matches the
            # original loop's work-skipping — adding 0·(L@ρ@L†) is wasted matmul,
            # which on a unitary-heavy group is most of the qubits.
            if m_active[q]:
                rm = r_minus[:, q][:, None, None]
                # σ_- channel: amplitude damping (unitary qubits) + thermal-down
                # (dissipative qubits) share this operator; rate is whichever applies.
                drho = drho + rm * (
                    Lm[q] @ rho @ Lm_dag[q]
                    - 0.5 * (LmdLm[q] @ rho + rho @ LmdLm[q])
                )
            if p_active[q]:
                rp = r_plus[:, q][:, None, None]
                # σ_+ channel: thermal-up (dissipative qubits only; rate 0 elsewhere).
                drho = drho + rp * (
                    Lp[q] @ rho @ Lp[q].conj().T
                    - 0.5 * (LpdLp[q] @ rho + rho @ LpdLp[q])
                )
        return drho

    def rk4_step(self, rho, H, r_minus, r_plus, dt, n_qubits):
        ops = _group_ops(n_qubits)
        # Which channels carry a nonzero rate for at least one cluster — skip the
        # rest (all-zero σ_+ on a unitary-only qubit position is pure waste).
        active = (np.any(r_minus != 0.0, axis=0), np.any(r_plus != 0.0, axis=0))
        dtb = dt[:, None, None]  # (B,1,1) — per-cluster step broadcast over (d,d)

        k1 = self._rhs(rho, H, ops, active, r_minus, r_plus)
        k2 = self._rhs(rho + 0.5 * dtb * k1, H, ops, active, r_minus, r_plus)
        k3 = self._rhs(rho + 0.5 * dtb * k2, H, ops, active, r_minus, r_plus)
        k4 = self._rhs(rho + dtb * k3, H, ops, active, r_minus, r_plus)
        rho_new = rho + (dtb / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Hermitize (batched) — drift correction, matches evolver.step.
        rho_new = 0.5 * (rho_new + rho_new.conj().transpose(0, 2, 1))
        # Renormalize trace to 1, per cluster.
        tr = np.trace(rho_new, axis1=1, axis2=2)  # (B,)
        safe = np.abs(tr) > 1e-15
        tr = np.where(safe, tr, 1.0)
        rho_new = rho_new / tr[:, None, None]
        return rho_new.astype(EVOLVE_DTYPE)

    def free_run(self, rho, H, r_minus, r_plus, dt, n_qubits, n_steps):
        """Reference free-run: loop rk4_step n_steps with the operators held fixed."""
        for _ in range(int(n_steps)):
            rho = self.rk4_step(rho, H, r_minus, r_plus, dt, n_qubits)
        return rho


class DispatchBackend:
    """The same batched RK4 as NumpyBackend, but every matmul runs through a NUMBER SYSTEM from
    bpu_dispatch — unifying the two batching seams (batched_evolve's Protocol + the bpu_dispatch
    backends). This is the bridge that lets the field's stacked (B,d,d) evolution run in the BPU's
    native format and, when the offline-compiled .bin kernel lands, on the BPU itself.

      backend='cpu'        float64, identity quantizer — bit-identical to NumpyBackend (the reference).
      backend='expansion'  the 2-term base-2 INT8 EXPANSION (the fib_fractal finding) — proves the REAL
                           batched field evolution survives the BPU-native shift+scale block-float format,
                           not just the toy Lindblad the experiment used.
      backend='bpu'        the compiled kernel (unavailable until HBDK3; falls back via bpu_dispatch).

    Intended for the LATENCY-TOLERANT, repeated workloads — forecast rollouts, dreaming, hindbrain
    replay — NOT the live field (which stays NumpyBackend on the CPU for responsiveness). Forecasts
    snapshot→roll-forward→restore, so a slower-but-offloadable backend there costs the live field nothing."""

    def __init__(self, backend: str = "expansion", levels: int = 2):
        from umwelt.foresight.bpu_dispatch import block_float_quantize, expansion_quantize
        self.backend = backend
        self.levels = int(levels)
        if backend == "cpu":
            self._q = lambda M: M                                      # float reference
        elif backend == "block":
            self._q = lambda M: block_float_quantize(M, levels=self.levels)   # BPU-native (shared scale)
        else:  # 'expansion' (per-element upper bound) / 'bpu' (sim until the .bin lands)
            self._q = lambda M: expansion_quantize(M, levels=self.levels)

    def _mm(self, a, b):
        # quantize operands → batched matmul → re-quantize the output (the on-chip number system,
        # batched: numpy `@` broadcasts a (d,d) op across (B,d,d), or (B,d,d)@(B,d,d) element-wise).
        return self._q(self._q(a) @ self._q(b))

    def _rhs(self, rho, H, ops, active, r_minus, r_plus):
        Lm, Lm_dag, LmdLm, Lp, LpdLp = ops
        m_active, p_active = active
        drho = -1j * (self._mm(H, rho) - self._mm(rho, H))
        for q in range(Lm.shape[0]):
            if m_active[q]:
                rm = r_minus[:, q][:, None, None]
                drho = drho + rm * (
                    self._mm(self._mm(Lm[q], rho), Lm_dag[q])
                    - 0.5 * (self._mm(LmdLm[q], rho) + self._mm(rho, LmdLm[q]))
                )
            if p_active[q]:
                rp = r_plus[:, q][:, None, None]
                drho = drho + rp * (
                    self._mm(self._mm(Lp[q], rho), Lp[q].conj().T)
                    - 0.5 * (self._mm(LpdLp[q], rho) + self._mm(rho, LpdLp[q]))
                )
        return drho

    def rk4_step(self, rho, H, r_minus, r_plus, dt, n_qubits):
        ops = _group_ops(n_qubits)
        active = (np.any(r_minus != 0.0, axis=0), np.any(r_plus != 0.0, axis=0))
        dtb = dt[:, None, None]
        k1 = self._rhs(rho, H, ops, active, r_minus, r_plus)
        k2 = self._rhs(rho + 0.5 * dtb * k1, H, ops, active, r_minus, r_plus)
        k3 = self._rhs(rho + 0.5 * dtb * k2, H, ops, active, r_minus, r_plus)
        k4 = self._rhs(rho + dtb * k3, H, ops, active, r_minus, r_plus)
        rho_new = rho + (dtb / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        rho_new = 0.5 * (rho_new + rho_new.conj().transpose(0, 2, 1))
        tr = np.trace(rho_new, axis1=1, axis2=2)
        safe = np.abs(tr) > 1e-15
        tr = np.where(safe, tr, 1.0)
        rho_new = rho_new / tr[:, None, None]
        return rho_new.astype(EVOLVE_DTYPE)

    def free_run(self, rho, H, r_minus, r_plus, dt, n_qubits, n_steps):
        """Free-run through the BPU-native number system — n_steps in the on-chip format. When the
        compiled .bin lands it runs this whole scan on the BPU; here it proves the format survives it."""
        for _ in range(int(n_steps)):
            rho = self.rk4_step(rho, H, r_minus, r_plus, dt, n_qubits)
        return rho


class JaxBackend:
    """XLA-FUSED batched RK4 — the dispatch-collapse the GIL actually needs (and the substrate for the
    decoupled free-run engine + the BPU on-ramp). The ENTIRE same-dim group step (4× RHS + combine +
    Hermitize + renorm) compiles to ONE XLA kernel: one GIL acquire for the whole evolution instead of
    the ~26k tiny numpy ops/tick that each grab it. Same contract + numerics (~float32 / ~1e-5 vs Numpy)
    as NumpyBackend. Measured on x86: 1.6–2.1× wall vs numpy, and the per-call GIL hold collapses to one
    kernel. JAX is an OPTIONAL dep — imported lazily, ONLY when this backend is selected — so the numpy
    brain (and the RDK, which has no JAX-on-ARM yet) runs untouched. Opt-in: make_backend('jax'); the live
    forebrain stays numpy. The all-channel compute is numerically equal to the loop's active-skip (the
    zero rates null inactive channels) and lets the kernel be one static-shape XLA graph per qubit count.

    See [[project_jax_metagraph_unification]] — this is the field half of the JAX substrate that also
    governs the datastream graph; the BPU lands later behind the same XLA front end."""

    def __init__(self):
        import jax  # noqa: F401 — lazy; raises ImportError if JAX absent (only when selected)
        import jax.numpy as jnp  # noqa: F401
        self._jax = jax
        self._jnp = jnp
        self._kernels: dict[int, object] = {}        # n → jit(single rk4 step)
        self._freerun_kernels: dict[int, object] = {}  # n → jit(fori_loop of n_steps)

    def _build_rk4(self, n: int):
        """The pure (un-jitted) one-step RK4 closure for an n-qubit group — shared by the single-step
        kernel and the fused free-run loop so both run identical numerics."""
        jnp = self._jnp
        Lm, Lm_dag, LmdLm, Lp, LpdLp = (jnp.asarray(np.asarray(o)) for o in _group_ops(n))
        Lp_dag = jnp.conj(jnp.transpose(Lp, (0, 2, 1)))

        def _rhs(rho, H, rm, rp):
            drho = -1j * (H @ rho - rho @ H)
            sand_m = jnp.einsum('qij,Bjk,qkl->qBil', Lm, rho, Lm_dag)
            anti_m = jnp.einsum('qij,Bjk->qBik', LmdLm, rho) + jnp.einsum('Bij,qjk->qBik', rho, LmdLm)
            drho = drho + jnp.einsum('Bq,qBij->Bij', rm, sand_m - 0.5 * anti_m)
            sand_p = jnp.einsum('qij,Bjk,qkl->qBil', Lp, rho, Lp_dag)
            anti_p = jnp.einsum('qij,Bjk->qBik', LpdLp, rho) + jnp.einsum('Bij,qjk->qBik', rho, LpdLp)
            return drho + jnp.einsum('Bq,qBij->Bij', rp, sand_p - 0.5 * anti_p)

        def _rk4(rho, H, rm, rp, dt):
            dtb = dt[:, None, None]
            k1 = _rhs(rho, H, rm, rp)
            k2 = _rhs(rho + 0.5 * dtb * k1, H, rm, rp)
            k3 = _rhs(rho + 0.5 * dtb * k2, H, rm, rp)
            k4 = _rhs(rho + dtb * k3, H, rm, rp)
            rn = rho + (dtb / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            rn = 0.5 * (rn + jnp.conj(jnp.transpose(rn, (0, 2, 1))))
            tr = jnp.trace(rn, axis1=1, axis2=2)
            tr = jnp.where(jnp.abs(tr) > 1e-15, tr, 1.0)
            return rn / tr[:, None, None]

        return _rk4

    def _kernel(self, n: int):
        fn = self._kernels.get(n)
        if fn is None:
            fn = self._kernels[n] = self._jax.jit(self._build_rk4(n))
        return fn

    def _freerun_kernel(self, n: int):
        """The DECOUPLED ENGINE: N RK4 steps with the operators held fixed, fused into ONE XLA kernel
        via lax.fori_loop. One GIL acquire + one dispatch for the whole free-run (vs N per the loop),
        and exactly the shape the BPU eventually churns. n_steps is a runtime arg (no retrace per H)."""
        fn = self._freerun_kernels.get(n)
        if fn is not None:
            return fn
        jax = self._jax
        rk4 = self._build_rk4(n)

        def _run(rho, H, rm, rp, dt, n_steps):
            body = lambda _i, r: rk4(r, H, rm, rp, dt)        # noqa: E731 — fixed operators in closure
            return jax.lax.fori_loop(0, n_steps, body, rho)

        fn = self._freerun_kernels[n] = jax.jit(_run, static_argnums=())
        return fn

    def rk4_step(self, rho, H, r_minus, r_plus, dt, n_qubits):
        jnp = self._jnp
        out = self._kernel(n_qubits)(
            jnp.asarray(rho), jnp.asarray(H), jnp.asarray(r_minus), jnp.asarray(r_plus), jnp.asarray(dt))
        return np.asarray(out).astype(EVOLVE_DTYPE)

    def free_run(self, rho, H, r_minus, r_plus, dt, n_qubits, n_steps):
        jnp = self._jnp
        out = self._freerun_kernel(n_qubits)(
            jnp.asarray(rho), jnp.asarray(H), jnp.asarray(r_minus), jnp.asarray(r_plus),
            jnp.asarray(dt), int(n_steps))
        return np.asarray(out).astype(EVOLVE_DTYPE)


_DEFAULT_BACKEND = NumpyBackend()
_JAX_BACKEND = None  # lazily constructed on first 'jax' selection (caches its jitted kernels)

# Backend registry — the live field uses 'numpy'; latency-tolerant workloads (forecasts/dreaming)
# can select a number-system backend that's ready for the BPU. `make_backend` keeps the choice in one place.
def make_backend(name: str = "numpy"):
    """Pick a batched-evolution backend. 'numpy' = the live-field default. 'jax' = XLA-fused (the GIL
    dispatch-collapse; optional dep, forecasts/dreams/experiments). Otherwise 'cpu' | 'expansion'
    | 'block' | 'bpu', with an optional ':<depth>' (e.g. 'block:3' = 3-level block float)."""
    if name in (None, "numpy", "cpu_float"):
        return _DEFAULT_BACKEND
    if name == "jax":
        global _JAX_BACKEND
        if _JAX_BACKEND is None:
            _JAX_BACKEND = JaxBackend()
        return _JAX_BACKEND
    backend, _, depth = name.partition(":")
    return DispatchBackend(backend=backend, levels=int(depth) if depth else 2)


def gather_group_rates(clusters, inputs_by_name):
    """Build the per-cluster (B,n) σ_- / σ_+ rate vectors and (B,) dt for a group
    of same-dim clusters, replicating evolver._lindblad_rhs's rate logic exactly.

    Returns (r_minus, r_plus, dt) ready for BatchedBackend.rk4_step.

    For qubit q in cluster b:
      σ_- rate = γ_b            if q is a UNITARY qubit (amplitude damping)
               = γ_diss·(1+v)/2 if q is DISSIPATIVE  (thermal-down toward |0>)
      σ_+ rate = γ_diss·(1-v)/2 if q is DISSIPATIVE  (thermal-up toward |1>)
               = 0              otherwise
    where v is the dissipative target (the sensor input for that qubit, 0 if none).
    """
    B = len(clusters)
    n = clusters[0].n_qubits
    r_minus = np.zeros((B, n), dtype=np.float64)
    r_plus = np.zeros((B, n), dtype=np.float64)
    dt = np.zeros(B, dtype=np.float64)

    for b, c in enumerate(clusters):
        ev = c.evolver
        dt[b] = ev.dt
        inp = inputs_by_name.get(c.zone_name)
        diss = ev._diss_qubits
        gd_map = ev._gamma_diss_per_qubit
        gd_def = ev._gamma_diss_default
        g = ev.gamma
        for q in range(n):
            if q in diss:
                v = float(inp[q]) if (inp is not None and q < len(inp)) else 0.0
                gd = gd_map.get(q, gd_def)
                if gd > 0:
                    r_minus[b, q] = gd * (1.0 + v) / 2.0
                    r_plus[b, q] = gd * (1.0 - v) / 2.0
            else:
                # Unitary qubit: amplitude damping at γ (σ_- channel).
                if g > 0:
                    r_minus[b, q] = g
    return r_minus, r_plus, dt


def build_group_H(clusters, inputs_by_name):
    """Stack each cluster's effective Hamiltonian H = H_base + Σ_q inputs[q]·op_q
    into (B,d,d), exactly as evolver.step builds it (σ_x kick on unitary qubits;
    dissipative qubits have null input ops so they don't enter H)."""
    Hs = []
    for c in clusters:
        ev = c.evolver
        H = ev.H_base.copy()
        inp = inputs_by_name.get(c.zone_name)
        if inp is not None and ev.input_ops:
            for val, op in zip(inp, ev.input_ops):
                if val:
                    H = H + float(val) * op
        Hs.append(H)
    return np.stack(Hs).astype(EVOLVE_DTYPE)


def evolve_groups(clusters_by_dim, inputs_by_name, dt_scale=1.0, backend=None):
    """Advance several same-dim groups of clusters one batched RK4 step in place.

    clusters_by_dim: {dim: [cluster, ...]} — clusters grouped by Hilbert dim.
    inputs_by_name:  {cluster.zone_name: input_array} — per-cluster sensor drive.
    dt_scale:        smooth-clock catch-up multiplier (scales every cluster's dt).

    Writes the evolved ρ back onto each cluster (cluster.rho setter). This is the
    drop-in batched replacement for the `for name, cluster: cluster.step(...)`
    loop in QuantumField.step Phase 1.
    """
    backend = backend or _DEFAULT_BACKEND
    for dim, clusters in clusters_by_dim.items():
        if not clusters:
            continue
        n_qubits = clusters[0].n_qubits
        rho = np.stack([c.rho for c in clusters]).astype(EVOLVE_DTYPE)
        H = build_group_H(clusters, inputs_by_name)
        r_minus, r_plus, dt = gather_group_rates(clusters, inputs_by_name)
        dt = dt * float(dt_scale)
        rho_new = backend.rk4_step(rho, H, r_minus, r_plus, dt, n_qubits)
        for b, c in enumerate(clusters):
            c.rho = rho_new[b]


def free_run_groups(clusters_by_dim, inputs_by_name, n_steps, dt_scale=1.0, backend=None):
    """Free-run several same-dim groups n_steps under their OWN fixed dynamics, in place.

    The decoupled engine for forecast/dream: gather each group's (ρ, H, rates) ONCE, then advance
    n_steps with the operators held constant (no fresh evidence between samples) via the backend's
    fused free_run. With JaxBackend each group's whole N-step run is ONE XLA kernel; with the dispatch
    backends it runs in the BPU-native number system. Writes the evolved ρ back onto each cluster.

    Returns the number of (B,d,d) kernel invocations issued — one per non-empty dim group, regardless
    of n_steps — which is the dispatch-count the GIL pays (vs n_steps×groups for the per-tick loop).
    """
    backend = backend or _DEFAULT_BACKEND
    n_steps = int(n_steps)
    kernels = 0
    for dim, clusters in clusters_by_dim.items():
        if not clusters or n_steps <= 0:
            continue
        n_qubits = clusters[0].n_qubits
        rho = np.stack([c.rho for c in clusters]).astype(EVOLVE_DTYPE)
        H = build_group_H(clusters, inputs_by_name)
        r_minus, r_plus, dt = gather_group_rates(clusters, inputs_by_name)
        dt = dt * float(dt_scale)
        rho_new = backend.free_run(rho, H, r_minus, r_plus, dt, n_qubits, n_steps)
        for b, c in enumerate(clusters):
            c.rho = rho_new[b]
        kernels += 1
    return kernels
