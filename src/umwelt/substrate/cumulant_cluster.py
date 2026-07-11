"""Cumulant cluster — a many-qubit cluster without the 2^n wall.

A QubitCluster (cluster.py) stores the full 2^n x 2^n density matrix and pays
O(8^n) per RK4 step, capping practical clusters at ~5 qubits. But the brain only
ever READS 1-body <σ_a^i> and 2-body connected <σ_a^i σ_b^j> (fractal level<=2),
and the generator is never more than 2-local (H = fields + ZZ; dissipators
σ_∓). So a cluster can track ONLY those expectations — O(n^2) numbers — and
evolve them under the same Lindblad equation, closing the 3-body cumulant ~0.

Cost: O(8^n) -> O(n^2). A 12-qubit cluster (a full 2^12=4096-dim density matrix,
16M complex entries, impossible) becomes ~66 pairs of 3x3 correlation blocks.
This is what puts large ENVIRONMENT and HUMAN-BIOMETRIC clusters within reach.

Accuracy: the 2-body closure is exact in the near-product / dissipation-dominated
regime our belief fields live in, and degrades only under strong COHERENT
multipartite entanglement (validated in experiments/cumulant_prototype.py —
closure error tracks log-negativity). Live adequacy is checked by the test rigs.

This module owns the exact Pauli-string EOM engine (no hand-derived commutators);
experiments/cumulant_prototype.py imports it for the full-density-matrix
validation harness.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import time
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from umwelt.spec.roles import role_input_mode

# axis codes: 1=x, 2=y, 3=z  (identity factors omitted from strings)
_AXES = (1, 2, 3)

# Topology → compiled dissipation channels, shared across same-topology clusters (population clones).
# Keyed (n_qubits, frozenset(dissipative-qubit indices)); the compiled channels are H-/param-independent.
# In-memory (per process) + a DISK cache so the (33s for the ~109-qubit MANIFOLD) compile is paid ONCE EVER,
# not on every boot/load/process: a cold boot compiles + persists; every later build loads the blob (~1s).
# Only EXPENSIVE compiles are persisted (small region clusters recompile in <1ms — not worth a file).
_CHANNEL_CACHE: dict = {}
_CHANNEL_FORMAT = "v1"          # bump when _compile_channels output structure changes → invalidates disk cache
_CHANNEL_DISK_MIN_S = 0.5       # only persist compiles slower than this (i.e. the big manifolds)


def _channel_cache_path(n_qubits: int, diss: frozenset) -> str:
    base = os.environ.get("UMWELT_CUMULANT_CACHE_DIR") or os.path.join("var", "cache", "cumulant")
    sig = f"{_CHANNEL_FORMAT}|n{n_qubits}|d{','.join(map(str, sorted(diss)))}"
    h = hashlib.sha1(sig.encode()).hexdigest()[:16]
    return os.path.join(base, f"chan_{_CHANNEL_FORMAT}_n{n_qubits}_{h}.pkl")

# ───────────────────────── Pauli-string algebra ─────────────────────────
# operator = dict {frozenset of (qubit, axis) : complex coeff}.  σ_aσ_b = δ_ab I + iε_abc σ_c
_EPS = {(1, 2): (1j, 3), (2, 1): (-1j, 3),
        (2, 3): (1j, 1), (3, 2): (-1j, 1),
        (3, 1): (1j, 2), (1, 3): (-1j, 2)}


def _pauli_mul_axis(a: int, b: int):
    return (1.0 + 0j, 0) if a == b else _EPS[(a, b)]


def op_mul(O1: dict, O2: dict) -> dict:
    out: dict = {}
    for s1, c1 in O1.items():
        d1 = dict(s1)
        for s2, c2 in O2.items():
            d2 = dict(s2)
            coeff = c1 * c2
            new = []
            for q in set(d1) | set(d2):
                a, b = d1.get(q), d2.get(q)
                if a and b:
                    cc, ax = _pauli_mul_axis(a, b)
                    coeff *= cc
                    if ax:
                        new.append((q, ax))
                else:
                    new.append((q, a or b))
            key = frozenset(new)
            out[key] = out.get(key, 0j) + coeff
    return out


def op_add(*ops: dict) -> dict:
    out: dict = {}
    for O in ops:
        for s, c in O.items():
            out[s] = out.get(s, 0j) + c
    # Prune exact/near-zero coeffs — disjoint commuting terms cancel in a
    # commutator and would otherwise leave 0-coeff high-body ghost strings.
    return {s: c for s, c in out.items() if abs(c) > 1e-14}


def op_scale(O: dict, k: complex) -> dict:
    return {s: c * k for s, c in O.items()}


def commutator(A: dict, B: dict) -> dict:
    return op_add(op_mul(A, B), op_scale(op_mul(B, A), -1.0))


# σ_- = |0><1| = (σx+iσy)/2 (the amplitude-damping jump op, matches cluster.py),
# σ_+ = |1><0| = (σx−iσy)/2.
def sigma_minus(q):
    return {frozenset([(q, 1)]): 0.5, frozenset([(q, 2)]): 0.5j}


def sigma_plus(q):
    return {frozenset([(q, 1)]): 0.5, frozenset([(q, 2)]): -0.5j}


def dissipator_adjoint(O: dict, channels) -> dict:
    """Σ_k γ_k ( L† O L − ½{L†L, O} ), Heisenberg adjoint. channels: (γ, L, L†)."""
    out: dict = {}
    for gamma, L, Ldag in channels:
        if gamma <= 0:
            continue
        LdOL = op_mul(op_mul(Ldag, O), L)
        LdL = op_mul(Ldag, L)
        anti = op_add(op_mul(LdL, O), op_mul(O, LdL))
        out = op_add(out, op_scale(op_add(LdOL, op_scale(anti, -0.5)), gamma))
    return out


# ───────────────────────── the cumulant state ─────────────────────────
# Stored as flat arrays for speed:
#   e1: (n, 3)            <σ_a^i>
#   e2: (n, n, 3, 3)      <σ_a^i σ_b^j>   (symmetric: e2[i,j,a,b]==e2[j,i,b,a])
# Only i<j is independent; we mirror for O(1) lookup in the closure.

def _e3_closure(e1, e2, i, a, j, b, l, c):
    """<σ_a^i σ_b^j σ_c^l> ≈ 2-body Wick closure (3-body cumulant = 0)."""
    return (e2[i, j, a, b] * e1[l, c]
            + e2[i, l, a, c] * e1[j, b]
            + e2[j, l, b, c] * e1[i, a]
            - 2.0 * e1[i, a] * e1[j, b] * e1[l, c])


def expect_string(items, e1, e2) -> float:
    """Expectation of a Pauli string [(qubit, axis), ...] under 2-body closure.
    axis 1/2/3 -> index 0/1/2."""
    k = len(items)
    if k == 0:
        return 1.0
    if k == 1:
        (q, a) = items[0]
        return e1[q, a - 1]
    if k == 2:
        (i, a), (j, b) = items
        return e2[i, j, a - 1, b - 1]
    if k == 3:
        (i, a), (j, b), (l, c) = items
        return _e3_closure(e1, e2, i, a - 1, j, b - 1, l, c - 1)
    raise ValueError(f"{k}-body string — a <=2-local generator cannot produce this")


def compile_op(O) -> list:
    """Freeze an operator dict into [(real_coeff, sorted_string_tuple), ...].
    The RHS operators are Hermitian (i[H,O] + D†[O]), so coeffs are real; we
    presort the strings once so the per-step hot loop never re-sorts."""
    out = []
    for s, c in O.items():
        if abs(c) <= 1e-14:
            continue
        out.append((c.real if isinstance(c, complex) else float(c),
                    tuple(sorted(s))))
    return out


def op_expect(compiled, e1, e2) -> float:
    tot = 0.0
    for c, items in compiled:
        tot += c * expect_string(items, e1, e2)
    return tot


class _EvolverShim:
    """Adapter so field/stack code that reaches `cluster.evolver.{gamma,dt,...}`
    keeps working against a CumulantCluster (which has no DensityMatrixEvolver).
    gamma/dt/gamma_diss are per-step scalars in the cumulant EOM, so writes need
    no recompile. H_base is materialized lazily (diagnostic / small-cluster)."""

    def __init__(self, cluster):
        object.__setattr__(self, "_c", cluster)

    @property
    def gamma(self):
        return self._c.gamma

    @gamma.setter
    def gamma(self, v):
        self._c.gamma = float(v)

    @property
    def dt(self):
        return self._c.dt

    @dt.setter
    def dt(self, v):
        self._c.dt = float(v)

    @property
    def _gamma_diss_per_qubit(self):
        return self._c._gamma_diss_per_qubit

    @_gamma_diss_per_qubit.setter
    def _gamma_diss_per_qubit(self, v):
        self._c._gamma_diss_per_qubit = v

    @property
    def _gamma_diss_default(self):
        return self._c._gamma_diss_default

    @_gamma_diss_default.setter
    def _gamma_diss_default(self, v):
        self._c._gamma_diss_default = v

    @property
    def H_base(self):
        return self._c._dense_H()

    @H_base.setter
    def H_base(self, H):
        self._c.set_hamiltonian(H)


class CumulantCluster:
    """A cluster carried by its 1- and 2-body cumulants instead of a 2^n ρ.

    Drop-in on the READ interface the field/readout use: n_qubits, dim,
    role_index, qubit_roles, qubit_rdm, qubit_bloch, role_bloch, features,
    step, observe_qubit, reset, rho-free.
    """

    is_cumulant = True   # the backend duck-flag the field/reservoir branch on

    def __init__(self, zone_name, qubit_roles, gamma=0.05, dt=0.01,
                 gamma_diss=5.0, role_modes=None, connectivity=None,
                 max_feature_level=2):
        self.zone_name = zone_name
        self.qubit_roles = list(qubit_roles)
        self.n_qubits = len(qubit_roles)
        self.dim = 2 ** self.n_qubits          # nominal only — never allocated
        self.role_index = {r: i for i, r in enumerate(qubit_roles)}
        self.max_feature_level = max(1, int(max_feature_level))

        if role_modes is None:
            role_modes = {r: role_input_mode(r) for r in qubit_roles}
        self._diss = {i for i, r in enumerate(qubit_roles)
                      if role_modes.get(r, role_input_mode(r)) == "dissipative"}
        self._unit = [i for i in range(self.n_qubits) if i not in self._diss]

        self.gamma = float(gamma)              # amplitude damping (unitary qubits)
        self.dt = float(dt)
        self._gamma_diss_per_qubit: dict[int, float] = {}
        self._gamma_diss_default = (gamma_diss.get("_default", 5.0)
                                    if isinstance(gamma_diss, dict) else float(gamma_diss))
        if isinstance(gamma_diss, dict):
            for i, r in enumerate(qubit_roles):
                if i in self._diss and r in gamma_diss:
                    self._gamma_diss_per_qubit[i] = gamma_diss[r]

        # Learned Hamiltonian, in cumulant-native form: 1-local fields h[i] = (hx,hy,hz)
        # and 2-local ZZ couplings J[(i,j)]. Start at zero (all drive from data).
        # connectivity (None=dense all-pairs / "chain" / an explicit edge set of
        # (i,j) or (role_a,role_b) pairs) restricts WHICH pairs couple — the sparse-
        # graph lever that keeps a big merged cluster (world manifold: region adjacency,
        # not all-pairs) cheap. resolve_zz_pairs gives the same pairs the H-tower basis
        # uses, so the two stay consistent.
        from umwelt.substrate.hamiltonian import resolve_zz_pairs
        self.connectivity = connectivity
        self._h = np.zeros((self.n_qubits, 3))
        self._zz_pairs = resolve_zz_pairs(connectivity, self.n_qubits, self.qubit_roles)
        self._zz = {p: 0.0 for p in self._zz_pairs}
        # 2-local EXCHANGE couplings (kxx·σxσx + kyy·σyσy) per pair. ZZ conserves populations
        # (commutes with σz) so it can't propagate a z-collapse between beliefs; exchange transfers
        # z↔z directly (no transverse field needed), which is what makes a collapse FORK the forecast.
        # Stored as (kxx, kyy): exchange=(J,J) follows, antiexchange=(J,−J) anti-follows. Starts at zero
        # → _H_op/_dense_H/norm are byte-identical to today until the dream coupling-learner sets one.
        self._xy = {p: (0.0, 0.0) for p in self._zz_pairs}

        # cumulant state
        n = self.n_qubits
        self.e1 = np.zeros((n, 3))             # ground state |0...0> -> <σz>=+1
        self.e1[:, 2] = 1.0
        self.e2 = np.zeros((n, n, 3, 3))
        self._sync_e2_product()                # start uncorrelated

        self.evolver = _EvolverShim(self)      # field/stack reach .evolver.{gamma,dt,H_base}
        self._compile_channels()               # H-independent, once
        self._compile_constant()               # unitary i[H,O] + vectorized arrays

    # ---- state helpers ----
    def _sync_e2_product(self):
        """Set e2 = e1 ⊗ e1 (uncorrelated) for every pair — used at init/reset."""
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                if i == j:
                    continue
                self.e2[i, j] = np.outer(self.e1[i], self.e1[j])

    def reset(self):
        self.e1[:] = 0.0
        self.e1[:, 2] = 1.0
        self._sync_e2_product()

    # ---- couplings (learned H) ----
    def set_couplings(self, h_fields=None, zz=None, xy=None):
        if h_fields is not None:
            self._h = np.asarray(h_fields, float).reshape(self.n_qubits, 3)
        if zz is not None:
            for p, J in zz.items():
                key = p if p in self._zz else (p[1], p[0])
                if key in self._zz:
                    self._zz[key] = float(J)
        if xy is not None:
            # xy[(i,j)] = scalar J (→ exchange (J,J)) or a (kxx, kyy) pair.
            for p, k in xy.items():
                key = p if p in self._xy else (p[1], p[0])
                if key in self._xy:
                    self._xy[key] = (float(k), float(k)) if np.isscalar(k) else (float(k[0]), float(k[1]))
        self._compile_constant()

    def set_hamiltonian(self, H):
        """Compat shim: decompose a dense Hermitian H (built from exactly the
        X/Y/Z/ZZ basis, incl. cross-cluster mean-field bridge terms) back into
        (h_fields, zz) by Pauli traces — EXACT. Used by the existing projection
        path (fractal_stack/field.apply_hamiltonian) for small clusters; large
        clusters should be fed via set_couplings directly (no dense matrix)."""
        from umwelt.substrate.density_matrix import pauli_x, pauli_y, pauli_z
        H = np.asarray(H)
        n = self.n_qubits
        dim = 2 ** n
        h = np.zeros((n, 3))
        for i in range(n):
            h[i, 0] = np.real(np.trace(H @ pauli_x(i, n))) / dim
            h[i, 1] = np.real(np.trace(H @ pauli_y(i, n))) / dim
            h[i, 2] = np.real(np.trace(H @ pauli_z(i, n))) / dim
        zz = {(i, j): np.real(np.trace(H @ (pauli_z(i, n) @ pauli_z(j, n)))) / dim
              for (i, j) in self._zz}
        self.set_couplings(h_fields=h, zz=zz)

    def sync_gamma_diss(self, bundle) -> None:
        """Live-read per-role gamma_diss from a ParameterBundle (mirror
        QubitCluster.sync_gamma_diss). Per-step scalar — no recompile."""
        default = bundle.get("gamma_diss", self._gamma_diss_default)
        self._gamma_diss_default = default
        for q in self._diss:
            key = f"gamma_diss_{self.qubit_roles[q]}"
            self._gamma_diss_per_qubit[q] = bundle.get(key, default)

    def clamp_physical(self) -> None:
        """Clamp each qubit's Bloch radius to ≤1 (1-RDM physicality) — the cheap
        cumulant analog of _enforce_physicality's eigenvalue clamp."""
        for i in range(self.n_qubits):
            r = float(np.linalg.norm(self.e1[i]))
            if r > 1.0:
                self.e1[i] /= r

    # ───────────────────── EOM (precompiled) ─────────────────────
    # The generator splits into a CONSTANT part (learned H + amplitude damping +
    # thermal channel STRUCTURE) and per-step SCALED parts (σ_x input kicks, and
    # thermal rates that depend on the sensor target). We precompile the exact
    # RHS operator G[O] for every basis observable O once; each step only
    # re-evaluates expectations (fast) and scales the input-dependent pieces.

    def _basis(self):
        """The tracked observables, in fractal.decompose_by_level order."""
        singles = [(q, a) for q in range(self.n_qubits) for a in _AXES]
        pairs = [((i, a), (j, b)) for (i, j) in self._zz_pairs_all()
                 for a in _AXES for b in _AXES]
        return singles, pairs

    def _zz_pairs_all(self):
        # all i<j pairs are tracked in e2 regardless of ZZ connectivity (we read
        # all pairwise correlations); ZZ connectivity only gates which pairs the
        # Hamiltonian couples.
        return list(combinations(range(self.n_qubits), 2))

    def _H_op(self):
        H: dict = {}
        for i in range(self.n_qubits):
            for a, h in zip(_AXES, self._h[i]):
                if h:
                    H = op_add(H, {frozenset([(i, a)]): complex(h)})
        for (i, j), J in self._zz.items():
            if J:
                H = op_add(H, {frozenset([(i, 3), (j, 3)]): complex(J)})
        for (i, j), (kxx, kyy) in self._xy.items():    # exchange: kxx·σxσx + kyy·σyσy (axes 1=x,2=y)
            if kxx:
                H = op_add(H, {frozenset([(i, 1), (j, 1)]): complex(kxx)})
            if kyy:
                H = op_add(H, {frozenset([(i, 2), (j, 2)]): complex(kyy)})
        return H

    def hamiltonian_norm(self) -> float:
        """Frobenius norm ‖H‖_F of the learned Hamiltonian, computed from the SPARSE
        couplings without materializing the 2ⁿ matrix. H = Σ h[i,a]·σ_a(i) + Σ J·σz σz is a
        sum of distinct (trace-orthogonal) Pauli strings, so Tr(H†H) = 2ⁿ·Σ(coeff²) exactly —
        the same number np.linalg.norm(dense_H) would give, but O(n²) and OOM-proof. This is
        what diagnostics (fractal_stack.stats) read on the big merged cumulant manifold cluster."""
        coeff_sq = (float(np.sum(self._h ** 2)) + float(sum(J * J for J in self._zz.values()))
                    + float(sum(kxx * kxx + kyy * kyy for kxx, kyy in self._xy.values())))
        return float(np.sqrt((2.0 ** self.n_qubits) * coeff_sq))

    # cap above which a dense 2ⁿ×2ⁿ H is refused (would OOM); the origin's merged manifold was 26 qubits.
    _DENSE_H_MAX_QUBITS = 16

    def _dense_H(self) -> NDArray:
        """Materialize the dense H matrix from the sparse couplings — diagnostic /
        small-cluster only (the .evolver.H_base shim). Refuses big clusters (use
        hamiltonian_norm() for the norm, or set_couplings() to feed H sparsely)."""
        from umwelt.substrate.density_matrix import pauli_x, pauli_y, pauli_z
        n = self.n_qubits
        if n > self._DENSE_H_MAX_QUBITS:
            raise ValueError(
                f"_dense_H refused for {n}-qubit cumulant cluster '{self.zone_name}' "
                f"(2^{n} matrix would OOM). Use hamiltonian_norm() for the norm or work "
                f"from the sparse (h_fields, zz) couplings directly.")
        H = np.zeros((2 ** n, 2 ** n), dtype=complex)
        for i in range(n):
            hx, hy, hz = self._h[i]
            if hx:
                H += hx * pauli_x(i, n)
            if hy:
                H += hy * pauli_y(i, n)
            if hz:
                H += hz * pauli_z(i, n)
        for (i, j), J in self._zz.items():
            if J:
                H += J * (pauli_z(i, n) @ pauli_z(j, n))
        for (i, j), (kxx, kyy) in self._xy.items():
            if kxx:
                H += kxx * (pauli_x(i, n) @ pauli_x(j, n))
            if kyy:
                H += kyy * (pauli_y(i, n) @ pauli_y(j, n))
        return H

    @staticmethod
    def _support(O):
        """The qubit indices an observable touches: (q,) for single, (i,j) for pair."""
        if isinstance(O[0], tuple):
            return (O[0][0], O[1][0])
        return (O[0],)

    def _compile_channels(self):
        """Precompile the DISSIPATION channels (amplitude damping σ_- on unitary
        qubits, σ_x input kicks, thermal σ_∓ on dissipative qubits). These depend
        ONLY on the qubit structure, NOT on the couplings (H) — so this runs ONCE
        at construction, never on a coupling change. A channel on qubit q is EXACTLY
        zero for any observable O that q isn't part of (disjoint dissipators leave
        ⟨O⟩ unchanged), so we loop only O's SUPPORT qubits, not all n — the O(n³)→
        O(n²) win that was the boot bottleneck for big clusters.

        TOPOLOGY CACHE: the compiled channels are a pure function of the topology —
        n_qubits, the dissipative-qubit set (unit = its complement), and the observable
        basis (singles + all i<j pairs, themselves a pure function of n). They carry NO
        learned/per-individual state (γ rates scale them at step time; H goes through the
        separate _compile_unitary). So the genetic POPULATION's clones — same topology,
        different params — share ONE compilation instead of each paying the (O(n²)
        observables × Pauli-string op_mul) cost. For the MANIFOLD's ~109-qubit monolith
        that turned an N-individual boot from a multi-minute hang into one compile; the
        shared lists are read-only after construction (only _compile_vectorized reads
        them, into per-individual arrays)."""
        singles, pairs = self._basis()
        self._obs = singles + pairs
        key = (self.n_qubits, frozenset(self._diss))   # _basis pairs depend only on n_qubits
        cached = _CHANNEL_CACHE.get(key)
        if cached is not None:
            self._damp, self._xkick, self._tdown, self._tup = cached
            return
        # disk cache — a prior process already paid the (33s manifold) compile; load the blob (~1s)
        disk = _channel_cache_path(*key)
        if os.path.exists(disk):
            try:
                with open(disk, "rb") as fh:
                    cached = pickle.load(fh)
                _CHANNEL_CACHE[key] = cached
                self._damp, self._xkick, self._tdown, self._tup = cached
                return
            except Exception:
                pass        # corrupt / format-mismatched cache → just recompile below
        t0 = time.perf_counter()
        unit_set = set(self._unit)
        self._damp, self._xkick, self._tdown, self._tup = [], [], [], []
        for O in self._obs:
            Oop = self._as_op(O)
            dmp, xk, td, tu = {}, {}, {}, {}
            for q in self._support(O):
                if q in unit_set:
                    d = dissipator_adjoint(Oop, [(1.0, sigma_minus(q), sigma_plus(q))])
                    if d:
                        dmp[q] = compile_op(d)
                    k = commutator(op_scale({frozenset([(q, 1)]): 1.0 + 0j}, 1j), Oop)
                    if k:
                        xk[q] = compile_op(k)
                elif q in self._diss:
                    d = dissipator_adjoint(Oop, [(1.0, sigma_minus(q), sigma_plus(q))])
                    u = dissipator_adjoint(Oop, [(1.0, sigma_plus(q), sigma_minus(q))])
                    if d:
                        td[q] = compile_op(d)
                    if u:
                        tu[q] = compile_op(u)
            self._damp.append(dmp)
            self._xkick.append(xk)
            self._tdown.append(td)
            self._tup.append(tu)
        # share this topology's compiled channels with every later same-topology cluster
        # (population clones) — read-only from here on.
        _CHANNEL_CACHE[key] = (self._damp, self._xkick, self._tdown, self._tup)
        # persist an EXPENSIVE compile so the next process/boot loads it instead of recompiling (best-effort)
        if time.perf_counter() - t0 > _CHANNEL_DISK_MIN_S:
            try:
                os.makedirs(os.path.dirname(disk), exist_ok=True)
                tmp = disk + f".tmp{os.getpid()}"
                with open(tmp, "wb") as fh:
                    pickle.dump(_CHANNEL_CACHE[key], fh, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, disk)          # atomic — concurrent forebrain/hindbrain boots stay safe
            except Exception:
                pass

    def _compile_unitary(self):
        """Precompile the unitary generator i[H,O] per observable — the ONLY part
        that changes on a coupling update (and the pure-H forecast_z generator).
        [P_k, O] = 0 unless H-term P_k shares a qubit with O, so we index H by qubit
        and commute O against ONLY the overlapping terms (O(n⁴)→O(n³) for dense H)."""
        H = self._H_op()
        by_qubit: dict[int, dict] = {}
        for s, c in H.items():
            for (q, _a) in s:
                by_qubit.setdefault(q, {})[s] = c
        self._unitary = []
        for O in self._obs:
            local: dict = {}
            for q in self._support(O):
                local.update(by_qubit.get(q, {}))
            if local:
                comm = commutator(op_scale(local, 1j), self._as_op(O))
                self._unitary.append(compile_op(comm))
            else:
                self._unitary.append([])

    def _compile_constant(self):
        """Recompile on a coupling (H) change: just the unitary generator + the
        flat vectorized arrays. The channels are H-independent (compiled once at
        construction by _compile_channels)."""
        self._compile_unitary()
        self._compile_vectorized()

    def _compile_vectorized(self):
        """Flatten the entire precompiled generator into monomial term arrays so a
        deriv is a handful of numpy gather/scatter ops instead of a Python loop over
        observables×strings×closure. Each compiled string becomes a monomial in
        (e1,e2) of degree ≤3 (the closure expands 3-body → e2·e1 + e1³), tagged with
        a per-step SCALAR index into S = [1, γ_q…, u_q…, rdown_q…, rup_q…]. Rebuilt
        with _compile_constant (on coupling change). Bit-identical to _deriv (gated)."""
        n = self.n_qubits
        lin = ([], [], [], [])          # k, scalar_idx, e1_flat, coeff
        q2 = ([], [], [], [])           # k, s, e2_flat, coeff      (2-body)
        qxl = ([], [], [], [], [])      # k, s, e2_flat, e1_flat, coeff  (e2·e1 closure)
        cub = ([], [], [], [], [], [])  # k, s, e1_i, e1_j, e1_l, coeff
        con = ([], [], [])              # k, s, coeff (identity terms)

        def e1f(q, a):
            return q * 3 + a            # a 0-based
        def e2f(i, j, a, b):
            return ((i * n + j) * 3 + a) * 3 + b

        def add(k, s, compiled):
            for c, items in compiled:
                m = len(items)
                if m == 0:
                    con[0].append(k); con[1].append(s); con[2].append(c)
                elif m == 1:
                    (q, a) = items[0]
                    lin[0].append(k); lin[1].append(s); lin[2].append(e1f(q, a - 1)); lin[3].append(c)
                elif m == 2:
                    (i, a), (j, b) = items
                    q2[0].append(k); q2[1].append(s); q2[2].append(e2f(i, j, a - 1, b - 1)); q2[3].append(c)
                elif m == 3:
                    (i, a), (j, b), (l, cc) = items
                    a -= 1; b -= 1; cc -= 1
                    for e2i, e1i in ((e2f(i, j, a, b), e1f(l, cc)),
                                     (e2f(i, l, a, cc), e1f(j, b)),
                                     (e2f(j, l, b, cc), e1f(i, a))):
                        qxl[0].append(k); qxl[1].append(s); qxl[2].append(e2i); qxl[3].append(e1i); qxl[4].append(c)
                    cub[0].append(k); cub[1].append(s)
                    cub[2].append(e1f(i, a)); cub[3].append(e1f(j, b)); cub[4].append(e1f(l, cc)); cub[5].append(-2.0 * c)

        for idx in range(len(self._obs)):
            add(idx, 0, self._unitary[idx])
            for q, op in self._damp[idx].items():
                add(idx, 1 + q, op)
            for q, op in self._xkick[idx].items():
                add(idx, 1 + n + q, op)
            for q, op in self._tdown[idx].items():
                add(idx, 1 + 2 * n + q, op)
            for q, op in self._tup[idx].items():
                add(idx, 1 + 3 * n + q, op)

        ia = lambda x: np.asarray(x, dtype=np.intp)
        fa = lambda x: np.asarray(x, dtype=float)
        self._vec = {
            "lin": (ia(lin[0]), ia(lin[1]), ia(lin[2]), fa(lin[3])),
            "q2": (ia(q2[0]), ia(q2[1]), ia(q2[2]), fa(q2[3])),
            "qxl": (ia(qxl[0]), ia(qxl[1]), ia(qxl[2]), ia(qxl[3]), fa(qxl[4])),
            "cub": (ia(cub[0]), ia(cub[1]), ia(cub[2]), ia(cub[3]), ia(cub[4]), fa(cub[5])),
            "con": (ia(con[0]), ia(con[1]), fa(con[2])),
        }
        # obs index → de1/de2 scatter targets (built once)
        sk, se1, pk, pij, pji = [], [], [], [], []
        for k, O in enumerate(self._obs):
            if isinstance(O[0], tuple):
                (i, a), (j, b) = O
                pk.append(k); pij.append(e2f(i, j, a - 1, b - 1)); pji.append(e2f(j, i, b - 1, a - 1))
            else:
                (q, a) = O
                sk.append(k); se1.append(e1f(q, a - 1))
        self._vec_scatter = (ia(sk), ia(se1), ia(pk), ia(pij), ia(pji))
        self._scalar = np.zeros(1 + 4 * n)
        self._scalar[0] = 1.0

    def _deriv_vec(self, e1, e2, S):
        """Vectorized deriv — numpy gather/scatter over the flattened monomials.
        S = per-step scalar vector [1, γ_q…, u_q…, rdown_q…, rup_q…]. For the
        pure-H forecast, pass S with only S[0]=1."""
        n = self.n_qubits
        e1f = e1.reshape(-1)
        e2f = e2.reshape(-1)
        vals = np.zeros(len(self._obs))
        V = self._vec
        k, s, c = V["con"]
        if k.size:
            np.add.at(vals, k, c * S[s])
        k, s, e, c = V["lin"]
        if k.size:
            np.add.at(vals, k, c * S[s] * e1f[e])
        k, s, e, c = V["q2"]
        if k.size:
            np.add.at(vals, k, c * S[s] * e2f[e])
        k, s, e2i, e1i, c = V["qxl"]
        if k.size:
            np.add.at(vals, k, c * S[s] * e2f[e2i] * e1f[e1i])
        k, s, i, j, l, c = V["cub"]
        if k.size:
            np.add.at(vals, k, c * S[s] * e1f[i] * e1f[j] * e1f[l])
        de1 = np.zeros((n, 3))
        de2 = np.zeros((n, n, 3, 3))
        sk, se1, pk, pij, pji = self._vec_scatter
        de1.reshape(-1)[se1] = vals[sk]
        d2 = de2.reshape(-1)
        d2[pij] = vals[pk]
        d2[pji] = vals[pk]
        return de1, de2

    def _scalar_vector(self, inputs, unitary_only=False):
        n = self.n_qubits
        S = self._scalar.copy()
        S[:] = 0.0
        S[0] = 1.0
        if unitary_only:
            return S
        for q in self._unit:
            S[1 + q] = self.gamma
            if inputs is not None and q < len(inputs):
                S[1 + n + q] = float(inputs[q])
        for q in self._diss:
            v = float(inputs[q]) if (inputs is not None and q < len(inputs)) else 0.0
            gd = self._gamma_diss_per_qubit.get(q, self._gamma_diss_default)
            if gd > 0:
                S[1 + 2 * n + q] = gd * (1.0 + v) / 2.0
                S[1 + 3 * n + q] = gd * (1.0 - v) / 2.0
        return S

    @staticmethod
    def _as_op(O):
        # O is (q,a) single or ((i,a),(j,b)) pair
        if isinstance(O[0], tuple):
            return {frozenset(O): 1.0 + 0j}
        return {frozenset([O]): 1.0 + 0j}

    def _deriv(self, e1, e2, gdamp, u, rdown, rup, unitary_only=False):
        """d e1/dt, d e2/dt at (e1,e2). gdamp[q]=amplitude-damping rate (γ) on
        unitary q; u[q]=σ_x input; rdown/rup[q]=thermal rates on dissipative q.
        unitary_only=True evaluates just i[H,·] (the pure-H forecast generator)."""
        n = self.n_qubits
        de1 = np.zeros((n, 3))
        de2 = np.zeros((n, n, 3, 3))
        for idx, O in enumerate(self._obs):
            val = op_expect(self._unitary[idx], e1, e2)
            if not unitary_only:
                for q, op in self._damp[idx].items():
                    if gdamp[q]:
                        val += gdamp[q] * op_expect(op, e1, e2)
                for q, op in self._xkick[idx].items():
                    if u[q]:
                        val += u[q] * op_expect(op, e1, e2)
                for q, op in self._tdown[idx].items():
                    if rdown[q]:
                        val += rdown[q] * op_expect(op, e1, e2)
                for q, op in self._tup[idx].items():
                    if rup[q]:
                        val += rup[q] * op_expect(op, e1, e2)
            if isinstance(O[0], tuple):
                (i, a), (j, b) = O
                de2[i, j, a - 1, b - 1] = val
                de2[j, i, b - 1, a - 1] = val
            else:
                (q, a) = O
                de1[q, a - 1] = val
        return de1, de2

    def step(self, inputs: NDArray | None = None, dt_scale: float = 1.0):
        """Evolve one timestep × dt_scale via cumulant RK4.
        inputs[i] drives qubit i: σ_x kick (unitary roles) or thermal target
        (dissipative roles), matching QubitCluster.step semantics."""
        S = self._scalar_vector(inputs)
        dt = self.dt * dt_scale
        e1, e2 = self.e1, self.e2
        d1a, d2a = self._deriv_vec(e1, e2, S)
        d1b, d2b = self._deriv_vec(e1 + 0.5 * dt * d1a, e2 + 0.5 * dt * d2a, S)
        d1c, d2c = self._deriv_vec(e1 + 0.5 * dt * d1b, e2 + 0.5 * dt * d2b, S)
        d1d, d2d = self._deriv_vec(e1 + dt * d1c, e2 + dt * d2c, S)
        self.e1 = e1 + (dt / 6.0) * (d1a + 2 * d1b + 2 * d1c + d1d)
        self.e2 = e2 + (dt / 6.0) * (d2a + 2 * d2b + 2 * d2c + d2d)

    def free_run(self, n_steps: int, inputs: NDArray | None = None,
                 dt_scale: float = 1.0, backend: str = "numpy"):
        """Roll the cumulant forward n_steps with the inputs (S) held FIXED — the decoupled free-run
        engine for the FOLDED MANIFOLD (the one root-node cumulant). One call advances the whole manifold
        n_steps under its own dynamics (no fresh evidence between samples), value-identical to n_steps ×
        step() at fixed inputs. backend='numpy' loops the RK4 (reference + the RDK); backend='jax' fuses
        the entire n-step rollout into ONE XLA kernel (one GIL acquire / one dispatch — the BPU on-ramp,
        the cumulant analog of batched_evolve.JaxBackend). Mutates e1/e2 in place. See forecast_freerun."""
        S = self._scalar_vector(inputs)
        dt = self.dt * dt_scale
        n_steps = int(n_steps)
        if backend == "jax":
            e1, e2 = self._jax_free_run(self.e1, self.e2, S, dt, n_steps)
            self.e1, self.e2 = np.asarray(e1, float), np.asarray(e2, float)
            return
        e1, e2 = self.e1, self.e2
        for _ in range(n_steps):
            d1a, d2a = self._deriv_vec(e1, e2, S)
            d1b, d2b = self._deriv_vec(e1 + 0.5 * dt * d1a, e2 + 0.5 * dt * d2a, S)
            d1c, d2c = self._deriv_vec(e1 + 0.5 * dt * d1b, e2 + 0.5 * dt * d2b, S)
            d1d, d2d = self._deriv_vec(e1 + dt * d1c, e2 + dt * d2c, S)
            e1 = e1 + (dt / 6.0) * (d1a + 2 * d1b + 2 * d1c + d1d)
            e2 = e2 + (dt / 6.0) * (d2a + 2 * d2b + 2 * d2c + d2d)
        self.e1, self.e2 = e1, e2

    def _jax_free_run(self, e1, e2, S, dt, n_steps):
        """XLA-fused n-step cumulant RK4 (lazy JAX import). The EOM is a fixed gather/scatter over the
        precomputed monomial tables (self._vec / self._vec_scatter) — static structure, so the whole
        n-step rollout compiles to one jit+fori_loop kernel, exactly the shape the BPU churns."""
        fn = getattr(self, "_jax_kernel_cache", None)
        if fn is None:
            fn = self._build_jax_kernel()
            self._jax_kernel_cache = fn
        import jax.numpy as jnp
        e1o, e2o = fn(jnp.asarray(e1), jnp.asarray(e2), jnp.asarray(S),
                      float(dt), int(n_steps))
        return e1o, e2o

    def _build_jax_kernel(self):
        import jax
        import jax.numpy as jnp
        n = self.n_qubits
        n_obs = len(self._obs)
        V = self._vec
        # static index/coeff tables → device constants (jnp)
        con_k, con_s, con_c = (jnp.asarray(a) for a in V["con"])
        lin_k, lin_s, lin_e, lin_c = (jnp.asarray(a) for a in V["lin"])
        q2_k, q2_s, q2_e, q2_c = (jnp.asarray(a) for a in V["q2"])
        qxl_k, qxl_s, qxl_e2, qxl_e1, qxl_c = (jnp.asarray(a) for a in V["qxl"])
        cub_k, cub_s, cub_i, cub_j, cub_l, cub_c = (jnp.asarray(a) for a in V["cub"])
        sk, se1, pk, pij, pji = (jnp.asarray(a) for a in self._vec_scatter)

        def deriv(e1f, e2f, S):
            vals = jnp.zeros(n_obs)
            if con_k.size:
                vals = vals.at[con_k].add(con_c * S[con_s])
            if lin_k.size:
                vals = vals.at[lin_k].add(lin_c * S[lin_s] * e1f[lin_e])
            if q2_k.size:
                vals = vals.at[q2_k].add(q2_c * S[q2_s] * e2f[q2_e])
            if qxl_k.size:
                vals = vals.at[qxl_k].add(qxl_c * S[qxl_s] * e2f[qxl_e2] * e1f[qxl_e1])
            if cub_k.size:
                vals = vals.at[cub_k].add(cub_c * S[cub_s] * e1f[cub_i] * e1f[cub_j] * e1f[cub_l])
            de1 = jnp.zeros(n * 3).at[se1].set(vals[sk])
            d2 = jnp.zeros((n * n * 9,))
            d2 = d2.at[pij].set(vals[pk]).at[pji].set(vals[pk])
            return de1.reshape(n, 3), d2.reshape(n, n, 3, 3)

        def rk4(e1, e2, S, dt):
            d1a, d2a = deriv(e1.reshape(-1), e2.reshape(-1), S)
            d1b, d2b = deriv((e1 + 0.5 * dt * d1a).reshape(-1), (e2 + 0.5 * dt * d2a).reshape(-1), S)
            d1c, d2c = deriv((e1 + 0.5 * dt * d1b).reshape(-1), (e2 + 0.5 * dt * d2b).reshape(-1), S)
            d1d, d2d = deriv((e1 + dt * d1c).reshape(-1), (e2 + dt * d2c).reshape(-1), S)
            return (e1 + (dt / 6.0) * (d1a + 2 * d1b + 2 * d1c + d1d),
                    e2 + (dt / 6.0) * (d2a + 2 * d2b + 2 * d2c + d2d))

        def run(e1, e2, S, dt, n_steps):
            body = lambda _i, c: rk4(c[0], c[1], S, dt)   # noqa: E731 — S,dt fixed in the loop
            return jax.lax.fori_loop(0, n_steps, body, (e1, e2))

        return jax.jit(run, static_argnums=())

    # ───────────────────── readout ─────────────────────
    def qubit_bloch(self, qubit_idx: int) -> NDArray:
        return self.e1[qubit_idx].copy()

    def role_bloch(self, role: str) -> NDArray:
        return self.qubit_bloch(self.role_index[role])

    def qubit_rdm(self, qubit_idx: int) -> NDArray:
        """2×2 reduced density matrix ρ_i = (I + r·σ)/2 — EXACT (1-RDM is tracked)."""
        x, y, z = self.e1[qubit_idx]
        return 0.5 * np.array([[1 + z, x - 1j * y], [x + 1j * y, 1 - z]], dtype=complex)

    def all_bloch(self) -> dict:
        """role → Bloch vector (dict, matching QubitCluster.all_bloch)."""
        return {role: self.e1[i].copy() for role, i in self.role_index.items()}

    def features_by_level(self, max_level=None) -> dict:
        """Same layout as fractal.decompose_by_level: {1:(n,3), 2:(pairs,9) connected}.
        Level 3 (triple cumulants) is intentionally EMPTY — the cumulant state
        closes 3-body ≈ 0, and the sparse readout caps at level 2 anyway; callers
        (fractal_signature, similarity) handle a missing/zero level-3 gracefully."""
        cap = self.max_feature_level if max_level is None else max_level
        out = {1: self.e1.copy()}
        if self.n_qubits >= 2 and cap >= 2:
            pairs = list(combinations(range(self.n_qubits), 2))
            lvl2 = np.zeros((len(pairs), 9))
            for idx, (i, j) in enumerate(pairs):
                lvl2[idx] = (self.e2[i, j] - np.outer(self.e1[i], self.e1[j])).reshape(-1)
            out[2] = lvl2
        if self.n_qubits >= 3 and cap >= 3:
            out[3] = np.zeros((len(list(combinations(range(self.n_qubits), 3))), 27))
        return out

    def features(self) -> NDArray:
        """Sparse fractal feature vector — byte-compatible with QubitCluster.features()
        (decompose_by_level layout flattened level-ascending)."""
        from umwelt.substrate.cluster import sparse_feature_vector
        return sparse_feature_vector(self.features_by_level(), self.max_feature_level)

    def forecast_z(self, horizon: int, dt: float | None = None,
                   e1: NDArray | None = None, e2: NDArray | None = None) -> NDArray:
        """Predict per-qubit ⟨σ_z⟩ `horizon` steps ahead under the LEARNED H ALONE
        (γ=0, no inputs, no dissipation) — the cumulant analog of the forecast
        probe's expm(-iH·dt·horizon)·ρ·U†. Evolves a COPY; does not mutate state.
        Optionally forecast from a GIVEN (e1,e2) start (the N-steps-ago rollout
        snapshot) instead of the current state. Returns array (n_qubits,) of z."""
        dt = self.dt if dt is None else dt
        e1 = self.e1.copy() if e1 is None else np.array(e1, dtype=float)
        e2 = self.e2.copy() if e2 is None else np.array(e2, dtype=float)
        S = self._scalar_vector(None, unitary_only=True)   # pure-H: S=[1,0,…]
        for _ in range(max(1, int(horizon))):
            d1a, d2a = self._deriv_vec(e1, e2, S)
            d1b, d2b = self._deriv_vec(e1 + 0.5 * dt * d1a, e2 + 0.5 * dt * d2a, S)
            d1c, d2c = self._deriv_vec(e1 + 0.5 * dt * d1b, e2 + 0.5 * dt * d2b, S)
            d1d, d2d = self._deriv_vec(e1 + dt * d1c, e2 + dt * d2c, S)
            e1 = e1 + (dt / 6.0) * (d1a + 2 * d1b + 2 * d1c + d1d)
            e2 = e2 + (dt / 6.0) * (d2a + 2 * d2b + 2 * d2c + d2d)
        return e1[:, 2].copy()

    @property
    def purity(self) -> float:
        """Exact single-qubit purities averaged — a cheap mixedness proxy (the full
        2^n purity isn't tracked; this is monotone with it for near-product states).
        The per-qubit form (1+|r|²)/2 is the atlas's qubit_purity over the e1 rows."""
        from umwelt.substrate.bloch import qubit_purity
        return float(np.mean(qubit_purity(self.e1)))

    @property
    def entropy(self) -> float:
        """Mean per-qubit von-Neumann entropy from each Bloch radius (diagnostic;
        the full 2^n entropy isn't tracked). λ = (1±r_i)/2."""
        tot = 0.0
        for i in range(self.n_qubits):
            r = min(float(np.linalg.norm(self.e1[i])), 1.0)
            for lam in ((1.0 + r) / 2.0, (1.0 - r) / 2.0):
                if lam > 1e-15:
                    tot -= lam * np.log2(lam)
        return tot / self.n_qubits if self.n_qubits else 0.0

    # ───────────────────── persistence ─────────────────────
    def snapshot(self) -> dict:
        """Serializable state for the brain pickle — the cumulants + learned H."""
        return {
            "kind": "cumulant",
            "zone_name": self.zone_name,
            "qubit_roles": list(self.qubit_roles),
            "e1": self.e1.tolist(),
            "e2": self.e2.tolist(),
            "h": self._h.tolist(),
            "zz": {f"{i},{j}": J for (i, j), J in self._zz.items()},
            # only persist non-zero exchange couplings (keeps existing pickles byte-identical until learned)
            "xy": {f"{i},{j}": [kxx, kyy] for (i, j), (kxx, kyy) in self._xy.items() if kxx or kyy},
        }

    def load(self, state: dict) -> bool:
        """Restore from snapshot(). Resets readout-incompatibly only if the role
        geometry changed (the caller should rebuild then); returns True on load."""
        if list(state.get("qubit_roles", [])) != self.qubit_roles:
            return False
        self.e1 = np.asarray(state["e1"], float).reshape(self.n_qubits, 3)
        self.e2 = np.asarray(state["e2"], float).reshape(
            self.n_qubits, self.n_qubits, 3, 3)
        self._h = np.asarray(state["h"], float).reshape(self.n_qubits, 3)
        for key, J in state.get("zz", {}).items():
            i, j = (int(x) for x in key.split(","))
            if (i, j) in self._zz:
                self._zz[(i, j)] = float(J)
        for key, k in state.get("xy", {}).items():     # absent in pre-exchange pickles → stays zero
            i, j = (int(x) for x in key.split(","))
            if (i, j) in self._xy:
                self._xy[(i, j)] = (float(k[0]), float(k[1]))
        self._compile_constant()
        return True

    # ───────────────────── observation (sensor injection) ─────────────────────
    def nudge_toward_rdm(self, qubit_idx, target_rdm, alpha):
        """Bridge/projection fiber connection (SubstrateBackend contract). There is no
        dense ρ to add a kron correction to; the marginal nudge toward a target
        single-qubit RDM IS observe_qubit — convert the 2×2 target → Bloch (x,y,z) and
        inject. (Moved verbatim from field._nudge_qubit's cumulant branch.)"""
        x = 2.0 * float(np.real(target_rdm[0, 1]))
        y = 2.0 * float(np.imag(target_rdm[1, 0]))
        z = float(np.real(target_rdm[0, 0] - target_rdm[1, 1]))
        self.observe_qubit(qubit_idx, (x, y, z), alpha=alpha)

    def observe_qubit(self, qubit_idx, target_bloch, alpha=0.5, confidence=None):
        """Partial collapse qubit i toward target_bloch with strength α.
        Models ρ → (1-α)ρ + α (ρ_target^i ⊗ ρ_rest): the qubit's marginal moves to
        the target and its correlations with every other qubit shrink by (1-α) —
        observing a qubit decouples it, the cumulant-consistent collapse.

        `confidence` is RECORDED as a read-only gauge quantity only (the caller has
        already folded it into `alpha` — the same contract as QubitCluster; the old
        internal α×conf fold here double-applied confidence on the flagship path)."""
        if confidence is not None:
            if not hasattr(self, "_obs_confidence"):
                self._obs_confidence: dict[int, float] = {}
            self._obs_confidence[qubit_idx] = float(confidence)
        a = alpha
        if a <= 0:
            return
        a = min(1.0, a)
        i = qubit_idx
        t = np.asarray(target_bloch, float)
        old_i = self.e1[i].copy()
        self.e1[i] = (1 - a) * old_i + a * t
        # correlations with i: <σ^i σ^j> -> (1-α)<σ^i σ^j> + α t_i <σ^j>
        for j in range(self.n_qubits):
            if j == i:
                continue
            self.e2[i, j] = (1 - a) * self.e2[i, j] + a * np.outer(t, self.e1[j])
            self.e2[j, i] = self.e2[i, j].T

    def measure_qubit(self, qubit_idx, record_z, strength, confidence=None):
        """Belavkin weak σ_z measurement of one qubit (docs/QUANTUM_KALMAN.md, rung L4).

        The principled sibling of observe_qubit: conditioned Kraus update with the
        bounded Wonham gain, coherence back-action, AND the second-cumulant
        cross-update — the mean of every correlated peer moves through the
        regression gain cov_zz/var, which observe_qubit's decorrelate-only blend
        never did. `strength` ≈ the old collapse_alpha (equator-matched, caller
        pre-folds confidence, same convention as observe_qubit); `confidence` is
        recorded as the gauge quantity only. strength ≤ 0 is the exact no-op.
        Returns the applied Δz on the measured qubit."""
        from umwelt.substrate.belavkin import measure_cumulant
        if confidence is not None:
            if not hasattr(self, "_obs_confidence"):
                self._obs_confidence = {}
            self._obs_confidence[qubit_idx] = float(confidence)
        return measure_cumulant(self.e1, self.e2, qubit_idx,
                                float(record_z), float(strength))
