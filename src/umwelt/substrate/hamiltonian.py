"""
Learnable Hamiltonian — sparse basis + coefficient fiber.

Graduates the field from H=0 (passive reservoir) to a learned
Hamiltonian whose autonomous dynamics predict the world.

The basis is physically motivated:
    Z_i     — natural frequency of role i (energy splitting)
    X_i     — spontaneous transition rate of role i (tunneling)
    ZZ_i_j  — pairwise correlation between roles (Ising coupling)

All basis operators are Hermitian, so H = Σ c_k O_k is Hermitian
for real coefficients c_k. Coefficients live in a ParameterBundle
and get Kalman updates + Berry phase tracking for free.
"""
from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from umwelt.substrate.density_matrix import ComplexMatrix, pauli_x, pauli_y, pauli_z
from umwelt.substrate.density_matrix import EVOLVE_DTYPE

logger = logging.getLogger(__name__)


def _two_qubit_op(
    op_a: ComplexMatrix,
    qubit_a: int,
    op_b: ComplexMatrix,
    qubit_b: int,
    n_qubits: int,
) -> ComplexMatrix:
    """Tensor product with two non-identity factors: op_a on qubit_a, op_b on qubit_b."""
    I2 = np.eye(2, dtype=EVOLVE_DTYPE)
    result = np.array([[1]], dtype=EVOLVE_DTYPE)
    for q in range(n_qubits):
        if q == qubit_a:
            result = np.kron(result, op_a)
        elif q == qubit_b:
            result = np.kron(result, op_b)
        else:
            result = np.kron(result, I2)
    return result


# 2×2 Pauli matrices (reused for basis construction)
_SX = np.array([[0, 1], [1, 0]], dtype=EVOLVE_DTYPE)
_SZ = np.array([[1, 0], [0, -1]], dtype=EVOLVE_DTYPE)


def resolve_zz_pairs(connectivity, n_qubits: int, roles: list[str]):
    """Which (i,j) qubit pairs get a ZZ coupling term.

    connectivity=None  → DENSE all-pairs (default, unchanged) — n(n-1)/2 terms.
    connectivity="chain" → nearest-neighbour (i,i+1) only — a tendril/gear-arm.
    connectivity=<iterable of (i,j) index pairs or (role_a, role_b) name pairs>
                       → only those edges (the graph's actual sparse connectivity).

    Trimming to the connected pairs is the secondary lever on the dimensionality
    explosion (the primary is qubits-per-cluster = 4^n operator size): a sparse
    graph cuts ZZ terms 24% (5q) → 36% (7q) → 59% (14q). The tendrils are the
    extreme — no edges reaching outside the arm."""
    if connectivity is None:
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    if connectivity == "chain":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    idx = {r: i for i, r in enumerate(roles)}
    pairs = set()
    for a, b in connectivity:
        i = a if isinstance(a, int) else idx[a]
        j = b if isinstance(b, int) else idx[b]
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


class HamiltonianBasis:
    """
    Sparse, physically motivated operator basis for a cluster Hamiltonian.

    For n qubits with semantic roles, the basis consists of:
        n       single-qubit Z terms (natural frequencies)
        n       single-qubit X terms (transition rates / tunneling)
        n       single-qubit Y terms (rotation / oscillation)
        ZZ terms for CONNECTED pairs (two-qubit correlations) — dense all-pairs
                n(n-1)/2 by default, or only the graph's edges when `connectivity`
                is given (a chain/tendril or an explicit edge set).

    Y terms are critical: sensor inputs enter via σ_x, which from |0⟩
    generates Y coherence in the meta-field. Without Y in the basis,
    the meta-field's learned rotation information can't project into H.
    Y terms in H drive Z-axis oscillation — exactly what periodic
    signals (a periodic driver's cycle, temperature cycle) require.

    Total (dense): n=2 → 7, n=3 → 12, n=5 → 25 terms.
    """

    def __init__(self, n_qubits: int, roles: list[str] | None = None,
                 connectivity=None, sparse: bool = False):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.roles = roles or [str(i) for i in range(n_qubits)]
        self.connectivity = connectivity
        # sparse=True builds ONLY labels + the coupling map (which qubit/axis or
        # ZZ pair each label drives), NOT the dense 2^n operators. For a Cumulant
        # cluster the dense ops are never needed AND would OOM (a 15-qubit H is
        # 32768² = 8 GB). build_couplings() reads the map → (h_fields, zz).
        self.sparse = sparse

        self.operators: list[ComplexMatrix] = []
        self.labels: list[str] = []
        # parallel to labels: ("field", qubit_idx, axis 0/1/2=x/y/z) or ("zz", i, j)
        self.coupling_map: list[tuple] = []

        # Single-qubit Z / X / Y terms. axis index matches CumulantCluster._h
        # columns: x=0, y=1, z=2. Label order (Z, then X, then Y, then ZZ) is
        # unchanged from the legacy dense basis so coefficient bundles still line up.
        for i in range(n_qubits):                       # Z (natural frequencies)
            if not sparse:
                self.operators.append(pauli_z(i, n_qubits))
            self.labels.append(f"Z_{self.roles[i]}")
            self.coupling_map.append(("field", i, 2))
        for i in range(n_qubits):                       # X (transition rates)
            if not sparse:
                self.operators.append(pauli_x(i, n_qubits))
            self.labels.append(f"X_{self.roles[i]}")
            self.coupling_map.append(("field", i, 0))
        for i in range(n_qubits):                       # Y (oscillation / rotation)
            if not sparse:
                self.operators.append(pauli_y(i, n_qubits))
            self.labels.append(f"Y_{self.roles[i]}")
            self.coupling_map.append(("field", i, 1))

        # Two-qubit ZZ terms — only for CONNECTED pairs (the graph's sparsity).
        for i, j in resolve_zz_pairs(connectivity, n_qubits, self.roles):
            if not sparse:
                self.operators.append(_two_qubit_op(_SZ, i, _SZ, j, n_qubits))
            self.labels.append(f"ZZ_{self.roles[i]}_{self.roles[j]}")
            self.coupling_map.append(("zz", i, j))

        self.n_terms = len(self.labels)

    def build(self, coefficients: NDArray[np.floating]) -> ComplexMatrix:
        """H = Σ c_k O_k. Result is Hermitian by construction."""
        if self.sparse:
            raise RuntimeError("sparse HamiltonianBasis has no dense operators — "
                               "use build_couplings() for cumulant clusters")
        H = np.zeros((self.dim, self.dim), dtype=EVOLVE_DTYPE)
        for c, op in zip(coefficients, self.operators):
            H += float(c) * op
        return H

    def build_couplings(self, coefficients: NDArray[np.floating]):
        """Map coefficients → (h_fields (n,3), zz dict) for CumulantCluster.
        set_couplings — the no-dense-matrix H path for big cumulant clusters."""
        import numpy as _np
        h = _np.zeros((self.n_qubits, 3))
        zz: dict = {}
        for c, m in zip(coefficients, self.coupling_map):
            cf = float(c)
            if m[0] == "field":
                h[m[1], m[2]] += cf
            else:
                zz[(m[1], m[2])] = cf
        return h, zz


# Process-level cache of HamiltonianBasis objects keyed by (n_qubits, roles).
# The basis operators are read-only (only summed in build()), so one table is
# safely shared by every consumer of the same signature — across FractalStack
# scales AND Population individuals AND production clusters. Without this, each
# of (2 scales × 8 individuals × N clusters) rebuilt a byte-identical operator
# list; for the larger clusters that was ~143 MB of pure duplication. The
# per-consumer coefficients live in separate ParameterBundles, not here.
# See project_construction_oom.
_BASIS_CACHE: dict[tuple, "HamiltonianBasis"] = {}


def _conn_key(connectivity, role_key):
    """Hashable, order-stable cache key for a connectivity spec."""
    if connectivity is None or connectivity == "chain":
        return connectivity
    return tuple(sorted(tuple(sorted((str(a), str(b)))) for a, b in connectivity))


def shared_basis(n_qubits: int, roles: list[str] | None = None,
                 connectivity=None, sparse: bool = False) -> "HamiltonianBasis":
    """Return the shared HamiltonianBasis for this (n_qubits, roles, connectivity)
    signature, building it once and caching it. Use this instead of
    `HamiltonianBasis(...)` everywhere the basis is read-only (the normal case).
    connectivity=None (default) is the dense all-pairs basis — unchanged.
    sparse=True returns a labels+coupling-map-only basis (no dense 2^n operators)
    for cumulant clusters, whose dense H would OOM."""
    role_key = tuple(roles) if roles is not None else tuple(str(i) for i in range(n_qubits))
    key = (n_qubits, role_key, _conn_key(connectivity, role_key), sparse)
    basis = _BASIS_CACHE.get(key)
    if basis is None:
        basis = HamiltonianBasis(n_qubits, list(role_key), connectivity=connectivity,
                                 sparse=sparse)
        _BASIS_CACHE[key] = basis
    return basis


class HamiltonianSpec:
    """
    Learnable Hamiltonian for one cluster.

    Wraps a HamiltonianBasis with a ParameterBundle of coefficients.
    All coefficients start at 0 (backward compatible with H=0).
    """

    def __init__(self, n_qubits: int, roles: list[str] | None = None):
        from umwelt.substrate.params import ParameterBundle

        # Shared read-only operator table (one per signature) — the coefficient
        # bundle below is per-spec, so specs stay independent. See shared_basis.
        self.basis = shared_basis(n_qubits, roles)
        # All coefficients start at zero — H=0 at birth
        specs = {
            label: (0.0, 0.1, -2.0, 2.0)
            for label in self.basis.labels
        }
        self.bundle = ParameterBundle.from_dict(specs)

    @property
    def n_terms(self) -> int:
        return self.basis.n_terms

    @property
    def coefficients(self) -> NDArray[np.floating]:
        """Current coefficient values as numpy array."""
        return np.array([
            self.bundle.get(label) for label in self.basis.labels
        ])

    def set_coefficients(self, values: NDArray[np.floating]):
        """Bulk set coefficients (for genetic crossover/mutation)."""
        for label, val in zip(self.basis.labels, values):
            param = self.bundle.params[label]
            param.value = float(val)
            # Clamp to bounds
            if param.lo is not None:
                param.value = max(param.lo, param.value)
            if param.hi is not None:
                param.value = min(param.hi, param.value)

    def build(self) -> ComplexMatrix:
        """Build the Hamiltonian matrix from current coefficients."""
        return self.basis.build(self.coefficients)

    def copy(self) -> HamiltonianSpec:
        """Deep copy — mutations to the copy don't affect the original."""
        new = HamiltonianSpec.__new__(HamiltonianSpec)
        new.basis = self.basis  # shared (immutable operators)
        new.bundle = deepcopy(self.bundle)
        return new

    def snapshot(self) -> dict:
        """Serializable state."""
        return {
            "n_qubits": self.basis.n_qubits,
            "roles": self.basis.roles,
            "coefficients": {
                label: self.bundle.get(label)
                for label in self.basis.labels
            },
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> HamiltonianSpec:
        """Reconstruct from snapshot."""
        spec = cls(n_qubits=data["n_qubits"], roles=data.get("roles"))
        for label, val in data["coefficients"].items():
            if label in spec.bundle.params:
                spec.bundle.params[label].value = val
        # legacy "berry_phase" in old snapshots is ignored (now state-fiber)
        return spec
