"""
Fractal Stack — the same quantum language at every scale.

The production field (scale 0) evolves density matrices under a Hamiltonian.
But where does H come from? From a meta-field (scale 1) whose density
matrix state IS the Hamiltonian — projected through the basis operators.

    scale 2 state  →  H for scale 1
    scale 1 state  →  H for scale 0 (production field)
    scale 0        →  sensor predictions

Each scale is a QuantumField on the same WorldGraph, with the same
topology and roles. The difference is timescale: higher scales tick
slower, dissipate slower, and modulate the dynamics of the scale below.

Timescale separation uses the golden ratio: Fibonacci strides ensure
adjacent scales never synchronize in simple resonant patterns. All
scale parameters (h_scale, gamma, dt, etc.) are themselves learnable
via ParameterBundle — parameters all the way down.

The projection is natural and exact:

    H = Σ_k  Tr(ρ_meta · O_k) · h_scale · O_k

where O_k are the Hamiltonian basis operators (Z_i, X_i, ZZ_ij).
The meta-field's single-qubit expectations become single-qubit H terms.
The meta-field's pairwise correlations become coupling H terms.
Same language, different scale.

Inputs flow UPWARD: prediction residuals at scale N become the sensor
signal at scale N+1. Parameters flow DOWNWARD: Bloch states at scale N
become the Hamiltonian at scale N-1.

    residuals↑     Bloch→H↓
    scale 0  →  scale 1  →  scale 0
                    ↕
                scale 2
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field as dc_field

import numpy as np
from numpy.typing import NDArray

from umwelt.substrate.density_matrix import ComplexMatrix, pauli_x, pauli_z
from umwelt.substrate.field import QuantumField
from umwelt.substrate.hamiltonian import HamiltonianBasis, shared_basis


def _is_couplings(H) -> bool:
    return isinstance(H, tuple) and len(H) == 3 and H[0] == "couplings"


def _apply_projected_h(cluster, H) -> None:
    """Apply a projected Hamiltonian to a cluster — a dense matrix (set_hamiltonian)
    or a ('couplings', h_fields, zz) tuple for cumulant clusters (set_couplings)."""
    if _is_couplings(H):
        cluster.set_couplings(h_fields=H[1], zz=H[2])
    else:
        cluster.set_hamiltonian(H)


def _scale_projected_h(H, gate: float):
    """Scale a projected H (matrix or couplings tuple) by a scalar gate."""
    if _is_couplings(H):
        return ("couplings", H[1] * gate, {k: v * gate for k, v in H[2].items()})
    return H * gate
from umwelt.substrate.params import BlochGeometricPhase, ParameterBundle
from umwelt.substrate.graph import WorldGraph
# Single source of the golden-ratio clock, shared with the parameter
# meta-tower (MetaStack). PHI / Fibonacci strides / effective_stride all live
# in phi_clock so both towers climb exactly the same ladder.
from umwelt.clocks.phi_clock import PHI, fib_strides, fib_strides_at, effective_stride as _effective_stride

logger = logging.getLogger(__name__)


# ================================================================
# Scale participation policy — which clusters climb the meta-tower
# ================================================================
# Simple device leaves (a light, a plug, a motion sensor) don't need a deep
# multi-scale Hamiltonian meta-tower learning their dynamics — their belief is
# driven by observe-collapse + the shared drift gamma (archetypes, Phase 2),
# not high-order meta-learning. Only RICH nodes (regions, the environment node,
# multi-child appliances, the synthetic _params/_clock/_preferences fibers)
# benefit from the full tower. So device leaves participate up to the BASE
# meta-scale (level 1) only; everything else climbs all scales. This is the
# "a leaf light isn't tied to the 4th-order meta parameters" cut: it
# omits those clusters from the higher-scale QuantumField clones entirely
# (memory) and from their H-learning/projection (compute). The production
# field is never filtered, so feature_dim — and the readout — are untouched.
# See project_depth_gating.
_DEVICE_KINDS = frozenset({"actuator", "sensor", "component", "appliance"})
_DEVICE_MAX_SCALE_LEVEL = 1  # device leaves climb to the base meta-scale only

# A big cluster cannot afford a full-rho meta-scale clone. The meta tower is
# full-rho (cluster_backend="qubit") for the XY-coherence closure the 2-body
# cumulant truncation doesn't cover — but an n-qubit full-rho clone is a
# 2^n x 2^n density matrix (2^(2n+4) bytes). At base/production level these
# big clusters are CUMULANT (O(n^2)) and fine, but cloning them into the
# full-rho tower OOMs the box (a 26-qubit region-merge manifold clone exhausted
# 5.5GB + swap on the A55). So clusters above this size stay at the production
# level only — they keep their internal cumulant dynamics but don't climb the
# tower. Protects the region merge (26q manifold), the person merge (15q entities),
# and similarity sub-domain parents. Current max live cluster is 7q (exterior),
# so this caps nothing today — it's a guard against the merges. See
# project_construction_oom + project_depth_gating.
_META_FULLRHO_MAX_QUBITS = 12

# JAX-oracle finding (heritage-braid experiment): the Hebbian δ(Z_i)=r_i rule ("Z affects Z
# directly") is NOT the gradient — the Z Hamiltonian term rotates the qubit ABOUT z, leaving
# ⟨σ_z⟩ ~invariant (the true grad is ~0, mildly anti-correlated under dissipation). The X/Y/ZZ
# rules ARE the exact gradient. Flag to DROP the δZ term, for the A/B that validates the fix
# (does forecast skill improve without the spurious detuning update?). Default = current
# behavior. Runtime-settable (the A/B flips this module global). See learning_router /
# experiments jax-oracle scripts.
_DROP_HEBBIAN_DZ = os.environ.get("UMWELT_DROP_HEBBIAN_DZ") == "1"


def cluster_max_scale_level(node) -> int:
    """Highest fractal scale LEVEL (1-indexed) a node's cluster participates in.

    An explicit `node.max_scale_level` attribute overrides the policy — the
    seam for the future similarity-grouped fractal-web, where a sub-domain
    parent might carry a different depth than its members.
    """
    explicit = getattr(node, "max_scale_level", None)
    if explicit is not None:
        return int(explicit)
    if node.is_leaf and node.kind in _DEVICE_KINDS:
        return _DEVICE_MAX_SCALE_LEVEL
    roles = getattr(node, "roles", None)
    if roles is not None and len(roles) > _META_FULLRHO_MAX_QUBITS:
        return 0  # too big for a full-rho meta clone → production level only
    return 1_000_000  # rich nodes: effectively unlimited depth


def scale_participates(node, level: int) -> bool:
    """Does this node's cluster live in the scale at `level`?"""
    return cluster_max_scale_level(node) >= level


# ================================================================
# Shared helpers (DRY for production field + meta-field residuals)
# ================================================================

def _snapshot_bloch_z(
    clusters: dict[str, object],
) -> dict[str, dict[str, float]]:
    """Snapshot Bloch-z for every qubit in every cluster."""
    return {
        name: {
            role: float(cluster.qubit_bloch(idx)[2])
            for role, idx in cluster.role_index.items()
        }
        for name, cluster in clusters.items()
    }


def _compute_residuals(
    clusters: dict[str, object],
    prev_z: dict[str, dict[str, float]],
) -> dict[str, NDArray[np.floating]]:
    """Signed surprise: (actual_z - predicted_z) / 2, normalized to [-1, 1]."""
    result = {}
    for name, cluster in clusters.items():
        prev = prev_z.get(name, {})
        n = cluster.n_qubits
        residuals = np.zeros(n)
        for i, (role, idx) in enumerate(cluster.role_index.items()):
            actual = float(cluster.qubit_bloch(idx)[2])
            predicted = prev.get(role, actual)
            residuals[i] = (actual - predicted) / 2.0
        result[name] = residuals
    return result


# ================================================================
# Residual normalization — amplifies weak signals so the meta-field
# sees contrast between roles even when absolute magnitudes are tiny.
# ================================================================

class ResidualNormalizer:
    """Per-role EMA of mean and variance, for adaptive normalization.

    Without this, production residuals of ~1e-6 leave the meta-field
    stuck at |0...0> and all H coefficients stay uniform. After
    normalization, the meta-field sees O(1) input whose *structure*
    reflects which roles are changing and which aren't — exactly the
    contrast needed to break symmetry.
    """

    def __init__(self, alpha: float = 0.05, gain: float = 0.1, eps: float = 1e-8):
        self._alpha = alpha  # EMA smoothing rate
        self._gain = gain    # post-normalization scale
        self._eps = eps
        self._mean: dict[str, NDArray] = {}
        self._var: dict[str, NDArray] = {}

    def normalize(
        self, residuals: dict[str, NDArray[np.floating]],
        alpha_override: float | None = None,
        gain_override: float | None = None,
    ) -> dict[str, NDArray[np.floating]]:
        """Normalize residuals to zero-mean, unit-variance, then scale by gain."""
        result = {}
        for name, raw in residuals.items():
            n = len(raw)
            if name not in self._mean or len(self._mean[name]) != n:
                self._mean[name] = np.zeros(n)
                self._var[name] = np.ones(n) * 1e-4
            mean = self._mean[name]
            var = self._var[name]
            a = alpha_override if alpha_override is not None else self._alpha

            # Update EMA stats
            delta = raw - mean
            mean += a * delta
            var = (1 - a) * var + a * (delta ** 2)
            self._mean[name] = mean
            self._var[name] = var

            # Normalize and scale
            g = gain_override if gain_override is not None else self._gain
            std = np.sqrt(np.maximum(var, self._eps))
            normalized = (raw - mean) / std
            result[name] = np.clip(normalized * g, -1.0, 1.0)
        return result

    def save_state(self) -> dict:
        return {
            "mean": {k: v.tolist() for k, v in self._mean.items()},
            "var": {k: v.tolist() for k, v in self._var.items()},
        }

    def load_state(self, data: dict):
        self._mean = {k: np.array(v) for k, v in data.get("mean", {}).items()}
        self._var = {k: np.array(v) for k, v in data.get("var", {}).items()}


# ================================================================
# Scale configs
# ================================================================

def phi_scales(
    n_levels: int = 2,
    base_dt: float = 0.01,
    base_gamma: float = 0.0,
    base_h_scale: float = 0.05,
    base_bridge: float = 0.5,
    dt_factor: float = 1.0,
) -> list[ScaleConfig]:
    """
    Generate scale configs with golden-ratio timescale separation.

    Strides follow the Fibonacci sequence (8, 13, 21, 34, ...),
    giving golden-ratio relationships between adjacent scales.
    Other parameters decay by 1/φ per level — each scale is φ×
    more stable and φ× more conservative than the one below.

    base_gamma=0.0: meta-fields evolve unitarily (no dissipation).
    They represent learnable parameters, not physical states.
    Lindblad dissipation drives all qubits toward |0⟩, making
    Z expectations uniform (+1) and killing X/Y coherences.
    Without dissipation, the meta-field's state truly reflects the
    input residual history — different roles get different expectations,
    and X terms survive to drive rotation in the production field.

    At 1 event/sec:
        Scale 1: ticks every ~8s,   learns on ~minute timescale
        Scale 2: ticks every ~104s, learns on ~10-minute timescale
        Scale 3: ticks every ~36m,  learns on ~hour timescale
    """
    # Fibonacci strides — sliding with dt_factor along the shared φ-clock ladder.
    # ContextState.dt_factor at construction picks how far up the ladder we
    # start; at 1.0 this reproduces the legacy [8, 13, 21, ...] sequence exactly.
    strides = fib_strides_at(dt_factor, n_levels)
    import math as _m
    level_offset = max(0, round(_m.log(max(1.0, float(dt_factor))) / _m.log(PHI)))

    scales = []
    for i in range(n_levels):
        level = i + 1 + level_offset
        scales.append(ScaleConfig(
            stride=strides[i],                         # 8, 13, 21, ... (already offset)
            dt=base_dt * PHI ** level,                 # φ× longer timestep
            gamma=base_gamma / PHI ** level,            # φ× more stable
            bridge_strength=base_bridge / PHI ** level,  # φ× weaker lateral
            h_scale=base_h_scale / PHI ** level,        # φ× softer projection
        ))
    return scales


@dataclass
class ScaleConfig:
    """Configuration for one level of the fractal stack.

    These are initial conditions — h_scale is actively learned,
    others are frozen priors that can be unfrozen later.
    """

    stride: int = 8
    dt: float = 0.016
    gamma: float = 0.031
    bridge_strength: float = 0.309
    h_scale: float = 0.012


@dataclass
class FractalStackConfig:
    """Configuration for the full fractal stack."""

    enabled: bool = False
    scales: list[ScaleConfig] = dc_field(default_factory=lambda: phi_scales(2))
    # N-step rollout training: when > 0, scale-0 Hebbian updates train H
    # against the pure-unitary N-step-ahead prediction rather than the
    # one-step dissipation-mixed residual. This forces H to encode
    # trajectory — the H-purity experiment showed one-step training leaves
    # H vestigial and γ_diss does all the prediction work.
    #
    # Cost: one expm(-iH·N·dt) per cluster per step once buffer is warm.
    # 0 disables; typical values: 5-20.
    rollout_horizon: int = 0


class FractalScale:
    """
    One level of the fractal stack.

    Wraps a QuantumField on the same graph as the production field.
    Learns H coefficients from production prediction residuals via
    Hebbian-Kalman updates. The meta-field still evolves and can
    modulate learning at higher scales.

    H learning rule (Hebbian gradient):
        For each cluster, production residuals r_i = (actual_z - predicted_z)
        provide gradient signal for each H coefficient:

            δ(Z_i)   = r_i * dt         (Z affects Z directly)
            δ(X_i)   = r_i * y_i * dt   (X rotates Z via Y)
            δ(Y_i)   = -r_i * x_i * dt  (Y rotates Z via X)
            δ(ZZ_ij) = (r_i * z_j + r_j * z_i) * dt  (correlation coupling)

        Each coefficient gets a Kalman update with the gradient as
        observation. Early (high sigma) → fast learning. Late
        (low sigma) → fine tuning. Thompson Sampling explores.

    Scale parameters live in a ParameterBundle — h_scale is actively
    learned from surprise, others are frozen priors.
    """

    def __init__(self, graph: WorldGraph, config: ScaleConfig, level: int):
        self.config = config
        self.level = level
        # Higher scales hold only the RICH clusters — device leaves are gated
        # to the base meta-scale (see scale_participates). The base scale
        # (level 1) keeps every cluster; level 2+ drops the simple devices,
        # shrinking the per-scale field clone and its H machinery.
        self.field = QuantumField(
            graph=graph,
            gamma=config.gamma,
            dt=config.dt,
            bridge_strength=config.bridge_strength,
            cluster_filter=(lambda n, _lvl=level: scale_participates(n, _lvl)),
            # Meta-scales stay full-ρ: they carry transverse XY coherence (the
            # rotation-gradient learning signal), NOT the dissipation-dominated
            # regime where the 2-body cumulant closure is validated. Stage-1
            # cumulant swap is production-field-only.
            cluster_backend="qubit",
        )

        # Learnable scale parameters — same fiber as everything else.
        self.params = ParameterBundle.from_dict(
            {
                "h_scale": (config.h_scale, config.h_scale * 0.3, 0.001, 0.5),
                "gamma": (config.gamma, max(config.gamma * 0.2, 1e-6), 0.0, 0.5),
                "dt": (config.dt, config.dt * 0.1, 0.001, 10.0),
                "bridge_strength": (config.bridge_strength, max(config.bridge_strength * 0.2, 1e-6), 0.0, 1.0),
                "stride": (float(config.stride), 1.0, 2.0, 100.0),
            },
            frozen_keys={"gamma", "dt", "bridge_strength", "stride"},
        )

        # Self-tuning state — on the fiber so they learn
        self.params.merge(ParameterBundle.from_dict({
            "surprise_target": (0.05, 0.02, 0.001, 0.5),
            "surprise_alpha": (0.1, 0.03, 0.01, 0.5),
            "normalizer_gain": (0.1, 0.03, 0.01, 2.0),
            "normalizer_alpha": (0.05, 0.02, 0.01, 0.5),
            # Hebbian learning rate — adapts from surprise trend.
            # High lr = fast H learning (exploration). Low lr = fine-tuning.
            "hebbian_lr": (0.5, 0.2, 0.001, 5.0),
            # Improvement/plateau thresholds for the lr trend classifier.
            # CALIBRATED by MetaStack tier 2 from classification persistence
            # (did "improving" actually keep improving over the tier-2 window?).
            # Tier 2 tightens when too many false-positive calls, loosens when
            # being too cautious.
            "lr_improve_thresh": (-0.002, 0.001, -0.5, -0.0001),
            "lr_plateau_thresh": (0.0005, 0.0002, 0.0001, 0.01),
            # Surprise-proportional plasticity: scale the per-step LR by
            # (1 + gain * inst_surprise). High surprise → learn more. 0 = off.
            # Tuning: 2.0 gives ~3x boost at surprise=1.0; keep below 10.
            "surprise_plasticity_gain": (2.0, 0.5, 0.0, 10.0),
            # Blank-slate boost: a fresh brain (h_norm ≈ 0) gets a plasticity
            # multiplier that fades as the model accumulates learned H structure.
            # At h_norm=0 → multiplier = 1 + blank_slate_boost; at h_norm >=
            # complexity_ceiling → multiplier = 1.0 (fully faded).
            # Tuning: boost=2.0 means a cold brain learns 3× faster than a
            # mature one. complexity_ceiling is the h_norm saturation point.
            "blank_slate_boost": (2.0, 0.5, 0.0, 5.0),
            "complexity_ceiling": (0.3, 0.05, 0.05, 2.0),
        }))
        self._surprise_ema: float = 0.0
        self._surprise_prev: float = 0.0  # for trend-based lr adaptation
        # Classification ring buffer for tier-2 grading: each entry is
        # (kind, surprise_at_call) — tier 2 reads it back later and asks
        # whether the surprise trend actually persisted in the right direction.
        from collections import deque
        self._lr_classifications: deque = deque(maxlen=64)

        # Hamiltonian basis and learnable coefficient bundles per cluster.
        self._h_bases: dict[str, HamiltonianBasis] = {}
        self._h_bundles: dict[str, ParameterBundle] = {}
        for name, cluster in self.field.clusters.items():
            # Shared read-only operator table — every scale of every cluster
            # with the same (n_qubits, roles) signature points at one basis;
            # only the coefficient bundle below is per-scale. See shared_basis.
            # Cumulant clusters get the SPARSE basis (labels + coupling map only,
            # no dense 2^n operators — a big cumulant cluster's dense H would OOM).
            # connectivity (e.g. the world graph's region adjacency) keeps the H-tower's ZZ
            # labels in lockstep with the cluster's coupled pairs (both via
            # resolve_zz_pairs) — so a sparse cluster learns only its real edges.
            basis = shared_basis(cluster.n_qubits, cluster.qubit_roles,
                                 connectivity=getattr(cluster, "connectivity", None),
                                 sparse=getattr(cluster, "is_cumulant", False))
            self._h_bases[name] = basis
            # Each H coefficient is a learnable ScalarParam: starts at 0,
            # sigma=0.1 allows early exploration via Thompson Sampling.
            specs = {
                label: (0.0, 0.1, -2.0, 2.0)
                for label in basis.labels
            }
            self._h_bundles[name] = ParameterBundle.from_dict(specs)

        # Bridge coupling bundles: one ParameterBundle per bridge, one coefficient per
        # shared role. Start at 0 — the brain discovers cross-cluster coupling from data
        # via the same Hebbian gradient rule as within-cluster H coefficients.
        # Bounds [-1, 1]: tighter than H coefficients to prevent coupling dominating
        # each cluster's own learned dynamics.
        self._bridge_bundles: dict[tuple[str, str], ParameterBundle] = {}
        self._tendril_keys: set[tuple[str, str]] = set()  # keys that use σx coupling
        for bridge in self.field.graph.bridges:
            key = (bridge.source, bridge.target)
            if bridge.is_tendril:
                # Bias negative so the equatorial decohere path (alpha_eq = abs(CC)*0.40)
                # fires immediately; training drives CC toward -1.0 from here.
                specs = {f"CC_{s}→{t}": (-0.5, 0.1, -1.0, 1.0)
                         for s, t in bridge.role_map.items()}
                if specs:
                    self._tendril_keys.add(key)
            else:
                if not bridge.shared_roles:
                    continue
                specs = {f"CC_{role}": (0.0, 0.1, -1.0, 1.0) for role in bridge.shared_roles}
            if specs:
                self._bridge_bundles[key] = ParameterBundle.from_dict(specs)

        self._prev_z: dict[str, dict[str, float]] = {}
        self._step = 0

        # Real geometric phase of THIS scale's own Bloch trajectory.
        # Accumulated per-qubit after each field step; returns to a prior
        # value only on genuine loop closure (see BlochGeometricPhase).
        self.bloch_berry = BlochGeometricPhase()

    @property
    def effective_stride(self) -> int:
        """Current stride (integer from learnable param), via the shared φ-clock."""
        return _effective_stride(self.params.get_param("stride"))

    @property
    def effective_h_scale(self) -> float:
        """Current h_scale point estimate (for diagnostics/API)."""
        return self.params.get("h_scale")

    @property
    def sampled_h_scale(self) -> float:
        """Thompson-sampled h_scale for evolution.

        Exploration is continuous: large sigma (early/uncertain) produces
        varied projections, small sigma (learned) produces stable ones.
        """
        return self.params.get("h_scale", explore=True)

    def record_predictions(self):
        """Snapshot Bloch-z before evolution for residual computation."""
        self._prev_z = _snapshot_bloch_z(self.field.clusters)

    def compute_residuals(self) -> dict[str, NDArray[np.floating]]:
        """Signed surprise: (actual_z - predicted_z) / 2, normalized to [-1, 1]."""
        return _compute_residuals(self.field.clusters, self._prev_z)

    def hebbian_update(
        self,
        production_residuals: dict[str, NDArray[np.floating]],
        production_clusters: dict[str, object],
        coherence_clusters: dict[str, object] | None = None,
    ):
        """
        Hebbian gradient update: learn H coefficients from production errors.

        Uses SGD with the learning rate on the parameter bundle (learnable).
        The gradient is derived from how each basis operator affects Z evolution:

            δ(Z_i)   = r_i                    (direct Z coupling)
            δ(X_i)   = r_i * y_i              (X rotates Z via Y)
            δ(Y_i)   = -r_i * x_i             (Y rotates Z via X)
            δ(ZZ_ij) = r_i * z_j + r_j * z_i  (correlation)

        `coherence_clusters` — optional dict of clusters whose Bloch X/Y values
        are used for the X_i and Y_i rotation gradients. Defaults to
        production_clusters (legacy behavior). Passing the meta-field's own
        clusters here is the fix for a pathology: production clusters with
        dissipative input collapse x, y toward 0, which zeros the X_i/Y_i
        gradients, so rotation-inducing H terms never learn. The meta-field
        is non-dissipative and carries genuine transverse coherence — using
        its Bloch xy for the rotation gradient keeps X/Y terms learnable.
        Z and ZZ gradients still use production (they depend on the real
        z-state we're trying to predict).

        Coefficients are clamped to [-2, 2] by the ScalarParam bounds.
        Berry phase tracks the parameter drift.
        """
        base_lr = self.params.get("hebbian_lr")

        # Surprise-proportional modulation: more surprise → more plasticity.
        all_residuals = [float(v) for rv in production_residuals.values()
                         for v in np.abs(rv)]
        inst_surprise = float(np.mean(all_residuals)) if all_residuals else 0.0
        surprise_gain = 1.0 + self.params.get("surprise_plasticity_gain") * inst_surprise

        # Blank-slate boost: cold brain learns faster; fades as H structure grows.
        h_vals = [abs(p.value)
                  for bundle in self._h_bundles.values()
                  for p in bundle.params.values()]
        h_norm = float(np.mean(h_vals)) if h_vals else 0.0
        ceiling = max(self.params.get("complexity_ceiling"), 1e-6)
        maturity = min(1.0, h_norm / ceiling)
        complexity_gain = 1.0 + self.params.get("blank_slate_boost") * (1.0 - maturity)

        lr = base_lr * surprise_gain * complexity_gain

        coh_map = coherence_clusters or {}
        for name, basis in self._h_bases.items():
            bundle = self._h_bundles.get(name)
            prod_cluster = production_clusters.get(name)
            residuals = production_residuals.get(name)
            if bundle is None or prod_cluster is None or residuals is None:
                continue

            n = prod_cluster.n_qubits
            prod_bloch = np.array([
                prod_cluster.qubit_bloch(i) for i in range(n)
            ])  # (n, 3): x, y, z from production
            coh_cluster = coh_map.get(name, prod_cluster)
            coh_bloch = prod_bloch if coh_cluster is prod_cluster else np.array([
                coh_cluster.qubit_bloch(i) for i in range(coh_cluster.n_qubits)
            ])

            for label in basis.labels:
                gradient = 0.0
                if label.startswith("Z_") and not _DROP_HEBBIAN_DZ:
                    role = label[2:]
                    idx = prod_cluster.role_index.get(role)
                    if idx is not None and idx < len(residuals):
                        gradient = residuals[idx]   # δZ=r — JAX-flagged; see _DROP_HEBBIAN_DZ
                elif label.startswith("X_"):
                    role = label[2:]
                    idx = prod_cluster.role_index.get(role)
                    coh_idx = coh_cluster.role_index.get(role)
                    if (idx is not None and idx < len(residuals)
                            and coh_idx is not None and coh_idx < len(coh_bloch)):
                        gradient = residuals[idx] * coh_bloch[coh_idx, 1]
                elif label.startswith("Y_"):
                    role = label[2:]
                    idx = prod_cluster.role_index.get(role)
                    coh_idx = coh_cluster.role_index.get(role)
                    if (idx is not None and idx < len(residuals)
                            and coh_idx is not None and coh_idx < len(coh_bloch)):
                        gradient = -residuals[idx] * coh_bloch[coh_idx, 0]
                elif label.startswith("ZZ_"):
                    parts = label[3:].split("_", 1)
                    if len(parts) == 2:
                        idx_i = prod_cluster.role_index.get(parts[0])
                        idx_j = prod_cluster.role_index.get(parts[1])
                        if (idx_i is not None and idx_j is not None
                                and idx_i < len(residuals)
                                and idx_j < len(residuals)):
                            gradient = (
                                residuals[idx_i] * prod_bloch[idx_j, 2]
                                + residuals[idx_j] * prod_bloch[idx_i, 2]
                            )

                if abs(gradient) > 1e-10:
                    param = bundle.get_param(label)
                    if param is not None and not param.frozen:
                        param.value += lr * gradient
                        # Clamp to bounds
                        if param.lo is not None:
                            param.value = max(param.lo, param.value)
                        if param.hi is not None:
                            param.value = min(param.hi, param.value)

        # Cross-cluster Hebbian: grow/shrink bridge coupling coefficients
        # from co-prediction failures. Same gradient form as within-cluster ZZ —
        # δ(CC_r) = r_src_r * z_tgt_r + r_tgt_r * z_src_r — but evaluated across
        # the bridge boundary instead of within a single cluster.
        for (src_name, tgt_name), cbundle in self._bridge_bundles.items():
            src_cluster = production_clusters.get(src_name)
            tgt_cluster = production_clusters.get(tgt_name)
            src_res = production_residuals.get(src_name)
            tgt_res = production_residuals.get(tgt_name)
            if src_cluster is None or tgt_cluster is None or src_res is None or tgt_res is None:
                continue
            src_bloch = np.array([src_cluster.qubit_bloch(i) for i in range(src_cluster.n_qubits)])
            tgt_bloch = np.array([tgt_cluster.qubit_bloch(i) for i in range(tgt_cluster.n_qubits)])
            for label, param in cbundle.params.items():
                if param.frozen or not label.startswith("CC_"):
                    continue
                raw = label[3:]
                if "→" in raw:
                    src_role, tgt_role = raw.split("→", 1)
                else:
                    src_role = tgt_role = raw
                src_idx = src_cluster.role_index.get(src_role)
                tgt_idx = tgt_cluster.role_index.get(tgt_role)
                if src_idx is None or tgt_idx is None:
                    continue
                if src_idx >= len(src_res) or tgt_idx >= len(tgt_res):
                    continue
                src_z = float(src_bloch[src_idx, 2])
                tgt_z = float(tgt_bloch[tgt_idx, 2])
                gradient = src_res[src_idx] * tgt_z + tgt_res[tgt_idx] * src_z
                if abs(gradient) > 1e-10:
                    param.value += lr * gradient
                    if param.lo is not None:
                        param.value = max(param.lo, param.value)
                    if param.hi is not None:
                        param.value = min(param.hi, param.value)

    def self_tune(self, residuals: dict[str, NDArray[np.floating]]):
        """
        Adapt h_scale from own surprise.

        Low surprise → increase h_scale (trust learned H).
        High surprise → decrease h_scale (be conservative).
        """
        all_residuals = []
        for r in residuals.values():
            all_residuals.extend(np.abs(r).tolist())
        if not all_residuals:
            return

        surprise = sum(all_residuals) / len(all_residuals)

        alpha = self.params.get("surprise_alpha")
        self._surprise_ema = (
            alpha * surprise + (1.0 - alpha) * self._surprise_ema
        )

        target = self.params.get("surprise_target")
        if target > 0 and self._surprise_ema > 1e-10:
            from umwelt.learning.meta_idioms import proportional_nudge
            observed_h_scale = proportional_nudge(
                self.params.get("h_scale"), target / self._surprise_ema,
            )
            self.params.update("h_scale", observed_h_scale, obs_sigma=0.01)

        # Adapt hebbian_lr from surprise trend.
        # Improving (delta < improve_thresh) → tighten lr (fine-tune).
        # Plateau (|delta| < plateau_thresh) → widen lr (break out of minimum).
        # The two thresholds are LIVE-READ fiber priors — MetaStack tier 2 grades
        # this classifier's persistence and tunes them. The "improving" call is
        # logged so tier 2 can ask: did surprise actually keep dropping after we
        # said so? (See _lr_classifications.) 0.97 / 1.05 step sizes are kept
        # finer than the standard step_down/up (a deliberate behavioural choice).
        improve_t = self.params.get("lr_improve_thresh")
        plateau_t = self.params.get("lr_plateau_thresh")
        surprise_delta = self._surprise_ema - self._surprise_prev
        self._surprise_prev = self._surprise_ema
        current_lr = self.params.get("hebbian_lr")
        if surprise_delta < improve_t:
            self.params.update("hebbian_lr", current_lr * 0.97, obs_sigma=0.1)
            self._lr_classifications.append(("improving", self._surprise_ema))
        elif abs(surprise_delta) < plateau_t and self._surprise_ema > target * 2:
            self.params.update("hebbian_lr", current_lr * 1.05, obs_sigma=0.1)
            self._lr_classifications.append(("plateau", self._surprise_ema))

    def project_hamiltonians(self, deterministic: bool = False,
                             production_clusters=None) -> dict[str, ComplexMatrix]:
        """
        Build Hamiltonians from learned coefficients.

        H = Σ_k c_k · O_k

        Coefficients are learned directly via Hebbian gradient updates.
        No h_scale multiplication — the coefficients ARE the learned values.
        Thompson Sampling explores when not deterministic.
        """
        result = {}
        cum = {}   # name -> [h_fields (n,3), zz dict] for cumulant clusters (no dense H)
        for name, basis in self._h_bases.items():
            bundle = self._h_bundles.get(name)
            if bundle is None:
                continue
            explore = not deterministic
            coeffs = np.array([
                bundle.get(label, default=0.0, explore=explore)
                for label in basis.labels
            ])
            if basis.sparse:
                h, zz = basis.build_couplings(coeffs)
                cum[name] = [h, zz]          # mean-field 1-local terms add into h below
            else:
                result[name] = basis.build(coeffs)

        # Mean-field cross-cluster coupling: for each learned bridge coefficient,
        # inject a bias-field term into both clusters proportional to the partner's
        # current Bloch-z. This makes cross-cluster coupling continuous and
        # Hamiltonian-driven (not a discrete density-matrix nudge).
        for (src_name, tgt_name), cbundle in self._bridge_bundles.items():
            src_cluster = self.field.clusters.get(src_name)
            tgt_cluster = self.field.clusters.get(tgt_name)
            if src_cluster is None or tgt_cluster is None:
                continue
            for label, param in cbundle.params.items():
                if not label.startswith("CC_"):
                    continue
                c = param.value if deterministic else param.sample()
                if abs(c) < 1e-9:
                    continue
                raw = label[3:]
                if "→" in raw:
                    src_role, tgt_role = raw.split("→", 1)
                else:
                    src_role = tgt_role = raw
                src_idx = src_cluster.role_index.get(src_role)
                tgt_idx = tgt_cluster.role_index.get(tgt_role)
                if src_idx is None or tgt_idx is None:
                    continue
                tgt_z_val = float(tgt_cluster.qubit_bloch(tgt_idx)[2])
                src_z_val = float(src_cluster.qubit_bloch(src_idx)[2])
                is_tendril = (src_name, tgt_name) in self._tendril_keys
                if is_tendril:
                    # Transverse (σx) coupling: z_src drives σx on tgt, creating
                    # population rotation toward +z when the source role is active.
                    # σz would only shift phase (commutes with mixed state); σx drives
                    # the qubit between |0⟩ and |1⟩.
                    # Use production field z (live sensor state) if available — the
                    # meta-field only ticks at stride intervals, so it's stale during
                    # short check runs.
                    if production_clusters is not None:
                        prod_src = production_clusters.get(src_name)
                        prod_idx = prod_src.role_index.get(src_role) if prod_src else None
                        if prod_src is not None and prod_idx is not None:
                            src_z_val = float(prod_src.qubit_bloch(prod_idx)[2])
                    # σx field on tgt_idx → hx (axis 0). Cumulant: add to h_fields
                    # directly (no dense pauli_x — would OOM for a big cluster).
                    if tgt_name in cum:
                        cum[tgt_name][0][tgt_idx, 0] += c * src_z_val
                    elif tgt_name in result:
                        result[tgt_name] = result[tgt_name] + c * src_z_val * pauli_x(
                            tgt_idx, tgt_cluster.n_qubits)
                else:
                    # σz fields → hz (axis 2) on each side.
                    if src_name in cum:
                        cum[src_name][0][src_idx, 2] += c * tgt_z_val
                    elif src_name in result:
                        result[src_name] = result[src_name] + c * tgt_z_val * pauli_z(
                            src_idx, src_cluster.n_qubits)
                    if tgt_name in cum:
                        cum[tgt_name][0][tgt_idx, 2] += c * src_z_val
                    elif tgt_name in result:
                        result[tgt_name] = result[tgt_name] + c * src_z_val * pauli_z(
                            tgt_idx, tgt_cluster.n_qubits)

        # Package cumulant couplings with a marker; _project_down feeds set_couplings.
        for name, (h, zz) in cum.items():
            result[name] = ("couplings", h, zz)
        return result

    def projected_coefficients(self) -> dict[str, dict[str, float]]:
        """Current H coefficients as readable dict (for API/diagnostics)."""
        result = {}
        for name, basis in self._h_bases.items():
            bundle = self._h_bundles.get(name)
            if bundle is None:
                continue
            result[name] = {
                label: round(bundle.get(label, default=0.0), 6)
                for label in basis.labels   # operators absent for sparse bases
            }
        return result

    def accumulate_berry(self) -> None:
        """Fold one step of this scale's Bloch trajectory into its real
        geometric phase. Call once per field.step()."""
        for name, cluster in self.field.clusters.items():
            for i in range(cluster.n_qubits):
                self.bloch_berry.update(f"{name}:{i}", cluster.qubit_bloch(i))

    def snapshot(self) -> dict:
        """Diagnostics for API."""
        return {
            "level": self.level,
            "step": self._step,
            "stride": self.effective_stride,
            "h_scale": round(self.effective_h_scale, 6),
            "h_scale_sigma": round(self.params.get_param("h_scale").sigma, 6),
            "hebbian_lr": round(self.params.get("hebbian_lr"), 6),
            "surprise_ema": round(self._surprise_ema, 6),
            "berry_phase": round(self.bloch_berry.total, 6),
            "gamma": self.config.gamma,
            "dt": self.config.dt,
            "cluster_purities": {
                name: round(float(cluster.purity), 6)
                for name, cluster in self.field.clusters.items()
            },
            "h_coefficients": self.projected_coefficients(),
            "bridge_couplings": {
                f"{s}↔{t}": {k: round(p.value, 6) for k, p in bun.params.items()}
                for (s, t), bun in self._bridge_bundles.items()
                if any(abs(p.value) > 0.001 for p in bun.params.values())
            },
        }

    def phase_align(
        self,
        production_clusters: dict[str, object],
        reference_series: list[tuple] | None = None,
    ):
        """Flip ALL rotation coefficients (Y and X) to match observed phase.

        After Hebbian learning, the frequency is correct but the sign
        may be wrong (180° phase error). This flips all Y and X terms
        together to reverse rotation direction.

        If reference_series is provided, measures correlation before/after
        flip and keeps the better one. Otherwise flips if Y_solar < 0
        (heuristic: sun rises = positive Y rotation).
        """
        # Check if we need to flip
        needs_flip = False
        for name, bundle in self._h_bundles.items():
            param = bundle.get_param("Y_solar")
            if param is not None and param.value < 0:
                needs_flip = True
                break

        if not needs_flip:
            logger.info("Phase align: Y_solar already positive, no flip needed")
            return

        # Flip ALL Y and X terms (they work together to create rotation)
        flipped = []
        for name, bundle in self._h_bundles.items():
            for label in list(bundle.params.keys()):
                if label.startswith(("Y_", "X_")):
                    param = bundle.get_param(label)
                    if param is not None and abs(param.value) > 1e-8:
                        param.value = -param.value
                        flipped.append(f"{name}.{label}")

        if flipped:
            logger.info("Phase align: flipped %d rotation terms", len(flipped))


class FractalStack:
    """
    Multi-scale fractal field stack with dynamic depth.

    Same graph, same quantum language, different timescales.
    Each scale's density matrix state becomes the Hamiltonian
    for the scale below. Residuals flow upward as input.

    The production field (scale 0) is external — it's the live
    field that processes sensor data. The fractal stack manages
    scales 1, 2, ... that learn its dynamics.

    The step function is recursive: the same function runs at
    every depth. Upward (residuals) on the way into the recursion,
    downward (H projection) on the way back out. The call stack
    IS the fractal hierarchy.

    The stack breathes: when the deepest scale stays surprised
    (can't model its inputs), it spawns a deeper scale. When a
    scale goes dormant, it gets pruned. Depth adapts to the
    complexity of the environment.

    Timescale separation uses golden-ratio Fibonacci strides.
    Scale parameters self-tune via Berry-phase-tracked Kalman
    updates — parameters learning parameters.
    """

    # Memory safety ceiling only — all other thresholds are learnable
    MAX_DEPTH = 6

    def __init__(
        self,
        graph: WorldGraph,
        production_field: QuantumField,
        config: FractalStackConfig | None = None,
    ):
        self.graph = graph
        self.production_field = production_field

        if config is None:
            config = FractalStackConfig()
        self.config = config

        self.scales: list[FractalScale] = []
        for i, sc in enumerate(config.scales):
            self.scales.append(FractalScale(graph, sc, level=i + 1))

        self._step = 0
        self._prev_prod_z: dict[str, dict[str, float]] = {}

        # Adaptive normalizer: amplifies tiny production residuals so the
        # meta-field sees O(1) contrast between roles.
        self._normalizer = ResidualNormalizer(alpha=0.05, gain=1.0)

        # Dynamic depth bookkeeping
        self._surprise_at_spawn: float = 0.0  # surprise when last scale was spawned
        self._surprise_at_prune: float = 0.0  # production surprise when last scale was pruned

        # Depth-breathing dynamics — BOTH directions are purely soft, no thresholds/patience/×hacks,
        # no fallback (Luke: a threshold means we aren't trusting the right soft qubit dynamic; a
        # comprehension engine must comprehend — no wheelchairs). SPAWN and PRUNE are duals of ONE
        # question — "is more depth worth it?" — so they SHARE the learned `spawn_marginal_value`
        # belief m: spawn grows on surprise·m, prune shrinks on dormancy·(1−m). See _maybe_spawn /
        # _maybe_prune + experiments/spawn_dissolution.py + spawn_live_validation.py.
        self.params = ParameterBundle.from_dict({
            # The marginal value of depth — "is more depth reducing surprise?" Learned + persisted;
            # the PRIOR (0.4) is where the retired ×1.05 futility-skepticism now lives (conservative
            # default, validated on live records: real single-scale surprise is low → spawning rare).
            # It learns UP if depth helps (m→1, spawn fires on surprise, prune is suppressed) and DOWN
            # if depth proves useless (m→0, spawn stalls, prune trims the dead scales). One belief,
            # both directions — the old prune_threshold/patience/×0.8 hacks are dissolved into it.
            "spawn_marginal_value": (0.4, 0.2, 0.0, 1.0),
            "spawn_leak": (0.97, 0.02, 0.5, 0.999),         # membrane leak — sets the sensitivity floor (shared)
            # The depth-learner's own collapse width — a gauge coordinate, not a literal (seed = the
            # old observe() obs_sigma=0.1). Completes the law's own-constant totality. See param_bundles.
            "spawn_obs_sigma": (0.1, 0.03, 0.02, 0.5),
        })
        self._spawn_pressure: float = 0.0   # leaky integrate-and-fire accumulators (transient state)
        self._prune_pressure: float = 0.0
        from umwelt.learning.universal_learner import UniversalLearner
        self._spawn_learner = UniversalLearner()

        # Last raw production residuals (exposed for sensor weight learning)
        self.last_raw_residuals: dict[str, NDArray] = {}
        # Per-(cluster, qubit) |production residual| EMA — the PRODUCTION prediction error,
        # resolved to roles. This is the per-cluster half of competence (b9.44): the global
        # scales[0]._surprise_ema is the META-field's surprise, but "is the brain predicting
        # THIS actuator's slice of the world" must read the production residuals at the roles
        # that actuator confounds (competence.actuator_competence via role_surprise()).
        # Persisted (save_state) so a redeployed learned brain reads skilled immediately,
        # matching learnedness's reload-surviving semantics.
        self._role_surprise_ema: dict[str, NDArray] = {}

        # Rollout buffer for N-step pure-H training. Each entry is a dict
        # of {cluster_name: ρ_snapshot} captured before that step's evolution.
        # When the buffer is `rollout_horizon` entries deep, the oldest entry
        # was captured N steps ago; we use it to build a pure-unitary
        # prediction that trains H against the observed N-step trajectory.
        from collections import deque as _deque
        self.rollout_horizon: int = int(config.rollout_horizon)
        self._rollout_buffer: _deque[dict[str, np.ndarray]] = _deque(
            maxlen=max(1, self.rollout_horizon + 1),
        )
        self._rollout_updates: int = 0  # diagnostic counter

        # Berry velocity (set by reservoir each step). High velocity =
        # system is learning fast → attenuate H projection (don't commit
        # to an H that's still moving). Low velocity → project confidently.
        self.berry_velocity: float = 0.0

        # Adaptive tick interval — the wall-clock rate at which the
        # main loop should call ingest(). Derived from surprise curvature:
        # high surprise = short interval (field is learning, needs fine steps),
        # low surprise = long interval (field is coasting, save CPU).
        # min/max are hard clamps; the actual value is computed from surprise.
        self.params.merge(ParameterBundle.from_dict({
            "tick_min_s": (5.0, 1.0, 1.0, 30.0),
            "tick_max_s": (120.0, 20.0, 30.0, 600.0),
            "tick_surprise_ref": (0.005, 0.002, 1e-4, 0.1),
        }, frozen_keys={"tick_min_s", "tick_max_s"}))

        n_terms = sum(
            b.n_terms for b in self.scales[0]._h_bases.values()
        ) if self.scales else 0
        logger.info(
            "FractalStack: %d scales, %d clusters, %d H-terms/scale, "
            "strides=%s (phi-separated)",
            len(self.scales),
            len(production_field.clusters),
            n_terms,
            [s.effective_stride for s in self.scales],
        )

    # ================================================================
    # Adaptive tick interval
    # ================================================================

    @property
    def recommended_tick_interval(self) -> float:
        """Wall-clock seconds between reservoir ingest calls.

        Maps surprise to interval via inverse proportionality:
            interval = ref / max(surprise, floor) clamped to [min, max]

        High surprise → short interval (field is learning, fine steps).
        Low surprise  → long interval  (field coasting, save CPU).

        ref, min, max live on the parameter fiber — ref is learnable
        so the stack can discover its own Courant condition.
        """
        if not self.scales:
            return 10.0
        surprise = self.scales[0]._surprise_ema
        ref = self.params.get("tick_surprise_ref")
        t_min = self.params.get("tick_min_s")
        t_max = self.params.get("tick_max_s")
        raw = ref / max(surprise, 1e-8)
        return float(np.clip(raw, t_min, t_max))

    # ================================================================
    # N-step rollout training helpers
    # ================================================================

    def _capture_rollout_snapshot(self) -> None:
        """Record each production-cluster's state for a future rollout. Dense clusters store ρ;
        cumulant clusters store (e1,e2) (no dense ρ); the param fiber (product) has no joint ρ (#309)."""
        snap = {}
        for name, cluster in self.production_field.clusters.items():
            if getattr(cluster, "is_product", False):
                continue                      # the param fiber has no joint rho (#309)
            if getattr(cluster, "is_cumulant", False):
                snap[name] = ("cum", cluster.e1.copy(), cluster.e2.copy())
            else:
                snap[name] = cluster.rho.copy()
        self._rollout_buffer.append(snap)

    def _rollout_residuals(self) -> dict[str, NDArray[np.floating]] | None:
        """Compute rollout residuals: observed z(t) minus pure-H N-step prediction.

        Returns None while the buffer is still warming up (first N steps).
        Otherwise, for each cluster:
            U = expm(−i · H · N · dt)
            ρ_pred = U · ρ_{t−N} · U†
            residual_i = (z_actual_i − z_predicted_i) / 2
        """
        if len(self._rollout_buffer) <= self.rollout_horizon:
            return None

        from scipy.linalg import expm

        N = self.rollout_horizon
        initial = self._rollout_buffer[0]  # ρ captured N steps ago
        residuals = {}

        for name, cluster in self.production_field.clusters.items():
            if getattr(cluster, "is_product", False):
                continue                                    # no joint rho on the param fiber (#309)
            snap0 = initial.get(name)
            if snap0 is None:
                continue
            if getattr(cluster, "is_cumulant", False):
                # native pure-H cumulant rollout from the N-steps-ago (e1,e2)
                _, e1_0, e2_0 = snap0
                pred_z = cluster.forecast_z(N, cluster.evolver.dt, e1=e1_0, e2=e2_0)
            else:
                rho0 = snap0
                H = cluster.evolver.H_base
                dt = cluster.evolver.dt
                U = expm(-1j * H * (N * dt))
                rho_pred = U @ rho0 @ U.conj().T
                # Temporarily swap ρ to read predicted Bloch-z per qubit.
                saved = cluster.rho
                cluster.rho = rho_pred
                pred_z = np.array([
                    cluster.qubit_bloch(i)[2] for i in range(cluster.n_qubits)
                ])
                cluster.rho = saved
            actual_z = np.array([
                cluster.qubit_bloch(i)[2] for i in range(cluster.n_qubits)
            ])
            residuals[name] = (actual_z - pred_z) / 2.0

        return residuals

    # ================================================================
    # Main step — recursive
    # ================================================================

    def step(self):
        """
        Called AFTER production field.step().

        Computes production residuals, normalizes them, then enters
        the recursive step. Same function at every depth:
            1. Going IN:  evolve this scale from parent residuals (upward)
            2. RECURSE:   pass this scale's residuals to the next depth
            3. Coming OUT: project this scale's H onto the scale below (downward)

        After recursion, check if the stack should breathe (spawn/prune).
        """
        self._step += 1

        # Production residuals → normalized → seed for recursion
        raw_residuals = _compute_residuals(
            self.production_field.clusters, self._prev_prod_z,
        )
        self.last_raw_residuals = raw_residuals  # expose for sensor weight learning
        self._update_role_surprise(raw_residuals)
        self._prev_prod_z = _snapshot_bloch_z(self.production_field.clusters)

        # Read normalizer params from scale 0 bundle (live-learnable)
        norm_alpha = None
        norm_gain = None
        if self.scales:
            norm_alpha = self.scales[0].params.get("normalizer_alpha")
            norm_gain = self.scales[0].params.get("normalizer_gain")
        residuals = self._normalizer.normalize(
            raw_residuals, alpha_override=norm_alpha, gain_override=norm_gain,
        )

        # Hebbian-Kalman: scale 0 learns H coefficients.
        #
        # One-step mode: residual = (z(t) − z(t−1)) tells H "what did the
        # dissipatively-evolved state just do differently from nothing."
        # Result (validated by the H-purity test): H stays near zero; γ_diss
        # does all the prediction work.
        #
        # Rollout mode: residual = (z(t) − z_pred_pure_H(t from ρ(t−N)))
        # forces H to produce the observed N-step drift under pure unitary
        # evolution. γ_diss can't substitute for H here because the prediction
        # runs with no dissipation at all.
        if self.scales:
            rollout_residuals = None
            if self.rollout_horizon > 0:
                rollout_residuals = self._rollout_residuals()
                if rollout_residuals is not None:
                    self._rollout_updates += 1
            training_residuals = rollout_residuals if rollout_residuals is not None else raw_residuals
            # Feed the meta-field's own Bloch as the coherence source for
            # X/Y rotation gradients. Production Bloch is dissipation-flattened
            # in xy so those gradients starve; meta-field is non-dissipative
            # and carries the transverse coherence that makes rotation terms
            # learnable.
            self.scales[0].hebbian_update(
                training_residuals,
                self.production_field.clusters,
                coherence_clusters=self.scales[0].field.clusters,
            )
            # Capture today's state for future rollouts.
            if self.rollout_horizon > 0:
                self._capture_rollout_snapshot()

        # Enter the recursion
        self._recurse(0, residuals)

        # Tendril injection: for directed region→actuator coupling, inject the
        # learned preference as a soft observation on the target cluster. Actuator
        # qubits are dissipative (Lindblad thermal), so H injection is overridden
        # by the thermal channel. Instead, we directly nudge ρ toward the
        # preferred state proportional to CC * z_src.
        if self.scales:
            self._project_tendrils(self.scales[0])

        # Dynamic depth: breathe
        self._maybe_spawn()
        self._maybe_prune()

    def _update_role_surprise(self, raw_residuals: dict[str, NDArray]) -> None:
        """EMA the |production residual| per (cluster, qubit) — the per-role prediction-error
        signal competence.actuator_competence reads. The rate is the SAME live-learnable
        surprise_alpha the global self_tune uses (one law, two resolutions). A cluster whose
        qubit count changed (topology growth) restarts its EMA at the current magnitudes."""
        if not self.scales:
            return
        alpha = float(self.scales[0].params.get("surprise_alpha"))
        ema = self._role_surprise_ema
        for name, r in raw_residuals.items():
            mag = np.abs(np.asarray(r, dtype=float))
            prev = ema.get(name)
            if prev is None or len(prev) != len(mag):
                ema[name] = mag
            else:
                ema[name] = alpha * mag + (1.0 - alpha) * prev

    def role_surprise(self, cluster_name: str) -> NDArray | None:
        """The per-qubit |production residual| EMA for one cluster (index by role via the
        cluster's role_index), or None before the first step / for an unknown cluster."""
        return getattr(self, "_role_surprise_ema", {}).get(cluster_name)

    def _recurse(self, level: int, residuals: dict[str, np.ndarray]):
        """
        The recursive heart. Same function, every depth.

        Going in  (upward):  evolve this scale's field from residuals.
        Recurse:             pass this scale's own residuals deeper.
        Coming out (downward): project this scale's state as H onto parent.

        The call stack IS the fractal. Depth = recursion depth.
        """
        if level >= len(self.scales):
            return  # base case: no deeper scale exists (yet)

        scale = self.scales[level]

        # ── Timing gate: this scale only ticks at its stride ──
        parent_step = self._step if level == 0 else self.scales[level - 1]._step
        if parent_step == 0 or parent_step % scale.effective_stride != 0:
            # Not our tick — but still project H downward (state is valid)
            self._project_down(level)
            return

        # ── Upward: evolve this scale from parent's residuals ──
        scale.record_predictions()
        scale.field.step(residuals)
        scale.accumulate_berry()
        scale._step += 1
        child_residuals = scale.compute_residuals()
        scale.self_tune(child_residuals)

        # ── Recurse deeper ──
        self._recurse(level + 1, child_residuals)

        # ── Downward: project this scale's density matrix as H ──
        self._project_down(level)

    def _project_down(self, level: int):
        """Project scale[level]'s state as Hamiltonian onto the scale below.

        Berry velocity gates the projection: when the system is learning
        fast (high velocity), the learned H is still moving — attenuate
        it so we don't commit to a transient. When stable (low velocity),
        project at full strength.

        Gate function: 1 / (1 + |velocity| * sensitivity)
        This is a soft sigmoid: velocity=0 → gate=1.0, velocity=∞ → gate→0.
        """
        scale = self.scales[level]
        prod_clusters = self.production_field.clusters if level == 0 else None
        h_matrices = scale.project_hamiltonians(production_clusters=prod_clusters)
        target = (
            self.production_field if level == 0
            else self.scales[level - 1].field
        )

        # Berry velocity gate (only for projection to production field)
        if level == 0 and abs(self.berry_velocity) > 1e-10:
            sensitivity = 10.0  # how much velocity attenuates projection
            gate = 1.0 / (1.0 + abs(self.berry_velocity) * sensitivity)
            h_matrices = {
                name: _scale_projected_h(H, gate) for name, H in h_matrices.items()
            }

        for name, H in h_matrices.items():
            cluster = target.clusters.get(name)
            if cluster is not None:
                _apply_projected_h(cluster, H)

    def _project_tendrils(self, scale: "FractalScale") -> None:
        """Soft-observe actuator qubits from learned tendril CC weights.

        Actuator clusters use Lindblad dissipation (thermal targets) rather
        than unitary H — adding σx to their H is overridden immediately by the
        thermal channel. Instead, for each learned tendril CC, we nudge the
        target qubit toward the preferred pole via observe_qubit with a small
        alpha proportional to CC * z_src. This is the "soft preference signal":
        a source region's occupancy-like role high → small pull on a target
        actuator's state role toward +z.

        A node whose actuation is owned by a COMMITTED output tendril (the egress layer's
        slow-pump/decay coupling) must not ALSO be fast-pumped here — no two tendrils fighting
        one output. That ownership rule is enforced at the egress layer, which drives its
        nodes out of band; the field-side drive below applies to every declared tendril edge.
        """
        TENDRIL_GAIN = 1.0  # alpha per step at CC=1, z_src=1 — drives to pole in 1-2 steps vs dissipation
        for (src_name, tgt_name) in scale._tendril_keys:
            cbundle = scale._bridge_bundles.get((src_name, tgt_name))
            if cbundle is None:
                continue
            src_cluster = self.production_field.clusters.get(src_name)
            tgt_cluster = self.production_field.clusters.get(tgt_name)
            if src_cluster is None or tgt_cluster is None:
                continue
            for label, param in cbundle.params.items():
                if not label.startswith("CC_") or "→" not in label:
                    continue
                c = param.value
                if abs(c) < 1e-4:
                    continue
                src_role, tgt_role = label[3:].split("→", 1)
                src_idx = src_cluster.role_index.get(src_role)
                tgt_idx = tgt_cluster.role_index.get(tgt_role)
                if src_idx is None or tgt_idx is None:
                    continue
                src_z = float(src_cluster.qubit_bloch(src_idx)[2])
                if abs(src_z) < 0.30:
                    # source is equatorial (uncertain) — actively decohere actuator
                    alpha_eq = float(np.clip(abs(c) * 0.40, 0.0, 0.40))
                    tgt_cluster.observe_qubit(tgt_idx, (0.0, 0.0, 0.0), alpha_eq)
                    continue
                # alpha scales with coupling strength and source activation
                alpha = float(np.clip(abs(c) * abs(src_z) * TENDRIL_GAIN, 0.0, 0.4))
                # target bloch: +z pole if src positive, -z if src negative
                target_bloch = (0.0, 0.0, 1.0) if src_z > 0 else (0.0, 0.0, -1.0)
                tgt_cluster.observe_qubit(tgt_idx, target_bloch, alpha)

    # ================================================================
    # Dynamic depth — the stack breathes
    # ================================================================

    def _spawn_scale(self) -> int:
        """Append a deeper scale (φ-scaled config). Shared by the hard + soft spawn policies."""
        parent = self.scales[-1].config
        new_config = ScaleConfig(
            stride=round(parent.stride * PHI),
            dt=parent.dt * PHI,
            gamma=parent.gamma / PHI,
            bridge_strength=parent.bridge_strength / PHI,
            h_scale=parent.h_scale / PHI,
        )
        new_level = len(self.scales) + 1
        child = FractalScale(self.graph, new_config, level=new_level)
        # Born BUSY: seed the new scale's surprise EMA from its parent so the soft prune gives it the
        # benefit of the doubt — a fresh scale (EMA=0) would otherwise read as instantly "dormant" and
        # be trimmed the next tick, fighting the spawn. It must EARN dormancy by its residual decaying.
        child._surprise_ema = float(self.scales[-1]._surprise_ema)
        self.scales.append(child)
        return new_level

    def _maybe_spawn(self):
        """Grow depth from a LEARNED BELIEF about the marginal value of depth — the ONLY spawn path.
        No threshold, no patience counter, no ×1.05 futility hack, no fallback (Luke: a threshold
        means we aren't trusting the right soft qubit dynamic; fallbacks cover problems — a
        comprehension engine must comprehend, no wheelchairs). A leaky integrate-and-fire of
        (deepest surprise × marginal_value): pressure leaks each tick and accumulates surprise·m;
        firing (≥1) spawns a scale. After a spawn we OBSERVE the marginal-value qubit toward whether
        surprise actually dropped — so on irreducible surprise m→0, surprise·m leaks away, and it
        SELF-STOPS. Validated in spawn_dissolution.py (sandbox) + spawn_live_validation.py (real
        records: conservative, no spurious growth)."""
        if not self.scales or len(self.scales) >= self.MAX_DEPTH:
            return
        deepest = self.scales[-1]
        s = float(deepest._surprise_ema)
        # learn the marginal value of depth from whether the LAST spawn reduced surprise
        if self._surprise_at_spawn > 0.0:
            helped = 1.0 if (self._surprise_at_spawn - s) > 1e-4 else 0.0
            self._spawn_learner.observe(self.params.get_param("spawn_marginal_value"), helped,
                                        obs_sigma=float(self.params.get("spawn_obs_sigma")))
            self._surprise_at_spawn = 0.0
        m = float(self.params.get("spawn_marginal_value"))
        leak = float(self.params.get("spawn_leak"))
        self._spawn_pressure = self._spawn_pressure * leak + s * m   # leaky integrate
        if self._spawn_pressure >= 1.0:                              # fire → spawn
            self._spawn_pressure = 0.0
            self._surprise_at_spawn = s
            new_level = self._spawn_scale()
            logger.info(
                "FractalStack SPAWN: depth %d → %d, deepest surprise=%.4f, marginal_value=%.3f",
                new_level - 1, new_level, s, m,
            )

    def _production_surprise(self) -> float:
        """Mean |residual| across the production field this tick — the global surprise the stack is
        responsible for. Used to judge whether removing a scale HURT (its residual reappeared)."""
        if not self.last_raw_residuals:
            return 0.0
        return float(np.mean([abs(v) for r in self.last_raw_residuals.values() for v in r]))

    def _maybe_prune(self):
        """Shrink depth from the SAME learned belief that grows it — the dual of _maybe_spawn, no
        threshold, no patience counter, no ×0.8 conservative hack, no fallback. A leaky integrate-and-
        fire of dormancy·(1−m): `m` = spawn_marginal_value (the shared marginal value of depth), and
        dormancy = how little the deepest scale catches versus its parent (a scale-free residual share,
        no threshold). When depth is valuable (m→1) prune pressure vanishes; when depth has proven
        useless (m→0) AND the deepest scale is dormant, pressure accumulates and fires a prune. After a
        prune we OBSERVE m toward whether the prune HURT (production surprise rose → that depth was
        load-bearing → m→1, the old '×0.8 be-conservative' rule dissolved into the one law); so a prune
        that hurts raises m, drains the prune pressure, and SELF-STOPS. Symmetric to the spawn."""
        if len(self.scales) <= 1:
            return  # always keep at least one scale

        # learn the marginal value of depth from whether the LAST prune hurt (surprise jumped back)
        if self._surprise_at_prune > 0.0:
            prod = self._production_surprise()
            hurt = 1.0 if (prod - self._surprise_at_prune) > 1e-4 else 0.0
            self._spawn_learner.observe(self.params.get_param("spawn_marginal_value"), hurt,
                                        obs_sigma=float(self.params.get("spawn_obs_sigma")))
            self._surprise_at_prune = 0.0

        deepest = self.scales[-1]
        parent = self.scales[-2]
        ds = float(deepest._surprise_ema)
        ps = float(parent._surprise_ema)
        # The absolute surprise GAP = residual the parent catches that the deepest scale does NOT —
        # i.e. how redundant the deepest scale is, in the same units spawn uses (small on quiet/real
        # data, so prune fires as rarely + deliberately as spawn; symmetric, threshold-free). Two
        # scales catching similar residual → gap≈0 → never pruned; a dead child under a busy parent →
        # gap≈parent surprise → pressure builds. Weighted by (1−m): only trim when depth is useless.
        gap = max(0.0, ps - ds)
        m = float(self.params.get("spawn_marginal_value"))
        leak = float(self.params.get("spawn_leak"))
        self._prune_pressure = self._prune_pressure * leak + gap * (1.0 - m)   # leaky integrate
        if self._prune_pressure >= 1.0:                                        # fire → prune
            self._prune_pressure = 0.0
            self._surprise_at_prune = self._production_surprise()
            removed = self.scales.pop()
            logger.info(
                "FractalStack PRUNE: depth %d → %d (gap=%.4f, marginal_value=%.3f, "
                "deepest surprise=%.6f)",
                len(self.scales) + 1, len(self.scales), gap, m, float(removed._surprise_ema),
            )

    # ================================================================
    # Diagnostics
    # ================================================================

    def stats(self) -> dict:
        """Full fractal stack diagnostics."""
        production_h_norms = {}
        for name, cluster in self.production_field.clusters.items():
            # hamiltonian_norm() is substrate-neutral: the cumulant backend computes ‖H‖
            # from sparse couplings instead of materializing a 2ⁿ matrix (the merged manifold
            # cluster is 26 qubits → dense H_base would OOM).
            production_h_norms[name] = round(float(cluster.hamiltonian_norm()), 6)

        return {
            "enabled": self.config.enabled,
            "step": self._step,
            "n_scales": len(self.scales),
            "max_depth": self.MAX_DEPTH,
            "tick_interval": round(self.recommended_tick_interval, 1),
            "tick_surprise_ref": round(self.params.get("tick_surprise_ref"), 6),
            "spawn_pressure": round(self._spawn_pressure, 4),
            "prune_pressure": round(self._prune_pressure, 4),
            "spawn_marginal_value": round(self.params.get("spawn_marginal_value"), 4),
            "scales": [s.snapshot() for s in self.scales],
            "production_h_norms": production_h_norms,
            "mean_h_norm": round(
                float(np.mean(list(production_h_norms.values()))), 6
            ) if production_h_norms else 0.0,
            "scale_1_coefficients": (
                self.scales[0].projected_coefficients()
                if self.scales else {}
            ),
        }

    # ================================================================
    # Persistence
    # ================================================================

    def save_state(self) -> dict:
        """Serializable state for persistence."""
        return {
            "step": self._step,
            "prev_prod_z": self._prev_prod_z,
            "normalizer": self._normalizer.save_state(),
            # per-role production surprise EMA (b9.44) — reload-surviving so a redeployed
            # learned brain reads per-actuator skill immediately (matches learnedness).
            "role_surprise_ema": {k: v.tolist() for k, v in self._role_surprise_ema.items()},
            "scales": [
                {
                    "step": scale._step,
                    "density_matrices": {
                        name: cluster.rho.copy()
                        for name, cluster in scale.field.clusters.items()
                        if not getattr(cluster, "is_product", False)      # #309 param fiber: no joint rho
                        and not getattr(cluster, "is_cumulant", False)    # cumulant: stored separately
                    },
                    "cumulant_states": {
                        name: cluster.snapshot()
                        for name, cluster in scale.field.clusters.items()
                        if getattr(cluster, "is_cumulant", False)
                    },
                    "prev_z": scale._prev_z,
                    "params": scale.params.snapshot(),
                    "surprise_ema": scale._surprise_ema,
                    "h_bundles": {
                        name: bundle.snapshot()
                        for name, bundle in scale._h_bundles.items()
                    },
                    "bridge_bundles": {
                        f"{s}__{t}": bundle.snapshot()
                        for (s, t), bundle in scale._bridge_bundles.items()
                    },
                }
                for scale in self.scales
            ],
        }

    def load_state(self, data: dict):
        """Restore from saved state."""
        self._step = data.get("step", 0)
        self._prev_prod_z = data.get("prev_prod_z", {})

        norm_data = data.get("normalizer")
        if norm_data is not None:
            self._normalizer.load_state(norm_data)

        self._role_surprise_ema = {
            k: np.asarray(v, dtype=float)
            for k, v in (data.get("role_surprise_ema") or {}).items()
        }

        for i, scale in enumerate(self.scales):
            if i >= len(data.get("scales", [])):
                break
            sd = data["scales"][i]
            scale._step = sd.get("step", 0)
            scale._prev_z = sd.get("prev_z", {})
            scale._surprise_ema = sd.get("surprise_ema", 0.0)

            # One param stamp for every bundle below. Prefer the full-precision
            # fields (`value_exact`/`sigma_exact`) — the rounded display values
            # are lossy, and a 6-decimal round-off in a restored H coefficient
            # forks an otherwise deterministic replay (the 2026-07-18
            # lease-drill chain fork's second cause). Legacy checkpoints
            # without the exact fields fall back to the rounded ones.
            def _stamp(param, pdata):
                param.value = pdata.get("value_exact", pdata.get("value", param.value))
                param.sigma = pdata.get("sigma_exact", pdata.get("sigma", param.sigma))
                param.update_count = pdata.get("updates", 0)

            # Restore learnable params
            params_snap = sd.get("params", {})
            for name, pdata in params_snap.get("params", {}).items():
                param = scale.params.get_param(name)
                if param is not None:
                    _stamp(param, pdata)
            # (legacy "berry_phase" in old checkpoints is ignored — the real
            # geometric phase now lives on the Bloch trajectory, not bundles.)

            # Restore density matrices (dense clusters) + cumulant states
            for name, rho in sd.get("density_matrices", {}).items():
                cluster = scale.field.clusters.get(name)
                # dense only — product .rho raises, cumulant has none (mirrors save).
                if (cluster is None or getattr(cluster, "is_product", False)
                        or getattr(cluster, "is_cumulant", False)):
                    continue
                if rho.shape == cluster.rho.shape:
                    cluster.rho = rho
            for name, cstate in sd.get("cumulant_states", {}).items():
                cluster = scale.field.clusters.get(name)
                if cluster is not None and getattr(cluster, "is_cumulant", False):
                    cluster.load(cstate)

            # Restore learned H coefficients
            for name, bdata in sd.get("h_bundles", {}).items():
                bundle = scale._h_bundles.get(name)
                if bundle is not None:
                    for pname, pdata in bdata.get("params", {}).items():
                        param = bundle.get_param(pname)
                        if param is not None:
                            _stamp(param, pdata)

            # Restore bridge coupling coefficients
            for key_str, bdata in sd.get("bridge_bundles", {}).items():
                parts = key_str.split("__", 1)
                if len(parts) != 2:
                    continue
                src, tgt = parts
                bundle = scale._bridge_bundles.get((src, tgt))
                if bundle is None:
                    continue
                for pname, pdata in bdata.get("params", {}).items():
                    param = bundle.get_param(pname)
                    if param is not None:
                        _stamp(param, pdata)

        # Re-project H from loaded meta-field states (deterministic — no sampling)
        for i in range(len(self.scales) - 1, -1, -1):
            scale = self.scales[i]
            h_matrices = scale.project_hamiltonians(deterministic=True)
            if i == 0:
                for name, H in h_matrices.items():
                    cluster = self.production_field.clusters.get(name)
                    if cluster is not None:
                        _apply_projected_h(cluster, H)
            else:
                target_field = self.scales[i - 1].field
                for name, H in h_matrices.items():
                    cluster = target_field.clusters.get(name)
                    if cluster is not None:
                        _apply_projected_h(cluster, H)

        logger.info(
            "FractalStack restored: step=%d, %d scales",
            self._step, len(self.scales),
        )
