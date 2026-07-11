"""
Quantum Probability Field — the sky layer.

A continuous, always-evolving probability field over the world graph.
Each node with qubit roles gets a QubitCluster (density matrix).

The field evolves in three phases per timestep:

    1. INPUT:      Sensor data drives cluster evolution
    2. BRIDGE:     Lateral reconciliation of shared qubits
    3. PROJECTION: Bottom-up propagation from children to parents

The field never resets on collapse. It holds the system's beliefs —
continuous, uncertain, anticipatory. The classical world model
(world_model.py) holds the committed ground truth.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from umwelt.substrate.graph import WorldGraph
from umwelt.substrate.cluster import QubitCluster
from umwelt.substrate.fractal import fractal_signature, fractal_dimension_estimate
from umwelt.projection.emoji import cluster_emoji, purity_emoji, field_summary, correlation_emoji

logger = logging.getLogger(__name__)


class QuantumField:
    """
    The quantum probability field over the world graph.

    Manages QubitClusters for every node with qubit roles.
    Evolves continuously. Never resets on collapse.
    """

    def __init__(
        self,
        graph: WorldGraph,
        gamma: float = 0.05,
        dt: float = 0.01,
        bridge_strength: float = 0.5,
        cluster_filter: "Callable[[object], bool] | None" = None,
        cluster_backend: "str | None" = None,
    ):
        self.graph = graph
        self.bridge_strength = bridge_strength
        self.gamma = gamma
        self.dt = dt

        # Cluster backend (Stage-1 cumulant-math swap). Resolution:
        #   explicit cluster_backend param → UMWELT_CUMULANT env → "qubit" (default).
        # "cumulant" routes nodes to CumulantCluster (1-RDM+2-RDM, O(n²)); optional
        # UMWELT_CUMULANT_MIN_QUBITS targets only clusters at/above that qubit count
        # (small clusters are cheaper/exact as full-ρ). Default OFF → byte-identical
        # to today. Meta-scales pass cluster_backend="qubit" explicitly (the closure
        # is validated only in the dissipation-dominated production field).
        import os as _os
        backend = cluster_backend or (
            "cumulant" if _os.environ.get("UMWELT_CUMULANT") == "1" else "qubit")
        self._cluster_backend = backend
        self._cumulant_min_qubits = int(_os.environ.get("UMWELT_CUMULANT_MIN_QUBITS", "1"))

        # Create a QubitCluster for every node that has roles.
        # Per-node gamma/dt/gamma_diss from param_bundle overrides globals.
        # Values are live-read from bundles each step (not baked at init).
        #
        # `cluster_filter`, when given, restricts which nodes get a cluster.
        # The fractal meta-scales use it to OMIT simple device leaves from the
        # higher-order scales ("a closet light isn't tied to the 4th-order
        # meta-tower") — see fractal_stack.scale_participates. Bridges and
        # projections to/from an omitted cluster are silently skipped (both
        # reconcile_bridges and _propagate_projections already `.get()`-guard).
        # The production field is NEVER filtered — it holds every belief — so
        # feature_dim is unaffected and no readout reset is triggered.
        self.clusters: dict[str, QubitCluster] = {}
        self._node_map: dict[str, object] = {}  # name → WorldNode, for live param reads
        for node in graph.nodes_with_roles():
            if cluster_filter is not None and not cluster_filter(node):
                continue
            if getattr(node, "folded", False):
                # A folded device edge (MANIFOLD): its STATE roles live in the root manifold
                # ({name}_{role}), so it gets NO separate cluster (that's the whole point — no per-step
                # projection/data-transfer). The node stays in the graph for IDENTITY (bindings,
                # actuator dispatch, de-confounding graph_node) + param reads; state routes via
                # merged_zone_role(name, role) → (root, '{name}_{role}'). Register it, skip the cluster.
                self._node_map[node.name] = node
                continue
            node_gamma = gamma
            node_dt = dt
            node_gamma_diss: float | dict[str, float] = 5.0
            if node.param_bundle is not None:
                node_gamma = node.param_bundle.get("gamma", gamma)
                node_dt = node.param_bundle.get("dt", dt)
                # Build per-role gamma_diss dict from param bundle.
                # gamma_diss_{role} overrides the node-level gamma_diss.
                default_gd = node.param_bundle.get("gamma_diss", 5.0)
                role_gd = {"_default": default_gd}
                for role in node.roles:
                    key = f"gamma_diss_{role}"
                    if node.param_bundle.get_param(key) is not None:
                        role_gd[role] = node.param_bundle.get(key)
                node_gamma_diss = role_gd if len(role_gd) > 1 else default_gd
            Cls = QubitCluster
            node_backend = getattr(node, "cluster_backend", None)
            if node_backend == "cumulant" or (
                    self._cluster_backend == "cumulant"
                    and node_backend != "qubit"
                    and len(node.roles) >= self._cumulant_min_qubits):
                from umwelt.substrate.cumulant_cluster import CumulantCluster
                Cls = CumulantCluster
            ckw = {}
            if getattr(node, "connectivity", None) is not None:
                ckw["connectivity"] = node.connectivity   # sparse ZZ for big merged clusters
            self.clusters[node.name] = Cls(
                zone_name=node.name,
                qubit_roles=node.roles,
                gamma=node_gamma,
                dt=node_dt,
                gamma_diss=node_gamma_diss,
                role_modes=getattr(node, "role_modes", None),
                **ckw,
            )
            self._node_map[node.name] = node

        self._step_count = 0

        # WIDE batched evolve hot path — THE path (opt-out flag deleted b9.64).
        # Phase-1 evolution stacks same-dim clusters into (B,d,d) arrays and does
        # one batched RK4 step per dim-group instead of a per-cluster matmul loop
        # — collapses the per-cluster BLAS dispatch the A55 pays dearly for. Math
        # is bit-equivalent to the loop (experiments/bench_batched_evolve.py, 2.5e-7).
        # A/B-measured live on the RDK: field.step 375→289 ms (1.30×). The
        # UMWELT_BATCHED_EVOLVE opt-out DELETED b9.64 (flag reckoning): the ledger
        # marked it dense-path-only/deletable since the C1 cutover — batched IS the path.
        # See umwelt/substrate/batched_evolve.py and the origin deployment's math-refactor notes.
        self._clusters_by_dim: dict[int, list] | None = None

        logger.info(
            "QuantumField: %d clusters, %d bridges [batched-evolve]",
            len(self.clusters), len(graph.bridges),
        )

    def _group_by_dim(self) -> dict[int, list]:
        """Same-dim QubitClusters grouped for the batched (B,d,d) path. Built once.
        CumulantClusters AND product fiber clusters are EXCLUDED (no dense ρ to stack:
        cumulant has none, product .rho is 2^N and raises) — they take the per-cluster
        loop in step()."""
        if self._clusters_by_dim is None:
            groups: dict[int, list] = {}
            for cluster in self.clusters.values():
                if getattr(cluster, "is_cumulant", False) or getattr(cluster, "is_product", False):
                    continue
                groups.setdefault(cluster.dim, []).append(cluster)
            self._clusters_by_dim = groups
        return self._clusters_by_dim

    # ================================================================
    # Hamiltonian injection
    # ================================================================

    def apply_hamiltonian(self, h_specs: dict):
        """Apply learned Hamiltonians to clusters.

        Args:
            h_specs: {cluster_name: HamiltonianSpec} — each spec's .build()
                     produces the Hermitian H matrix for that cluster.
        """
        for name, spec in h_specs.items():
            cluster = self.clusters.get(name)
            if cluster is not None:
                cluster.set_hamiltonian(spec.build())

    # ================================================================
    # Evolution
    # ================================================================

    def step(self, inputs: dict[str, NDArray[np.floating]] | None = None,
             dt_scale: float = 1.0):
        """
        Evolve the field by one timestep × dt_scale.

        Args:
            inputs: node_name -> input array (one value per qubit role).
                    Missing nodes get zero input.
            dt_scale: smooth-clock catch-up multiplier (default 1.0 = unchanged); >1
                    advances proportionally more simulated time after skipped calm ticks.
        """
        inputs = inputs or {}

        # Phase 0: Live-read dynamic params from bundles.
        # This is the heartbeat — calibration, training, and population
        # update bundle values; this step propagates them to the physics.
        self._sync_params()

        # Phase 1: Evolve each cluster with its inputs.
        # Build the full per-cluster input map (missing → zeros, the thermal-
        # toward-v=0 drive) once, so the loop and the batched path see identical
        # inputs and stay bit-equivalent.
        step_inputs = {}
        for name, cluster in self.clusters.items():
            inp = inputs.get(name)
            step_inputs[name] = inp if inp is not None else np.zeros(cluster.n_qubits)

        from umwelt.substrate.batched_evolve import evolve_groups
        # The evolve backend is selectable per-context: the LIVE field uses NumpyBackend (None →
        # default, responsive on the CPU); a FORECAST rollout or dream can set `_evolve_backend` to a
        # number-system backend (the BPU-native expansion, eventually the .bin) for the latency-
        # tolerant offload. getattr-default-None keeps the live path byte-identical to today.
        evolve_groups(self._group_by_dim(), step_inputs, dt_scale=dt_scale,
                      backend=getattr(self, "_evolve_backend", None))
        # CumulantClusters + the product param fiber can't be stacked into (B,d,d) — step
        # them directly (matches the non-batched loop; product evolves the fiber qubits #309).
        for name, cluster in self.clusters.items():
            if getattr(cluster, "is_cumulant", False) or getattr(cluster, "is_product", False):
                cluster.step(step_inputs[name], dt_scale=dt_scale)

        # Phase 2: Lateral bridge reconciliation
        if self.bridge_strength > 0:
            self._reconcile_bridges()

        # Phase 3: Bottom-up projection (children -> parents)
        self._propagate_projections()

        # Phase 4: Enforce physicality periodically (eigendecomp is expensive)
        phys_interval = 10
        root = self.graph.root
        if root.param_bundle is not None:
            phys_interval = int(root.param_bundle.get("physicality_interval", 10))
        if phys_interval > 0 and self._step_count % phys_interval == 0:
            self._enforce_physicality()

        self._step_count += 1

    def _sync_params(self):
        """Live-read dynamic parameters from world graph param bundles.

        This closes the loop: calibration/training write to bundles,
        _sync_params reads them back into the physics each step.
        Without this, bundle updates are invisible to the actual evolution.
        """
        root = self.graph.root

        # Global bridge_strength from root bundle
        if root.param_bundle is not None:
            self.bridge_strength = root.param_bundle.get(
                "bridge_strength", self.bridge_strength
            )

        # Per-cluster gamma, gamma_diss (per-role), and dt from node bundles
        for name, cluster in self.clusters.items():
            node = self._node_map.get(name)
            if node is None or node.param_bundle is None:
                continue
            cluster.evolver.gamma = node.param_bundle.get(
                "gamma", cluster.evolver.gamma
            )
            cluster.evolver.dt = node.param_bundle.get(
                "dt", cluster.evolver.dt
            )
            # Per-role gamma_diss: reads gamma_diss_{role} for each
            # dissipative qubit, falling back to node-level gamma_diss.
            cluster.sync_gamma_diss(node.param_bundle)

    def _reconcile_bridges(self):
        """Reconcile shared qubits between bridged nodes."""
        alpha = self.bridge_strength

        for bridge in self.graph.bridges:
            if bridge.is_tendril:
                continue  # directed coupling — CC bundles handle it, not symmetric nudge
            cluster_a = self.clusters.get(bridge.source)
            cluster_b = self.clusters.get(bridge.target)
            if cluster_a is None or cluster_b is None:
                continue

            for role in bridge.shared_roles:
                if role not in cluster_a.role_index:
                    continue
                if role not in cluster_b.role_index:
                    continue

                idx_a = cluster_a.role_index[role]
                idx_b = cluster_b.role_index[role]

                rdm_a = cluster_a.qubit_rdm(idx_a)
                rdm_b = cluster_b.qubit_rdm(idx_b)

                # Symmetric average as reconciliation target
                rdm_avg = 0.5 * (rdm_a + rdm_b)
                tr = np.trace(rdm_avg)
                if abs(tr) > 1e-15:
                    rdm_avg /= tr

                # Nudge strength scaled by connection type
                nudge_alpha = alpha * bridge.coupling_base
                _nudge_qubit(cluster_a, idx_a, rdm_avg, nudge_alpha)
                _nudge_qubit(cluster_b, idx_b, rdm_avg, nudge_alpha)

    def _propagate_projections(self):
        """Bottom-up: child qubit states project onto parent roles."""
        # Process deepest nodes first
        nodes = sorted(self.graph.nodes_with_roles(), key=lambda n: -n.depth)

        for node in nodes:
            if node.projection is None or node.parent is None:
                continue

            child_cluster = self.clusters.get(node.name)
            parent_cluster = self.clusters.get(node.parent.name)
            if child_cluster is None or parent_cluster is None:
                continue

            # Projection coupling, honoring the hierarchy: per-node override →
            # root bundle (the live, learnable global default) → 0.3 constant.
            # configure_param_bundles registers projection_coupling on root only,
            # so without the root fallback the registered param is dead and every
            # projecting node silently uses the 0.3 constant.
            proj_factor = 0.3
            root = getattr(self.graph, "root", None)
            if root is not None and root.param_bundle is not None:
                proj_factor = root.param_bundle.get("projection_coupling", proj_factor)
            if node.param_bundle is not None:
                proj_factor = node.param_bundle.get("projection_coupling", proj_factor)

            for child_role, parent_role in node.projection.items():
                child_idx = child_cluster.role_index.get(child_role)
                parent_idx = parent_cluster.role_index.get(parent_role)
                if child_idx is None or parent_idx is None:
                    continue

                child_rdm = child_cluster.qubit_rdm(child_idx)
                _nudge_qubit(
                    parent_cluster, parent_idx, child_rdm,
                    alpha=self.bridge_strength * proj_factor,
                )

    # ================================================================
    # Readout
    # ================================================================

    def global_features(self) -> NDArray[np.floating]:
        """Concatenated feature vector from all clusters."""
        parts = [cluster.features() for cluster in self.clusters.values()]
        return np.concatenate(parts)

    def zone_features(self, zone_name: str) -> NDArray[np.floating] | None:
        """Feature vector for a specific node (region)."""
        cluster = self.clusters.get(zone_name)
        return cluster.features() if cluster else None

    def fractal_features(self) -> dict[str, dict[int, NDArray[np.floating]]]:
        """Per-node fractal decomposition.

        Explicitly requests the full level-3 decomposition (independent of each
        cluster's readout `max_feature_level` cap) so this diagnostic surface
        keeps its complete fractal view — the readout-feature pruning shouldn't
        silently shrink the dimension/signature analysis.
        """
        return {
            name: cluster.features_by_level(max_level=3)
            for name, cluster in self.clusters.items()
        }

    def global_fractal_signature(self) -> dict[str, dict[int, float]]:
        """Per-node fractal signatures (energy at each correlation level)."""
        result = {}
        for name, cluster in self.clusters.items():
            levels = cluster.features_by_level(max_level=3)
            result[name] = fractal_signature(levels)
        return result

    def bridge_correlations(self) -> dict[tuple[str, str], float]:
        """Correlation strength across each bridge."""
        result = {}
        for bridge in self.graph.bridges:
            ca = self.clusters.get(bridge.source)
            cb = self.clusters.get(bridge.target)
            if ca is None or cb is None:
                continue

            similarities = []
            for role in bridge.shared_roles:
                if role not in ca.role_index or role not in cb.role_index:
                    continue
                bloch_a = ca.role_bloch(role)
                bloch_b = cb.role_bloch(role)
                norm_a = np.linalg.norm(bloch_a)
                norm_b = np.linalg.norm(bloch_b)
                if norm_a > 1e-10 and norm_b > 1e-10:
                    sim = float(np.dot(bloch_a, bloch_b) / (norm_a * norm_b))
                else:
                    sim = 0.0
                similarities.append(sim)

            result[(bridge.source, bridge.target)] = (
                float(np.mean(similarities)) if similarities else 0.0
            )
        return result

    # ================================================================
    # Visualization
    # ================================================================

    def emoji_state(self) -> str:
        """Emoji summary of the full field state."""
        zone_emojis = {}
        zone_purities = {}
        for name, cluster in self.clusters.items():
            zone_emojis[name] = cluster_emoji(cluster.all_bloch())
            zone_purities[name] = cluster.purity

        bridge_strengths = self.bridge_correlations()
        return field_summary(zone_emojis, zone_purities, bridge_strengths)

    def status(self) -> dict:
        """Full status dict for API/debugging."""
        fractal_sigs = self.global_fractal_signature()
        return {
            "step": self._step_count,
            "nodes": {
                name: {
                    "purity": cluster.purity,
                    "entropy": cluster.entropy,
                    "n_qubits": cluster.n_qubits,
                    "roles": cluster.qubit_roles,
                    "bloch": {
                        role: cluster.qubit_bloch(i).tolist()
                        for role, i in cluster.role_index.items()
                    },
                    "fractal_signature": fractal_sigs.get(name, {}),
                    "fractal_dimension": fractal_dimension_estimate(
                        fractal_sigs.get(name, {})
                    ),
                }
                for name, cluster in self.clusters.items()
            },
            "bridges": {
                f"{a}\u2194{b}": strength
                for (a, b), strength in self.bridge_correlations().items()
            },
        }

    def _enforce_physicality(self):
        """
        Ensure all clusters remain physical after nudging.

        Each backend re-projects in its own representation (substrate.py): the dense
        backend does the Hermitian + PSD-eigenvalue-clamp + renormalize, the cumulant
        backend a cheap 1-RDM Bloch-radius clamp, the product fiber a no-op. The field
        no longer reaches into a dense `.rho` here.
        """
        for cluster in self.clusters.values():
            cluster.clamp_physical()

    def reset(self):
        """Reset all clusters to |0...0>."""
        for cluster in self.clusters.values():
            cluster.reset()
        self._step_count = 0


# ============================================================================
# Internal helpers
# ============================================================================


def _nudge_qubit(
    cluster,
    qubit_idx: int,
    target_rdm: NDArray,
    alpha: float,
):
    """Nudge a single qubit in a cluster toward a target reduced state — the fiber
    connection that keeps shared qubits consistent across clusters.

    Thin shim over the backend's `nudge_toward_rdm` (substrate.py): the dense kron
    correction, the cumulant Bloch→observe, and the product-fiber no-op now live with
    their respective backends, so the field is substrate-neutral here. Kept as a free
    function so the bridge/projection call sites read unchanged."""
    cluster.nudge_toward_rdm(qubit_idx, target_rdm, alpha)
