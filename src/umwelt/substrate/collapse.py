"""
Selective Collapse — the bridge between sky and ground.

The quantum field (sky) evolves continuously, holding beliefs
as probability amplitudes. The classical world model (ground)
holds definite commitments.

Collapse projects the field onto the ground. It can be triggered by:

    PERIODIC    -- regular interval check (simplest policy)
    ACTION      -- the system needs to decide something
    QUERY       -- someone asked about the state
    MEASUREMENT -- a sensor gave a definitive reading
    CONFIDENCE  -- the field became very certain on its own

Collapse is selective: only the relevant qubits/nodes get projected.
The field keeps evolving -- it never resets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from umwelt.spec.roles import is_analog_role

logger = logging.getLogger(__name__)


class CollapseReason(str, Enum):
    """Why a collapse was triggered."""
    PERIODIC = "periodic"
    ACTION = "action"
    QUERY = "query"
    MEASUREMENT = "measurement"
    CONFIDENCE = "confidence"


@dataclass
class CollapsePolicy:
    """Configures automatic collapse behavior."""
    periodic_interval: int = 10         # 0 = disabled
    confidence_threshold: float = 0.9   # auto-collapse when |z| exceeds this
    measurement_collapse: bool = True    # sensor events force collapse
    # A background projection near zero is only orbital ringing if it is ALSO
    # not moving. transition_floor is the |z| below which a role counts as
    # "small"; motion_eps is the per-collapse |Δz| below which it counts as
    # "static". A role is suppressed only when small AND static — a genuine
    # swing through zero (small |z|, large |Δz|) is kept. Both are priors,
    # overridable from the root node's param_bundle so they calibrate rather
    # than stay asserted; see configure_param_bundles().
    transition_floor: float = 0.3
    motion_eps: float = 0.02


class CollapseEngine:
    """
    Projects quantum field state onto the classical world model.

    Measures sigma_z for specified qubits, applies hysteresis,
    and updates the world model with committed values.
    """

    def __init__(
        self,
        policy: CollapsePolicy | None = None,
        hysteresis: float = 0.1,
    ):
        self.policy = policy or CollapsePolicy()
        self.hysteresis = hysteresis
        self._prev_z: dict[tuple[str, str], float] = {}
        # (node, role) pairs that have ever been fed by a sensor. When set,
        # periodic/auto collapse is suppressed for any pair not in this set —
        # orphan qubit roles free-evolve and their zero-crossings are not
        # commitments. Populated by the reservoir from SensorBridge.touched_roles.
        self.touched_roles: set[tuple[str, str]] | None = None

    def collapse_node(
        self,
        field,                                          # QuantumField
        world,                                          # ClassicalWorldModel
        node_name: str,
        reason: CollapseReason = CollapseReason.QUERY,
        roles: list[str] | None = None,
    ) -> list:
        """
        Collapse specific roles of a specific node.

        Per-node hysteresis can be overridden via param_bundle on the
        corresponding WorldGraph node (key: "hysteresis").

        Returns list of Transitions that occurred.
        """
        cluster = field.clusters.get(node_name)
        if cluster is None:
            return []

        # Per-node hysteresis from param_bundle, or fall back to engine default
        node_hysteresis = self.hysteresis
        graph_node = field.graph.find(node_name)
        if graph_node is not None and graph_node.param_bundle is not None:
            node_hysteresis = graph_node.param_bundle.get(
                "hysteresis", self.hysteresis
            )

        target_roles = roles or list(cluster.role_index.keys())

        # Background-collapse modes apply two extra gates to suppress
        # orbital-noise transitions on roles that aren't actually being
        # measured. Explicit QUERY/ACTION/MEASUREMENT collapses bypass these
        # gates — when something asks, we answer with what the field says.
        background = reason in (CollapseReason.PERIODIC, CollapseReason.CONFIDENCE)
        # Movement gate thresholds. Learnable priors from the root bundle so
        # they self-calibrate (no magic constants); fall back to policy.
        floor = self.policy.transition_floor if background else 0.0
        motion_eps = self.policy.motion_eps
        root = getattr(field.graph, "root", None)
        if background and root is not None and root.param_bundle is not None:
            floor = root.param_bundle.get("transition_floor", floor)
            motion_eps = root.param_bundle.get("motion_eps", motion_eps)

        for role in target_roles:
            idx = cluster.role_index.get(role)
            if idx is None:
                continue

            # Orphan-role gate: in background modes, skip roles that have
            # never received a sensor input. Free-evolving qubits cross zero
            # on reservoir orbital wobble, and projecting that as a commitment
            # is deceptive.
            if (
                background
                and self.touched_roles is not None
                and (node_name, role) not in self.touched_roles
            ):
                continue

            bloch = cluster.qubit_bloch(idx)
            x, y, z = float(bloch[0]), float(bloch[1]), float(bloch[2])

            # Analog roles (periodic drivers, temperature, ...) are continuous: commit
            # the real position, never a ±1 pole. world.update emits no
            # transition for these (analog set), so the collapse log stays an
            # event journal instead of churning on every horizon crossing. We
            # skip the movement/hysteresis gates — a tracked position has no
            # "orbital ringing" to suppress; we always store where it is.
            if is_analog_role(role):
                purity = (x * x + y * y + z * z) ** 0.5
                world.update(
                    node_name, role, 1 if z >= 0 else -1, purity,
                    field._step_count, reason.value,
                    analog=z, bloch=(x, y, z),
                )
                self._prev_z[(node_name, role)] = z
                continue

            confidence = abs(z)

            prev_z = self._prev_z.get((node_name, role), 0.0)
            prev_val = 1 if prev_z >= 0 else -1

            # Movement gate: |z| below the floor is orbital ringing ONLY if it
            # is also static. A small |z| that moved a lot since the last
            # commit is a genuine swing through zero — keep it. Don't commit or
            # update prev_z on a true skip, so the hysteresis baseline doesn't
            # drift on noise.
            if background and confidence < floor and abs(z - prev_z) < motion_eps:
                continue

            # Hysteresis: keep previous value if change is small

            if abs(z - prev_z) < node_hysteresis:
                value = prev_val
            else:
                value = 1 if z >= 0 else -1

            world.update(
                node_name, role, value, confidence,
                field._step_count, reason.value,
            )
            self._prev_z[(node_name, role)] = z

        return world.pop_transitions()

    def collapse_all(
        self,
        field,
        world,
        reason: CollapseReason = CollapseReason.PERIODIC,
    ) -> list:
        """Collapse all nodes in the field."""
        all_transitions = []
        for name in field.clusters:
            t = self.collapse_node(field, world, name, reason)
            all_transitions.extend(t)
        return all_transitions

    def auto_collapse(self, field, world) -> list:
        """Confidence-triggered: collapse qubits that are very certain.

        Per-node confidence_threshold can be set via param_bundle
        (key: "confidence_threshold"), falling back to policy default.
        """
        default_threshold = self.policy.confidence_threshold
        transitions = []

        for name, cluster in field.clusters.items():
            # Per-node threshold from param_bundle
            node_threshold = default_threshold
            graph_node = field.graph.find(name)
            if graph_node is not None and graph_node.param_bundle is not None:
                node_threshold = graph_node.param_bundle.get(
                    "confidence_threshold", default_threshold
                )

            for role, idx in cluster.role_index.items():
                bloch = cluster.qubit_bloch(idx)
                z = float(bloch[2])
                if abs(z) > node_threshold:
                    t = self.collapse_node(
                        field, world, name,
                        CollapseReason.CONFIDENCE, roles=[role],
                    )
                    transitions.extend(t)

        return transitions

    def check_periodic(self, field, world) -> list:
        """Check if periodic collapse is due, and do it if so."""
        interval = self.policy.periodic_interval
        if interval <= 0:
            return []
        if field._step_count % interval != 0:
            return []
        return self.collapse_all(field, world, CollapseReason.PERIODIC)
