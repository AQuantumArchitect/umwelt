"""Ingress membrane — maps external signals to field inputs.

Each signal targets a specific node in the world graph and drives a specific qubit
role. The bridge normalizes raw values to [-1, +1] and routes them to the correct
cluster input channel:

    external signal -> normalize -> node.role -> input operator -> density matrix

Signal registration is dynamic (add signals at runtime as they're discovered or
configured) and declarative (apply_spec_bindings applies a DomainSpec's bindings).
The bridge handles heterogeneous node roles — different nodes can have different
qubit axes. Role classification (unitary/dissipative/observe/driver/analog) is
registry data in umwelt.spec.roles; normalizers are registry data in
umwelt.spec.normalizers. The engine hard-codes NO domain vocabulary here.
"""
from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import NDArray

from umwelt.spec.roles import is_driver_role, is_observe_role, role_input_mode
from umwelt.spec.schema import match_ignored

if TYPE_CHECKING:  # annotations only (PEP 563) — deferring these keeps ingress
    # importable standalone (ScalarParam is imported function-locally where constructed).
    from umwelt.spec.schema import DomainSpec
    from umwelt.substrate.graph import WorldGraph
    from umwelt.substrate.params import ParameterBundle, ScalarParam

logger = logging.getLogger(__name__)

# The CONFIDENCE BRAKE on collapse strength: alpha = collapse_alpha × conf**gamma. gamma=1 is the
# historical LINEAR brake (a low-confidence reading rotates the belief less). Measured in the origin
# deployment: the linear brake HURTS tracking at a live collapse_alpha (~0.15) — the base rate
# already smooths noise, so braking just adds lag — and only helps near full-snap collapse. So the
# exponent is not a hardcoded 1: gamma=0 ignores confidence (no brake), gamma>1 distrusts it harder.
# Default 1.0 = bit-exact with the historical behaviour; flip via UMWELT_CONF_BRAKE_GAMMA to A/B live.
CONF_BRAKE_GAMMA = float(os.environ.get("UMWELT_CONF_BRAKE_GAMMA", "1.0"))

# The LOW-PASS scaffolding: observations partial-collapse at `binding.collapse_alpha` — a
# per-observation EMA that smooths the belief. UMWELT_COLLAPSE_ALPHA, if set, OVERRIDES every
# binding so the low-pass can be stripped (→1.0 = snap, no smoothing) to test whether the field
# is sticky on its OWN — output hysteresis + Lindblad dissipation should hold it. Unset = default.
_COLLAPSE_ALPHA_OVERRIDE = os.environ.get("UMWELT_COLLAPSE_ALPHA")


def effective_collapse_alpha(binding_alpha: float) -> float:
    """The per-observation collapse rate, with the strip-the-low-pass override applied if set."""
    return float(_COLLAPSE_ALPHA_OVERRIDE) if _COLLAPSE_ALPHA_OVERRIDE else binding_alpha


def conf_brake(conf: float, gamma: float | None = None) -> float:
    """The confidence factor on collapse alpha. conf<=0 → 0 (invalid reading is always a no-op);
    gamma=1 → conf (exact passthrough, the historical linear brake); gamma=0 → 1 (ignore confidence
    magnitude for any valid reading); else conf**gamma."""
    if conf <= 0.0:
        return 0.0
    g = CONF_BRAKE_GAMMA if gamma is None else gamma
    return conf if g == 1.0 else float(conf ** g)


# ── Role → event-type classification (journal stamps) ───────────────────────────────
# Registry, empty by default: a domain declares which roles stamp which event types on
# the journal/tape. Unregistered roles stamp the neutral "sensor".
_ROLE_EVENT_TYPE: dict[str, str] = {}


def register_event_type(role: str, event_type: str) -> None:
    """Declare the journal event-type a role's readings stamp (e.g. a contact-like role
    → "contact"). Domain vocabulary is data, never engine code."""
    _ROLE_EVENT_TYPE[role] = event_type


# ── Signal-type presets ──────────────────────────────────────────────────────────────
# Registry, empty by default: a domain may register shorthand presets so register() can
# infer (qubit_role, normalize) from a `sensor_type` string. Declarative specs
# (apply_spec_bindings) don't need presets — this is a convenience for imperative wiring.
SENSOR_PRESETS: dict[str, dict] = {}


def register_sensor_preset(name: str, *, qubit_role: str,
                           normalize: Callable[[float], float]) -> None:
    """Register a signal-type preset for SensorBridge.register(sensor_type=...)."""
    SENSOR_PRESETS[name] = {"qubit_role": qubit_role, "normalize": normalize}


# ── Live-calibrating normalizer ──────────────────────────────────────────────────────

def param_range_norm(
    lo_param: "ScalarParam",
    hi_param: "ScalarParam",
    auto_calibrate: bool = True,
    calibrate_sigma: float = 5.0,
) -> Callable[[float], float]:
    """Create a normalizer backed by learnable ScalarParam bounds.

    If auto_calibrate is True, observations that fall outside the current
    [lo, hi] range trigger a Kalman update on the bound, widening the
    range to accommodate the new data.
    """
    def _norm(val: float) -> float:
        lo = lo_param.value
        hi = hi_param.value
        if auto_calibrate:
            if val < lo:
                lo_param.kalman_update(val, calibrate_sigma)
            if val > hi:
                hi_param.kalman_update(val, calibrate_sigma)
            lo = lo_param.value
            hi = hi_param.value
        if hi == lo:
            return 0.0
        clamped = max(lo, min(hi, val))
        return 2.0 * (clamped - lo) / (hi - lo) - 1.0
    return _norm


# Map qubit roles to param-bundle key prefixes for upgrade_to_param_norms — registry,
# empty by default. A domain registers e.g. its environment role → ("temp", "humidity")
# so static range normalizers get promoted to live param_range_norm over
# sensor_{prefix}_lo/hi fiber params.
ROLE_PARAM_MAP: dict[str, list[str]] = {}


def register_range_param_prefixes(role: str, *prefixes: str) -> None:
    """Declare which sensor_{prefix}_lo/hi param pairs back a role's normalizer."""
    ROLE_PARAM_MAP.setdefault(role, []).extend(prefixes)


@dataclass
class SensorBinding:
    """Maps a signal to a qubit input channel."""
    sensor_id: str
    node: str            # target node name
    qubit_role: str
    normalize: Callable[[float], float] = None
    weight: float = 1.0
    description: str = ""
    # Display / semantic metadata — mirrors the output-binding shape for parity
    kind:  str = ""      # sensor_type preset key preserved at registration
    label: str = ""      # human name for UI/LLMs; falls back to description
    weight_param: "ScalarParam | None" = None  # learnable weight (created by upgrade_weights)
    event_type_override: str | None = None  # per-binding override of the role→type map
    # Per-stream DARK ttl for the health spine. None → resolved by role at read time
    # (change-driven roles day-scale, continuous roles the loop default). Measured lesson
    # from the origin deployment: _last_raw_ts sits downstream of the noise gates, so a
    # switch nobody flips is silent for days while perfectly healthy — heartbeat-scale
    # ttls would cry wolf.
    dark_ttl_s: float | None = None
    # Observe-collapse params (only meaningful for observe roles). r_obs is the
    # magnitude |r| the belief collapses to on observation (1 = pure eigenstate,
    # <1 = partial collapse / how much we trust the reading). collapse_alpha is
    # the per-observation mixing strength (1 = full snap, <1 = nudge).
    # Defaults assume a DIRECTLY observed device (a switch reporting its own
    # state) → trust it: snap hard (alpha=1) to near-pure (r_obs=0.95).
    r_obs: float = 0.95
    collapse_alpha: float = 1.0
    force_observe: bool = False  # route through observe path even for non-observe roles

    @property
    def is_event_driven(self) -> bool:
        """True for signals that fire on state transitions. Generalized from the origin
        deployment's hand-listed role set: a unitary-channel role IS event-driven by
        definition (see umwelt.spec.roles) — it kicks on events and free-evolves between
        them; dissipative roles are continuously driven."""
        return role_input_mode(self.qubit_role) == "unitary"

    @property
    def is_observe(self) -> bool:
        """True for belief qubits driven by observation/partial-collapse."""
        return self.force_observe or is_observe_role(self.qubit_role)

    @property
    def event_type(self) -> str:
        """Classification for journal stamps. Per-binding override wins; otherwise
        infer from qubit_role via the register_event_type registry."""
        if self.event_type_override is not None:
            return self.event_type_override
        return _ROLE_EVENT_TYPE.get(self.qubit_role, "sensor")


class SensorBridge:
    """
    Routes signal readings to field input channels.

    Works with the world graph to handle heterogeneous node roles.
    Each node can have different qubit axes — the bridge knows which
    roles exist on each node and routes accordingly.

    Usage:
        bridge = SensorBridge(graph)
        bridge.register("hall_motion", zone="hall", qubit_role="occupancy",
                        normalize=binary_norm)
        inputs = bridge.process({"hall_motion": 1})
        # -> {"hall": np.array([+1.0, 0.0, 0.0, 0.0, 0.0])}
    """

    def __init__(self, graph: "WorldGraph"):
        self.graph = graph
        self.bindings: dict[str, SensorBinding] = {}
        # Signal/driver trust-weight learning is the IMMEDIATE gradient (Δw = lr·input·residual),
        # routed through the ONE learner object (gradient_step = the eligibility law with no trace,
        # credit being immediate). See umwelt.learning.universal_learner.
        from umwelt.learning.universal_learner import UniversalLearner
        self._learner = UniversalLearner()

        # Build per-node role info from the graph
        self._node_info: dict[str, dict] = {}
        self._node_params: dict[str, "ParameterBundle | None"] = {}
        for node in graph.nodes_with_roles():
            self._node_info[node.name] = {
                "roles": node.roles,
                "role_index": {r: i for i, r in enumerate(node.roles)},
                "n_roles": len(node.roles),
            }
            self._node_params[node.name] = node.param_bundle

        # Last pre-weight normalized input per signal — used for weight learning.
        # Stores what the signal "said" independent of its current weight.
        self._last_normed: dict[str, float] = {}

        # Last raw (un-normalized) value per signal — used for dashboards.
        self._last_raw: dict[str, float] = {}

        # Wall-clock timestamp (time.time()) of the last raw update per signal.
        # Used by app surfaces to compute readback freshness (dim stale state).
        self._last_raw_ts: dict[str, float] = {}

        # Last raw value that was actually FED to the field for event-driven
        # signals. Used to drop heartbeat republishes (some devices emit the
        # same state on a fixed cadence even when nothing changed). Continuous /
        # dissipative roles bypass this — thermalization wants the steady-
        # state value every tick.
        self._last_event_value: dict[str, float] = {}

        # UNMATCHED signals — readings whose sensor_id has NO binding (foreign / mislabelled /
        # not-yet-wired). Dropping them silently would leave a foreign-domain ingest with zero
        # feedback about what arrived and bound to nothing. Track {sensor_id: {count, last_value,
        # last_ts}}, bounded — so a dev (and, later, an auto-adapt organ) can SEE the gap between
        # the data's vocabulary and the engine's bindings. See unmatched_snapshot().
        self._unmatched: "OrderedDict[str, dict]" = OrderedDict()
        self._unmatched_total: int = 0
        self._ignored: list = []     # (pattern, reason) — deliberately-unbound, declared by the spec

        # (node, role) pairs that have ever received a signal input. Used by
        # the collapse engine to suppress orbital-noise collapses on roles
        # with no grounded data source.
        self.touched_roles: set[tuple[str, str]] = set()

        # Per-leaf TRUST WEB fusion (foresight): when attached, observe_targets FUSES every input to a
        # leaf (redundant signals + forecast brains) through one learned conditional-trust web per
        # (node, role) instead of last-wins. None = off → last-wins behaviour, byte-unchanged. Opt-in
        # exactly as the origin gated it: UMWELT_TRUST_WEB auto-attaches at construction (UMWELT_TRUST_WEB_LEARN
        # enables conservative consensus-grounded online learning; UMWELT_TRUST_QUBIT → qubit-backed reliability).
        import os
        self.trust_webs: "dict[tuple[str, str], object] | None" = None
        self._trust_learn: bool = False
        self._trust_qubit: bool = bool(os.environ.get("UMWELT_TRUST_QUBIT"))
        if os.environ.get("UMWELT_TRUST_WEB"):
            self.attach_trust_web(learn=bool(os.environ.get("UMWELT_TRUST_WEB_LEARN")))

    def attach_trust_web(self, *, learn: bool = False, qubit: bool | None = None) -> None:
        """Turn on per-leaf trust-web fusion in observe_targets. `learn` also enables conservative
        consensus-grounded online learning. `qubit` (default: read UMWELT_TRUST_QUBIT) makes per-source
        reliability a learned qubit instead of a scalar — a value-preserving upgrade (day-1 identical,
        plus a purity = confidence-in-reliability DOF). Prior-initialized so day-1 fusion == today's
        confidence-weighted observation; webs-off stays last-wins. Idempotent."""
        import os
        if self.trust_webs is None:
            self.trust_webs = {}
        self._trust_learn = bool(learn)
        self._trust_qubit = (bool(os.environ.get("UMWELT_TRUST_QUBIT")) if qubit is None
                             else bool(qubit))

    def _new_trust_web(self):
        """Factory for a per-leaf fuser — the qubit-backed variant when enabled, else classical."""
        if getattr(self, "_trust_qubit", False):
            from umwelt.foresight.qubit_trust_web import QubitTrustWeb
            return QubitTrustWeb()
        from umwelt.foresight.trust_web import TrustWeb
        return TrustWeb()

    def trust_web_snapshot(self) -> dict:
        """Per-leaf web state ("node.role" → web.snapshot()) for the heritage pickle."""
        if not self.trust_webs:
            return {}
        return {f"{n}.{r}": w.snapshot() for (n, r), w in self.trust_webs.items()}

    def node_params(self, zone: str) -> "ParameterBundle | None":
        """Get the ParameterBundle for a node, if any."""
        return self._node_params.get(zone)

    def refresh_node_params(self) -> None:
        """Refresh cached ParameterBundle references from the world graph.

        SensorBridge caches node param bundles at construction time. If param
        bundles are attached to graph nodes after construction, call this to
        pull the updated references in. Must be called before
        upgrade_to_param_norms() reads sensor_*_lo/hi off the bundles.
        """
        for node in self.graph.nodes_with_roles():
            self._node_params[node.name] = node.param_bundle

    def register(
        self,
        sensor_id: str,
        zone: str,
        sensor_type: str | None = None,
        qubit_role: str | None = None,
        normalize: Callable[[float], float] | None = None,
        weight: float = 1.0,
        description: str = "",
        label: str = "",
        event_type: str | None = None,
        r_obs: float = 0.95,
        collapse_alpha: float = 1.0,
        force_observe: bool = False,
    ) -> SensorBinding:
        """
        Register a signal -> node.role binding.

        Args:
            sensor_id: Unique signal identifier.
            zone: Target node name (kwarg name kept for signature-compatibility
                  with the origin seam and BindingSpec.zone).
            sensor_type: Preset type for auto-config (see register_sensor_preset).
            qubit_role: Which qubit role to drive. Inferred from sensor_type if not given.
            normalize: Raw value -> [-1, +1] function. Inferred from sensor_type if not given.
            weight: Scaling factor (multiple signals can feed one qubit).
            description: Human-readable description.
            r_obs: (observe roles) magnitude the belief collapses to on observation.
            collapse_alpha: (observe roles) per-observation mixing strength.
            force_observe: route this binding through the observe/collapse path even if
                           qubit_role is not a registered observe role.
        """
        if sensor_type and sensor_type in SENSOR_PRESETS:
            preset = SENSOR_PRESETS[sensor_type]
            qubit_role = qubit_role or preset["qubit_role"]
            normalize = normalize or preset["normalize"]
        elif normalize is None:
            normalize = lambda x: float(np.tanh(x))

        if qubit_role is None:
            raise ValueError(
                f"Must specify qubit_role or a known sensor_type for {sensor_id}"
            )

        node_info = self._node_info.get(zone)
        if node_info and qubit_role not in node_info["role_index"]:
            logger.warning(
                "Role '%s' not in node '%s' roles %s — signal %s may be ignored",
                qubit_role, zone, node_info["roles"], sensor_id,
            )

        binding = SensorBinding(
            sensor_id=sensor_id,
            node=zone,
            qubit_role=qubit_role,
            normalize=normalize,
            weight=weight,
            description=description or f"{sensor_type or qubit_role} in {zone}",
            kind=sensor_type or "",
            label=label,
            event_type_override=event_type,
            r_obs=r_obs,
            collapse_alpha=collapse_alpha,
            force_observe=force_observe,
        )
        self.bindings[sensor_id] = binding
        logger.debug("Registered signal %s -> %s.%s", sensor_id, zone, qubit_role)
        return binding

    def unregister(self, sensor_id: str):
        """Remove a signal binding."""
        self.bindings.pop(sensor_id, None)

    def process(
        self, readings: dict[str, float], confidence: dict[str, float] | None = None,
    ) -> dict[str, NDArray[np.floating]]:
        """
        Convert raw signal readings to per-node input arrays.

        Args:
            readings: sensor_id -> raw value
            confidence: sensor_id -> [0,1] edge-supplied validity (None → all 1.0). A
                conf=0 (null/failed) reading is DROPPED — the dissipative qubit gets no
                input and free-evolves (the engine "overlooks" it). 0<conf<1 scales the
                input amplitude (a half-trusted reading half-thermalizes).

        Returns:
            node_name -> np.array of shape (n_roles,) with normalized inputs.
            Multiple signals feeding the same qubit are averaged.
        """
        accum: dict[str, dict[int, list[float]]] = {}

        for sensor_id, raw_value in readings.items():
            binding = self.bindings.get(sensor_id)
            if binding is None:
                self._note_unmatched(sensor_id, raw_value)  # foreign / unwired — see, don't silently drop
                continue

            node_info = self._node_info.get(binding.node)
            if node_info is None:
                continue

            role_idx = node_info["role_index"].get(binding.qubit_role)
            if role_idx is None:
                continue

            self._last_raw[sensor_id] = raw_value  # track raw value for dashboards
            self._last_raw_ts[sensor_id] = time.time()

            # Observe and driver roles are NOT driven through the continuous
            # input array — they are corrected by partial collapse out of band
            # (observe_targets / driver_observations + engine ingest).
            # Skip them here so they free-evolve (drift) between observations
            # instead of being double-driven.
            if binding.is_observe or is_driver_role(binding.qubit_role):
                continue

            conf = 1.0 if confidence is None else float(confidence.get(sensor_id, 1.0))
            if conf <= 0.0:
                continue  # null/failed read → no input; the qubit free-evolves (overlook)

            # Heartbeat dedup for event-driven signals. Some devices republish
            # the same value on a polling cadence even when nothing changed;
            # left ungated they generate spurious σ_x kicks that drive orbital
            # noise. Continuous roles are dissipative and need the steady-state
            # value every tick to thermalize correctly, so they bypass this.
            if binding.is_event_driven:
                prev = self._last_event_value.get(sensor_id)
                if prev is not None and raw_value == prev:
                    continue
                self._last_event_value[sensor_id] = raw_value

            pre_weight = binding.normalize(raw_value)
            # Energy-convention sign. The excited state |1⟩ (Bloch z = -1) is
            # the high-energy / active pole; the ground state |0⟩ (z = +1) is
            # the calm / idle pole. Continuous (dissipative) roles thermalize
            # to z = input, so a high reading (hot, busy, bright) must map to
            # z = -1. Normalizers emit high → +1, so flip dissipative inputs
            # here. Event (unitary) roles are σ_x kicks, not thermal targets —
            # their sign means rotation direction, so leave them untouched.
            if role_input_mode(binding.qubit_role) == "dissipative":
                pre_weight = -pre_weight
            self._last_normed[sensor_id] = pre_weight  # track for Hebbian update
            self.touched_roles.add((binding.node, binding.qubit_role))

            w = binding.weight_param.value if binding.weight_param else binding.weight
            normalized = pre_weight * w * conf   # validity scales the dissipative push

            if binding.node not in accum:
                accum[binding.node] = {}
            if role_idx not in accum[binding.node]:
                accum[binding.node][role_idx] = []
            accum[binding.node][role_idx].append(normalized)

        result = {}
        for node_name, role_vals in accum.items():
            node_info = self._node_info[node_name]
            arr = np.zeros(node_info["n_roles"])
            for role_idx, vals in role_vals.items():
                arr[role_idx] = np.mean(vals)
            result[node_name] = arr

        return result

    def observe_targets(
        self, readings: dict[str, float], confidence: dict[str, float] | None = None,
    ) -> dict[tuple[str, str], tuple[tuple[float, float, float], float, float]]:
        """Produce observation-collapse targets for observe-role bindings.

        `confidence` (sensor_id → [0,1], edge-supplied; None → all 1.0) is the reading's
        validity. It SCALES the collapse strength: alpha = collapse_alpha × conf_brake(conf).
        conf=0 → alpha=0 → observe_qubit is a no-op → the belief free-evolves (the engine
        "overlooks" a null/failed read). The contract is enforced in the gauge-math; the
        value comes from the edge, not the engine.

        Returns {(node, role): (target_bloch, alpha, conf)} for each observe binding.
        The belief qubit drifts freely between observations; an observation partially
        collapses it toward the seen state. Every delivered reading re-anchors the belief
        (including a republish of the same state); the journal keeps its own change-only
        dedup so it records events, not every heartbeat. Last observation of a given
        (node, role) this tick wins.

        Direction comes from the binding's normalizer (high/active → +1),
        flipped to the energy convention (active → z = -1 = |1⟩, the glowing
        pole). Magnitude is r_obs (observation trust). The engine applies
        these via QubitCluster.observe_qubit after field.step.
        """
        targets: dict[tuple[str, str], tuple[tuple[float, float, float], float, float]] = {}
        # When the trust web is on, accumulate EVERY input per leaf (instead of last-wins) so the web
        # can fuse them: leaf → {sensor_id: (target_z, conf)} + the strongest collapse_alpha among
        # contributors (sets the snap rate).
        fusing = self.trust_webs is not None
        acc: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
        amax: dict[tuple[str, str], float] = {}
        for sensor_id, raw_value in readings.items():
            binding = self.bindings.get(sensor_id)
            if binding is None or not binding.is_observe:
                continue
            conf = 1.0 if confidence is None else float(confidence.get(sensor_id, 1.0))
            node_info = self._node_info.get(binding.node)
            if node_info is None or binding.qubit_role not in node_info["role_index"]:
                continue

            self._last_raw[sensor_id] = raw_value
            self._last_raw_ts[sensor_id] = time.time()

            # Every delivered reading is an observation — including a republish
            # of the same state ("I'm still on"). Re-affirming keeps the belief
            # anchored to reality between sparse device reports; without it the
            # belief drifts away and never re-anchors.
            self._last_event_value[sensor_id] = raw_value

            direction = max(-1.0, min(1.0, float(binding.normalize(raw_value))))
            target_z = -binding.r_obs * direction          # active → z = -1
            self._last_normed[sensor_id] = target_z
            leaf = (binding.node, binding.qubit_role)
            self.touched_roles.add(leaf)

            if fusing:
                acc.setdefault(leaf, {})[sensor_id] = (target_z, conf)
                amax[leaf] = max(amax.get(leaf, 0.0),
                                 effective_collapse_alpha(binding.collapse_alpha))
            else:
                # Confidence scales the collapse strength (conf=0 → alpha=0 → no-op) and
                # rides along so the engine can record it as a gauge quantity.
                targets[leaf] = ((0.0, 0.0, target_z),
                                 effective_collapse_alpha(binding.collapse_alpha) * conf_brake(conf),
                                 conf)

        if fusing:
            targets.update(self._fuse_leaves(acc, amax))
        return targets

    def _fuse_leaves(
        self, acc: dict[tuple[str, str], dict[str, tuple[float, float]]],
        amax: dict[tuple[str, str], float],
    ) -> dict[tuple[str, str], tuple[tuple[float, float, float], float, float]]:
        """Fuse this tick's per-leaf inputs through each leaf's TrustWeb. One fused (target_z, conf) per
        leaf, replacing last-wins; sources absent this tick but seen before are "down" and trigger their
        peers' compensation. The collapse alpha rides the strongest contributor's collapse_alpha × the
        fused confidence, so a single full-confidence signal reproduces the last-wins behaviour exactly."""
        out: dict[tuple[str, str], tuple[tuple[float, float, float], float, float]] = {}
        webs = self.trust_webs
        if webs is None:
            return out
        for leaf, contribs in acc.items():
            web = webs.get(leaf)
            if web is None:
                web = webs[leaf] = self._new_trust_web()
            inputs = {sid: (z, conf, True) for sid, (z, conf) in contribs.items()}
            z_f, conf_f = web.fuse(inputs)
            # Conservative consensus-grounded online learning: only when ≥2 sources corroborate at high
            # fused confidence (never train on a lone unverified read).
            if self._trust_learn and len(inputs) >= 2 and conf_f > 0.6:
                web.learn(inputs, z_f)
            if conf_f <= 0.0:
                continue  # fused no-op → leaf free-evolves
            alpha = amax.get(leaf, 1.0) * conf_brake(conf_f)
            out[leaf] = ((0.0, 0.0, z_f), alpha, conf_f)
        return out

    def latest_value(
        self, node: str, role: str
    ) -> tuple[float, float] | None:
        """Return (raw_value, wall_clock_ts) of the freshest reading for (node, role).

        Read-only — does not touch ingest or _last_event_value. Used by app
        surfaces to show real device state. Multiple signals can bind the same
        (node, role); we return the most recently updated one.
        """
        best: tuple[float, float] | None = None
        for b in self.bindings.values():
            if b.node != node or b.qubit_role != role:
                continue
            ts = self._last_raw_ts.get(b.sensor_id)
            if ts is None:
                continue
            raw = self._last_raw.get(b.sensor_id)
            if raw is None:
                continue
            if best is None or ts > best[1]:
                best = (float(raw), float(ts))
        return best

    def latest_raw(self, sensor_id: str) -> tuple[float, float] | None:
        """Return (raw_value, wall_clock_ts) of the freshest reading for ONE
        sensor_id, or None if it never reported. Per-signal analog of
        ``latest_value`` (which is per node+role); used where many signals share
        a role and the individual contributor matters."""
        ts = self._last_raw_ts.get(sensor_id)
        raw = self._last_raw.get(sensor_id)
        if ts is None or raw is None:
            return None
        return (float(raw), float(ts))

    def fresh_nodes(
        self, role: str, ttl_seconds: float, now: float | None = None
    ) -> set[str]:
        """Nodes where `role` has at least one signal that reported within ttl.

        Signal-outage resilience (generalized from the origin deployment's
        occupancy-specific form): a node NOT in this set has no live evidence
        for the role (its signals are offline / never paired, or the activity
        ended long ago) — the caller can re-pin it to the ground state so an
        offline node can't drift back to the asserted fixed point. A
        still-active source keeps its node fresh as long as ttl > the gap
        between pings, so sticky beliefs are preserved (pick ttl on that
        horizon).
        """
        now = time.time() if now is None else now
        fresh: set[str] = set()
        for sid, b in self.bindings.items():
            if b.qubit_role != role:
                continue
            ts = self._last_raw_ts.get(sid)
            if ts is not None and (now - ts) <= ttl_seconds:
                fresh.add(b.node)
        return fresh

    def health_snapshot(
        self, ttl_seconds: float = 1800.0, now: float | None = None
    ) -> dict:
        """Per-signal liveness from the last raw report time — operator visibility
        into which signals are actually reporting. A signal that never reported
        (age=None) or is older than its ttl is 'stale'; the engine stays correct
        without it (silent signals contribute nothing to the continuous-input
        field), this just surfaces the gaps."""
        now = time.time() if now is None else now
        out: dict[str, dict] = {}
        live = 0
        for sid, b in self.bindings.items():
            ts = self._last_raw_ts.get(sid)
            age = None if ts is None else round(now - ts, 1)
            ttl = self.resolve_dark_ttl(b, default=ttl_seconds)
            ok = age is not None and age <= ttl
            live += 1 if ok else 0
            out[sid] = {"node": b.node, "role": b.qubit_role, "age_s": age, "live": ok,
                        "ttl_s": ttl}
        return {"ts": round(now, 1), "ttl_s": ttl_seconds, "n": len(out),
                "live": live, "stale": len(out) - live, "sensors": out}

    # Roles whose streams only speak on CHANGE (noise gates drop unchanged reports) — a
    # quiet stream is normal for a day, not dark in half an hour. CONTINUOUS_ROLES is a
    # registry the domain extends (empty by default): roles listed here use the caller's
    # short default ttl; every other role is treated as change/event-driven → day-scale.
    EVENT_DRIVEN_TTL_S = 26 * 3600.0
    CONTINUOUS_ROLES: set[str] = set()

    def resolve_dark_ttl(self, binding, default: float = 1800.0) -> float:
        """explicit binding.dark_ttl_s > role default (continuous roles use the caller's
        default; every other role is change/event-driven → day-scale)."""
        if binding.dark_ttl_s is not None:
            return float(binding.dark_ttl_s)
        if binding.qubit_role in self.CONTINUOUS_ROLES:
            return float(default)
        return self.EVENT_DRIVEN_TTL_S

    def hydrate_last_seen(self, ages: dict[str, float]) -> int:
        """Boot hydration: seed _last_raw_ts from a durable source (stream tape MAX(t)
        per signal) so a deploy restart doesn't read as every stream going DARK.
        Only fills gaps — a live report always wins. Returns the number seeded."""
        n = 0
        for sid, ts in ages.items():
            if sid in self.bindings and sid not in self._last_raw_ts:
                self._last_raw_ts[sid] = float(ts)
                n += 1
        return n

    def list_bindings(self) -> list[dict]:
        """List all registered signal bindings."""
        return [
            {
                "sensor_id": b.sensor_id,
                "node": b.node,
                "role": b.qubit_role,
                "weight": b.weight,
                "description": b.description,
            }
            for b in self.bindings.values()
        ]

    # Cap on distinct unmatched signal IDs retained (a foreign/noisy stream could be unbounded).
    _UNMATCHED_CAP = 256

    def _note_unmatched(self, sensor_id: str, raw_value) -> None:
        """Record a reading whose sensor_id has no binding — the foreign-data signal. Bounded LRU so a
        chatty foreign stream can't grow this without limit; counts are still exact per retained id."""
        self._unmatched_total += 1
        e = self._unmatched.get(sensor_id)
        if e is None:
            if len(self._unmatched) >= self._UNMATCHED_CAP:
                self._unmatched.popitem(last=False)      # evict oldest-touched
            e = {"count": 0, "last_value": None, "last_ts": 0.0}
            self._unmatched[sensor_id] = e
        e["count"] += 1
        try:
            e["last_value"] = float(raw_value)
        except (TypeError, ValueError):
            e["last_value"] = None
        e["last_ts"] = time.time()
        self._unmatched.move_to_end(sensor_id)

    def register_ignored(self, ignored: tuple) -> None:
        """Declare which unmatched sensor_ids are DELIBERATELY unbound (each a (pattern, reason) —
        exact id or `prefix_*`). Turns the ingest gap from a scary 'N unbound' into the TRUTH:
        'A actionable, I explained'. Applied from the spec's `ignored` by apply_spec_bindings."""
        self._ignored = list(ignored or ())

    def unmatched_snapshot(self, limit: int = 50) -> dict:
        """What arrived that bound to NOTHING, SPLIT into actionable vs declared-ignored (the honest
        ingest gap). `sensors` = genuinely-unexplained (the real onboarding work for ANY domain);
        `ignored` = deliberately-unbound, each with a reason (retired / redundant / producer-alias /
        deferred). Cheap read. See _note_unmatched, register_ignored."""
        actionable, ignored = [], []
        for sid, e in sorted(self._unmatched.items(), key=lambda kv: -kv[1]["count"]):
            reason = match_ignored(sid, tuple(self._ignored))
            row = {"sensor_id": sid, "count": e["count"], "last_value": e["last_value"], "last_ts": e["last_ts"]}
            (ignored if reason is not None else actionable).append({**row, "reason": reason} if reason else row)
        return {
            "distinct": len(self._unmatched),
            "actionable": len(actionable),
            "explained": len(ignored),
            "total_readings": self._unmatched_total,
            "truncated": len(self._unmatched) >= self._UNMATCHED_CAP,
            "sensors": actionable[:max(0, limit)],          # the real gaps (loudest first)
            "ignored": ignored[:max(0, limit)],             # deliberately unbound, with reasons
        }

    def upgrade_weights(self):
        """Promote fixed signal weights to learnable ScalarParams.

        After this call, each binding's weight becomes a ScalarParam on
        the parameter fiber. Any learning loop can update it via
        binding.weight_param.kalman_update(observed_weight, sigma).

        The initial sigma is 30% of the weight, so Thompson Sampling
        will explore different weight configurations early on.
        """
        # Driver trust floor (safety guarantee: the anchor never vanishes) is a
        # named fiber prior, not a buried literal. Read it from the root bundle.
        from umwelt.substrate.params import ScalarParam  # deferred (cycle-break)
        driver_floor = 0.15
        root = getattr(self.graph, "root", None)
        if root is not None and root.param_bundle is not None:
            driver_floor = root.param_bundle.get("driver_trust_floor", driver_floor)

        upgraded = 0
        for binding in self.bindings.values():
            if binding.weight_param is not None:
                continue
            # Periodic drivers are ground truth — their trust can be down-weighted
            # by learning but never to zero, so the anchor stays on.
            lo = driver_floor if is_driver_role(binding.qubit_role) else 0.0
            binding.weight_param = ScalarParam(
                name=f"w_{binding.sensor_id}",
                value=binding.weight,
                sigma=max(binding.weight * 0.3, 0.05),
                lo=lo,
                hi=3.0,
            )
            upgraded += 1
        if upgraded:
            logger.info("Promoted %d signal weights to learnable ScalarParams", upgraded)

    def upgrade_to_param_norms(self):
        """Replace static range_norm closures with live param_range_norm.

        After all signals and param bundles are configured, call this once.
        For each binding whose node has matching sensor_{prefix}_lo/hi params
        on its ParameterBundle (prefixes declared via
        register_range_param_prefixes), the static normalizer is replaced with
        a param_range_norm that reads bounds live from the bundle.

        This closes the loop: calibration updates sensor_*_lo/hi on the
        bundle, and the normalizer reads them each time it's called.
        """
        upgraded = 0
        for binding in self.bindings.values():
            bundle = self._node_params.get(binding.node)
            if bundle is None:
                continue

            # Find matching lo/hi params for this binding's role
            prefixes = ROLE_PARAM_MAP.get(binding.qubit_role, [])
            for prefix in prefixes:
                lo_key = f"sensor_{prefix}_lo"
                hi_key = f"sensor_{prefix}_hi"
                lo_param = bundle.get_param(lo_key)
                hi_param = bundle.get_param(hi_key)
                if lo_param is not None and hi_param is not None:
                    # Check if sensor_id contains the prefix (e.g. "east_climate_temp" has "temp")
                    if prefix in binding.sensor_id or prefix in binding.description.lower():
                        binding.normalize = param_range_norm(lo_param, hi_param)
                        upgraded += 1
                        logger.debug(
                            "Upgraded %s normalizer to param_range_norm (%s/%s)",
                            binding.sensor_id, lo_key, hi_key,
                        )
                        break

        if upgraded:
            logger.info("Upgraded %d signal normalizers to live param_range_norm", upgraded)

    def _root_read(self, key: str, default: float) -> float:
        """Live-read a hyperparameter off the root bundle (the gauge), seed default pre-attach/in tests."""
        root = getattr(self.graph, "root", None)
        pb = getattr(root, "param_bundle", None) if root is not None else None
        return float(pb.get(key, default)) if pb is not None else default

    def hebbian_weight_update(
        self,
        residuals: dict[str, np.ndarray],
        lr: float | None = None,
    ):
        """Update signal weights from production residuals via Hebbian rule.

        For each signal: if its normalized input agrees in sign with the
        cluster's surprise (the field needed to move in the same direction
        the signal pushed it), increase the weight. If it disagreed, decrease.

        Update rule:
            Δw = lr * pre_weight_input * role_residual

        Uses a soft Kalman update (high obs_sigma) so weights drift slowly
        — Thompson Sampling continues to explore even with many updates.

        Call this after each fractal_stack.step() with last_raw_residuals.
        """
        if lr is None:
            lr = self._root_read("hebbian_lr", 0.01)          # the gradient lr, a gauge coordinate
        obs_sigma = self._root_read("hebbian_obs_sigma", 0.5)
        for sensor_id, pre_weight_val in self._last_normed.items():
            if abs(pre_weight_val) < 1e-10:
                continue  # signal was silent this step

            binding = self.bindings.get(sensor_id)
            if binding is None or binding.weight_param is None:
                continue

            node_info = self._node_info.get(binding.node)
            if node_info is None:
                continue

            role_idx = node_info["role_index"].get(binding.qubit_role)
            if role_idx is None:
                continue

            cluster_residuals = residuals.get(binding.node)
            if cluster_residuals is None or role_idx >= len(cluster_residuals):
                continue

            role_residual = float(cluster_residuals[role_idx])
            if abs(role_residual) < 1e-10:
                continue  # field didn't move — no learning signal

            # Hebbian gradient (influence × surprise) via the ONE learner's immediate gradient mode:
            # target = w − lr·(−residual)·input = w + lr·input·residual, then collapse (clamped to the
            # param's [lo,hi]). lr/obs_sigma are gauge coordinates.
            self._learner.gradient_step(binding.weight_param, pre_weight_val, -role_residual,
                                        lr=lr, obs_sigma=obs_sigma)

    def hebbian_driver_update(
        self,
        residuals: dict[str, np.ndarray],
        lr: float = 0.01,
    ):
        """Credit periodic drivers' TRUST weight, the way signals are credited.

        Same Hebbian rule as hebbian_weight_update (Δw = lr · input · residual,
        soft Kalman) but for driver bindings only, driven each tick from
        `_last_normed` set by the engine's driver anchor (target z). If
        anchoring to the driver moved the field the way reality needed (residual
        agrees in sign), trust rises; if it fought the field, trust falls — but
        the weight floor (set in upgrade_weights) keeps the ground-truth anchor
        always on. Distinct from hebbian_weight_update so it can run every tick
        (the driver loop sends no sensor readings).
        """
        for sensor_id, binding in self.bindings.items():
            if not is_driver_role(binding.qubit_role) or binding.weight_param is None:
                continue
            pre_weight_val = self._last_normed.get(sensor_id)
            if pre_weight_val is None or abs(pre_weight_val) < 1e-10:
                continue
            node_info = self._node_info.get(binding.node)
            if node_info is None:
                continue
            role_idx = node_info["role_index"].get(binding.qubit_role)
            cluster_residuals = residuals.get(binding.node)
            if role_idx is None or cluster_residuals is None or role_idx >= len(cluster_residuals):
                continue
            role_residual = float(cluster_residuals[role_idx])
            if abs(role_residual) < 1e-10:
                continue
            # Same immediate gradient as signal trust, via the ONE learner (clamped to param [lo,hi]).
            # obs_sigma shares the signal-trust gauge coordinate; lr is the driver_hebbian_lr passed in.
            self._learner.gradient_step(binding.weight_param, pre_weight_val, -role_residual,
                                        lr=lr, obs_sigma=self._root_read("hebbian_obs_sigma", 0.5))

    @property
    def bound_nodes(self) -> set[str]:
        """All nodes that have at least one signal."""
        return {b.node for b in self.bindings.values()}


def apply_spec_bindings(bridge: SensorBridge, spec: "DomainSpec") -> None:
    """Apply a spec's declarative bindings + ignored set onto a bridge — the domain-AGNOSTIC
    binding seam. Each BindingSpec's declarative normalizer resolves through the registry
    (BindingSpec.build_normalizer). Membrane-guarded — a bad spec binding must never break
    the bindings already registered."""
    for b in (spec.bindings or ()):
        try:
            # measurement_alpha() = k·η when the binding declares a measurement model,
            # else its collapse_alpha, else None → the bridge default.
            _alpha = b.measurement_alpha() if hasattr(b, "measurement_alpha") else b.collapse_alpha
            bridge.register(
                b.sensor_id, zone=b.zone, qubit_role=b.role,
                normalize=b.build_normalizer(), weight=b.weight,
                event_type=(b.event_type or None),
                **({"collapse_alpha": _alpha} if _alpha is not None else {}),
                **({"force_observe": True} if b.force_observe else {}),
            )
        except Exception as exc:
            logger.warning("spec binding %s skipped: %s", b.sensor_id, exc)
    try:
        bridge.register_ignored(spec.ignored)
    except Exception:
        pass
