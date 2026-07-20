"""
BeliefEngine — the full system.

Two layers:

    SKY:    Quantum probability field (continuous, always evolving)
    GROUND: Classical world model (discrete, committed facts)

The field holds beliefs. The world model holds commitments.
Collapse bridges the two — selectively, when needed.

    Signals -> SensorBridge -> QuantumField (sky)
                                    |  collapse
                                    v
                               WorldModel (ground) -> context -> actions
                                    |
                                    v
                               features(t) -> forecasters -> predictions

Curated copy of the origin deployment's reservoir seam (see tools/RENAMES.md): the domain layer (its device actuators, person/forecast
runners, ephemeris clocks and the hard-coded location gear) is CUT; what the
domain used to wire imperatively is now injected as data — `drivers` (periodic
clocks from DriverSpec), `tendrils` (outputs, P3), `forecasters` (P4), and
named `anchors` with pluggable value codecs.
"""
from __future__ import annotations
from umwelt._util import clamp01

import logging
import pickle
import random
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from umwelt.substrate.belavkin import env_belavkin_enabled
from umwelt.substrate.graph import WorldGraph
from umwelt.substrate.field import QuantumField
from umwelt.membranes.ingress import SensorBridge
from umwelt.substrate.ground import ClassicalWorldModel, Transition
from umwelt.substrate.collapse import CollapseEngine, CollapsePolicy, CollapseReason
from umwelt.learning.calibration import CalibrationLoop, CalibrationConfig
from umwelt.substrate.fractal import fractal_dimension_estimate
from umwelt.substrate.population import Population, PopulationConfig
from umwelt.substrate.fractal_stack import FractalStack, FractalStackConfig
from umwelt.learning.training import TrainingRunner, TrainingConfig  # noqa: F401 (config re-export)
from umwelt.clocks.berry_tape import BerryTape
from umwelt.learning.surprise_tape import SurpriseTape
from umwelt.substrate.params import BlochGeometricPhase
from umwelt.clocks.adaptive_clock import AdaptiveClock
from umwelt.learning.competence import competence_score
from umwelt.substrate.bloch import bloch_to_phase

logger = logging.getLogger(__name__)

# Node kinds whose leaves may pool dynamics params when behaviorally similar
# (see _maybe_pool_dynamics). Canonical spec kinds only — a domain's dialect
# ("sensor", "appliance") canonicalizes before the graph is built.
_GROUPABLE_KINDS = frozenset({"actuator", "signal", "component"})


class BeliefEngine:
    """
    The belief engine: field (sky) + world model (ground).

    The field evolves continuously — it never stops, never resets
    on collapse. It holds the system's probabilistic beliefs about
    the world.

    The world model holds committed facts — discrete, definite states
    that downstream reasoning and the action system can act on.

    Collapse bridges the two: projecting field state into world model
    commitments. Collapse is selective (per-node, per-role) and can
    be triggered by various reasons (periodic, confidence, query, action).
    """

    def __init__(
        self,
        gamma: float = 0.05,
        dt: float = 0.01,
        bridge_strength: float = 0.5,
        collapse_interval: int = 10,
        confidence_threshold: float = 0.9,
        hysteresis: float = 0.1,
        seed: int | None = None,
        graph: WorldGraph | None = None,
        calibration: CalibrationConfig | bool | None = None,
        population: PopulationConfig | None = None,
        fractal_stack: FractalStackConfig | None = None,
        cluster_filter: "Callable[[object], bool] | None" = None,
        subdomains: bool = False,
    ):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(int(seed) % (2**32 - 1))

        # The world graph is REQUIRED — the engine has no default world. Boot
        # builds it from a DomainSpec (build_graph_from_spec) and passes it in.
        if graph is None:
            raise ValueError(
                "BeliefEngine requires a graph (build one from a DomainSpec via "
                "umwelt.spec.build.build_graph_from_spec, or boot via umwelt.boot.build_engine)"
            )
        self.graph = graph
        self.subdomains = bool(subdomains)

        # The sky: quantum probability field (continuous beliefs). `cluster_filter`,
        # when given, SCOPES the production field to a subgraph — a lightweight brain
        # that holds only the clusters in its scope (the sparse graph partitions
        # cleanly). Bridges/projections to omitted clusters are .get()-guarded, and
        # out-of-scope signals are skipped in observe_targets/ingest. Used by SCOPED
        # FORECAST BRAINS; the live forebrain passes None (the full graph).
        self.field = QuantumField(
            graph=self.graph,
            gamma=gamma,
            dt=dt,
            bridge_strength=bridge_strength,
            cluster_filter=cluster_filter,
        )

        # The ground: classical world model (committed facts)
        self.world = ClassicalWorldModel()

        # The bridge between sky and ground
        policy = CollapsePolicy(
            periodic_interval=collapse_interval,
            confidence_threshold=confidence_threshold,
        )
        self.collapse_engine = CollapseEngine(
            policy=policy,
            hysteresis=hysteresis,
        )

        # Signal routing (perception: physical -> field)
        self.sensor_bridge = SensorBridge(graph=self.graph)

        # Share the bridge's touched_roles set with the collapse engine so it
        # can suppress background collapses on orphan qubit roles (those that
        # no signal has ever fed). The set is mutated live as new signals fire,
        # so the gate auto-relaxes as coverage grows.
        self.collapse_engine.touched_roles = self.sensor_bridge.touched_roles

        # Action routing (world model -> physical) is P3's egress surface. The
        # attribute survives for surface-compat with the origin seam; nothing in
        # the engine reads it — Actions come from the tendril loop in ingest().
        self.actuator_bridge = None

        # Calibration loop (feedback: reality -> parameter fiber)
        if calibration is True:
            calibration = CalibrationConfig()
        if isinstance(calibration, CalibrationConfig):
            self.calibration = CalibrationLoop(
                graph=self.graph,
                field=self.field,
                config=calibration,
            )
        else:
            self.calibration: CalibrationLoop | None = None

        # Genetic Hamiltonian population (Channel 4)
        self.population: Population | None = None
        if population is not None and population.enabled:
            self.population = Population(
                graph=self.graph,
                config=population,
                gamma=gamma,
                dt=dt,
                bridge_strength=bridge_strength,
                seed=seed,
            )
            if self.calibration is not None:
                self.calibration.population = self.population

        # Fractal stack (multi-scale H-learning). Read context.dt_factor and
        # slide the φ-ladder at construction — the same gauge axis that gates
        # actuate/learn/persist also chooses where the H-tower samples its
        # Fibonacci strides. See ContextState + phi_clock.fib_strides_at.
        self.fractal_stack: FractalStack | None = None
        if fractal_stack is not None and fractal_stack.enabled:
            try:
                from umwelt.learning.context import ContextState
                from umwelt.substrate.fractal_stack import phi_scales as _phi_scales
                _ctx_dt = ContextState.from_bundle(
                    getattr(getattr(self.graph, "root", None), "param_bundle", None)
                ).dt_factor
                if _ctx_dt != 1.0 and fractal_stack.scales:
                    fractal_stack.scales = _phi_scales(
                        len(fractal_stack.scales), dt_factor=_ctx_dt
                    )
            except Exception:
                # dt_factor sliding is opt-in; never block stack construction.
                pass
            self.fractal_stack = FractalStack(
                graph=self.graph,
                production_field=self.field,
                config=fractal_stack,
            )

        # Training runner: adapts learning rate, manages training lifecycle
        self.training: TrainingRunner | None = None
        if self.fractal_stack is not None:
            self.training = TrainingRunner()

        # Berry tape: phenomenological record of the field's learning journey.
        # Stamps significant events ordered by geometric phase, not clock time.
        self.berry_tape = BerryTape()

        # The real Berry phase: accumulated from the qubits' Bloch trajectories
        # (dγ = -½(1-cosθ)dφ·|r|), summed across qubits. Feeds the berry tape.
        self.bloch_berry = BlochGeometricPhase()

        # ── PERIODIC DRIVERS — the domain's clocks, injected as data ──
        # Boot fills this from DriverSpec via clocks/drivers.build_driver. Each
        # ingest re-anchors every driver's (node, role) qubit toward its
        # deterministic target_bloch(now) so the phase HOLDS its sphere position
        # against field dephasing — a continuous pursuit; the field learns the
        # cycle's COMPREHENSION (anticipation), never the phase itself.
        self.drivers: list = []
        # Cached driver targets ("node:role" -> unit (x,y,z) Bloch position),
        # refreshed from the drivers each tick (and overridable per-call via
        # ingest(driver_observations=...) — explicit payloads win).
        self.driver_targets: dict[str, tuple[float, float, float]] = {}
        # Horizon-ahead labels ("node:role" -> future (x,y,z)) that supervise a
        # forecaster (the field learning to ANTICIPATE the cycle) — P4 consumes.
        self.driver_forecast_labels: dict[str, tuple[float, float, float]] = {}
        # Anticipation skill EMA per driver anchor: how close the field's own
        # pre-anchor state already is to the driver truth (1 = perfect).
        self.driver_anticipation: dict[str, float] = {}
        # Raw (un-smoothed) per-tick anticipation error — lets the calibrator
        # estimate the signal's SNR and adapt the anticipation EMA bandwidth.
        self._anticipation_raw: deque[float] = deque(maxlen=64)
        # Per-driver phase skill EMA + last computed target (for clock_snapshot).
        self.clock_skill: dict[str, float] = {}
        self._clock_last_target: dict[str, dict] = {}
        self._driver_applied: dict[str, float] = {}   # sensor_id -> target z (Hebbian credit)

        # ── OUTPUT TENDRILS — decisions leaving the field, injected as data ──
        # P3 fills this from OutputSpec (membranes/egress.build_tendrils). The
        # uniform surface is already live: ONE step loop (emit Actions) + ONE
        # override loop (learn from the operator) in ingest(), no bespoke
        # per-actuator blocks. Empty list = both loops no-op.
        self.tendrils: list = []

        # ── FORECASTERS — anticipation runners, injected as data (P4) ──
        # The origin wired its forward models imperatively here; the generic
        # Forecaster protocol lands with foresight/ (P4). Empty = no-op.
        self.forecasters: list = []

        # BPU forecast OFFLOAD probe — foresight/ is a later phase (P4); until
        # that module exists the probe is None and the engine runs CPU-only.
        try:
            from umwelt.foresight.bpu_forecast import make_bpu_forecast_probe
            self.bpu_forecast = make_bpu_forecast_probe()
        except ImportError:
            self.bpu_forecast = None

        # Let the calibration loop see the driver-comprehension learners (like
        # population). Channel 6's forecast hyperparameter block stays dormant
        # until P4 injects a forecaster (driver_forecast stays None).
        if self.calibration is not None:
            self.calibration.driver_anticipation = self.driver_anticipation
            self.calibration.driver_anticipation_snr = self.anticipation_snr
            # Cross-tower: tier 2 grades each H-tower scale's improving/plateau
            # classifications, so it needs the list of scales (or empty if the
            # H-tower isn't enabled).
            self.calibration.fractal_scales = (
                self.fractal_stack.scales if self.fractal_stack is not None else []
            )
        # Move the WHOLE learnable parameter fiber into qubit space (ProductQubitCluster `_fiber`).
        # After this, every `bundle.get(param)` / `bundle.update(...)` silently routes through a
        # per-param qubit (Thompson wriggle, purity=confidence, Berry history). See _bind_param_fiber.
        self._bind_param_fiber()

        # ── NAMED ANCHORS — grounded coordinates, injected as data ──
        # An anchor is one qubit holding a slowly-grounded world coordinate (the
        # origin's location gear, generalized): evidence fixes accumulate into
        # their Bloch centroid via ground_anchor, purity rising as fixes agree.
        # Codecs (encode(value)->bloch, decode(bloch)->value) are registered by
        # the app/spec; the engine never knows a coordinate system.
        self.anchor_codecs: dict[str, object] = {}
        # anchor name -> (node, role) qubit address; register_anchor overrides
        # the `("_"+name, name)` convention.
        self.anchor_nodes: dict[str, tuple[str, str]] = {}
        # Which anchors have been grounded from evidence (vs still on their
        # build floor). Persisted in the state snapshot.
        self._grounded_anchors: set[str] = set()

        # Agency qubit: the act↔listen axis as one Bloch qubit in the meta-param stack —
        # driven by the silence lever, healing toward 'recommend' over weeks. The smooth/axiomatic
        # layer atop the scalar actuation_silenced floor; downstream reads its |act⟩/|listen⟩.
        from umwelt.learning.agency import AgencyQubit
        # tau_days (the weeks-scale agency relaxation constant) is a fiber prior
        # (calibratable + gauge-tracked), default 10.0.
        self.agency = AgencyQubit(tau_days=self._root_param("agency_tau_days", 10.0))

        # Surprise tape: phase-indexed record of prediction errors across all
        # learners (fractal scales, calibration channels, training runner).
        # Shannon-rate-matched gating + weighted reservoir sampling keep write
        # pressure negligible. Wire in via `attach_surprise_tape()`.
        self.surprise_tape: SurpriseTape | None = None

        # Per-source last-emitted surprise value. Substrate learners (fractal
        # scales, training) only tick at their stride, so their EMAs stay
        # frozen between ticks. Without this gate the recorder emits the
        # frozen value every engine step, creating wallclock-cadence spam on
        # the timeline. We only forward to the tape when the EMA actually moved.
        self._last_observed_surprise: dict[str, float] = {}
        # For substrate learners: only emit when their discrete tick counter
        # (epoch / step) advances. Smooth EMA drift between ticks is the
        # learner settling, not a surprise event.
        self._last_emit_counter: dict[str, int] = {}

        # Belavkin measurement path (rung L4). Default OFF → the observe path is
        # the byte-unchanged α-blend; ON swaps in the conditioned Kraus update
        # (bounded Wonham gain + cumulant cross-update).
        self._belavkin = env_belavkin_enabled()

        self._step = 0
        self._web_topology = None   # fractal-web (lazy; gated UMWELT_LEARNED_TOPOLOGY)

        # Stasis lever (the pause switch). When True, ingest() caches signal
        # readings (the surface stays live) but does NOT evolve the field,
        # collapse, learn, or run population/fractal — the brain is frozen,
        # disconnected from the signal surface. Set by the app.
        self.paused = False

        # Recipe-config attrs (the artifact-carried opt-ins, not a runtime profile
        # branch). apply_seed_profile writes these; the blank floor leaves them off.
        self.home_lock = False

        # Adaptive clock — compute-compression gate. None = OFF (default; the live
        # path is unchanged). The rig/lab sets this to an AdaptiveClock to enable
        # coasting through calm. See clocks/adaptive_clock.py.
        self.adaptive: AdaptiveClock | None = None
        self._coasted_steps = 0
        self._since_full = 0   # ticks since the last full (non-skipped) step — the smooth-clock
                               # cadence counter; drives the dt_scale catch-up. See ingest().

        # Membrane cadence (spec.ingest_hold_s; None = OFF → ingest-driven, the
        # origin behavior: one compute step per batch no matter how much wall time
        # passed). NOT a time model: the world's clocks are drivers (in-universe
        # qubits, opt-in), dt is compute step size — this knob only adapts SPARSE
        # FEED CADENCE at the ingest membrane, holding each batch's inputs across
        # the gap it closes in bounded unit substeps. Measured need: the first
        # foreign-cadence world (daily market bars) under-drove the field ~15x.
        # See docs/TIME.md and ingest().
        import os as _os
        self.ingest_hold_s: float | None = None
        self._last_wall_now = None
        self._last_bridged_inputs: dict | None = None
        self._wall_catchup_steps = 0        # lifetime counter (the off-means-off pin)
        self._wall_catchup_max = int(_os.environ.get("UMWELT_WALL_CATCHUP_MAX", "32") or 32)

        # field.step timing instrument (UMWELT_TICK_TIMING=1) — EMA + periodic
        # journal log so dtype/batching A/Bs are measurable on a live board
        # without building a second engine (OOM risk). Zero overhead off.
        import os as _os
        self._tick_timing = _os.environ.get("UMWELT_TICK_TIMING") == "1"
        self._tick_ema: float | None = None
        self._tick_n = 0
        self._tick_every = int(_os.environ.get("UMWELT_TICK_TIMING_EVERY", "100"))

        # ── LIVE STREAMING LEARNING (UMWELT_STREAM_LEARN=N, default 0=off) ──
        # The handoff design is "always learning — only ACTUATE gates"; the forebrain freeze
        # (learn≈0, skip the heavy learner block) is a PERF compromise for slow boards. Streaming
        # learning is the middle path: a FROZEN forebrain still runs the live learner block every
        # Nth FULL tick — learning ONLINE from the live world (not the hindbrain's backlog replay),
        # throttled so the heavy tick lands 1-in-N and real-time holds. N is the plasticity dial an
        # offload eventually turns to 1 (learn every tick). Opt-in; default 0 leaves forebrain frozen.
        self._stream_learn = int(_os.environ.get("UMWELT_STREAM_LEARN", "0"))
        self._stream_tick = 0
        self._stream_count = 0   # online learn ticks fired (visibility — see ingest STREAM-LEARN log)

        logger.info(
            "BeliefEngine: %d clusters, %d features, %d drivers",
            len(self.field.clusters), self.feature_dim, len(self.drivers),
        )

    # ================================================================
    # Named anchors (grounded coordinates)
    # ================================================================

    def register_anchor(self, name: str, *, node: str | None = None,
                        role: str | None = None, codec: object | None = None) -> None:
        """Declare an anchor: which (node, role) qubit holds it and (optionally) the
        value codec (encode(value)->(x,y,z), decode((x,y,z))->value). Defaults follow
        the `("_"+name, name)` node/role convention."""
        self.anchor_nodes[name] = (node or f"_{name}", role or name)
        if codec is not None:
            self.anchor_codecs[name] = codec

    def _anchor_address(self, name: str) -> tuple[str, str]:
        return self.anchor_nodes.get(name, (f"_{name}", name))

    def _anchor_qubit(self, name: str):
        """(cluster, role_index) for an anchor, or (None, None) if the graph lacks it."""
        node, role = self._anchor_address(name)
        cluster = self.field.clusters.get(node)
        if cluster is None:
            return None, None
        idx = cluster.role_index.get(role)
        if idx is None:
            return None, None
        return cluster, idx

    def delocate_anchor(self, name: str) -> None:
        """Set an anchor to the MAXIMALLY-MIXED state (Bloch r=0) — genuinely
        un-grounded, awaiting evidence. The blank floor for every anchor: a fresh
        qubit's pure default pole would be a definite-but-meaningless coordinate,
        so a blank build explicitly de-locates it."""
        cluster, idx = self._anchor_qubit(name)
        if cluster is None:
            return
        cluster.observe_qubit(idx, (0.0, 0.0, 0.0), alpha=1.0)
        self._grounded_anchors.discard(name)

    def ground_anchor(self, name: str, value, *, alpha: float | None = None,
                      codec: object | None = None):
        """Ground an anchor from an evidence FIX. Observes the anchor qubit toward the
        fix's Bloch point; accumulating fixes converge the qubit to their spherical
        centroid (the qubit IS the running centroid), with purity rising as fixes agree
        = the system growing sure of the coordinate. Marks the anchor grounded.

        `value` is either a Bloch triple (x, y, z) or a domain value the codec encodes.
        `alpha` is the per-fix gain; default small (root `anchor_ground_alpha`, ~0.1) so
        no single fix dominates and the centroid averages out jitter. Returns the
        anchor's new decoded value (codec) or its Bloch point (no codec)."""
        cluster, idx = self._anchor_qubit(name)
        if cluster is None:
            return None
        codec = codec or self.anchor_codecs.get(name)
        if codec is not None:
            target = codec.encode(value)
        else:
            target = tuple(float(v) for v in value)
        a = float(self._root_param("anchor_ground_alpha", 0.1) if alpha is None else alpha)
        cluster.observe_qubit(idx, target, clamp01(a))
        self._grounded_anchors.add(name)
        return self.anchor_value(name, codec=codec)

    def anchor_bloch(self, name: str) -> tuple[float, float, float]:
        """An anchor's Bloch point — the system's coordinate, read from its qubit.
        A graph with NO such anchor (a scoped brain, or a spec that never declared
        one) is UN-GROUNDED — maximally mixed, r=0. The coordinate must never
        assume a value it was not given."""
        cluster, idx = self._anchor_qubit(name)
        if cluster is None:
            return (0.0, 0.0, 0.0)
        b = cluster.qubit_bloch(idx)
        return float(b[0]), float(b[1]), float(b[2])

    def anchor_value(self, name: str, *, codec: object | None = None):
        """The anchor decoded through its codec (or its raw Bloch point without one)."""
        codec = codec or self.anchor_codecs.get(name)
        b = self.anchor_bloch(name)
        return codec.decode(b) if codec is not None else b

    def anchor_pin_target(self, name: str) -> tuple[float, float, float] | None:
        """The Bloch point a slow app loop should re-anchor this qubit toward each tick.

        GROUNDED (the qubit has a direction — evidence-grounded, or restored from an
        artifact that carries one) → hold the anchor's OWN current direction
        (unit-normalized): keep position + gently purify against field dephasing,
        without dragging it anywhere. UN-GROUNDED (the blank floor's maximally-mixed
        qubit) → None: stay that way until evidence arrives (no assumed coordinate).
        No profile check — it reads the qubit's state, which the artifact carries."""
        b = np.asarray(self.anchor_bloch(name), dtype=float)
        n = float(np.linalg.norm(b))
        grounded = (name in self._grounded_anchors) or n > 0.5
        return tuple(b / n) if (grounded and n > 1e-9) else None

    # ── Origin-seam compat: the "geo" anchor as (lat, lon) ──
    # projection.gauge_name._place reads these; without a registered geo codec
    # (an app concern — the engine ships no geography) the decoded value is the
    # (0.0, 0.0) un-grounded floor.
    def location_bloch(self) -> tuple[float, float, float]:
        return self.anchor_bloch("geo")

    def location_latlon(self) -> tuple[float, float]:
        codec = self.anchor_codecs.get("geo")
        if codec is None:
            return (0.0, 0.0)
        lat, lon = codec.decode(self.location_bloch())
        return float(lat), float(lon)

    def context_gauge(self) -> dict:
        """The engine's fully self-describing context: seed-profile + anchors + run-mode.
        A snapshot is enough to say 'I'm a <profile> brain, anchored at <coords>, in
        <mode>' — every axis is tracked in the graph/pickle (profile in the pickle
        header, anchors in their qubits, run-mode on the root bundle), so the
        description is the engine's own state, not out-of-band metadata."""
        from umwelt.learning.context import ContextState
        root = getattr(self.graph, "root", None)
        bundle = root.param_bundle if root is not None else None
        anchors = {}
        for name in sorted(set(self.anchor_nodes) | self._grounded_anchors | {"geo"}):
            cluster, _ = self._anchor_qubit(name)
            if cluster is None and name not in self._grounded_anchors:
                continue
            b = self.anchor_bloch(name)
            anchors[name] = {
                "bloch": tuple(round(float(v), 4) for v in b),
                "grounded": name in self._grounded_anchors,
            }
        return {
            "seed_profile": getattr(self, "seed_profile", "blank"),
            "anchors": anchors,
            "run_mode": ContextState.from_bundle(bundle).snapshot(),
            # the manifold view: per-axis value + the brain's CONFIDENCE (purity) in its run-mode
            "run_mode_beliefs": ContextState.beliefs(bundle),
        }

    # ================================================================
    # Periodic drivers (the domain's clocks)
    # ================================================================

    def _anchor_drivers(self, now: datetime | None,
                        explicit: dict[str, tuple[float, float, float]] | None = None) -> None:
        """Anchor every driver's (node, role) qubit toward its deterministic phase.

        Re-anchors EVERY tick (a continuous pursuit): the field dephases the qubit
        between ticks, so a sparse observe washes out — per-tick anchoring holds the
        phase on its point of the sphere while still letting the field wobble
        (alpha < 1). Full position, not a pole. Effective anchor strength =
        base driver_alpha × per-driver learnable TRUST weight: a clock earns its
        influence the way a signal does (trust is Hebbian-credited after the fractal
        step). Before applying the anchor we record the qubit's pre-anchor Bloch —
        the field's OWN forecast — to measure anticipation skill against the truth.

        `explicit` payloads ("node:role" -> (x,y,z), the ingest driver_observations
        kwarg) win over computed targets — an app may supply richer phase sources
        (an ephemeris service, an exchange calendar) out of band.
        """
        self._driver_applied = {}   # sensor_id -> target z (for Hebbian credit)
        # Refresh the cached targets from the injected drivers (deterministic).
        if now is not None:
            for d in self.drivers:
                key = f"{d.node}:{d.role}"
                try:
                    self.driver_targets[key] = tuple(float(v) for v in d.target_bloch(now))
                    self._clock_last_target[d.name] = {
                        "phase": round(float(d.phase(now)), 6),
                        "bloch": self.driver_targets[key],
                    }
                except Exception as e:
                    logger.warning("driver %s target failed: %s", d.name, e)
        targets = {**self.driver_targets, **(explicit or {})}
        if not targets:
            return
        # All knobs live-read from the fiber (no buried literals). The
        # alpha clamp uses the driver_alpha param's own bounds.
        base_alpha = self._root_param("driver_alpha", 0.35)
        antic_ema = self._root_param("driver_anticipation_ema", 0.02)
        alpha_lo, alpha_hi = 0.05, 0.95
        root = getattr(self.field.graph, "root", None)
        ap = (root.param_bundle.get_param("driver_alpha")
              if root is not None and root.param_bundle is not None else None)
        if ap is not None:
            alpha_lo, alpha_hi = ap.lo, ap.hi
        for key, target_bloch in targets.items():
            node_name, _, role = key.partition(":")
            cluster = self.field.clusters.get(node_name)
            if cluster is None:
                continue
            idx = cluster.role_index.get(role)
            if idx is None:
                continue
            sensor_id = f"{node_name}_{role}"
            binding = self.sensor_bridge.bindings.get(sensor_id)
            trust = (binding.weight_param.value
                     if binding is not None and binding.weight_param is not None
                     else 1.0)
            alpha_eff = float(min(alpha_hi, max(alpha_lo, base_alpha * trust)))

            # Anticipation: how close the field already was, pre-anchor.
            pre = cluster.qubit_bloch(idx)
            tgt = np.asarray(target_bloch, dtype=float)
            err = 0.5 * float(np.linalg.norm(pre - tgt))
            prev = self.driver_anticipation.get(key, 1.0 - err)
            self.driver_anticipation[key] = antic_ema * (1.0 - err) + (1.0 - antic_ema) * prev
            self._anticipation_raw.append(err)   # raw series for SNR-adaptive EMA
            prev_skill = self.clock_skill.get(key, 1.0 - err)
            self.clock_skill[key] = antic_ema * (1.0 - err) + (1.0 - antic_ema) * prev_skill

            cluster.observe_qubit(idx, tuple(float(v) for v in target_bloch), alpha_eff)

            # Bookkeeping so the Hebbian credit + dashboards see this driver.
            self.sensor_bridge.touched_roles.add((node_name, role))
            self.sensor_bridge._last_normed[sensor_id] = float(tgt[2])
            self.sensor_bridge._last_raw[sensor_id] = float(tgt[2])
            self._driver_applied[sensor_id] = float(tgt[2])

    def clock_snapshot(self) -> dict | None:
        """Per-driver view of the learned phase qubits (position, purity, skill)."""
        if not self.drivers:
            return None
        out: dict[str, dict] = {}
        for d in self.drivers:
            cluster = self.field.clusters.get(d.node)
            if cluster is None:
                continue
            idx = cluster.role_index.get(d.role)
            if idx is None:
                continue
            x, y, z = (float(v) for v in cluster.qubit_bloch(idx))
            key = f"{d.node}:{d.role}"
            out[d.name] = {
                "phase": round(bloch_to_phase(x, y), 6),
                "bloch": {"x": round(x, 6), "y": round(y, 6), "z": round(z, 6)},
                "purity": round(float(np.linalg.norm([x, y, z])), 6),
                "skill": round(float(self.clock_skill.get(key, 0.0)), 6),
                "target": self._clock_last_target.get(d.name),
                "source": "driver",
            }
        return out or None

    def set_driver_targets(
        self,
        targets: dict[str, tuple[float, float, float]],
        forecast_labels: dict[str, tuple[float, float, float]] | None = None,
    ) -> None:
        """Update the cached driver phase targets ("node:role" -> (x,y,z)).

        Called by a slow app loop when phase sources live out of band; every
        subsequent ingest re-anchors the qubits toward these targets so they hold
        position between the loop's sparse updates. (Engine-built drivers refresh
        their own entries each tick — this seam is for external sources.)

        forecast_labels (optional) give the same anchors' positions one horizon
        ahead (deterministic phase-time, not clock time). They supervise a
        forecaster — the field learning to anticipate the cycle (P4).
        """
        self.driver_targets.update(dict(targets))
        if forecast_labels is not None:
            self.driver_forecast_labels = dict(forecast_labels)

    # ================================================================
    # Parameter fiber
    # ================================================================

    def _bind_param_fiber(self) -> dict | None:
        """Move the WHOLE learnable parameter fiber into qubit space, SECTORED BY REWARD CHANNEL: every
        non-frozen, bounded param across every bundle becomes a QubitBackedParam on the ProductQubitCluster
        of its reward sector (`_fiber_surprise/_skill/_override/_unlearned`) — Thompson-wriggle on read,
        purity = confidence, Berry-phase history. Each param carries a receptor profile (its sector,
        the neuromodulator that collapses it). Every `bundle.get`/`update` callsite is unchanged (the
        facade). Frozen/unbounded params stay scalar; shared archetype params bind ONCE (dedup by id).

        IDEMPOTENT: re-running skips already-qubit-backed params (safe after the post-load re-configure;
        merge is non-destructive). The sector clusters live in field.clusters directly, NOT graph nodes."""
        from umwelt.substrate.product_cluster import ProductQubitCluster
        from umwelt.substrate.qubit_param import QubitBackedParam
        from umwelt.learning.reward.registry import CHANNELS, channel_for, receptor_for
        root = getattr(self.graph, "root", None)
        if root is None:
            return None
        # Ensure one ProductQubitCluster per reward sector exists in the field.
        sectors: dict[str, ProductQubitCluster] = {}
        for ch in CHANNELS.values():
            c = self.field.clusters.get(ch.fiber_cluster)
            if not getattr(c, "is_product", False):
                c = ProductQubitCluster(ch.fiber_cluster)
                self.field.clusters[ch.fiber_cluster] = c
            sectors[ch.name] = c
        seen: set[int] = set()
        bound: dict[str, int] = {name: 0 for name in sectors}
        for node in root.walk():
            bundle = getattr(node, "param_bundle", None)
            if bundle is None:
                continue
            for key in list(bundle.params.keys()):
                p = bundle.params[key]
                if id(p) in seen:
                    continue
                if isinstance(p, QubitBackedParam):       # already on a qubit (dedup / re-run)
                    seen.add(id(p)); continue
                if getattr(p, "frozen", False) or p.lo is None or p.hi is None or p.lo >= p.hi:
                    continue                              # physical constant / unbounded → stays scalar
                ch_name = channel_for(node.name, key)     # the reward sector that owns this param
                fiber = sectors[ch_name]
                role = f"_param_{node.name}_{key}"         # identity independent of cluster (re-homable)
                idx = fiber.add_role(role)
                try:
                    bundle.bind_qubit(key, fiber, idx)
                    qp = bundle.params[key]
                    qp.receptor = receptor_for(node.name, key)   # stamp the neuromodulator receptor
                    seen.add(id(qp))
                    bound[ch_name] += 1
                except Exception as exc:
                    logger.warning("fiber: failed to bind %s.%s: %s — staying scalar", node.name, key, exc)
        logger.info("param fiber sectored: %s", {n: bound[n] for n in bound if bound[n]})
        return sectors

    def reward_snapshot(self) -> dict:
        """Per-reward-channel view of the fiber: each neuromodulator's sector cluster, how many
        params it owns, and their mean purity (settledness ∈ [0,1] — how confident the brain is in that
        sector) + the channel's release tone. The inspectable face of the multi-reward substrate."""
        from umwelt.learning.reward.registry import CHANNELS
        out: dict = {}
        for ch in CHANNELS.values():
            c = self.field.clusters.get(ch.fiber_cluster)
            if not getattr(c, "is_product", False):
                continue
            n = c.n_qubits
            if n:
                mean_purity = float(np.mean([float(np.linalg.norm(c.qubit_bloch(i))) for i in range(n)]))
            else:
                mean_purity = 0.0
            out[ch.name] = {"cluster": ch.fiber_cluster, "qubits": n,
                            "mean_purity": round(mean_purity, 4),
                            "release_level": ch.release_level, "timescale": ch.timescale}
        return out

    def _collect_berry_phases(self) -> list[float]:
        """The real per-qubit geometric phases, summed by the ticker.

        Each value is γ = ∮ -½(1-cosθ)dφ·|r| accumulated along one qubit's
        Bloch trajectory under the Lindblad dynamics (see BlochGeometricPhase).
        The ticker sums them into the global phase and tracks its velocity.
        """
        return list(self.bloch_berry.phases.values())

    def _compact_bloch_z(self) -> dict[str, float]:
        """Compact snapshot: one representative Bloch-z per cluster.

        Uses the mean z-component across all qubits in each cluster.
        This is the "what does the field believe right now" snapshot
        that gets stamped onto the Berry tape.
        """
        snap = {}
        for name, cluster in self.field.clusters.items():
            z_vals = [
                float(cluster.qubit_bloch(i)[2])
                for i in range(cluster.n_qubits)
            ]
            snap[name] = round(sum(z_vals) / max(len(z_vals), 1), 4)
        return snap

    def attach_surprise_tape(self, tape: SurpriseTape) -> None:
        """Wire a SurpriseTape into the ingest loop.

        Call once after engine construction. After this, every ingest()
        records per-learner prediction errors onto the tape.
        """
        self.surprise_tape = tape
        logger.info("SurpriseTape attached: %s", tape.db_path)

    def attach_stream_tape(self, tape) -> None:
        """Wire a StreamTape (bounded gauge-pruned per-stream VALUE history) — the learner's data source.
        After this, the app's ingest service tees raw signal readings onto it, so an offline learner
        reads a compact fiber store in ~ms instead of scanning the raw events firehose."""
        self.stream_tape = tape
        logger.info("StreamTape attached (fiber history): %s", tape.db_path)

    # Minimum EMA delta to register as a new surprise observation. Below this,
    # the learner hasn't moved and we treat the emit as a heartbeat.
    _SURPRISE_DELTA_EPS = 1e-6

    def _substrate_emit_deadband(self) -> float:
        """Relative deadband for substrate surprise emission (learnable prior)."""
        root = getattr(self.graph, "root", None)
        if root is not None and root.param_bundle is not None:
            return root.param_bundle.get("substrate_emit_deadband", 0.1)
        return 0.1

    def _emit_surprise(self, source: str, surprise: float, phase: float,
                       metadata: dict, tick: int | None = None) -> None:
        """Emit to the surprise tape only when something actually happened.

        Gates:
          - If `tick` is given (substrate learners: epoch/step counter),
            emit only when that counter advances AND the surprise *level* has
            shifted by more than a relative deadband since the last emit. A
            learner settling at a steady rate ticks every step with a
            near-constant value — that floods the feed and isn't news.
          - Otherwise, emit only when the value moved by more than EPS.
        """
        if tick is not None:
            prev_tick = self._last_emit_counter.get(source)
            if prev_tick is not None and tick == prev_tick:
                return
            self._last_emit_counter[source] = tick
            # Relative deadband vs the last *emitted* value (learnable prior).
            prev = self._last_observed_surprise.get(source)
            if prev is not None:
                deadband = self._substrate_emit_deadband()
                if abs(surprise - prev) < deadband * max(abs(prev), 1e-9):
                    return
        else:
            prev = self._last_observed_surprise.get(source)
            if prev is not None and abs(surprise - prev) < self._SURPRISE_DELTA_EPS:
                return
        self._last_observed_surprise[source] = surprise
        self.surprise_tape.observe(
            source=source,
            surprise=surprise,
            berry_phase=phase,
            metadata=metadata,
        )

    def _record_surprise(self) -> None:
        """Pull EMAs from each learner and push onto the surprise tape.

        Called once per ingest(), after all components have stepped. Reads
        the already-updated EMAs — no recomputation, no extra work.
        """
        tape = self.surprise_tape
        if tape is None:
            return
        phase = self.berry_tape.ticker.phase

        # Fractal stack: one observation per scale
        if self.fractal_stack is not None:
            for scale in self.fractal_stack.scales:
                s = float(scale._surprise_ema)
                if abs(s) < 1e-10:
                    continue
                self._emit_surprise(
                    source=f"fractal_scale_{scale.level}",
                    surprise=s,
                    phase=phase,
                    metadata={
                        "category": "substrate",
                        "h_scale": round(scale.params.get("h_scale"), 6),
                        "hebbian_lr": round(scale.params.get("hebbian_lr"), 6),
                        "step": scale._step,
                    },
                    tick=scale._step,
                )

        # Calibration Channel 2 — per-node dynamics surprise
        if self.calibration is not None:
            for name, ema in self.calibration._surprise_ema.items():
                s = float(ema)
                if abs(s) < 1e-10:
                    continue
                node = self.graph.find(name)
                gamma = None
                if node and node.param_bundle and "gamma" in node.param_bundle:
                    gamma = round(node.param_bundle.get("gamma"), 6)
                meta = {"category": "grounded"}
                if gamma is not None:
                    meta["gamma"] = gamma
                self._emit_surprise(
                    source=f"calibration_dynamics:{name}",
                    surprise=s,
                    phase=phase,
                    metadata=meta,
                )

            # Calibration Channel 5 — per-role tracking error
            for key, ema in self.calibration._tracking_ema.items():
                s = float(ema)
                if abs(s) < 1e-10:
                    continue
                self._emit_surprise(
                    source=f"calibration_tracking:{key}",
                    surprise=s,
                    phase=phase,
                    metadata={"category": "grounded"},
                )

        # Training runner
        if self.training is not None:
            s = float(self.training._surprise_ema)
            if abs(s) >= 1e-10:
                self._emit_surprise(
                    source="training",
                    surprise=s,
                    phase=phase,
                    metadata={
                        "category": "substrate",
                        "lr": round(self.training.effective_lr, 6),
                        "epoch": self.training._epoch,
                    },
                    tick=self.training._epoch,
                )

    @property
    def feature_dim(self) -> int:
        """Total feature dimensionality across all clusters."""
        return sum(c.features().shape[0] for c in self.field.clusters.values())

    # ================================================================
    # Main loop: ingest -> evolve -> collapse -> features
    # ================================================================

    def _root_param(self, key: str, default: float) -> float:
        """Read a fiber prior off the root bundle (with default if absent).

        The root bundle holds the tower's global priors (driver_alpha,
        forecast_lr/l2/ema, nudge bounds, etc.). This helper kills the
        five-line `getattr(...) ... .param_bundle.get(...)` guard that was
        sprinkled across the driver block — one source of truth, one
        guard, no duplication.
        """
        root = getattr(self.field.graph, "root", None)
        if root is None or root.param_bundle is None:
            return default
        return root.param_bundle.get(key, default)

    def anticipation_snr(self) -> float | None:
        """SNR proxy of the raw anticipation-error series: var(x)/var(Δx).

        Reads the un-smoothed series (non-circular) so the calibrator can match
        the anticipation EMA's bandwidth to the signal. None until enough samples.
        """
        if len(self._anticipation_raw) < 16:
            return None
        x = np.asarray(self._anticipation_raw, dtype=float)
        var_diff = float(np.var(np.diff(x)))
        if var_diff < 1e-12:
            return None
        return float(np.var(x)) / var_diff

    def ingest(
        self,
        sensor_readings: dict[str, float] | None = None,
        raw_inputs: dict[str, NDArray] | None = None,
        driver_observations: dict[str, tuple[float, float, float]] | None = None,
        now: datetime | None = None,
        coast_secs_to_event: float | None = None,
        confidence: dict[str, float] | None = None,
    ) -> dict:
        """
        Process one timestep of signal data through the engine.

        The field always evolves (sky never stops).
        Collapse happens per policy (periodic + confidence-triggered).

        Args:
            sensor_readings: sensor_id -> raw value (goes through SensorBridge)
            raw_inputs: node_name -> input array (bypasses SensorBridge)
            driver_observations: "node:role" -> (x, y, z) unit Bloch target.
                Deterministic driver phase (an ephemeris, a session calendar)
                partially collapses the anchor qubit toward its true point via
                observe_qubit after field.step — the same SKY-update path as
                signal observe, but the target is a full position, not a pole.
                Explicit payloads win over the injected drivers' computed ones.

        Returns:
            dict with keys:
                features: current feature vector
                prediction: reserved (always None — forecasting lives in the
                    injected forecasters, P4)
                collapsed: bool (whether any collapse happened)
                transitions: list[Transition]
                actions: list[Action] (tendril commands)
                step: current step number
        """
        # Route signals to cluster inputs (confidence gates/scales the dissipative push)
        if sensor_readings:
            bridged = self.sensor_bridge.process(sensor_readings, confidence)
        else:
            bridged = {}

        inputs = {**bridged, **(raw_inputs or {})}

        # ── STASIS GATE ──
        # sensor_bridge.process() above already cached the raw readings (dashboards
        # / signal echo stay live — the surface keeps sensing). But when paused the
        # brain is DISCONNECTED from that surface: no field evolution, no collapse,
        # no calibration/fractal/training/population/surprise, no step-clock tick.
        # Every ingest caller funnels through here, so this one gate is full stasis.
        # Return the documented result shape with empty actions/transitions.
        if self.paused:
            return {
                "features": None, "prediction": None, "collapsed": False,
                "transitions": [], "actions": [], "step": self._step, "paused": True,
            }

        # ── SMOOTH ADAPTIVE CLOCK (cadence) × FOREBRAIN/HINDBRAIN (work-split) ──
        # Two orthogonal compute levers, composed (the smooth clock + the forebrain/
        # hindbrain split):
        #  • SMOOTH CLOCK (self.adaptive; None = OFF → every tick, dt_scale=1): when calm
        #    AND no input, run the full tick only every N=round(dt_factor) ticks; between,
        #    SKIP (advance the deterministic fibers only) and catch the skipped sim-time
        #    up via dt_scale on the next full step. Binary coast is its calm limit.
        #  • FOREBRAIN/HINDBRAIN (context_learn): even on a FULL tick, the FOREBRAIN
        #    (learn≈0) skips global_features (the kron hog) + every learner —
        #    evolve→collapse→ACTUATE only — while the HINDBRAIN (learn=1, REPLAY) pays it
        #    to learn off the backlog. See learning/context.py + phi_clock.fib_strides_at.
        _skip = False
        _dt_scale = 1.0
        if self.adaptive is not None and not sensor_readings and not raw_inputs:
            _speed = self.berry_tape.ticker.speed
            _surprise = (self.fractal_stack.scales[0]._surprise_ema
                         if self.fractal_stack is not None and self.fractal_stack.scales
                         else 0.0)
            _N = max(1, int(round(self.adaptive.decide(
                coast_secs_to_event, _speed, _surprise).dt_factor)))
            self._since_full += 1
            if self._since_full < _N:
                _skip = True
            else:
                _dt_scale = float(self._since_full)   # catch up the skipped sim-time
                self._since_full = 0
        else:
            self._since_full = 0                       # input (or no clock) → full step
        _learning = self._root_param("context_learn", 1.0) > 0.5

        # LIVE STREAMING LEARNING: a frozen forebrain still learns ONLINE from the live tick every Nth
        # full tick (1-in-N keeps real-time). Only kicks in when learning is otherwise gated off (the
        # forebrain) and the tick isn't skipped by the smooth clock — so it never double-runs the
        # hindbrain (already learn=1) and never pays cost on a coast tick. N→1 once the field offloads.
        _stream_learned = False
        if self._stream_learn and not _learning and not _skip:
            self._stream_tick += 1
            if self._stream_tick >= self._stream_learn:
                _learning = True
                _stream_learned = True
                self._stream_tick = 0
                self._stream_count += 1
                _surp = (self.fractal_stack.scales[0]._surprise_ema
                         if self.fractal_stack is not None and self.fractal_stack.scales else 0.0)
                logger.info("STREAM-LEARN: online learn tick #%d (step=%d, surprise=%.4f) — "
                            "the brain learning from the live world", self._stream_count, self._step, _surp)

        # ── MEMBRANE CADENCE (spec.ingest_hold_s; None = OFF → ingest-driven,
        # byte-identical to the origin behavior). A sparse feed's batches arrive with
        # wall gaps the dense-polled origin never had; when the spec declares a hold
        # window, the previous batch's delivered inputs persist across the gap they
        # close (a close stands for its session; a reading stands until the next
        # one), advanced in bounded unit compute substeps before this batch's input
        # step. A channel absent from the previous batch contributes nothing during
        # the gap and relaxes — holds span ONE gap, never inventing a longer past.
        # This is cadence plumbing at the ingest membrane, NOT a time model — the
        # world's clocks are drivers (opt-in in-universe qubits), and dt is compute
        # step size (docs/TIME.md). Orthogonal to the adaptive clock (compute
        # compression on calm ticks); driver anchoring still happens once per ingest.
        if (self.ingest_hold_s and now is not None and self._last_wall_now is not None
                and not _skip):
            _gap = (now - self._last_wall_now).total_seconds() / float(self.ingest_hold_s)
            _catchup = min(int(_gap) - 1, self._wall_catchup_max)
            _held = self._last_bridged_inputs or {}
            for _ in range(max(0, _catchup)):
                self.field.step(_held, dt_scale=1.0)
                self._wall_catchup_steps += 1
        if now is not None:
            self._last_wall_now = now
        if self.ingest_hold_s:
            self._last_bridged_inputs = dict(inputs) if inputs else None

        # Evolve the quantum field (sky never stops).
        # All signals go through the unified pipeline:
        #   - Unitary roles (motion, contact): σ_x Hamiltonian kick
        #   - Dissipative roles (continuous readings): thermal Lindblad thermalization
        # No classical bypass — everything through the Lindblad master equation.
        if not _skip:
            if self._tick_timing:
                import time as _t
                _t0 = _t.perf_counter()
                self.field.step(inputs, dt_scale=_dt_scale)
                _ms = (_t.perf_counter() - _t0) * 1000.0
                # EMA over full ticks; periodic log so each A/B config's steady
                # state is readable in the journal. Gated by UMWELT_TICK_TIMING=1.
                self._tick_ema = _ms if self._tick_ema is None else (
                    0.95 * self._tick_ema + 0.05 * _ms)
                self._tick_n += 1
                if self._tick_n % self._tick_every == 0:
                    # Report the LIVE dtype, not a re-read of the env var — the real
                    # branch lives in density_matrix.EVOLVE_DTYPE (default fp32).
                    from umwelt.substrate.density_matrix import EVOLVE_DTYPE as _dtype
                    logger.info(
                        "TICK field.step EMA=%.2f ms (last=%.2f) [dtype=%s batched] n=%d",
                        self._tick_ema, _ms,
                        "fp64" if _dtype == np.complex128 else "fp32",
                        self._tick_n,
                    )
            else:
                self.field.step(inputs, dt_scale=_dt_scale)
        self._step += 1

        # ── Observation collapse (SKY update) ──
        # Discrete-quality signals (a switch's on/off, a plug's draw) hold a
        # *belief* qubit that drifts via field.step (Hamiltonian + decoherence).
        # When reality is actually observed, we partially collapse that belief
        # toward the seen state — distinct from the read-only SKY→GROUND
        # projection below. This is what makes a just-flipped switch read sharp
        # and certain, then soften as the belief drifts until the next reading.
        if sensor_readings:
            for (node_name, role), (target_bloch, alpha, conf) in (
                self.sensor_bridge.observe_targets(sensor_readings, confidence).items()
            ):
                cluster = self.field.clusters.get(node_name)
                if cluster is None:
                    continue
                idx = cluster.role_index.get(role)
                if idx is None:
                    continue
                # Direction convention: a spec-FORCED observe binding on a regular
                # (unitary/dissipative) role keeps its normalizer's own direction —
                # asserted → +z, the ground model's asserted() pole. The device-echo
                # energy flip the bridge applies (active → z = -1, the glowing pole)
                # belongs only to REGISTERED observe roles (the origin's device-state
                # vocabulary), so undo it here for everything else.
                from umwelt.spec.roles import is_observe_role
                if not is_observe_role(role):
                    target_bloch = (float(target_bloch[0]), float(target_bloch[1]),
                                    -float(target_bloch[2]))
                # DISSOLUTION (UMWELT_LEARN_COLLAPSE, default off = parity): earn the collapse rate
                # instead of the hand-set 0.95/0.15. The innovation |obs − belief| over time IS the
                # signal's learned noise; a reliable leaf earns a high alpha (snap), a noisy one a low
                # alpha (smooth). Confidence still rides on top via conf_brake. See observation_trust.py.
                from umwelt.learning.observation_trust import LEARN_COLLAPSE
                # ALWAYS track observation innovation so per-leaf RELIABILITY (the
                # learned trust coordinate) is readable everywhere (host.api.beliefs
                # and the hearth /beliefs endpoint). Tracking is a cheap per-leaf EMA;
                # when LEARN_COLLAPSE is off the learned alpha is NOT used for collapse,
                # so the dynamics stay BIT-IDENTICAL to the hand-set collapse_alpha
                # (exact parity). It only DRIVES the collapse rate when the gate is set.
                trust = getattr(self, "_obs_trust", None)
                if trust is None:
                    from umwelt.learning.observation_trust import ObservationTrust
                    trust = self._obs_trust = ObservationTrust()
                belief_z = float(cluster.role_bloch(role)[2])
                learned_alpha = trust.learned_alpha(
                    (node_name, role), float(target_bloch[2]), belief_z)
                if LEARN_COLLAPSE:
                    from umwelt.membranes.ingress import conf_brake
                    alpha = learned_alpha * conf_brake(float(conf) if conf is not None else 1.0)
                # alpha already folds in confidence; pass conf so it's recorded as a
                # gauge quantity (gauge.cluster_gauge), symmetric with purity.
                if self._belavkin:
                    # Belavkin weak measurement (UMWELT_BELAVKIN=1): conditioned
                    # Kraus update with the bounded Wonham gain + the cumulant
                    # cross-update — strength reuses the pre-folded alpha
                    # (equator-matched to the blend).
                    cluster.measure_qubit(idx, float(target_bloch[2]), alpha,
                                          confidence=conf)
                else:
                    cluster.observe_qubit(idx, target_bloch, alpha, confidence=conf)

        # ── Driver anchoring (SKY update) ──
        # Re-anchor every periodic driver's phase qubit toward its true (x, y, z)
        # point EVERY tick (computed targets ∪ any explicitly passed this call).
        # See _anchor_drivers. Guarded — a bad driver never crashes the tick.
        try:
            self._anchor_drivers(now, driver_observations)
        except Exception as e:
            logger.warning("driver anchoring failed: %s", e)

        # One override loop over every tendril: each detects the operator overriding its last
        # dispatch (reading its own readback signal) and pulls the manifold toward reality.
        # SKIPPED while LISTENING (|act⟩ low): we didn't dispatch, so the operator's manual changes
        # are CLEAN observation (flowing in via signals), not overrides of a phantom command.
        # The agency read is narrowed to the one boot-shaped failure (agency not built yet) and
        # logged once; True stays the conservative default (overrides still apply when the read
        # is unavailable).
        _agency_acting_ov = True
        try:
            _agency_acting_ov = self.agency.is_acting   # qubit measurement (anchor-derived), not a magic 0.3
        except AttributeError:
            if not getattr(self, "_agency_read_warned", False):
                self._agency_read_warned = True
                logger.warning("agency.is_acting unavailable (boot?) — defaulting to acting=True")
        if _agency_acting_ov:
            for tendril in getattr(self, "tendrils", []):
                try:
                    tendril.apply_override()
                except Exception as e:
                    logger.warning("%s override check failed: %s", getattr(tendril, "name", "tendril"), e)

        # ── SMOOTH-CLOCK COAST: on a calm SKIPPED tick the deterministic fibers
        # (the drivers) already advanced analytically; skip the whole learn +
        # collapse + output path. The field held its parked state — nothing to
        # learn or actuate. This is the bulk of the smooth-clock compute compression.
        if _skip:
            self._coasted_steps += 1
            return {
                "features": None, "prediction": None, "collapsed": False,
                "transitions": [], "actions": [], "step": self._step,
                "berry_phase": self.berry_tape.ticker.phase,
                "berry_velocity": self.berry_tape.ticker.velocity,
                "coasted": True,
            }

        # ── HINDBRAIN-ONLY: the learners + feature decomposition (the heavy cost) ──
        # On a FULL tick the FOREBRAIN (context_learn≈0) still skips this — the
        # `if _learning:` block below pays global_features (the kron hog), the fractal
        # stack, calibration only when learn=1 (the HINDBRAIN, REPLAY). So forebrain =
        # evolve→collapse→actuate; hindbrain learns off the backlog.
        features = None
        prediction = None
        if _learning:
            # Fractal stack: meta-fields evolve from residuals, project H downward
            if self.fractal_stack is not None:
                # Feed Berry velocity so the stack can gate H projection
                self.fractal_stack.berry_velocity = self.berry_tape.ticker.velocity
                self.fractal_stack.step()
                # Signal weight Hebbian learning: credit/penalize signals that
                # helped/hurt the field's ability to track reality.
                if sensor_readings:
                    self.sensor_bridge.hebbian_weight_update(
                        self.fractal_stack.last_raw_residuals
                    )
                # Driver trust learning: credit each anchor every tick it was applied.
                if self._driver_applied:
                    self.sensor_bridge.hebbian_driver_update(
                        self.fractal_stack.last_raw_residuals,
                        lr=self._root_param("driver_hebbian_lr", 0.01),
                    )

            # Fractal-web width breathing: pool device-bank dynamics on a slow
            # φ-clock (gated UMWELT_SIMILARITY_GROUPING, default OFF). Slower than the
            # depth spawn/prune above so the two breathing axes don't oscillate.
            self._maybe_pool_dynamics()
            # Fractal-web topology breathing: grow/prune learned bridges on an
            # even slower clock (gated UMWELT_LEARNED_TOPOLOGY, default OFF).
            self._maybe_evolve_web()

            # Training runner: adapt learning rate, track metrics
            if self.training is not None:
                self.training.step(self.fractal_stack)

            # Calibrate parameter fiber from prediction residuals
            if self.calibration is not None:
                self.calibration.step(sensor_readings, self.sensor_bridge)

            # Extract features (the expensive Gell-Mann/kron decomposition)
            features = self.field.global_features()

            # (The origin trained its anticipation forecaster here on the
            # driver_forecast_labels horizon; the generic Forecaster protocol
            # lands with foresight/ (P4) and will loop self.forecasters.)

        # Collapse policy
        transitions = []
        collapsed = False

        # Periodic collapse
        periodic_t = self.collapse_engine.check_periodic(self.field, self.world)
        if periodic_t:
            transitions.extend(periodic_t)
            collapsed = True

        # Auto-collapse high-confidence qubits
        auto_t = self.collapse_engine.auto_collapse(self.field, self.world)
        if auto_t:
            transitions.extend(auto_t)
            collapsed = True

        # Calibrate collapse thresholds from outcomes. This is a LEARNER (it updates the
        # confidence_threshold fiber param via observe_qubit), so it must respect the freeze
        # gate like every other learner — otherwise a frozen forebrain keeps learning
        # confidence_threshold and the non-training certificate cannot hold. Live (learn=1)
        # is unaffected.
        if _learning and self.calibration is not None:
            self.calibration.calibrate_collapse(auto_t, self.field)

        # Actions accumulate from the tendril loop below. (The origin routed
        # collapse transitions through an actuator catalog here; in the engine
        # that surface is P3's egress OutputSpec — until then transitions only
        # commit to the ground.)
        actions: list = []

        # Agency qubit: drive the act↔listen axis from the silence lever, else heal toward
        # 'recommend' over wall-clock weeks. Read-only on the field — deliberately driven, never
        # coupled to the noisy world clusters. The dispatcher reads its |act⟩ as the smooth silence
        # layer atop the scalar floor; the non-divergence invariant holds (the scalar drives it).
        try:
            import time as _t_agency
            _rb_ag = getattr(getattr(self.graph, "root", None), "param_bundle", None)
            _silenced = (float(_rb_ag.get("actuation_silenced", 0.0)) >= 0.5) if _rb_ag else False
            # The operator's listen↔act PREFERENCE (stimulus into the graph belief). When set
            # (alpha>0) it drives the qubit toward the wish; the binary silence is the legacy special
            # case (a listen preference). No preference → the qubit relaxes on its own.
            _pref_z = float(_rb_ag.get("agency_pref_z", 0.0)) if _rb_ag else 0.0
            _pref_alpha = float(_rb_ag.get("agency_pref_alpha", 0.0)) if _rb_ag else 0.0
            # Confidence→act coupling (opt-in): released silence heals to RECOMMEND by default; with
            # agency_auto_act_enabled, the brain's LEARNED COMPETENCE folds it up toward ACT — the
            # real "trust earns agency" handoff. competence = learnedness × prediction-skill
            # (competence.py): the reload-surviving signal that the model has both SETTLED and
            # stopped being SURPRISED, so the brain takes over because it LEARNED, not on a wall-clock.
            _auto_act = (float(_rb_ag.get("agency_auto_act_enabled", 0.0)) >= 0.5) if _rb_ag else False
            _conf = competence_score(self) if _auto_act else 0.0
            # The two remaining hand-set anchors read LIVE from the fiber each tick —
            # recommend_z (the resting stance; moves the regime geometry) and the competence
            # knee (below it, competence buys no climb toward act). Both default to the old
            # law's values (0.0) → byte-identical until learned/set.
            if _rb_ag:
                self.agency.recommend_z = float(_rb_ag.get("agency_recommend_z", self.agency.recommend_z))
            _knee = float(_rb_ag.get("agency_competence_knee", 0.0)) if _rb_ag else 0.0
            self.agency.tick(_t_agency.time(), silenced=_silenced, confidence=_conf, auto_act=_auto_act,
                             prefer_z=_pref_z, prefer_alpha=_pref_alpha, knee=_knee)
        except Exception as e:
            logger.warning("agency tick failed: %s", e)

        # INTENTION / pre-side gate: when the agency qubit is LISTENING the brain
        # does not even FORM actions — so no phantom dispatch_echo is stamped and apply_override
        # can't misfire on the operator's clean manual changes. The dispatcher is the hard
        # backstop; this is the brain not forming an intention it can't carry out
        # (no thinking-it-acted-when-it-didn't).
        _agency_acting = True
        try:
            _agency_acting = self.agency.is_acting       # qubit measurement (anchor-derived), not a magic 0.3
        except AttributeError:                           # narrowed from except-Exception (see above)
            if not getattr(self, "_agency_read_warned", False):
                self._agency_read_warned = True
                logger.warning("agency.is_acting unavailable (boot?) — defaulting to acting=True")
        # Tendril actuator loop: ONE path for every output. Each tendril reads the comprehension
        # manifold (its relevant qubits), decodes to a device command, gates, and emits an Action
        # whose reason="<name>_auto" tells the egress surface to route it APPROVED (auto-executed)
        # rather than SUGGESTED. Shadow until each tendril's enable gate flips. New outputs append
        # to self.tendrils with no new ingest block (P3 builds them from OutputSpec). See
        # membranes/tendril.py.
        if _agency_acting:
            # Tendrils tick in EVENT time (the batch's `now`), wall clock only as
            # the no-timestamp fallback. Wall clock here made the rate-limit gate
            # replay-speed-dependent: a from-log replay compresses hours into
            # seconds, so its dispatch pattern (and everything a dispatch feeds)
            # diverged run-to-run — the deterministic-replay contract leaked
            # (found root-causing the 2026-07-18 lease-drill chain fork).
            _tendril_now = now.timestamp() if now is not None else None
            for tendril in getattr(self, "tendrils", []):
                try:
                    auto_action = tendril.step(_tendril_now)
                    if auto_action is not None:
                        actions = actions + [auto_action]
                except Exception as e:
                    logger.warning("%s tendril step failed: %s", getattr(tendril, "name", "tendril"), e)
        # Route this tick's decisions through the egress surface when one is attached:
        # shadow/recommend decisions are recorded for the app (the ghost layer), auto
        # non-shadow ones go to the injected dispatcher. The result dict still carries
        # every action either way — the surface routes, it never hides.
        if actions and getattr(self, "output_surface", None) is not None:
            try:
                self.output_surface.route(actions, getattr(self, "tendrils", []))
            except Exception as e:
                logger.warning("output surface routing failed: %s", e)

        # ── Berry tape: advance the real geometric phase, then tick ──
        for name, cluster in self.field.clusters.items():
            for i in range(cluster.n_qubits):
                self.bloch_berry.update(f"{name}:{i}", cluster.qubit_bloch(i))
        self.berry_tape.tick(self._collect_berry_phases())

        # ── Surprise tape: record per-learner prediction errors ──
        # Reads the EMA from each component; gating + reservoir handle
        # sampling, so we can call observe() every step without write pressure.
        if self.surprise_tape is not None:
            self._record_surprise()

        # Stamp signal events on the berry tape. Only event-driven signals
        # (motion, contact, lock, state) earn a journal entry — continuous
        # readings drive the field via sensor_bridge but would otherwise
        # overwhelm the tape with substrate jitter.
        bloch_snap = None
        if sensor_readings:
            for sensor_id, raw_value in sensor_readings.items():
                binding = self.sensor_bridge.bindings.get(sensor_id)
                if binding is None or not (binding.is_event_driven or binding.is_observe):
                    continue
                if bloch_snap is None:
                    bloch_snap = self._compact_bloch_z()
                # Observe events are measurements (reality correcting belief);
                # stamp them as such so the journal shows the collapse moment.
                event_type = "measurement" if binding.is_observe else binding.event_type
                self.berry_tape.stamp_sensor(
                    sensor_id, raw_value,
                    event_type=event_type,
                    event_driven=True,
                    bloch_z_snap=bloch_snap,
                )

        # Stamp collapse transitions on the berry tape AND cross-feed them
        # into the surprise tape — a collapse is, by construction, a moment
        # the world model surprised itself enough to commit to a new state.
        for t in transitions:
            if bloch_snap is None:
                bloch_snap = self._compact_bloch_z()
            self.berry_tape.stamp_collapse(
                t.node, t.role,
                str(t.from_state), str(t.to_state),
                bloch_snap,
            )
        if self.surprise_tape is not None and transitions:
            phase = self.berry_tape.ticker.phase
            for t in transitions:
                mag = float(t.confidence) * abs(int(t.to_state) - int(t.from_state))
                if mag <= 0:
                    continue
                self.surprise_tape.observe(
                    source=f"collapse:{t.node}:{t.role}",
                    surprise=mag,
                    berry_phase=phase,
                    metadata={
                        "category": "grounded",
                        "from": str(t.from_state),
                        "to": str(t.to_state),
                    },
                )

        return {
            "features": features,
            "prediction": prediction,
            "collapsed": collapsed,
            "transitions": transitions,
            "actions": actions,
            "step": self._step,
            "berry_phase": self.berry_tape.ticker.phase,
            "berry_velocity": self.berry_tape.ticker.velocity,
        }

    # ================================================================
    # Human feedback as observation
    # ================================================================

    def observe_feedback(
        self,
        node: str,
        role: str,
        value: int,
        *,
        alpha: float = 1.0,
        confidence: float = 1.0,
        decision: str = "confirm",
    ) -> float:
        """Apply a human accept/decline of a recommendation as a hard
        observation of the belief qubit (human = ground truth).

        `value` is in actuator convention (+1 on/active, -1 off); the energy
        convention flips it to the glowing pole (active → z = -1), mirroring
        SensorBridge.observe_targets so a confirmed switch reads the same as a
        signal-echoed one. The belief is partially collapsed toward that pole
        with strength `alpha` (1.0 = full snap), and the distance the belief had
        to jump is stamped on the surprise tape — a confident-but-wrong
        prediction the human overturns lands as large surprise (→ learning); a
        prediction the human confirms that the field already believed lands as
        ~zero. Returns the surprise magnitude in [0, 1] (0.0 if the qubit is
        unknown or nothing moved).
        """
        cluster = self.field.clusters.get(node)
        if cluster is None:
            return 0.0
        idx = cluster.role_index.get(role)
        if idx is None:
            return 0.0
        target_z = -1.0 if value > 0 else 1.0
        pre_z = float(cluster.qubit_bloch(idx)[2])
        cluster.observe_qubit(idx, (0.0, 0.0, target_z), float(alpha))
        self.sensor_bridge.touched_roles.add((node, role))
        mag = 0.5 * abs(target_z - pre_z)            # belief→truth jump, [0,1]
        if self.surprise_tape is not None and mag > 1e-6:
            self.surprise_tape.observe(
                source=f"feedback:{node}:{role}",
                surprise=mag,
                berry_phase=self.berry_tape.ticker.phase,
                metadata={
                    "category": "feedback",
                    "decision": decision,
                    "confidence": round(float(confidence), 3),
                    "pre_z": round(pre_z, 3),
                    "target_z": target_z,
                },
            )
        return round(mag, 4)

    # ================================================================
    # Explicit collapse API
    # ================================================================

    def collapse(
        self,
        node: str | None = None,
        roles: list[str] | None = None,
        reason: CollapseReason = CollapseReason.QUERY,
    ) -> list[Transition]:
        """
        Explicitly collapse part of the field.

        Args:
            node: Node name to collapse (None = all nodes).
            roles: Specific roles to collapse (None = all roles on node).
            reason: Why the collapse is happening.

        Returns:
            List of state transitions that occurred.
        """
        if node is None:
            return self.collapse_engine.collapse_all(
                self.field, self.world, reason
            )
        return self.collapse_engine.collapse_node(
            self.field, self.world, node, reason, roles
        )

    # Fractal-web similarity-driven dynamics pooling.
    _POOL_INTERVAL = 512        # slow φ-clock: runs MUCH rarer than depth spawn/prune
    _POOL_THRESHOLD = 0.05      # conservative: device banks are fp-degenerate (dist 0); regions differ
    _POOL_KEYS = ("gamma",)     # dissipation rate: same device TYPE → same physics → safe to pool

    def _maybe_pool_dynamics(self):
        """Fractal-web width breathing (gated UMWELT_SIMILARITY_GROUPING, default OFF): on a
        slow φ-clock, discover behaviorally-similar DEVICE banks and pool their dynamics
        ScalarParams — the self-organizing extension of the archetype layers (the modular-
        reservoir idea: sub-fields sharing per-module hyperparameters). One learned γ
        per bank instead of one per leaf → more data per param, faster convergence, smaller
        state snapshot.

        Scope is deliberately conservative (validated in the origin deployment):
          • Only groupable DEVICE-kind clusters are pooled. Rich clusters (regions, entities)
            are EXCLUDED — they are deliberately distinct and their fingerprints are too
            close to safely auto-merge; pooling them would erase real differences (the
            over-sharing risk). They climb the meta-tower individually.
          • require_same_roles + the fingerprint act as a DIVERGENCE guard: two same-role
            devices that start behaving differently won't be newly pooled.
        Runs φ-slower than the depth breathing (the meta-tower rule: a tuner runs slower
        than what it tunes), and only in the HINDBRAIN (learn=1) — pooling is a learning act.
        """
        from umwelt._util import env_flag
        if not env_flag("UMWELT_SIMILARITY_GROUPING"):
            return
        if self._step % self._POOL_INTERVAL != 0:
            return
        from umwelt.substrate.similarity import similarity_groups, apply_discovered_sharing
        # Exclude synthetic fibers + every non-device cluster (regions/entities/subdomains/root).
        exclude = set()
        for nm in self.field.clusters:
            node = self.graph.find(nm)
            if node is None or node.kind not in _GROUPABLE_KINDS:
                exclude.add(nm)
        groups = similarity_groups(
            self.field, threshold=self._POOL_THRESHOLD,
            require_same_roles=True, exclude=exclude,
        )
        applied = apply_discovered_sharing(self.graph, groups, shared_keys=self._POOL_KEYS)
        if applied:
            logger.info(
                "web pool: shared %s across %d device bank(s): %s",
                self._POOL_KEYS, len(applied),
                ", ".join(f"{rep}(×{len(m)})" for rep, m in applied.items()),
            )

    # Fractal-web learnable topology.
    _WEB_SAMPLE_INTERVAL = 32    # how often to sample co-movement into the rolling buffer
    _WEB_EVOLVE_INTERVAL = 2048  # grow/prune clock: φ-slower than width pooling (512)

    def _maybe_evolve_web(self):
        """Fractal-web topology breathing (gated UMWELT_LEARNED_TOPOLOGY, default OFF): grow a weak
        learned bridge between clusters that co-MOVE, decay+prune learned edges that don't.
        Emergent connectivity, not declared. A grown edge carries the same ScalarParam
        coupling as every declared bridge → it is an in-gauge coordinate, so the halo stays
        closed (N=0); only kind='learned' edges are ever touched. Sparse by a hard cap.
        Runs slower than width-pooling (the meta-tower rule), HINDBRAIN-only. See
        substrate/web_topology.py."""
        from umwelt._util import env_flag
        if not env_flag("UMWELT_LEARNED_TOPOLOGY"):
            return
        if self._web_topology is None:
            from umwelt.substrate.web_topology import WebTopology
            self._web_topology = WebTopology()
        # Sample co-movement on a fast-ish clock; grow/prune on a much slower one.
        if self._step % self._WEB_SAMPLE_INTERVAL == 0:
            self._web_topology.observe(self.field)
        if self._step % self._WEB_EVOLVE_INTERVAL == 0:
            delta = self._web_topology.maybe_evolve(self.field, self.graph)
            if delta["grown"] or delta["pruned"]:
                logger.info("web topology: grew %s, pruned %s", delta["grown"], delta["pruned"])

    # ================================================================
    # Context (for downstream reasoning and the action system)
    # ================================================================

    def context(self) -> dict:
        """
        Full context combining quantum field and classical world model.

        Downstream reasoning gets both views:
          - 'world': definite commitments (act on these)
          - 'field': probability state (reason about uncertainty)
        """
        fractal = self.field.global_fractal_signature()
        bridge_corr = self.field.bridge_correlations()

        return {
            "world": self.world.context(),
            "field": {
                "step": self._step,
                "fractal_signatures": {
                    name: {
                        "levels": sig,
                        "dimension": fractal_dimension_estimate(sig),
                    }
                    for name, sig in fractal.items()
                },
                "bridges": {
                    f"{a}↔{b}": round(s, 3)
                    for (a, b), s in bridge_corr.items()
                },
            },
            "calibration": self.calibration.stats() if self.calibration else None,
            "population": self.population.stats() if self.population else None,
            "fractal_stack": self.fractal_stack.stats() if self.fractal_stack else None,
            "berry_tape": self.berry_tape.snapshot(),
        }

    def emoji_state(self) -> str:
        """Quick emoji summary of the field (sky view)."""
        return self.field.emoji_state()

    def world_summary(self) -> str:
        """Emoji summary of the world model (ground view)."""
        return self.world.emoji_summary()

    # ================================================================
    # Persistence
    # ================================================================

    def _snapshot_param_fiber(self) -> dict[str, dict[str, tuple[float, float, int]]]:
        """Per-node {param_key: (value, sigma, update_count)} for every node
        with a ParameterBundle. Values-only — bounds and prior come from
        `configure_param_bundles` at boot, so adding/removing/renaming a key
        across versions is graceful (loader skips unknown, defaults missing).

        Shared archetype params (one ScalarParam object referenced by many
        leaves — see param_bundles._attach_archetypes) are stored ONCE: the
        first node that holds a given object in DFS order owns the snapshot
        entry. On restore the shared object is stamped via that one entry and
        every member sees it through the shared reference, so the snapshot
        reflects the parameter-sharing optimization instead of duplicating it.
        """
        from umwelt.substrate.qubit_param import QubitBackedParam
        out: dict[str, dict[str, tuple[float, float, int]]] = {}
        seen: set[int] = set()
        for node in self.graph.root.walk():
            bundle = getattr(node, "param_bundle", None)
            if bundle is None:
                continue
            snap: dict[str, tuple[float, float, int]] = {}
            for key, p in bundle.params.items():
                if id(p) in seen:
                    continue  # shared param already captured under an earlier node
                if isinstance(p, QubitBackedParam):
                    continue  # owned by `param_fiber_qubits` (the 2×2 matrices) — skip the scalar triple
                try:
                    snap[key] = (
                        float(p.value),
                        float(getattr(p, "sigma", 0.0)),
                        int(getattr(p, "update_count", 0)),
                    )
                except Exception:
                    # QubitBackedParam-style accessors that fail without a
                    # cluster context: skip. Their qubit state IS captured
                    # via density_matrices, so the brain still restores fully.
                    continue
                seen.add(id(p))
            if snap:
                out[node.name] = snap
        return out

    def _restore_param_fiber(self, snapshot: dict[str, dict[str, tuple[float, float, int]]]) -> int:
        """Stamp pickled values onto the currently-attached bundles. Skips
        keys whose node doesn't exist or whose ScalarParam isn't on the live
        bundle (newer code may have moved/renamed knobs). Returns the count
        of params actually restored — useful for boot logs."""
        if not isinstance(snapshot, dict):
            return 0
        from umwelt.substrate.qubit_param import QubitBackedParam
        restored = 0
        for node in self.graph.root.walk():
            per_node = snapshot.get(node.name)
            if not per_node:
                continue
            bundle = getattr(node, "param_bundle", None)
            if bundle is None:
                continue
            for key, triple in per_node.items():
                p = bundle.get_param(key)
                if p is None:
                    continue
                if isinstance(p, QubitBackedParam):
                    # owned by param_fiber_qubits — DON'T stamp p.value (routes through observe(α=1),
                    # a pole reset that destroys the restored purity). The matrices restore last.
                    continue
                try:
                    value, sigma, update_count = triple
                    p.value = max(p.lo if p.lo is not None else -float("inf"),
                                  min(p.hi if p.hi is not None else float("inf"), float(value)))
                    if sigma:
                        p.sigma = float(sigma)
                    p.update_count = int(update_count)
                    restored += 1
                except Exception:
                    continue
        return restored

    def as_manifold(self, *, node: str = "manifold"):
        """The whole live field as ONE connected '1-matrix' MANIFOLD cluster — the forecast/dream brain's view.
        The forebrain runs CLUSTERED (fast, decoupled = hub-and-spokes); this folds the SAME state into one
        connected cumulant cluster (the webway) on demand, so the native clustered save/load stays cheap.
        Byte-identical to the clusters at the decoupled state; keeps every learned cross-cluster coupling."""
        from umwelt.substrate import field_unify
        return field_unify.to_manifold(field_unify.pack_field(self.field), node=node)

    def field_canon_hash(self) -> str:
        """Content hash of the field's canonical state — its GAUGE COORDINATE (clock-tape gauge ↔ git). The
        clustered and manifold forms share it; unchanged across a save = empty diff = provable non-training."""
        from umwelt.substrate import field_unify
        return field_unify.canon_hash(field_unify.pack_field(self.field))

    def save(self, path: str | Path):
        """Save engine state."""
        path = Path(path)
        try:                                   # the field's GAUGE COORDINATE (content hash of the learned
            _canon_hash = self.field_canon_hash()   # state) — persisted so the gauge/git layer can track it:
        except Exception:                      # an unchanged hash across a save = empty diff = provable
            _canon_hash = None                 # non-training (clock-tape gauge ↔ git). Best-effort; never fatal.
        data = {
            "field_canon_hash": _canon_hash,
            # Global RNG stream positions AT THE SNAPSHOT CURSOR. The replay path
            # consumes the process-global `random` stream (qubit_param/params
            # Thompson samples, surprise-tape reservoir draws), so an incremental
            # boot (snapshot + log tail) can only reproduce a from-log replay's
            # field_canon_hash if the stream resumes exactly where the snapshot
            # left it — the 2026-07-18 lease-drill chain fork. numpy's stream was
            # measured NOT to move during replay; it rides along as a guard.
            "rng_state": {
                "random": random.getstate(),
                "numpy": np.random.get_state(),
            },
            # Snapshot format version + the feature geometry this brain exposes
            # (self-describing; lets a loader spot a topology mismatch — the
            # LAUNCH-RULE guard). v1 = legacy/unversioned. Origin-era keys that
            # no longer ship (readout/buffers, device/person/forecast state)
            # load fine from old snapshots (ignored).
            "format_version": 2,
            # The initial-condition gauge this brain was built under (blank vs magic). Anchors live in
            # their qubits' density matrices below; this makes the snapshot fully self-describe its context.
            "seed_profile": getattr(self, "seed_profile", "blank"),
            # Which anchors have been grounded from evidence — travels with the anchor qubits so a
            # grounded brain keeps holding its learned coordinates on reload.
            "anchors_grounded": sorted(self._grounded_anchors),
            "feature_dim": self.feature_dim,
            "step": self._step,
            "prev_z": self.collapse_engine._prev_z,
            "world_nodes": self.world._nodes,
            # Field memory by cluster kind — each cluster is exactly one of dense / product / cumulant:
            #  • dense → density_matrices + hamiltonians (full ρ + H_base)
            #  • product (the qubit param fiber) → param_fiber_qubits (per-role 2×2 matrices)
            #  • cumulant (the rho-free O(n²) backend) → cumulant_states (e1/e2 + couplings)
            # Density matrices (field memory) — dense clusters only (NOT product, NOT cumulant).
            "density_matrices": {
                name: cluster.rho.copy()
                for name, cluster in self.field.clusters.items()
                if not getattr(cluster, "is_product", False)
                and not getattr(cluster, "is_cumulant", False)
            },
            # Learned Hamiltonians — dense clusters only (product has no H_base; cumulant H rides in
            # the cumulant snapshot + the fractal _h_bundles).
            "hamiltonians": {
                name: cluster.evolver.H_base.copy()
                for name, cluster in self.field.clusters.items()
                if not getattr(cluster, "is_product", False)
                and not getattr(cluster, "is_cumulant", False)
                and hasattr(cluster.evolver, "H_base")
            },
            # The qubit-backed parameter fiber: per-cluster {role: 2×2 matrix} — the AUTHORITATIVE
            # state of every qubit-backed param (value + purity + Bloch phase). See _bind_param_fiber.
            "param_fiber_qubits": {
                name: cluster.state_matrices()
                for name, cluster in self.field.clusters.items()
                if getattr(cluster, "is_product", False)
            },
            # Cumulant cluster states (e1/e2 + couplings) — the rho-free backend.
            "cumulant_states": {
                name: cluster.snapshot()
                for name, cluster in self.field.clusters.items()
                if getattr(cluster, "is_cumulant", False)
            },
            # Population state
            "population": self.population.snapshot() if self.population else None,
            # Fractal stack state
            "fractal_stack": self.fractal_stack.save_state() if self.fractal_stack else None,
            "clock_skill": dict(self.clock_skill),
            "clock_last_target": dict(self._clock_last_target),
            # Parameter fiber (every node's ScalarParam values). The console
            # tunes these, the calibration tower learns these — they MUST
            # survive a restart or replay loses its anchor. Captured as a
            # per-node {key: (value, sigma, update_count)} dict so the loader
            # can re-stamp values onto whatever bundle shape exists on disk;
            # the bundle itself is rebuilt fresh at boot from
            # `configure_param_bundles` so newly-added keys land naturally.
            "param_fiber": self._snapshot_param_fiber(),
            # Fractal-web learned bridges — runtime-grown topology. The node-keyed
            # param_fiber can't capture these (they are graph EDGES, not node params), so the
            # learned web travels separately and is re-grown onto the fresh graph on load.
            # Empty on brains that never grew one (gated default-OFF) → zero effect on every
            # existing snapshot. Each carries its learned coupling so the web survives a restart.
            "learned_bridges": [
                {"source": b.source, "target": b.target,
                 "shared_roles": list(b.shared_roles),
                 "coupling": b.coupling_param.value if b.coupling_param is not None else 0.1}
                for b in self.graph.bridges if b.kind == "learned"
            ],
            # Agency qubit (act↔listen) — so the weekly heal survives a restart, not just the
            # persisted silence scalar (which only restores the on/off, not the decay position).
            "agency": self.agency.state() if getattr(self, "agency", None) is not None else None,
            # Egress tendril continuation state (commit qubit + dispatch memory + learned
            # rise/fall geometry). A tendril's committed level feeds back into the tick's
            # learning surface, so continued evolution after a load only matches a
            # never-stopped run when this rides the snapshot — measured on the 2026-07-18
            # lease-drill chain fork (an incremental evolve forked from the from-log
            # referee on its FIRST tail batch until tendril state was restored).
            "tendrils": {
                t.name: t.state_dict()
                for t in (getattr(self, "tendrils", None) or [])
                if hasattr(t, "state_dict")
            },
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Engine state saved to %s", path)

    def load(self, path: str | Path):
        """Load a saved engine state."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        # The brain's initial-condition gauge travels with the snapshot (self-describing context).
        self.seed_profile = data.get("seed_profile", getattr(self, "seed_profile", "blank"))
        # Which anchors were evidence-grounded (so a slow pin loop holds the learned spots, not a seed).
        # Origin snapshots carried the single "location_grounded" bool → map it onto the "geo" anchor.
        grounded = data.get("anchors_grounded")
        if grounded is not None:
            self._grounded_anchors = set(grounded)
        elif data.get("location_grounded"):
            self._grounded_anchors.add("geo")
        # Agency qubit: restore the act↔listen position so a mid-heal silence resumes where it left off.
        _ag = data.get("agency")
        if _ag and getattr(self, "agency", None) is not None:
            try:
                self.agency.load(_ag)
            except Exception as e:
                logger.warning("agency state restore failed: %s", e)

        # Feature-geometry visibility (the LAUNCH-RULE guard). Everything restored
        # below — density matrices, param fiber, Hamiltonians, population, fractal
        # stack — keys by node name/shape and is independent of feature layout, so
        # a feature_dim mismatch is not fatal; it just means this snapshot was cut
        # from a different topology, which deserves a loud line in the log.
        # (Origin-era keys for retired subsystems are simply ignored.)
        stored_feature_dim = data.get("feature_dim")
        if stored_feature_dim is not None and stored_feature_dim != self.feature_dim:
            logger.warning(
                "feature geometry changed (snapshot=%s, live=%d) — this snapshot was cut "
                "from a different topology (pick a topology-matching brain)",
                stored_feature_dim, self.feature_dim,
            )
        self._step = data.get("step", 0)
        self.collapse_engine._prev_z = data.get("prev_z", {})
        if "world_nodes" in data:
            self.world._nodes = data["world_nodes"]

        # Fractal-web: re-grow persisted learned bridges onto the fresh graph
        # (the spec build only made the DECLARED topology). Defensive: skip a pair
        # already bridged or whose endpoints don't exist on this graph. Absent key
        # → no-op, so every older snapshot is unaffected.
        learned = data.get("learned_bridges")
        if learned:
            from umwelt.substrate.web_topology import make_learned_bridge
            existing = {frozenset((b.source, b.target)) for b in self.graph.bridges}
            for lb in learned:
                pair = frozenset((lb["source"], lb["target"]))
                if pair in existing:
                    continue
                if self.graph.find(lb["source"]) is None or self.graph.find(lb["target"]) is None:
                    continue
                self.graph.bridges.append(make_learned_bridge(
                    lb["source"], lb["target"], lb["shared_roles"],
                    coupling=float(lb.get("coupling", 0.1))))
                existing.add(pair)

        # Restore the live FIELD (density matrices) — load-only; the forebrain handoff
        # (reload_learned_params) deliberately skips this to keep its present field.
        for name, rho in data.get("density_matrices", {}).items():
            cluster = self.field.clusters.get(name)
            # dense only — a product cluster's .rho RAISES, a cumulant cluster has none (mirrors save).
            if (cluster is None or getattr(cluster, "is_product", False)
                    or getattr(cluster, "is_cumulant", False)):
                continue
            if rho.shape == cluster.rho.shape:
                cluster.rho = rho

        # Restore cumulant clusters (e1/e2 + couplings) — the rho-free backend.
        for name, state in data.get("cumulant_states", {}).items():
            cluster = self.field.clusters.get(name)
            if cluster is not None and getattr(cluster, "is_cumulant", False):
                cluster.load(state)

        # The slow-learned params (param fiber, Hamiltonians, population, fractal)
        # — the block shared verbatim with the forebrain handoff.
        self._restore_learned(data)

        self.clock_skill = dict(data.get("clock_skill", {}))
        _clt = data.get("clock_last_target")
        self._clock_last_target = dict(_clt) if isinstance(_clt, dict) else {}

        # Egress tendrils: restore the committed actuation qubits + dispatch memory
        # (matched by name; absent block = older snapshot = tendrils keep their boot
        # seed, exactly the pre-fix behavior).
        _tstates = data.get("tendrils") or {}
        for t in (getattr(self, "tendrils", None) or []):
            st = _tstates.get(getattr(t, "name", None))
            if st and hasattr(t, "load_state_dict"):
                try:
                    t.load_state_dict(st)
                except Exception as e:
                    logger.warning("tendril state restore failed for %s: %s", t.name, e)

        # Resume the global RNG streams at the snapshot's cursor — LAST, so any
        # RNG a restore step above consumed cannot shift the resumed position.
        # This is what lets a snapshot + log-tail boot land on the same
        # field_canon_hash as a from-log replay (the packet referee's contract).
        rng = data.get("rng_state")
        if rng:
            try:
                random.setstate(rng["random"])
                np.random.set_state(rng["numpy"])
                logger.info("Global RNG streams resumed at the snapshot cursor")
            except (KeyError, TypeError, ValueError) as e:
                logger.warning("rng_state restore failed (%s) — streams keep "
                               "their current position", e)
        else:
            logger.warning(
                "LEGACY snapshot: no rng_state — global RNG streams keep their "
                "seed-once position, so an incremental replay from this snapshot "
                "is NOT guaranteed to reproduce a from-log replay's "
                "field_canon_hash (referee such chains from the log alone)")

        logger.info("Engine state loaded from %s (%d steps)", path, self._step)

    def _restore_learned(self, data: dict) -> None:
        """Stamp the slow-learned parameters from a snapshot dict onto the live brain:
        param fiber, learned Hamiltonians, population, fractal stack.
        Does NOT touch the live field (density matrices), step clock, or world nodes.
        Shared verbatim by load() (which also restores the field + clock skill) and
        reload_learned_params() (the forebrain handoff)."""
        # Param fiber: re-stamp pickled ScalarParam values onto the live bundles — the
        # seam that makes the console/calibration knobs survive a restart/handoff.
        param_fiber = data.get("param_fiber")
        if param_fiber:
            n = self._restore_param_fiber(param_fiber)
            logger.info("Restored %d param fiber values from snapshot", n)
        # The qubit-backed fiber: restore the per-qubit 2×2 matrices into the product cluster — the
        # AUTHORITATIVE state (value + purity + Bloch phase) of every qubit-backed param. Done AFTER
        # the scalar param_fiber (which skips QB params) so nothing clobbers the matrices. Old
        # snapshots lack this block → those params keep their value-preserving bind seed.
        for name, mats in (data.get("param_fiber_qubits") or {}).items():
            cluster = self.field.clusters.get(name)
            if cluster is not None and getattr(cluster, "is_product", False):
                cluster.load_matrices(mats)
        # Population (genetic H search) + fractal stack — BEFORE the H_base/coupling
        # stamps below: FractalStack.load_state ends by RE-PROJECTING H from its
        # restored meta-fields onto the production clusters (a blend against whatever
        # H_base they hold), so the snapshot's authoritative H_base must land LAST or
        # the re-projection drifts the field off its saved gauge coordinate
        # (field_canon_hash covers h/zz/xy — the roundtrip contract).
        if data.get("population") is not None and self.population is not None:
            self.population.load_snapshot(data["population"])
        if data.get("fractal_stack") is not None and self.fractal_stack is not None:
            self.fractal_stack.load_state(data["fractal_stack"])
        # Learned Hamiltonians (per-cluster H_base).
        for name, H in (data.get("hamiltonians") or {}).items():
            cluster = self.field.clusters.get(name)
            if cluster is not None and H.shape == (cluster.dim, cluster.dim):
                cluster.set_hamiltonian(H)
        # GRAFT the CUMULANT clusters' learned COUPLINGS (h / zz / xy — incl. the grown 'web' cross-couplings
        # the dream/surprise-min learner adds) WITHOUT clobbering the live beliefs (e1/e2). The cumulant analogue
        # of set_hamiltonian above — the wire that lets an offline learner's grown topology reach the live
        # forebrain at the rest-window handoff (reload_learned_params calls this without cluster.load, so
        # couplings would otherwise never graft for a big merged cumulant). load() also runs cluster.load →
        # same values.
        for name, state in (data.get("cumulant_states") or {}).items():
            cluster = self.field.clusters.get(name)
            if cluster is None or not getattr(cluster, "is_cumulant", False):
                continue
            try:
                h = (np.asarray(state["h"], float).reshape(cluster.n_qubits, 3)
                     if state.get("h") is not None else None)
                zz = {tuple(int(x) for x in k.split(",")): float(v) for k, v in (state.get("zz") or {}).items()}
                xy = {tuple(int(x) for x in k.split(",")): (float(v[0]), float(v[1]))
                      for k, v in (state.get("xy") or {}).items()}
                cluster.set_couplings(h_fields=h, zz=zz, xy=xy)
            except Exception as e:
                logger.warning("cumulant coupling graft failed for %s: %s", name, e)

    def reload_learned_params(self, path: str | Path) -> bool:
        """FOREBRAIN handoff: graft the HINDBRAIN's freshly-learned parameters onto
        this live brain WITHOUT touching the present-tense field (density matrices),
        the step clock, or prev_z. The forebrain keeps sensing/actuating in the now;
        only the slow-learned knobs are updated. Restores: param fiber, learned
        Hamiltonians, population, fractal stack. Returns True if a snapshot was read.

        This is the read side of the forebrain↔hindbrain snapshot handoff: the hindbrain
        writes engine.save(path) atomically; the forebrain calls this on a slow loop.
        """
        path = Path(path)
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            logger.warning("reload_learned_params: could not read %s: %s", path, e)
            return False

        # Slow-learned params (param fiber, Hamiltonians, population, fractal)
        # — the same block load() uses, minus the live field it would reset.
        self._restore_learned(data)
        if data.get("clock_skill"):
            self.clock_skill = dict(data["clock_skill"])
        logger.info("Forebrain grafted learned params from hindbrain snapshot %s", path.name)
        return True

    def reset(self, clear_training: bool = False):
        """Reset quantum state and world model. (`clear_training` kept for API
        compatibility with the origin seam.)"""
        self.field.reset()
        self.world.reset()
        self._step = 0
        self.collapse_engine._prev_z.clear()


# Origin-seam alias: ported tests and consumers written against the deployment's
# class name keep working (RENAMES.md: reservoir.py -> engine.py).
QuantumReservoir = BeliefEngine
