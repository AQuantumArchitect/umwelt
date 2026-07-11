"""
Calibration Loop — feedback from reality to the parameter fiber.

After each ingest cycle, the calibration loop compares what the field
predicted vs what sensors reported.  The residual drives Kalman updates
on the parameter fiber, causing the field to learn its own dynamics.

Five calibration channels:

    1. SENSOR RANGE  — raw values widen/tighten normalization bounds
    2. DYNAMICS      — prediction surprise adjusts gamma & hysteresis
    3. COUPLING      — bridge correlation adjusts coupling weights
    4. HAMILTONIAN   — genetic population learns H from prediction residuals
    5. TRACKING      — dissipative qubit tracking error adjusts gamma_diss_{role}

All updates flow through ParameterBundle.update(), so Berry phase
accumulates automatically.  The loop is lightweight: a few scalar
Kalman updates per step, not a training pass.
"""
from __future__ import annotations
from umwelt._util import clamp01

import logging
from dataclasses import dataclass

import numpy as np

from umwelt.substrate.graph import WorldGraph
from umwelt.substrate.field import QuantumField
from umwelt.clocks.phi_clock import fib_strides, effective_stride
def _calib_node_key(binding, pname):
    """(node, key) for a per-node calibration param of a signal binding. NOTE on folded
    topologies: under a fold transform the binding is remapped onto the surviving node with a
    '{node}_{base}' role and the calib param must re-key the same way — that redirect arrives
    with the generic fold transform (see param_bundles._attach note). Un-folded: identity."""
    return binding.node, pname
from umwelt.learning.meta_idioms import proportional_nudge, tower_steps
from umwelt.learning.meta_stack import MetaStack, MetaTier
from umwelt.membranes.ingress import SensorBridge

logger = logging.getLogger(__name__)


# ============================================================================
# Config
# ============================================================================

@dataclass
class CalibrationConfig:
    """Tuning knobs for the three calibration channels."""

    # Channel 1: sensor range
    range_obs_sigma: float = 5.0
    range_enabled: bool = True

    # NOTE: the *_interval fields below SEED the learnable φ-strides
    # (phi_clock); defaults are Fibonacci ladder rungs (de-magicked). Once
    # seeded they self-tune — the interval is a prior, not a fixed cadence.

    # Channel 2: field dynamics
    dynamics_obs_sigma: float = 0.02
    dynamics_ema_alpha: float = 0.1
    dynamics_surprise_target: float = 0.05
    dynamics_enabled: bool = True
    dynamics_interval: int = 8         # φ-stride seed (fast rung)

    # Channel 3: bridge coupling
    coupling_obs_sigma: float = 0.15
    coupling_enabled: bool = True
    coupling_interval: int = 21        # φ-stride seed

    # Channel 4: Hamiltonian learning (genetic population)
    hamiltonian_enabled: bool = False
    hamiltonian_interval: int = 50     # evolve population every N steps
    hamiltonian_inject_interval: int = 100  # inject best H every N steps

    # Channel 5: dissipative tracking — adjusts gamma_diss_{role}
    # based on tracking error for continuous sensors.
    tracking_obs_sigma: float = 1.0
    tracking_ema_alpha: float = 0.1
    tracking_error_target: float = 0.05  # target |z - input| per role
    tracking_enabled: bool = True
    tracking_interval: int = 21          # φ-stride seed

    # Channel 6: driver comprehension — projection_coupling (driver↔temporal
    # correlation), driver_alpha (shrinks as anticipation improves), and the
    # forecast hyperparameters (lr/l2/ema). See _calibrate_projection/_calibrate_drivers.
    driver_enabled: bool = True
    driver_interval: int = 21         # φ-stride seed

    # Meta level: tune the calibrator's own knobs a φ-step slower than level-0.
    meta_interval: int = 55              # φ-stride seed
    # Higher (tier 2): tune tier-1's thresholds / clamps / EMA-α a φ-step
    # slower again (fib_strides[5] = 89). See _meta_meta_learn.
    higher_interval: int = 89


# ============================================================================
# Sensor-range keyword map
# ============================================================================

# (keyword found in sensor_id) -> (lo_param | None, hi_param | None)
# in the target node's ParameterBundle.
RANGE_KEYWORDS: dict[str, tuple[str | None, str | None]] = {
    "temp":     ("sensor_temp_lo", "sensor_temp_hi"),
    "humidity": ("sensor_humidity_lo", "sensor_humidity_hi"),
    "voc":      (None, "sensor_voc_hi"),
    "lux":      (None, "sensor_lux_hi"),
    "power":    (None, "sensor_power_hi"),
}


# ============================================================================
# CalibrationLoop
# ============================================================================

class CalibrationLoop:
    """Teaches the parameter fiber from prediction residuals.

    Channels
    --------
    1. **Sensor range** -- when a raw value exceeds a node's
       ``sensor_*_lo`` / ``sensor_*_hi`` param, Kalman-update the
       bound to accommodate reality.

    2. **Dynamics** -- track how "surprised" the field is at each
       step (mean |dz| across roles).  Sustained high surprise means
       the field can't keep up: nudge gamma up.  Sustained low
       surprise means the field is over-responsive: nudge gamma down.

    3. **Coupling** -- measure Bloch-vector correlation between
       bridged nodes.  Kalman-update ``coupling_param`` toward the
       observed correlation.

    4. **Hamiltonian** -- a genetic population of competing H candidates
       evolves in parallel.  The best individual's Hamiltonian is
       periodically injected into the production field.

    5. **Tracking** -- for dissipative-input qubits (continuous sensors),
       compare Bloch-z with the normalized reading.  Sustained tracking
       error nudges ``gamma_diss_{role}`` up or down, so each role
       discovers its own thermalization timescale.
    """

    def __init__(
        self,
        graph: WorldGraph,
        field: QuantumField,
        config: CalibrationConfig | None = None,
    ):
        from umwelt.substrate.params import ParameterBundle

        self.graph = graph
        self.field = field
        self.config = config or CalibrationConfig()
        self._step = 0

        # Learnable calibration parameters — all on the fiber.
        # These were hardcoded in CalibrationConfig; now they learn
        # from the field's own dynamics via the calibration loop.
        c = self.config
        # Channel tick cadences are LEARNABLE φ-strides (phi_clock) seeded from
        # the config intervals (whose defaults are the Fibonacci ladder — the
        # magic intervals are gone). The meta level seeds a φ-step higher so it
        # sees the consequences of level-0's nudges, not the noise.
        self.params = ParameterBundle.from_dict({
            "dynamics_ema_alpha": (c.dynamics_ema_alpha, 0.03, 0.01, 0.5),
            "dynamics_surprise_target": (c.dynamics_surprise_target, 0.02, 0.001, 0.5),
            "dynamics_obs_sigma": (c.dynamics_obs_sigma, 0.005, 0.001, 0.2),
            "coupling_obs_sigma": (c.coupling_obs_sigma, 0.03, 0.01, 0.5),
            # Channel 6 observation noises (how softly each driver knob updates).
            "driver_obs_sigma": (0.05, 0.02, 0.005, 0.3),
            "projection_obs_sigma": (0.15, 0.03, 0.01, 0.5),
            # Tier-2 observation noise (how softly the higher-order knobs move).
            "higher_obs_sigma": (0.05, 0.02, 0.005, 0.3),
            # Learnable φ-strides (ticks between runs), seeded from config.
            "dynamics_stride": (float(c.dynamics_interval), 2.0, 1.0, 200.0),
            "coupling_stride": (float(c.coupling_interval), 3.0, 1.0, 300.0),
            "tracking_stride": (float(c.tracking_interval), 3.0, 1.0, 300.0),
            "driver_stride": (float(c.driver_interval), 3.0, 1.0, 300.0),
            "meta_stride": (float(c.meta_interval), 5.0, 8.0, 600.0),  # φ above level-0
            "higher_stride": (float(c.higher_interval), 8.0, 13.0, 800.0),  # φ above meta
        })

        # Channel 2 state
        self._prev_bloch: dict[str, dict[str, float]] = {}
        # surprise/tracking/collapse EMAs: gated qubit-backed banks (exact-EMA parity) or plain dicts.
        # mean |Δz| ∈ [0, 2] for surprise/tracking; collapse rate ∈ [0, 1]. See qubit_ema.py.
        from umwelt.substrate.qubit_ema import make_ema_bank
        self._surprise_ema = make_ema_bank("_calib_surprise", 0.0, 2.0)
        # Every calibration nudge is a SUPERVISED-target collapse toward a MEASURED value (a sensor
        # range, a tracked γ, an observed correlation) — already the universal principle. Routed
        # through the ONE learner object (supervised mode) so calibration is on the same law as the
        # dwell and trust. Parity-exact in the no-channel path. See project_universal_learning_law.
        from umwelt.learning.universal_learner import UniversalLearner
        self._learner = UniversalLearner()

        # Channel 4 state
        self.population = None  # set externally by reservoir
        self._hamiltonian_updates = 0

        # Channel 6 state (drivers) — refs set externally by the engine.
        self.driver_forecast = None
        self.driver_anticipation = None          # dict: key -> skill EMA
        self.driver_anticipation_snr = None       # callable -> float | None
        self._projection_updates = 0
        self._driver_updates = 0
        self._prev_forecast_err: float | None = None
        self._prev_forecast_wnorm: float | None = None
        # (Channel 6b — output-anchor calibration — retired with the origin's hand-drawn polynomial.)

        # ── Tier-2 (meta-meta) state ──
        # Cross-tower ref to the H-tower scales (set by reservoir), so tier 2
        # can read each scale's _lr_classifications ring buffer and write its
        # thresholds (lr_improve_thresh / lr_plateau_thresh).
        self.fractal_scales = None
        # Per-channel clamp saturation tracking (tier-2 widens/narrows the
        # wide-nudge bounds based on how often we're saturating the clamp).
        from collections import deque
        self._sat_hits: dict[str, int] = {}
        self._sat_total: dict[str, int] = {}
        # Per-node raw collapse-rate series for adaptive EMA bandwidth.
        self._collapse_rate_raw: dict[str, deque] = {}
        self._higher_updates = 0

        # Channel 5 state: per-role tracking error EMA
        # key: "node_name:role" -> EMA of |z - input|
        self._tracking_ema = make_ema_bank("_calib_tracking", 0.0, 2.0)
        self._tracking_updates = 0
        # Channel 5b: per-node collapse-rate EMA (eager-init so it shares the gated bank type)
        self._collapse_rate_ema = make_ema_bank("_calib_collapse", 0.0, 1.0)

        # Counters
        self._range_updates = 0
        self._dynamics_updates = 0
        self._coupling_updates = 0

        # Meta-learning state: track whether calibration is helping.
        # Cadence is the learnable meta_stride (phi_clock), not a fixed interval.
        self._prev_mean_surprise = 0.0
        self._prev_mean_coupling_error = 0.0

        # The φ-clocked meta-tower over the parameter fiber (companion to the
        # FractalStack's operator tower). Tier 0 tunes world/fiber params; tier 1
        # tunes tier 0's own knobs (φ-strides, obs_sigmas, targets), a φ-step
        # slower. Depth-capped at the levels with a real signal — see meta_stack.
        self.meta_stack = MetaStack([
            MetaTier(
                name="world",
                learns="field/fiber: sensor ranges, gamma, bridge coupling, "
                       "gamma_diss, projection_coupling, driver_alpha, forecast lr/l2/ema",
                run=self._run_world_tier,
                strides=lambda: {ch: self._stride(ch) for ch in (
                    "dynamics_stride", "coupling_stride",
                    "tracking_stride", "driver_stride")},
            ),
            MetaTier(
                name="meta",
                learns="tier-0 knobs: channel φ-strides + obs_sigmas + surprise target",
                run=self._run_meta_tier,
                strides=lambda: {"meta_stride": self._stride("meta_stride")},
            ),
            # Tier 2 (higher) — a φ-step above the meta tier. Learns tier-1's
            # own thresholds + the clamp widths + the EMA bandwidths from real
            # non-circular signals (classification persistence, saturation rate,
            # raw-series SNR). Honest depth cap: tier 3 is intentionally NOT
            # built — no genuine "is tier 2 helping?" signal exists yet; the
            # day it does, this is the seam.
            MetaTier(
                name="higher",
                learns="tier-1 thresholds (lr_improve/plateau on each scale), wide-clamp "
                       "bounds, collapse-rate EMA bandwidth",
                run=self._run_higher_tier,
                strides=lambda: {"higher_stride": self._stride("higher_stride")},
            ),
        ])

    # ================================================================
    # Public API
    # ================================================================

    def _stride(self, name: str) -> int:
        """Current learnable φ-stride (ticks between runs) for a channel.

        floor=1 so a channel seeded with interval=1 runs every step; the meta
        level stays slow via its param's own lower bound (8), not this floor.
        """
        return effective_stride(self.params.get_param(name), floor=1)

    def _steps(self) -> dict[str, float]:
        """The tower's proportional-nudge bounds + binary step factors, live-read
        from the root bundle (single source for the optimizer step idiom)."""
        root = getattr(self.graph, "root", None)
        return tower_steps(root.param_bundle if root is not None else None)

    def _white_noise_snr_ref(self) -> float:
        """The white-noise SNR reference (var(x)/var(Δx)=0.5) for adaptive EMA."""
        root = getattr(self.graph, "root", None)
        rb = root.param_bundle if root is not None else None
        return rb.get("snr_white_noise_ref", 0.5) if rb is not None else 0.5

    def _root(self):
        root = getattr(self.graph, "root", None)
        return root.param_bundle if root is not None else None

    def _wide_bounds(self) -> tuple[float, float]:
        """Live-read the wider clamp used by dynamics gamma / tracking gamma_diss.
        These are wider than the standard (0.8, 1.2) because gamma can legitimately
        swing further; tier 2 tunes them from per-channel saturation rate."""
        rb = self._root()
        if rb is None:
            return 0.5, 2.0
        return rb.get("wide_nudge_lo", 0.5), rb.get("wide_nudge_hi", 2.0)

    def _wide_nudge_tracked(self, channel: str, current: float, ratio: float) -> float:
        """Apply the wide clamp + multiply, AND count whether we saturated.

        Tier 2 reads (_sat_hits[channel] / _sat_total[channel]) and widens the
        clamp if saturation > target, narrows if below. A purely behavioural
        signal for the clamp width.
        """
        lo, hi = self._wide_bounds()
        self._sat_total[channel] = self._sat_total.get(channel, 0) + 1
        if ratio <= lo or ratio >= hi:
            self._sat_hits[channel] = self._sat_hits.get(channel, 0) + 1
        return current * max(lo, min(hi, ratio))

    def step(
        self,
        sensor_readings: dict[str, float] | None,
        sensor_bridge: SensorBridge,
    ):
        """Run the φ-clocked meta-tower for one timestep.

        Call **after** ``field.step()`` so Bloch vectors reflect the latest
        input. The MetaStack drives tier 0 (world/fiber params) then tier 1
        (tier-0's own knobs); each tier self-gates on its learnable φ-stride.
        The Hamiltonian/population channel belongs to the H-tower, not this
        parameter tower, so it stays here.
        """
        self.meta_stack.step(self._step, sensor_readings, sensor_bridge)

        if (
            self.config.hamiltonian_enabled
            and self.population is not None
            and self._step % self.config.hamiltonian_interval == 0
        ):
            bridged = sensor_bridge.process(sensor_readings) if sensor_readings else {}
            self._calibrate_hamiltonian(bridged)

        self._snapshot_bloch()
        self._step += 1

    # ── Meta-tower tiers (driven by self.meta_stack) ──────────────────────────

    def _run_world_tier(self, host_step, sensor_readings, sensor_bridge):
        """Tier 0: tune the world/fiber params. Each channel self-gates on its
        own learnable φ-stride."""
        if sensor_readings and self.config.range_enabled:
            self._calibrate_ranges(sensor_readings, sensor_bridge)
        if (
            self.config.dynamics_enabled
            and self._prev_bloch
            and host_step % self._stride("dynamics_stride") == 0
        ):
            self._calibrate_dynamics()
        if (
            self.config.coupling_enabled
            and host_step % self._stride("coupling_stride") == 0
        ):
            self._calibrate_coupling()
        if (
            sensor_readings
            and self.config.tracking_enabled
            and host_step % self._stride("tracking_stride") == 0
        ):
            self._calibrate_tracking(sensor_readings, sensor_bridge)
        if (
            self.config.driver_enabled
            and host_step % self._stride("driver_stride") == 0
        ):
            self._calibrate_projection()
            self._calibrate_drivers()

    def _run_meta_tier(self, host_step, sensor_readings, sensor_bridge):
        """Tier 1: tune tier-0's own knobs, a φ-step slower (meta_stride ≈ 55)."""
        if host_step > 0 and host_step % self._stride("meta_stride") == 0:
            self._meta_learn()

    def _run_higher_tier(self, host_step, sensor_readings, sensor_bridge):
        """Tier 2: tune tier-1's thresholds + the clamp widths + EMA bandwidths,
        a φ-step slower again (higher_stride ≈ 89). Only learners with real
        signals; depth-cap is honest — tier 3 is intentionally unbuilt."""
        if host_step > 0 and host_step % self._stride("higher_stride") == 0:
            self._meta_meta_learn()

    def stats(self) -> dict:
        """Calibration diagnostics for logging / API."""
        result = {
            "step": self._step,
            "range_updates": self._range_updates,
            "dynamics_updates": self._dynamics_updates,
            "coupling_updates": self._coupling_updates,
            "hamiltonian_updates": self._hamiltonian_updates,
            "tracking_updates": self._tracking_updates,
            "projection_updates": self._projection_updates,
            "driver_updates": self._driver_updates,
            "higher_updates": self._higher_updates,
            "surprise_ema": dict(self._surprise_ema),
            "tracking_ema": dict(self._tracking_ema),
            "strides": {
                name: self._stride(name)
                for name in ("dynamics_stride", "coupling_stride",
                             "tracking_stride", "driver_stride",
                             "meta_stride", "higher_stride")
            },
            # The φ-clocked meta-tower (depth-ordered tiers). Tier 1 meta-tunes
            # the dynamics/coupling strides today; tracking/driver strides are
            # learnable priors not yet meta-tuned (no clean effectiveness signal
            # wired — left honest rather than faked).
            "meta_tower": self.meta_stack.snapshot(),
        }
        if self.population is not None:
            result["population"] = self.population.stats()
        return result

    # ================================================================
    # Channel 1: Sensor Range
    # ================================================================

    def _calibrate_ranges(
        self,
        sensor_readings: dict[str, float],
        sensor_bridge: SensorBridge,
    ):
        """Widen normalization bounds when raw values exceed them."""
        obs_sigma = self.config.range_obs_sigma

        for sensor_id, raw_value in sensor_readings.items():
            binding = sensor_bridge.bindings.get(sensor_id)
            if binding is None:
                continue

            sid_lower = sensor_id.lower()

            for keyword, (lo_name, hi_name) in RANGE_KEYWORDS.items():
                if keyword not in sid_lower:
                    continue

                # Resolve (node, key) per param honoring folded topologies: under a fold
                # transform a node's sensor params live on the surviving node keyed '{node}_sensor_*'
                # (param_bundles._attach), so graph.find(binding.node)="bedroom" would miss them.
                for pname, is_lo in ((lo_name, True), (hi_name, False)):
                    if not pname:
                        continue
                    node_name, key = _calib_node_key(binding, pname)
                    node = self.graph.find(node_name)
                    if node is None or node.param_bundle is None:
                        continue
                    bundle = node.param_bundle
                    if key not in bundle:
                        continue
                    cur = bundle.get(key)
                    if (is_lo and raw_value < cur) or (not is_lo and raw_value > cur):
                        self._learner.observe(bundle.get_param(key), raw_value, obs_sigma)
                        self._range_updates += 1
                        logger.debug("Range%s %s.%s ← %.1f",
                                     "↓" if is_lo else "↑", node_name, key, raw_value)

                break  # first keyword match only

    # ================================================================
    # Channel 2: Field Dynamics
    # ================================================================

    def _snapshot_bloch(self):
        """Record current Bloch z-values for next step's delta."""
        self._prev_bloch.clear()
        for name, cluster in self.field.clusters.items():
            self._prev_bloch[name] = {
                role: float(cluster.qubit_bloch(idx)[2])
                for role, idx in cluster.role_index.items()
            }

    def _calibrate_dynamics(self):
        """Nudge gamma based on sustained surprise level.

        Surprise = mean |dz| across roles since last snapshot.

        * High sustained surprise -> field lags reality -> raise gamma
        * Low sustained surprise  -> field is over-responsive -> lower gamma

        The update is a soft proportional nudge: the "observed gamma"
        is the current gamma scaled by (ema / target), clamped.
        """
        target = self.params.get("dynamics_surprise_target")
        alpha = self.params.get("dynamics_ema_alpha")
        obs_sigma = self.params.get("dynamics_obs_sigma")

        for name, cluster in self.field.clusters.items():
            prev = self._prev_bloch.get(name)
            if prev is None:
                continue

            node = self.graph.find(name)
            if node is None or node.param_bundle is None:
                continue

            bundle = node.param_bundle
            if "gamma" not in bundle:
                continue

            # Mean |dz| across roles
            deltas = []
            for role, idx in cluster.role_index.items():
                prev_z = prev.get(role, 0.0)
                curr_z = float(cluster.qubit_bloch(idx)[2])
                deltas.append(abs(curr_z - prev_z))

            if not deltas:
                continue

            surprise = sum(deltas) / len(deltas)

            # EMA (qubit partial-collapse when gated on; exact-parity scalar otherwise)
            ema = self._surprise_ema.observe(name, surprise, alpha, target)

            # Proportional nudge with the wider (saturation-tracked) clamp.
            current_gamma = bundle.get("gamma")
            if target > 0 and current_gamma > 0:
                observed_gamma = self._wide_nudge_tracked(
                    "dynamics", current_gamma, ema / target,
                )
                self._learner.observe(bundle.get_param("gamma"), observed_gamma, obs_sigma)
                self._dynamics_updates += 1

    # ================================================================
    # Channel 3: Bridge Coupling
    # ================================================================

    def _calibrate_coupling(self):
        """Update bridge coupling from observed Bloch correlations."""
        obs_sigma = self.params.get("coupling_obs_sigma")

        for bridge in self.graph.bridges:
            if bridge.coupling_param is None:
                continue

            ca = self.field.clusters.get(bridge.source)
            cb = self.field.clusters.get(bridge.target)
            if ca is None or cb is None:
                continue

            correlations = []
            for role in bridge.shared_roles:
                if role not in ca.role_index or role not in cb.role_index:
                    continue
                ba = ca.role_bloch(role)
                bb = cb.role_bloch(role)
                na = float(np.linalg.norm(ba))
                nb = float(np.linalg.norm(bb))
                if na > 1e-10 and nb > 1e-10:
                    correlations.append(float(np.dot(ba, bb) / (na * nb)))
                else:
                    correlations.append(0.0)

            if not correlations:
                continue

            # Map mean correlation [-1, 1] -> coupling [0, 1]
            mean_corr = sum(correlations) / len(correlations)
            observed_coupling = (mean_corr + 1.0) / 2.0

            self._learner.observe(bridge.coupling_param, observed_coupling, obs_sigma)
            self._coupling_updates += 1

    # ================================================================
    # Channel 6: Driver comprehension
    # ================================================================

    def _calibrate_projection(self):
        """Calibrate a projecting node's projection_coupling from how strongly
        the projected child role actually co-varies with the parent role it
        feeds (Bloch correlation). Same fixed-point idiom as channel 3 — a
        projection that tracks a real relationship earns a stronger nudge.
        """
        obs_sigma = self.params.get("projection_obs_sigma")
        for name, child in self.field.clusters.items():
            node = self.graph.find(name)
            if node is None or node.projection is None or node.parent is None:
                continue
            if node.param_bundle is None or "projection_coupling" not in node.param_bundle:
                continue
            parent = self.field.clusters.get(node.parent.name)
            if parent is None:
                continue
            corrs = []
            for crole, prole in node.projection.items():
                if crole not in child.role_index or prole not in parent.role_index:
                    continue
                ba, bb = child.role_bloch(crole), parent.role_bloch(prole)
                na, nb = float(np.linalg.norm(ba)), float(np.linalg.norm(bb))
                if na > 1e-10 and nb > 1e-10:
                    corrs.append(float(np.dot(ba, bb) / (na * nb)))
            if not corrs:
                continue
            observed = (sum(corrs) / len(corrs) + 1.0) / 2.0   # [-1,1] → [0,1]
            self._learner.observe(node.param_bundle.get_param("projection_coupling"), observed, obs_sigma)
            self._projection_updates += 1

    def _calibrate_drivers(self):
        """Learn the driver comprehension knobs from real signals:
          - driver_alpha     ← anticipation skill (anchor earns autonomy)
          - forecast_lr      ← forecast error trend
          - forecast_l2      ← overfit detector (weights grew, error didn't fall)
          - forecast_ema / anticipation_ema ← adaptive bandwidth from raw SNR
        Step sizes (0.9–1.1 ratios) follow the existing proportional-nudge idiom
        (cf. _calibrate_dynamics); they're optimizer steps, not model beliefs.
        """
        root = getattr(self.graph, "root", None)
        rb = root.param_bundle if root is not None else None
        if rb is None:
            return
        cs = self.params.get("driver_obs_sigma")

        # (a) driver_alpha shrinks as the field's anticipation improves.
        # PILOT (one-species fractal): driver_alpha is qubit-backed when the
        # _params cluster exists (reservoir._bind_pilot_qubit_params). The
        # rb.update() call below is unchanged — it dispatches polymorphically
        # to QubitBackedParam.kalman_update, which converts (target, σ) into
        # cluster.observe_qubit(target_bloch, α). The math stays the analog
        # continuous (1-α)·old + α·target fuse, but now lives on a real qubit.
        # See qubit_param.py + plans/noble-sleeping-yao.md.
        if self.driver_anticipation and "driver_alpha" in rb:
            skill = sum(self.driver_anticipation.values()) / len(self.driver_anticipation)
            ap = rb.get_param("driver_alpha")
            if ap is not None:
                observed = ap.hi * (1.0 - clamp01(skill))  # high skill → low anchor
                self._learner.observe(ap, observed, cs)              # qubit-routed via the one learner
                self._driver_updates += 1

        steps = self._steps()
        white = self._white_noise_snr_ref()

        fc = self.driver_forecast
        if fc is not None and getattr(fc, "error_ema", None) is not None:
            err, wn = fc.error_ema, fc.weight_norm
            pe, pw = self._prev_forecast_err, self._prev_forecast_wnorm
            # (b) forecast_lr ← error trend: worsening → bold up, improving → settle.
            if pe is not None and "forecast_lr" in rb:
                factor = (steps["step_up_bold"] if err > pe
                          else steps["step_down"] if err < pe else 1.0)
                rb.update("forecast_lr", rb.get("forecast_lr") * factor, cs)   # META-step (forecast hyperparam)
                self._driver_updates += 1
            # (c) forecast_l2 ← overfit: weights grew while error didn't improve →
            # regularize; stuck-high error with small weights → underfit → relax.
            if pe is not None and pw is not None and "forecast_l2" in rb:
                if wn > pw and err >= pe:
                    rb.update("forecast_l2", rb.get("forecast_l2") * steps["step_up_bold"], cs)
                    self._driver_updates += 1
                elif err > pe and wn <= pw:
                    rb.update("forecast_l2", rb.get("forecast_l2") * steps["step_down_bold"], cs)
                    self._driver_updates += 1
            self._prev_forecast_err, self._prev_forecast_wnorm = err, wn
            # (d) forecast_ema ← adaptive bandwidth: SNR>white_noise_ref (structure)
            # widens, <ref (noise) narrows. proportional_nudge clamps to ±20%.
            snr = fc.raw_error_snr() if hasattr(fc, "raw_error_snr") else None
            if snr is not None and "forecast_ema" in rb:
                rb.update("forecast_ema",
                          proportional_nudge(rb.get("forecast_ema"), snr / white,
                                             lo=steps["nudge_lo"], hi=steps["nudge_hi"]), cs)

        # (d') anticipation_ema ← adaptive bandwidth from raw anticipation SNR.
        if self.driver_anticipation_snr is not None and "driver_anticipation_ema" in rb:
            snr = self.driver_anticipation_snr()
            if snr is not None:
                rb.update("driver_anticipation_ema",
                          proportional_nudge(rb.get("driver_anticipation_ema"), snr / white,
                                             lo=steps["nudge_lo"], hi=steps["nudge_hi"]), cs)
                self._driver_updates += 1

    # (Channel 6b — the output-anchor calibrator — retired with the origin's hand-authored
    # polynomial. The preference β is no longer anchored to a curve, so there is no anchor-α
    # to dial down; β is learned solely from the operator's overrides.)

    # ================================================================
    # Channel 5: Collapse Thresholds
    # ================================================================

    def calibrate_collapse(self, transitions: list, field):
        """Steer each node's confidence_threshold toward a target collapse RATE.

        Called after auto_collapse with the resulting transitions. Signal: per-node collapse
        frequency (EMA). The threshold collapses toward the value that hits the target rate via the
        ONE universal law (a proportional gradient_step: raising the threshold lowers the rate), not
        a hand-wired ±step deadband. The target rate + gain are gauge coordinates (collapse_rate_*).
        """
        # Count collapses per node
        collapse_counts: dict[str, int] = {}
        for t in transitions:
            node_name = getattr(t, 'node_name', None) or getattr(t, 'node', None)
            if node_name:
                collapse_counts[node_name] = collapse_counts.get(node_name, 0) + 1

        # Track collapse rate per node via EMA (qubit-backed bank when gated on; init in __init__)
        for name, cluster in field.clusters.items():
            node = self.graph.find(name)
            if node is None or node.param_bundle is None:
                continue
            if "confidence_threshold" not in node.param_bundle:
                continue

            collapsed = 1.0 if name in collapse_counts else 0.0
            # EMA alpha is a LIVE-READ fiber prior — tier 2 adapts its bandwidth
            # from the raw-series SNR (same idiom as forecast_ema). We buffer
            # the raw signal here so tier 2 can compute the SNR honestly.
            alpha_rate = self._root().get("collapse_rate_ema_alpha", 0.05) if self._root() else 0.05
            rate = self._collapse_rate_ema.observe(name, collapsed, alpha_rate, 0.5)
            buf = self._collapse_rate_raw.setdefault(name, __import__("collections").deque(maxlen=64))
            buf.append(collapsed)

            # THE ONE LAW: steer confidence_threshold toward the value that hits the target collapse
            # RATE — a proportional controller as an immediate gradient (raising the threshold lowers
            # the rate → influence = −1; surprise = rate − target). Replaces the hand-wired ±0.01
            # deadband nudge (the 0.2/0.05 bands + 0.6/0.99 clamps were crutches; the param's own
            # [0.5,1.0] bounds clamp now). Continuous, no deadband. See project_universal_learning_law.
            rb = self._root()
            tgt = rb.get("collapse_rate_target", 0.1) if rb is not None else 0.1
            lr = rb.get("collapse_thresh_lr", 0.1) if rb is not None else 0.1
            self._learner.gradient_step(node.param_bundle.get_param("confidence_threshold"),
                                        influence=-1.0, surprise=(rate - tgt), lr=lr, obs_sigma=0.02)

    # ================================================================
    # Meta-learning: calibration params learn from their own effect
    # ================================================================

    def _meta_learn(self):
        """Update calibration meta-parameters from their own effectiveness.

        The signal: are calibration updates making things better?

        For dynamics: if mean surprise is trending down, the current
        dynamics_ema_alpha and dynamics_surprise_target are working.
        If surprise is stuck or rising, adjust them.

        For coupling: if bridge correlation errors are shrinking, the
        coupling_obs_sigma is appropriate. If errors are stuck, adjust.
        """
        # ── Dynamics meta-learning ──
        if self._surprise_ema:
            mean_surprise = sum(self._surprise_ema.values()) / len(self._surprise_ema)
            target = self.params.get("dynamics_surprise_target")

            # Did surprise move toward target since last meta-step?
            prev = self._prev_mean_surprise
            if prev > 0:
                # If surprise decreased toward target, current params are good
                # If surprise increased away from target, params need adjustment
                improved = abs(mean_surprise - target) < abs(prev - target)

                ds = self.params.get("dynamics_stride")
                current_obs_sigma = self.params.get("dynamics_obs_sigma")
                steps = self._steps()
                if not improved:
                    # Surprise isn't converging — widen obs_sigma to explore more,
                    # nudge the target toward reality, and shorten the φ-stride
                    # (run more often to gather signal faster).
                    self.params.update("dynamics_obs_sigma",
                                       current_obs_sigma * steps["step_up_bold"], 0.01)
                    self.params.update(
                        "dynamics_surprise_target",
                        0.8 * target + 0.2 * mean_surprise,
                        0.01,
                    )
                    self.params.update("dynamics_stride", ds * steps["step_down_bold"], 0.5)
                else:
                    # Improving — tighten obs_sigma (exploit) and lengthen the
                    # stride (ease the cadence, more timescale separation). The
                    # cadence of learning is itself learned by the tier above.
                    self.params.update("dynamics_obs_sigma",
                                       current_obs_sigma * steps["step_down"], 0.01)
                    self.params.update("dynamics_stride", ds * steps["step_up"], 0.5)

            self._prev_mean_surprise = mean_surprise

        # ── Coupling meta-learning ──
        bridge_corrs = self.field.bridge_correlations()
        if bridge_corrs:
            # Coupling error: how far is each bridge from its coupling_param?
            errors = []
            for bridge in self.graph.bridges:
                if bridge.coupling_param is None:
                    continue
                observed = bridge_corrs.get((bridge.source, bridge.target), 0.0)
                expected = (bridge.coupling_param.value * 2.0) - 1.0  # [0,1] -> [-1,1]
                errors.append(abs(observed - expected))

            if errors:
                mean_error = sum(errors) / len(errors)
                prev_error = self._prev_mean_coupling_error

                if prev_error > 0:
                    improved = mean_error < prev_error
                    current_obs_sigma = self.params.get("coupling_obs_sigma")
                    cs = self.params.get("coupling_stride")
                    steps = self._steps()
                    if not improved:
                        self.params.update("coupling_obs_sigma",
                                           current_obs_sigma * steps["step_up_bold"], 0.01)
                        self.params.update("coupling_stride", cs * steps["step_down_bold"], 0.5)
                    else:
                        self.params.update("coupling_obs_sigma",
                                           current_obs_sigma * steps["step_down"], 0.01)
                        self.params.update("coupling_stride", cs * steps["step_up"], 0.5)

                self._prev_mean_coupling_error = mean_error

    # ================================================================
    # Tier 2 (meta-meta) — learns tier-1's thresholds + clamp widths +
    # EMA bandwidth from real, non-circular signals. Honest depth cap:
    # tier 3 is intentionally not built (no signal for "is tier 2 helping").
    # ================================================================

    def _meta_meta_learn(self):
        """Three sub-learners, each paired with a named signal:
        (1) per-scale lr_improve/plateau thresholds  ← classification persistence;
        (2) wide-clamp lo/hi (calibration ch 2 & 5)  ← per-channel saturation rate;
        (3) collapse_rate_ema_alpha                    ← raw-series SNR (adaptive bw).
        """
        rb = self._root()
        if rb is None:
            return
        cs = self.params.get("higher_obs_sigma")
        steps = self._steps()

        # (1) Threshold tuning per H-tower scale. The supervisor is FUTURE
        # surprise: an "improving" call should be followed by surprise actually
        # continuing to drop; a "plateau" call by surprise actually staying flat.
        # Hit rate vs classification_target drives loosen/tighten direction.
        if self.fractal_scales:
            target_hit = rb.get("classification_target", 0.7)
            for scale in self.fractal_scales:
                buf = getattr(scale, "_lr_classifications", None)
                if not buf:
                    continue
                here = float(scale._surprise_ema)
                imp = [s for k, s in buf if k == "improving"]
                pla = [s for k, s in buf if k == "plateau"]
                if imp:
                    # improving persisted = surprise actually below the call point
                    hit = sum(1 for s in imp if here < s) / len(imp)
                    cur = scale.params.get("lr_improve_thresh")  # negative
                    # hit > target ⇒ too cautious ⇒ loosen (less negative ⇒ * step_down)
                    factor = steps["step_down"] if hit > target_hit else steps["step_up"]
                    scale.params.update("lr_improve_thresh", cur * factor, cs)
                    self._higher_updates += 1
                if pla:
                    # Grade with a band WIDER than the classifier's own threshold
                    # (5×) — keeps the persistence check non-circular: we ask
                    # "did the system stay in an obviously-plateau regime?" not
                    # "did our own threshold still classify it?". The 5× factor
                    # is a pragmatic prior; a future tier could tune it.
                    cur = scale.params.get("lr_plateau_thresh")  # positive
                    band = 5.0 * cur
                    hit = sum(1 for s in pla if abs(here - s) < band) / len(pla)
                    factor = steps["step_up"] if hit > target_hit else steps["step_down"]
                    scale.params.update("lr_plateau_thresh", cur * factor, cs)
                    self._higher_updates += 1
                buf.clear()

        # (2) Clamp tuning from per-channel saturation rate (aggregated across
        # the two channels that use the wide clamp). Saturation > target ⇒
        # widen bounds; very low ⇒ narrow.
        total = sum(self._sat_total.values())
        if total >= 32:
            hits = sum(self._sat_hits.values())
            sat = hits / total
            target_sat = rb.get("clamp_saturation_target", 0.15)
            lo, hi = self._wide_bounds()
            if sat > target_sat:
                # Widen: lo toward 0 (× step_down), hi up (× step_up)
                rb.update("wide_nudge_lo", lo * steps["step_down"], cs)
                rb.update("wide_nudge_hi", hi * steps["step_up"], cs)
                self._higher_updates += 1
            elif sat < target_sat * 0.5:
                # Narrow: lo toward 1 (× step_up), hi toward 1 (× step_down)
                rb.update("wide_nudge_lo", lo * steps["step_up"], cs)
                rb.update("wide_nudge_hi", hi * steps["step_down"], cs)
                self._higher_updates += 1
            self._sat_hits.clear()
            self._sat_total.clear()

        # (3) Collapse-rate EMA bandwidth ← raw-series SNR (white-noise ref).
        snrs = []
        for buf in self._collapse_rate_raw.values():
            if len(buf) < 16:
                continue
            import numpy as np
            x = np.asarray(buf, dtype=float)
            var_diff = float(np.var(np.diff(x)))
            if var_diff > 1e-12:
                snrs.append(float(np.var(x)) / var_diff)
        if snrs:
            snr = sum(snrs) / len(snrs)
            white = self._white_noise_snr_ref()
            alpha = rb.get("collapse_rate_ema_alpha", 0.05)
            rb.update("collapse_rate_ema_alpha",
                      proportional_nudge(alpha, snr / white,
                                         lo=steps["nudge_lo"], hi=steps["nudge_hi"]), cs)
            self._higher_updates += 1

    # ================================================================
    # Channel 4: Hamiltonian Learning
    # ================================================================

    def _calibrate_hamiltonian(self, inputs: dict | None = None):
        """Evolve the genetic population and periodically inject best H."""
        self.population.step(inputs)

        if self._step % self.config.hamiltonian_inject_interval == 0:
            self.population.inject(self.field)
            self._hamiltonian_updates += 1

    # ================================================================
    # Channel 5: Dissipative Tracking
    # ================================================================

    def _calibrate_tracking(
        self,
        sensor_readings: dict[str, float],
        sensor_bridge: SensorBridge,
    ):
        """Adjust per-role gamma_diss based on tracking error.

        For each dissipative sensor, compare the qubit's Bloch-z with
        the normalized sensor reading. If the qubit consistently lags
        (error > target), nudge gamma_diss_{role} up. If it's too
        responsive (error < target), nudge down.

        Same proportional-nudge logic as _calibrate_dynamics.
        """
        from umwelt.spec.roles import role_input_mode

        target = self.config.tracking_error_target
        alpha = self.config.tracking_ema_alpha
        obs_sigma = self.config.tracking_obs_sigma

        for sensor_id, raw_value in sensor_readings.items():
            binding = sensor_bridge.bindings.get(sensor_id)
            if binding is None:
                continue

            # Only calibrate dissipative roles
            if role_input_mode(binding.qubit_role) != "dissipative":
                continue

            cluster = self.field.clusters.get(binding.node)
            if cluster is None:
                continue

            q_idx = cluster.role_index.get(binding.qubit_role)
            if q_idx is None:
                continue

            node = self.graph.find(binding.node)
            if node is None or node.param_bundle is None:
                continue

            # Tracking error: |Bloch-z - normalized_input|
            curr_z = float(cluster.qubit_bloch(q_idx)[2])
            norm_val = binding.normalize(raw_value)
            error = abs(curr_z - norm_val)

            # EMA (qubit partial-collapse when gated on; exact-parity scalar otherwise)
            ema_key = f"{binding.node}:{binding.qubit_role}"
            ema = self._tracking_ema.observe(ema_key, error, alpha, target)

            # Proportional nudge of gamma_diss_{role}
            param_key = f"gamma_diss_{binding.qubit_role}"
            bundle = node.param_bundle
            if bundle.get_param(param_key) is None:
                continue  # no per-role param to calibrate

            current_gd = bundle.get(param_key)
            if target > 0 and current_gd > 0:
                # error > target → qubit lags → increase gamma_diss
                # error < target → qubit overshoots → decrease gamma_diss
                observed_gd = self._wide_nudge_tracked(
                    "tracking", current_gd, ema / target,
                )
                self._learner.observe(bundle.get_param(param_key), observed_gd, obs_sigma)
                self._tracking_updates += 1
