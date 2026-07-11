"""Parameter bundles: learnable priors for every node in the world graph."""
from __future__ import annotations

import logging

from umwelt.substrate.params import ParameterBundle, ScalarParam
from umwelt.substrate.graph import WorldGraph

logger = logging.getLogger(__name__)


def purge_legacy_calibration_keys(graph: WorldGraph) -> int:
    """Remove `cal_*` keys from the studio bundle if a stale pickle restored
    them. Calibration moved out of the brain into CalibrationStore (the
    structure-level state file); leaving the keys around would orphan them in
    the pickle forever and confuse anyone wondering where the live truth is.

    Returns the count purged (for boot-log clarity). Safe to call repeatedly.
    """
    studio = graph.find("studio")
    if studio is None or studio.param_bundle is None:
        return 0
    bundle = studio.param_bundle
    stale = [k for k in list(bundle.params.keys()) if k.startswith("cal_")]
    for k in stale:
        del bundle.params[k]
    if stale:
        logger.info("Purged %d legacy cal_* keys from studio bundle "
                    "(calibration lives in CalibrationStore now)", len(stale))
    return len(stale)


def _attach(
    graph: WorldGraph,
    node_name: str,
    specs: dict[str, tuple],
    frozen_keys: set[str] | None = None,
):
    """Helper: find a node in the graph and attach a ParameterBundle.

    NOTE on folded topologies: when a graph transform folds a node into a merged parent,
    its params must be redirected (re-keyed "{node}_{param}") onto the surviving node so
    learned priors survive the fold. That redirect map is a property of the FOLD TRANSFORM
    and must come from the spec/graph, never from a module of literals — it lands with the
    generic fold transform. Until then a missing node is a silent no-op (graph.find → None)."""
    node = graph.find(node_name)
    if node is None:
        return
    bundle = ParameterBundle.from_dict(specs, frozen_keys=frozen_keys)
    if node.param_bundle is not None:
        node.param_bundle.merge(bundle)
    else:
        node.param_bundle = bundle


# ── Archetype "parent layers": shared hyperparameters across similar leaves ──
# An archetype is a shared ParameterBundle whose ScalarParam objects are
# referenced by every member leaf, so the class learns ONE shared response
# rather than one-per-leaf. Only shared DYNAMICS hyperparameters belong here;
# per-leaf semantics stay private. Extend with finer classes (kasa_dimmer,
# motion_sensor, ...) as their genuinely-shared params emerge.
_ARCHETYPE_SPECS: dict[str, dict[str, tuple]] = {
    # Actuator belief-stores: slow-drift dynamics so a sparse device report
    # (~60s republish) PERSISTS instead of decohering to idle. One learnable
    # drift rate for the whole class — was one near-frozen gamma per leaf.
    "actuator": {
        "gamma": (0.005, 0.002, 0.0, 0.1),
    },
}


def _device_class(node) -> str | None:
    """Archetype key for a node, or None if it carries no shared parent layer.
    Leaves of kind 'actuator' share the 'actuator' archetype today; the seam
    generalizes to finer behavioral classes (by kind + name prefix) later."""
    if node.is_leaf and node.kind == "actuator":
        return "actuator"
    return None


def _attach_archetypes(graph: WorldGraph) -> dict[str, ParameterBundle]:
    """Build one shared bundle per archetype and point every member leaf's
    bundle at its SHARED ScalarParam objects (the parent layer).

    Idempotent: re-running (e.g. the post-load re-attach in main.py) rebuilds
    each parent layer and re-points all members at the current shared objects,
    so sharing survives a restart. A leaf that needs its own value for an
    archetype key can attach a private param for that key AFTER this call,
    which replaces the shared reference for that one leaf.
    """
    archetypes: dict[str, ParameterBundle] = {}
    for node in graph.root.walk():
        cls = _device_class(node)
        if cls is None:
            continue
        parent = archetypes.get(cls)
        if parent is None:
            parent = ParameterBundle.from_dict(_ARCHETYPE_SPECS[cls])
            archetypes[cls] = parent
        if node.param_bundle is None:
            node.param_bundle = ParameterBundle(params={})
        for key, shared_param in parent.params.items():
            # Reference the SAME object — the shared parent layer. Always
            # re-point so a second configure() pass doesn't strand members on a
            # stale object from the first pass.
            node.param_bundle.params[key] = shared_param
    return archetypes


def configure_param_bundles(graph: WorldGraph, spec=None) -> WorldGraph:
    """Attach ParameterBundles to every node in the world graph.

    Each bundle carries a node's learnable priors. THE SPLIT:
      - The ROOT bundle below is ENGINE DNA — the field dynamics, collapse policy,
        learning-law constants, tower-optimizer steps, context gauge, clocks and agency
        priors every domain shares. Spec-agnostic.
      - PER-NODE priors are DOMAIN DATA: each NodeSpec may declare `params`
        ({name: (default, sigma, lo, hi)}), attached here when a spec is given. A root
        NodeSpec's params MERGE over the engine defaults (spec wins).
    Everything starts at its prior and is learned away from it; a frozen fiber is
    provably inert (the empty-diff discipline).
    """

    # ── Root — engine-generic priors ─────────────────────────────
    # Merge-style attach (preserves pickle-restored values on re-configure): a second
    # configure_param_bundles() call after load(pkl) must not trash live values.
    root = graph.root
    _attach(graph, root.name, {
        # Field dynamics
        "gamma": (0.05, 0.01, 0.001, 0.3),
        "gamma_diss": (5.0, 1.0, 0.5, 50.0),
        "dt": (0.01, 0.002, 0.001, 0.1),
        "bridge_strength": (0.5, 0.1, 0.0, 1.0),
        "physicality_interval": (10.0, 2.0, 1.0, 50.0),
        # Collapse
        "confidence_threshold": (0.9, 0.05, 0.5, 1.0),
        "hysteresis": (0.1, 0.02, 0.01, 0.5),
        # Background-collapse movement gate (see CollapseEngine.collapse_node):
        # a role near zero is suppressed only if it's also not moving.
        "transition_floor": (0.3, 0.05, 0.05, 0.8),
        "motion_eps": (0.02, 0.01, 0.0, 0.2),
        # Projection (child → parent nudge strength)
        "projection_coupling": (0.3, 0.1, 0.01, 1.0),
        # ── Fractal-web topology fiber ──
        # web_min_activity: the LEARNED noise floor for co-movement. A pair of clusters
        # earns a learned bridge only if BOTH genuinely move (z-std) above this. A real
        # gauge coordinate: _bind_param_fiber qubit-backs it, it freezes when the engine
        # freezes, and WebTopology learns it toward the observed quiescent-activity floor
        # each evolve tick. The 0.15 is only the prior the field learns away from.
        "web_min_activity": (0.15, 0.05, 0.0, 1.0),
        # The rest of the web-topology grow/prune policy, gauge-backed — every constant
        # that decides WHEN an edge grows/decays/prunes is a live-read coordinate, not a
        # buried literal (the totality-of-constants move). Int-valued ones are read via
        # int(...) at the call site.
        "web_window": (16.0, 2.0, 4.0, 64.0),
        "web_grow_threshold": (0.6, 0.1, 0.2, 0.95),
        "web_keep_threshold": (0.3, 0.05, 0.1, 0.9),
        "web_decay": (0.7, 0.1, 0.3, 0.99),
        "web_floor": (0.02, 0.01, 0.001, 0.2),
        "web_grow_value": (0.1, 0.05, 0.01, 0.5),
        "web_max_learned": (8.0, 1.0, 2.0, 32.0),
        "web_min_samples": (8.0, 1.0, 2.0, 32.0),
        # ── Periodic-driver comprehension fiber ──
        # A driver's PHASE is deterministic physics (fixed); its comprehension is learned.
        # driver_alpha: base per-tick anchor strength (× per-driver learnable trust
        # weight), CALIBRATED down as the field learns to anticipate the cycle.
        "driver_alpha": (0.35, 0.1, 0.05, 0.95),
        # anchor_ground_alpha: per-fix gain for grounding an anchor gear from evidence.
        # Deliberately SMALL so no single fix dominates — the gear accumulates fixes into
        # their centroid and jitter averages out.
        "anchor_ground_alpha": (0.1, 0.03, 0.01, 0.5),
        # Hebbian lr that credits each driver's trust weight (stable hyperparam).
        "driver_hebbian_lr": (0.01, 0.003, 0.001, 0.1),
        # ── The learning law's OWN constants, as gauge coordinates ──
        # The UniversalLearner's hyperparameters are live-read fiber params (totality:
        # every constant of the one law is itself a diff-witnessed coordinate).
        "hebbian_obs_sigma": (0.5, 0.1, 0.05, 1.0),   # trust-gradient collapse width
        "hebbian_lr": (0.01, 0.003, 0.001, 0.1),
        # Smoothing window for the anticipation-skill EMA (adaptive bandwidth).
        "driver_anticipation_ema": (0.02, 0.01, 0.005, 0.3),
        # Forecast model (anticipation) hyperparameters.
        "forecast_lr": (0.02, 0.005, 0.001, 0.2),       # CALIBRATED (error trend)
        "forecast_l2": (1e-4, 5e-5, 1e-6, 1e-2),        # CALIBRATED (weight norm)
        "forecast_ema": (0.02, 0.01, 0.005, 0.3),       # CALIBRATED (adaptive SNR)
        # FIXED named priors (deliberately not calibrated):
        #   driver_trust_floor — a safety GUARANTEE that the driver anchor never
        #     vanishes (ground truth stays on); a bound, not an optimization target.
        #   forecast_horizon_min — a task SPEC: how far ahead we choose to predict.
        "driver_trust_floor": (0.15, 0.03, 0.05, 0.5),
        "forecast_horizon_min": (30.0, 5.0, 5.0, 180.0),
        # ── Tower optimizer steps ──
        # The proportional-nudge bounds and binary step factors used by every
        # meta-learner. Even the optimizer's own step sizes are named priors here
        # rather than buried literals (see learning/meta_idioms.py). Not actively
        # calibrated (calibrating the optimizer's step bounds from its own
        # effectiveness is meta-meta-meta and risks theater; honest priors).
        "nudge_lo": (0.8, 0.05, 0.5, 0.99),
        "nudge_hi": (1.2, 0.05, 1.01, 2.0),
        "step_down": (0.95, 0.02, 0.5, 0.999),
        "step_up": (1.05, 0.02, 1.001, 2.0),
        "step_down_bold": (0.9, 0.03, 0.5, 0.99),
        "step_up_bold": (1.1, 0.03, 1.01, 2.0),
        # White-noise SNR reference: var(x) / var(Δx) = 0.5 for a white-noise series —
        # the principled neutral point for adaptive-bandwidth EMA smoothing. A math
        # fact, registered as a named prior so it has one named home.
        "snr_white_noise_ref": (0.5, 0.05, 0.3, 0.7),
        # ── Tier-2 (higher-order) priors ──
        # Wide-clamp for calibration channels whose ratio variance is genuinely wider
        # than the standard nudge bounds. CALIBRATED by tier 2 from saturation rate.
        "wide_nudge_lo": (0.5, 0.05, 0.1, 0.95),
        "wide_nudge_hi": (2.0, 0.1, 1.05, 5.0),
        # Collapse-rate EMA smoothing window (adaptive SNR, same idiom as forecast_ema).
        "collapse_rate_ema_alpha": (0.05, 0.02, 0.005, 0.3),
        # The auto-collapse confidence_threshold is steered toward a target collapse
        # RATE via the one law (a proportional gradient_step). Its control knobs:
        "collapse_rate_target": (0.1, 0.03, 0.01, 0.5),
        "collapse_thresh_lr":   (0.1, 0.03, 0.01, 0.5),
        # Attention transcribe-hysteresis as GEOMETRY: the run/skip band half-width =
        # (1−purity)·this, centred on the 0.5 measurement midpoint — confident attend
        # flips sharply, uncertain stays sticky. See learning/attention.py.
        "attend_hysteresis_scale": (0.5, 0.1, 0.0, 1.0),
        # Tier-2 control targets (what counts as "well-tuned" for each learner).
        "classification_target": (0.7, 0.05, 0.5, 0.95),
        "clamp_saturation_target": (0.15, 0.05, 0.02, 0.5),
        # Surprise feed: a settling substrate learner emits a near-constant value every
        # tick — relative deadband below which that tick is not news and isn't stamped.
        "substrate_emit_deadband": (0.1, 0.03, 0.0, 0.5),
        # ── MASTER ACTUATION SILENCE (the "Listen" override) ──
        # actuation_silenced=1 → the engine stops actuating ENTIRELY while observation +
        # learning keep running — clean-data LISTEN mode. The hard guarantee is at the
        # dispatcher (refuses to publish + drains the queue). The scalar floor of the
        # agency qubit.
        "actuation_silenced": (0.0, 0.0, 0.0, 1.0),
        # agency_auto_act_enabled: opt-in to auto-fold — a released silence climbs past
        # RECOMMEND toward ACT as confidence earns it. Default OFF → heals to recommend
        # and stops (the safe floor); the operator re-enables deliberately.
        "agency_auto_act_enabled": (0.0, 0.0, 0.0, 1.0),
        # The operator's listen↔act PREFERENCE — a wish the user sends; the act/listen
        # CONCEPT lives in the AgencyQubit (graph), this only nudges it. agency_pref_z ∈
        # [−1,+1] (−1 listen / 0 recommend / +1 act); agency_pref_alpha = how hard,
        # 0 = NO active preference (the qubit relaxes on its own — the default floor).
        "agency_pref_z":     (0.0, 0.0, -1.0, 1.0),
        "agency_pref_alpha": (0.0, 0.0, 0.0, 1.0),
        # tau_days: the weeks-scale agency relaxation constant (a released silence heals
        # back toward 'recommend' over ~tau_days).
        "agency_tau_days":   (10.0, 2.0, 1.0, 60.0),
        # agency_recommend_z — the RESTING stance between listen(-1) and act(+1); the
        # regime boundaries are midpoints to this anchor, so learning it moves the
        # geometry. agency_competence_knee — competence below the knee buys NO climb
        # toward act (soft ramp above); 0.0 = the exact prior law.
        "agency_recommend_z":      (0.0, 0.3, -0.8, 0.8),
        "agency_competence_knee":  (0.0, 0.1, 0.0, 0.9),
        # ── ContextState gauge (the run-mode fiber) ──
        # Four axes the engine reads to know what KIND of run this is. Default LIVE —
        # actuate/learn/persist=1.0, dt_factor=1.0. Sliding context_dt_factor moves the
        # whole φ-clock ladder along the Fibonacci sequence in unison.
        "context_actuate":   (1.0, 0.05, 0.0, 1.0),
        "context_dt_factor": (1.0, 0.1,  1.0, 10000.0),
        "context_learn":     (1.0, 0.05, 0.0, 1.0),
        "context_persist":   (1.0, 0.05, 0.0, 1.0),
        # The engine learns its OWN run-mode: the `learn` axis is observed from the
        # field's surprise rate (plasticity follows novelty). Opt-in UMWELT_CONTEXT_LEARN;
        # shadow-safe (floored above the 0.5 gate so learning can't switch itself off).
        "context_learn_floor":     (0.6, 0.1, 0.5, 0.95),
        "context_surprise_ref":    (0.05, 0.02, 0.005, 0.5),
        "context_learn_obs_sigma": (0.2, 0.05, 0.02, 0.5),
        # ── Smooth adaptive clock — the sampling DIAL, as LEARNED meta-params ──
        # dt_factor_max = max sampling stride during deep calm (the compute↔surprise
        # dial). Read off the fiber each tick. Tuned by a skill-per-compute meta-loop,
        # NOT the within-life tower — whose surprise-only objective would drive
        # dt_factor_max→1 (never coast), discarding the compute win.
        "dt_factor_max": (8.0, 2.0, 1.0, 32.0),
        "coast_eps":     (0.05, 0.02, 0.005, 0.30),
        # ── Breathing cadence dial (the dual-process shared clock) ──
        # Couplings for clocks/cadence_dial.py: the LIVE loop's wall-clock cadence
        # (demand speeds up, CPU stress + replay-lag slow down) + the REPLAY learner's
        # geometry (dormancy + lag coarsen the φ-ladder to catch up).
        "cadence_w_demand": (1.0, 0.2, 0.0, 3.0),    # decode-demand → faster live loop
        "cadence_w_stress": (0.8, 0.2, 0.0, 3.0),    # CPU/heat → slower (protect substrate)
        "cadence_w_lag":    (0.6, 0.2, 0.0, 3.0),    # replay lag → slower, but only in lulls
        "cadence_w_dorm":   (0.7, 0.2, 0.0, 2.0),    # dormancy → coarser replay (catch up)
        "cadence_w_lagH":   (0.7, 0.2, 0.0, 2.0),    # lag → coarser replay
        "cadence_k_max":    (6.0, 1.0, 1.0, 10.0),   # max φ-rungs of replay compression
        "cadence_l_ref":    (3600.0, 600.0, 60.0, 86400.0),  # lag horizon (s): backlog this deep ⇒ L=1
        # The dial's band edges + easing (learnable).
        "cadence_fast_s":   (0.5, 0.2, 0.1, 5.0),     # wall-clock FAST end (s)
        "cadence_slow_s":   (20.0, 5.0, 5.0, 120.0),  # wall-clock SLOW end (s)
        "cadence_attack":   (0.8, 0.1, 0.2, 1.0),     # fast-attack easing (catch transitions)
        "cadence_release":  (0.12, 0.05, 0.01, 0.6),  # slow-release easing (anti-thrash)
        # Replay throughput catch-up.
        "cadence_throughput_mult":   (2.0, 0.5, 0.0, 5.0),     # max extra batches when urgent
        "cadence_busy_ref":          (120.0, 30.0, 10.0, 600.0), # events/window ⇒ "busy"
        "cadence_dormancy_window_s": (120.0, 30.0, 30.0, 600.0), # dormancy-estimate window (s)
        # ── Signal-rate dial easing (see clocks/sensor dial) ──
        # attack = crank-to-fast when belief uncertainty rises (snap — never miss a
        # transition); release = relax-to-slow when settled (ease — no write-storm).
        "sensor_dial_attack":  (0.85, 0.1, 0.1, 1.0),
        "sensor_dial_release": (0.15, 0.05, 0.01, 1.0),
    })

    # ── Synthetic _params node (parameter memory cells) ────────────
    # No amplitude damping (gamma=0): a parameter qubit must HOLD whatever
    # observe_qubit puts on it between updates — otherwise it'd drift toward
    # |0⟩ (z→+1) and silently bias the scalar. gamma_diss is irrelevant here
    # because all _param_* roles are unitary in role_input_mode.
    if graph.find("_params") is not None:
        _attach(graph, "_params", {
            "gamma": (0.0, 0.0, 0.0, 0.001),
        }, frozen_keys={"gamma"})

    # ── Synthetic _clock node (periodic-driver phase leaf) ─────────
    # Holds the learned driver phase on a Bloch equator. No damping: the engine
    # anchors it from the driver and the H-tower learns to carry it; amplitude
    # damping would bias the phase.
    if graph.find("_clock") is not None:
        _attach(graph, "_clock", {
            "gamma": (0.0, 0.0, 0.0, 0.001),
        }, frozen_keys={"gamma"})

    # ── Per-node priors from the SPEC (domain data, never engine code) ──
    # A NodeSpec's `params` dict ({name: (default, sigma, lo, hi)}) attaches here;
    # the root node's spec params MERGE over the engine defaults above (spec wins,
    # via ParameterBundle.merge semantics in _attach).
    if spec is not None:
        for ns in getattr(spec, "nodes", ()) or ():
            node_params = getattr(ns, "params", None)
            if node_params:
                _attach(graph, ns.name, dict(node_params))

    # ── Archetype parent layers — shared dynamics across similar leaves ───
    # Similar leaves don't need UNIQUE dynamics hyperparameters. Each archetype is a
    # shared "parent layer" (a single ParameterBundle) whose ScalarParam objects are
    # REFERENCED by every member leaf — so the class learns ONE response, not
    # one-per-leaf. Cuts the distinct learnable-parameter count (more data per param,
    # faster convergence) and shrinks the state snapshot (each shared param stored
    # once). Per-leaf SEMANTICS stay private in spec params; only shared DYNAMICS
    # hyperparameters live on the parent.
    _attach_archetypes(graph)

    # ── Bridge coupling params ────────────────────────────────────
    _kind_defaults = {"open": 1.0, "gated": 0.7, "wall": 0.3}
    for bridge in graph.bridges:
        default = _kind_defaults.get(bridge.kind, 0.5)
        bridge.coupling_param = ScalarParam(
            name=f"coupling_{bridge.source}_{bridge.target}",
            value=default,
            sigma=0.1,
            lo=0.0,
            hi=1.5,
        )
