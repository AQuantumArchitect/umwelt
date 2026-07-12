"""Seed profile — the INITIAL-CONDITION gauge (blank vs magic).

The brain has two run-mode knobs already: does-it-actuate (`context.actuate`) and which-pipe/run-config
(the ContextState gauge). This is the THIRD, orthogonal knob — not a runtime mode but a STARTING POINT:

  • magic — the origin inhabitant's mined preferences baked into the seeds.
  • blank — a brain that biases NOTHING: it learns the inhabitant's tastes, and (Phase 3) the
    world's anchors, from scratch. The deliverable: leave a blank build running, it learns enough
    from one inhabitant over ~30 days to take over the controls.

Every learnable seed in `param_bundles.py` is one of four CLASSES — this module is the executable audit map:

  mechanism  — how the quantum substrate works (field dynamics, collapse, projection, calibration
               learning rates, the cadence/context gauges, evidence-FUSION knobs, actuation gates, device
               clamps, the driver COUPLING priors [a cycle's phase is deterministic; how it couples is
               generic], tower-optimizer steps, per-kind gamma/hysteresis, bridge couplings). Generic —
               identical for any world. KEPT in both profiles.
  ephemeris  — where/when the world is: latitude, longitude. A blank build LEARNS this from the body
               (a fix stream), it does not assume it. Midpoint-neutralizing a coordinate is meaningless, so this module
               AUDITS + flags the ephemeris params but does NOT neutralize them; the live coordinate lives
               in a gauge-tracked anchor qubit, which a blank build leaves UN-GROUNDED (maximally mixed)
               and grounds from evidence fixes via engine.ground_anchor.
  preference — the inhabitant's mined tastes: the comfort manifold + per-region dwell. The thing a blank
               build must learn from scratch. NEUTRALIZED in blank.
  gear       — hand-drawn preference SHAPES: the origin's "wellness" polynomial + its anchor. Forward-only,
               but the H-tower learns to ANTICIPATE the curve, so the hand-drawn shape IS the curriculum.
               NEUTRALIZED here (Phase 1) AND demoted to a shadow-teacher in code (Phase 2).

`blank` is the DEFAULT and the FLOOR — the maximally-mixed slate a build starts from: value at the midpoint
of [lo,hi] (Bloch z=0 → purity 0 → full-width wriggle), update_count reset, the hand-drawn gears + the
home-lock absent. Nothing is assumed. `magic` is the explicit OPT-IN (UMWELT_SEED_PROFILE=magic) that ADDS
the origin's mined seeds + the gears + its home-lock back — it is not 'the default with blank gating it
off', it is the floor PLUS opted-in shapes. The Fibonacci watch-first handoff (separate feature) means
blank's uninformative point estimates are OBSERVED into shape before they ever actuate.

Mirrors the reward-registry idiom (reward/registry.py): one human-authored, diffable manifest is the single
place a developer assigns a seed to a class.
"""
from __future__ import annotations

import logging

from umwelt._util import env_flag  # noqa: F401  (kept for symmetry; profile read below)

logger = logging.getLogger(__name__)

PROFILE_MAGIC = "magic"
PROFILE_BLANK = "blank"
PROFILES = (PROFILE_MAGIC, PROFILE_BLANK)

# Classes that a `blank` profile resets to max-entropy. ephemeris is audited but body-sourced (Phase 3),
# so it is NOT midpoint-neutralized here (a midpoint latitude is nonsense).
NEUTRALIZED_IN_BLANK = frozenset({"preference", "gear"})

# ── EPHEMERIS — where/when the world is (body-sourced in blank) ──
_EPHEMERIS = frozenset({"latitude", "longitude"})

# ── PREFERENCE — the inhabitant's mined tastes (learn from scratch in blank) ──
_PREFERENCE = frozenset({
    "comfort_temp_pref", "comfort_bed_shift", "comfort_couch_shift",
    "comfort_valence_coef", "comfort_arousal_coef",
    "presence_decay",            # per-region DWELL — learned DOWN from the inhabitant's re-declares
    "presence_decay_default",    # the global dwell fallback
})

# ── GEAR — hand-drawn preference shapes ──
# Empty since task #298: the origin's "wellness" polynomial (13 coefs + anchor/skill alphas) that used to
# live here was retired from the machine. The hand-drawn curve survives only as a shadow-cassette
# teacher in the lab — it is no longer a live param class to neutralize. The set is kept (empty) so the
# class label and the audit summary still resolve; future hand-drawn shapes can re-populate it.
_GEAR: frozenset = frozenset()


def seed_class(node_name: str, key: str) -> str:
    """The class a seed belongs to. `node_name` is accepted for future per-node rules (e.g. exterior
    climate sensor ranges → ephemeris); Phase-1 classification is key-based. Default: mechanism."""
    # Folded-topology keys ('{node}_{param}') would need the spec's param_key_normalizer
    # hook here; un-folded keys classify as-is.
    if key in _EPHEMERIS:
        return "ephemeris"
    if key in _GEAR:
        return "gear"
    if key in _PREFERENCE:
        return "preference"
    return "mechanism"


def _neutralize(param) -> None:
    """Reset a learnable param to MAX ENTROPY: value at the midpoint of its range (Bloch z=0 → purity 0,
    full-width posterior), confidence cleared. Works for QubitBackedParam (value setter → maximally-mixed
    qubit) and ScalarParam alike."""
    mid = (float(param.lo) + float(param.hi)) / 2.0
    param.value = mid                       # qubit → (0,0,0) maximally mixed (purity 0); scalar → mid
    if hasattr(param, "update_count"):
        param.update_count = 0
    if hasattr(param, "prior_mean"):
        param.prior_mean = mid


def _mix_analog_dissipative_beliefs(reservoir) -> list[str]:
    """Blank's slate for the BELIEF qubits themselves.

    Clusters construct in the pure ground state |0...0> (z=+1). For an event/unitary
    or observe role that IS the honest floor — |0> means 'off/vacant/unseen', the
    semantic rest state. But for an ANALOG dissipative role (a temperature, a rate, a
    drift) the ground pole carries no meaning: 'the reading sits at the cold pole' is
    a false certainty, not ignorance — the first foreign-cadence world (the market
    run) measured it as a ~15-tick boot transient every belief had to relax through
    before evidence won. Blank starts those qubits maximally mixed — genuinely
    unknown — and leaves every other role's ground start untouched."""
    from umwelt.spec.roles import is_analog_role, is_driver_role, is_observe_role
    mixed: list[str] = []
    field = getattr(reservoir, "field", None)
    for name, cluster in (getattr(field, "clusters", None) or {}).items():
        diss = getattr(cluster, "_dissipative_indices", None)
        if diss is None:
            diss = getattr(cluster, "_diss", set())
        for i, role in enumerate(getattr(cluster, "qubit_roles", ())):
            if (i in diss and is_analog_role(role)
                    and not is_observe_role(role) and not is_driver_role(role)):
                cluster.observe_qubit(i, (0.0, 0.0, 0.0), alpha=1.0)
                mixed.append(f"{name}:{role}")
    return mixed


def apply_seed_profile(reservoir, profile: str = PROFILE_MAGIC) -> dict:
    """Post-build pass over the param fiber. `magic` → NO-OP. `blank` → neutralize every preference+gear
    param to max-entropy (frozen params skipped; ephemeris classified + reported but left for body-sourcing)
    and start analog-dissipative belief qubits maximally mixed (_mix_analog_dissipative_beliefs).
    Returns a summary. Call AFTER `_bind_param_fiber` so the params are qubit-backed."""
    summary = {"profile": profile, "neutralized": [], "deferred_ephemeris": [], "counts": {}}
    if profile not in PROFILES:
        raise ValueError(f"unknown seed profile {profile!r} (choose {PROFILES})")
    # Record the recipe as provenance (the gauge-name reads it; the pickle carries it).
    reservoir.seed_profile = profile
    # THE RECIPE IS DATA, NOT A RUNTIME BRANCH: instead of an is_magic() check scattered through the brain,
    # the recipe writes the shapes it brings as plain config ATTRS on the engine. ac_use_seed_baseline
    # persists in the pickle so a SELECTED artifact carries it; home_lock is build-only (the anchor qubit
    # persists the locked coordinate). A blank floor leaves them off. The consumers (a domain's comfort
    # output, the anchor init) read these attrs — no recipe string. (The origin's output-anchor flag was
    # retired: preferences are learned from overrides, never seeded from a hand-authored polynomial.)
    magic = (profile == PROFILE_MAGIC)
    reservoir.ac_use_seed_baseline = magic     # AC bias = the mined baseline (else neutral device midpoint)
    reservoir.home_lock = magic                # lock the earth gear to home at build (else unlocated floor)
    counts: dict[str, int] = {}
    for node in reservoir.graph.root.walk():
        bundle = getattr(node, "param_bundle", None)
        if bundle is None:
            continue
        for key, param in bundle.params.items():
            cls = seed_class(node.name, key)
            counts[cls] = counts.get(cls, 0) + 1
            if profile != PROFILE_BLANK:
                continue
            if cls == "ephemeris":
                summary["deferred_ephemeris"].append(f"{node.name}.{key}")
                continue
            if cls in NEUTRALIZED_IN_BLANK and not getattr(param, "frozen", False):
                _neutralize(param)
                summary["neutralized"].append(f"{node.name}.{key}")
    summary["counts"] = counts
    if profile == PROFILE_BLANK:
        summary["mixed_beliefs"] = _mix_analog_dissipative_beliefs(reservoir)
        logger.info("seed profile BLANK: neutralized %d preference+gear params (%d ephemeris deferred to "
                    "body-sourcing); %d analog-dissipative beliefs mixed; mechanism seeds kept",
                    len(summary["neutralized"]), len(summary["deferred_ephemeris"]),
                    len(summary["mixed_beliefs"]))
    return summary


def seed_profile_snapshot(reservoir) -> dict:
    """The executable bias map: per-class param counts + a sample of each class's keys. Console-surfaceable
    — answers 'how much of this world is baked into the seeds?' without mutating anything."""
    by_class: dict[str, list[str]] = {"mechanism": [], "ephemeris": [], "preference": [], "gear": []}
    for node in reservoir.graph.root.walk():
        bundle = getattr(node, "param_bundle", None)
        if bundle is None:
            continue
        for key in bundle.params:
            by_class[seed_class(node.name, key)].append(f"{node.name}.{key}")
    return {
        "counts": {cls: len(keys) for cls, keys in by_class.items()},
        "neutralized_in_blank": sorted(by_class["preference"] + by_class["gear"]),
        "deferred_ephemeris": sorted(by_class["ephemeris"]),
        "sample": {cls: sorted(keys)[:8] for cls, keys in by_class.items()},
    }


# NOTE: there is deliberately NO is_magic()/is_blank() runtime predicate. "magic" is not a code special-case
# — it's the set of config attrs apply_seed_profile writes (ac_use_seed_baseline / home_lock) + the learned
# state, all carried by the artifact's pickle. Consumers read those attrs directly.
# `seed_profile` survives only as provenance (the gauge-name recipe component).


def seed_profile_from_env() -> str:
    """Read UMWELT_SEED_PROFILE. The DEFAULT is BLANK — the maximally-mixed floor: a build assumes
    NOTHING about its inhabitant or its coordinates unless something is explicitly OPTED IN. The mined
    magic build (the origin's preferences + the hand-drawn gears + its home-lock) is opted in with
    UMWELT_SEED_PROFILE=magic. Unknown → blank (the safe floor). DEPLOYMENT NOTE: a live magic build
    must set UMWELT_SEED_PROFILE=magic; with it unset the brain boots blank (preferences/gears absent)."""
    import os
    p = (os.environ.get("UMWELT_SEED_PROFILE") or PROFILE_BLANK).strip().lower()
    return p if p in PROFILES else PROFILE_BLANK
