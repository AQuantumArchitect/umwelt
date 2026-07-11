"""
ContextState — a 4-axis gauge that says what KIND of run this is.

The brain doesn't fundamentally distinguish "live service" from "test demo"
from "training replay" from "parallel sim." Those are *positions on a single
manifold* with four orthogonal axes:

    actuate    ∈ [0, 1]    how much do my decisions affect the physical world
                           (0 = shadow / dry-run; 1 = full dispatch)
    dt_factor  ∈ [1, ∞)    how fast does my internal time move vs wallclock
                           (1 = real-time; 1440 = 24h compressed to 60s)
    learn      ∈ [0, 1]    how much should this experience update my model
                           (0 = frozen weights; 1 = full kalman + hebbian)
    persist    ∈ [0, 1]    does this experience get stamped in long-term memory
                           (0 = no surprise tape, no DB; 1 = full audit)

Named operating modes are corners (or convenient interior points) on this
manifold:

    live      = (1, 1, 1, 1)         normal autonomous operation
    shadow    = (0, 1, 1, 1)         brain thinks but does not act
    test      = (1, 1, 0, 0)         demo: dispatch happens, but no learning,
                                     no surprise stamps — keeps the brain from
                                     interpreting test commands as user prefs
    replay    = (0, N, 1, 0)         feed historical events at compressed time,
                                     learn from them, don't actuate, don't
                                     pollute the persistence store
    sandbox   = (0, ∞, 1, 0)         the limit of replay — pure mental simulation

The state lives on `root.param_bundle` as four scalars (context_actuate,
context_dt_factor, context_learn, context_persist). Defaults are LIVE so the
existing system behaviour is unchanged.

Downstream loops (DimmerActuator, OutputSurface, SurpriseTape, calibration,
fractal_stack) read the context via `ContextState.from_bundle(root_bundle)`
and gate side-effects on it. The phi-clock's stride choices are themselves
samples of `dt_factor` at fixed φ-ladder positions; once context is wired,
the strides can be DERIVED from dt_factor instead of being hard-set (future
work — see project_context_gauge memory).

THE MANIFOLD, MADE REAL (2026-06-17). The four axes are not thin scalars — they are four qubits
of ONE shared cluster on the root fiber (a 4-qubit sub-manifold). So context is a genuine BELIEF:
each axis carries a value (the qubit's z, mapped to its range) AND a confidence (the qubit's purity
|r|). Named modes are CORNERS of that manifold; the interior is every mixture. Two ways the manifold
moves, and they are different physics:

  • IMPOSE (`write_to_bundle`) — the operator/run-harness pins a corner: a HARD collapse to full
    confidence, no learning (the membrane — an exogenous run-condition the brain does not infer).
  • OBSERVE (`observe`) — the brain softly collapses an axis toward evidence it read (surprise rate,
    user-touch frequency, time-of-day …): a PARTIAL collapse via the ONE universal law
    (UniversalLearner.observe), confidence rising only as far as the evidence supports. This is the
    brain LEARNING its own mode. The hook is complete + tested; no live signal wires it yet (that is
    the next pass — give the axes a tuning signal). See project_context_gauge, universal_learner.

GENERAL by construction: the axes live in one `_AXES` registry, so read/impose/observe/snapshot all
iterate it — a fifth axis (e.g. an `attend` mode) is one registry line, not code. Because the four
qubits share a cluster, axis COUPLING (learn↔persist correlation, say) is a future geometric
deepening that needs no new substrate.
"""
from __future__ import annotations

from dataclasses import dataclass

from umwelt.learning.universal_learner import UniversalLearner


# Root-bundle param keys. Kept central so other modules don't drift on naming.
PARAM_ACTUATE = "context_actuate"
PARAM_DT_FACTOR = "context_dt_factor"
PARAM_LEARN = "context_learn"
PARAM_PERSIST = "context_persist"

# The manifold's axis registry — (attr on ContextState, root-bundle param key, LIVE default).
# Single source of truth: from_bundle / write_to_bundle / beliefs / observe / snapshot all iterate it.
_AXES = (
    ("actuate", PARAM_ACTUATE, 1.0),
    ("dt_factor", PARAM_DT_FACTOR, 1.0),
    ("learn", PARAM_LEARN, 1.0),
    ("persist", PARAM_PERSIST, 1.0),
)
_AXIS_KEYS = {attr: key for attr, key, _ in _AXES}

# The brain learns its own mode by the SAME law as every other interior coordinate (totality):
# context observation is just UniversalLearner.observe on a context axis qubit.
_LEARNER = UniversalLearner()


@dataclass
class ContextState:
    """The four-axis run-mode gauge. Reads default to LIVE."""

    actuate: float = 1.0
    dt_factor: float = 1.0
    learn: float = 1.0
    persist: float = 1.0

    # ── Named modes (factory methods) ────────────────────────────────

    @classmethod
    def live(cls) -> ContextState:
        """Normal autonomous operation. The default everywhere."""
        return cls(actuate=1.0, dt_factor=1.0, learn=1.0, persist=1.0)

    @classmethod
    def shadow(cls) -> ContextState:
        """Brain thinks but doesn't dispatch. Useful for watching what the
        brain *would* do without giving it control of physical fixtures."""
        return cls(actuate=0.0, dt_factor=1.0, learn=1.0, persist=1.0)

    @classmethod
    def test(cls) -> ContextState:
        """Demo/calibration: actuation fires (we WANT the bulb to move),
        but learning is frozen and surprise-tape is muted so the test
        commands don't get interpreted as user preferences."""
        return cls(actuate=1.0, dt_factor=1.0, learn=0.0, persist=0.0)

    @classmethod
    def replay(cls, dt_factor: float = 10.0) -> ContextState:
        """Feed historical events at compressed wallclock-time, learn from
        them, but don't fire actuators or pollute the persistence store."""
        return cls(actuate=0.0, dt_factor=max(1.0, float(dt_factor)),
                   learn=1.0, persist=0.0)

    @classmethod
    def sandbox(cls) -> ContextState:
        """Pure mental simulation — no real-world coupling at all. The
        limiting case of replay with dt_factor → ∞."""
        return cls(actuate=0.0, dt_factor=float("inf"),
                   learn=1.0, persist=0.0)

    @classmethod
    def forebrain(cls) -> ContextState:
        """The live FOREBRAIN run-mode: actuate the world in real-time on a FROZEN
        brain — the expensive feature decomposition + every learner gated off so the
        tick stays real-time. Learning is delegated to the HINDBRAIN (REPLAY), which
        hands a fresh pickle back at the siesta. Distinct from test() (a transient
        demo with persist off) — this is the standing live brain."""
        return cls(actuate=1.0, dt_factor=1.0, learn=0.0, persist=1.0)

    @classmethod
    def forecast(cls) -> ContextState:
        """The FORECAST brain run-mode (the third brain in the braid). It NEVER
        actuates — strictly-forecast-without-intervention keeps the learning signal
        clean (human intervention muddies cause/effect). It DOES learn its leaves
        online (the per-leaf forward models train on delayed labels). It runs at the
        PRESENT (dt_factor=1) and predicts forward — the +H time-shift lives in each
        LeafForecaster's own features, not the reservoir clock. It owns no live
        pickle (persist=0): its product is the published forecast stream, not a saved
        body. Its belief flows downstream into the live brain as a confidence-gated
        observation (confidence = forecast skill × purity) — brain-chaining is the
        same gauge-math as a sensor read. See project_forebrain_hindbrain,
        project_confidence_gauge_braid, plan jaunty-finding-pizza."""
        return cls(actuate=0.0, dt_factor=1.0, learn=1.0, persist=0.0)

    # ── Bundle I/O ───────────────────────────────────────────────────

    @classmethod
    def from_bundle(cls, bundle) -> ContextState:
        """Read the current context from the root parameter bundle.

        Missing bundle (e.g. test fixtures without configure_param_bundles
        wiring) returns the LIVE default. Missing individual keys default to
        their live value — never silently swap a gate to "off."
        """
        if bundle is None:
            return cls.live()
        try:
            return cls(**{attr: float(bundle.get(key, dflt)) for attr, key, dflt in _AXES})
        except Exception:
            return cls.live()

    @classmethod
    def beliefs(cls, bundle) -> dict:
        """The manifold view: per-axis {value, confidence}. The brain's CURRENT belief about what
        kind of run this is, and how SURE it is (confidence = the axis qubit's purity |r|). Missing
        bundle/param → the LIVE value at full confidence (an imposed/undefined axis is certain).
        This is the read side the console + gauge show; the value side stays `from_bundle`."""
        out: dict[str, dict] = {}
        for attr, key, dflt in _AXES:
            p = bundle.get_param(key) if bundle is not None else None
            if p is None:
                out[attr] = {"value": dflt, "confidence": 1.0}
            else:
                out[attr] = {"value": round(float(p.value), 6),
                             "confidence": round(float(getattr(p, "purity_r", 1.0)), 4)}
        return out

    def write_to_bundle(self, bundle) -> None:
        """Stamp this context onto the root parameter bundle.

        Context is an IMPOSED run-condition (the operator's choice of mode), not a learned
        observation — so we HARD-SET each axis directly. The previous soft-Kalman write
        (`bundle.update(key, val, 0.01)`) was silently swallowed once the param fiber went
        qubit-backed (#309): a context param pinned pure (σ=0) gives Kalman gain
        α=σ_q²/(σ_q²+σ_o²)=0, so the write did nothing and `context_learn` stayed at its LIVE
        default — meaning forebrain/shadow/replay never actually engaged (the live forebrain
        was not freezing). Setting `param.value` works for both ScalarParam (plain field) and
        QubitBackedParam (observe α=1, no `update_count` bump — imposing a mode is not learning),
        and `value_to_bloch_z` clips, so dt_factor=inf (sandbox) is safe. Clamp to the param's own
        bounds when present so the old bounds-respecting behaviour is preserved.
        """
        if bundle is None:
            return
        for attr, key, _dflt in _AXES:
            p = bundle.get_param(key)
            if p is None:
                continue
            val = getattr(self, attr)
            lo, hi = getattr(p, "lo", None), getattr(p, "hi", None)
            if lo is not None:
                val = max(lo, val)
            if hi is not None:
                val = min(hi, val)
            p.value = val

    @classmethod
    def observe(cls, bundle, axis: str, target: float, obs_sigma: float = 0.15) -> float | None:
        """The brain OBSERVES its own run-mode: soft-collapse one axis toward evidence it read.

        This is the moonshot hook — context as something the brain LEARNS, not only something the
        operator imposes. Unlike `write_to_bundle` (a hard collapse pinning a corner at full
        confidence, the membrane), this is a PARTIAL collapse through the ONE universal law
        (UniversalLearner.observe → kalman_update): the axis moves toward `target` and confidence
        (purity) rises only as far as repeated consistent evidence supports. So mode-learning is the
        same physics as every other interior coordinate (totality), and it composes with imposition —
        an operator-pinned corner is pure (gain≈0) and won't drift; a never-imposed axis is mixed and
        free to be learned. `axis` is an attr name in `_AXES`. No live signal calls this yet — it is
        the complete, tested substrate the next pass fills (give the axes a tuning signal). Returns
        the post-collapse value, or None if the bundle/axis/param is absent."""
        if bundle is None or axis not in _AXIS_KEYS:
            return None
        return _LEARNER.observe(bundle.get_param(_AXIS_KEYS[axis]), float(target), obs_sigma)

    # ── Convenience: boolean gate projections ────────────────────────
    #
    # Today the four existing gates collapse a continuous axis to a bool.
    # Until we have continuous-mixed downstream logic, these properties
    # provide the same projection: ≥0.5 means "the gate is open."

    @property
    def actuate_allowed(self) -> bool:
        return self.actuate >= 0.5

    @property
    def learn_allowed(self) -> bool:
        return self.learn >= 0.5

    @property
    def persist_allowed(self) -> bool:
        return self.persist >= 0.5

    @property
    def is_live(self) -> bool:
        return (self.actuate_allowed and self.learn_allowed
                and self.persist_allowed and self.dt_factor == 1.0)

    # ── Diagnostic ───────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "actuate": round(self.actuate, 4),
            "dt_factor": round(self.dt_factor, 4)
                if self.dt_factor != float("inf") else "inf",
            "learn": round(self.learn, 4),
            "persist": round(self.persist, 4),
            "is_live": self.is_live,
        }
