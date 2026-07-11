"""The brain learns its OWN run-mode — context observation from evidence (the moonshot's learning).

`context.py` made run-mode a qubit-backed BELIEF manifold (actuate/dt_factor/learn/persist as four
qubits of one cluster) and gave it `observe` — a soft collapse via the ONE universal law. That was the
complete, tested SUBSTRATE; nothing observed it. This organ is the first evidence→observe wire: the
brain reads how surprising the world has been and collapses its `learn` axis toward the matching
PLASTICITY belief — novelty makes it plastic, confirmed prediction makes it rigid (the design-memory
vision, project_context_gauge). Mode-learning is therefore the same physics as every other interior
coordinate (UniversalLearner.observe → kalman_update); the manifold densifies with no new mechanism.

SAFE BY CONSTRUCTION, behaviorally SHADOW for now:
  • Opt-in: gated `UMWELT_CONTEXT_LEARN`, default OFF — the blank floor is unchanged.
  • The plasticity target is FLOORED at `context_learn_floor` (≥ the 0.5 gate threshold), so a learned
    `learn` belief can NEVER switch the learners off — it only expresses HOW plastic within the
    always-learning regime. Letting that continuous value drive the learning RATE (not just the gate)
    is the deliberate next step; today the belief moves + is visible (run_mode_beliefs) but does not
    change the binary gate.
  • Respects the membrane: an operator-IMPOSED freeze (learn pinned ≤ floor, e.g. forebrain) is left
    alone — the organ never fights an imposed run-mode.

Coupling weights live on the root bundle (context_* keys) so the meta-loop can tune them, exactly like
cadence_dial. Self-test: `python -m umwelt.learning.context_learning`. See context.py, cadence_dial.py.
"""
from __future__ import annotations

from umwelt._util import clamp01

from umwelt.learning.context import ContextState, PARAM_LEARN

# Fallback defaults (pre-attach / sandbox); param_bundles.py seeds mirror them.
DEFAULTS: dict[str, float] = {
    "context_learn_floor":     0.6,   # plasticity never falls below this — the gate (0.5) stays open
    "context_surprise_ref":    0.05,  # surprise EMA that saturates plasticity to 1.0 (the "very novel" scale)
    "context_learn_obs_sigma": 0.2,   # collapse width for the learn-axis observation (gentle)
}


class ContextLearner:
    """Observes the run-mode `learn` axis from the field's surprise rate — the brain inferring how
    plastic it should be. Reads its knobs live from an optional root ParameterBundle (meta-tunable),
    falling back to DEFAULTS. One method per tick; gated + shadow-safe by the caller + the floor."""

    def __init__(self, bundle=None):
        self.bundle = bundle
        self._relaxed = False     # have we handed the learn axis to the brain yet (opt-in, once)?

    def _g(self, key: str) -> float:
        if self.bundle is not None and key in getattr(self.bundle, "params", {}):
            return float(self.bundle.get(key))
        return float(DEFAULTS[key])

    def plasticity_target(self, surprise_rate: float) -> float:
        """Map a surprise EMA → a plasticity belief in [floor, 1]. Saturating: surprise at the
        reference scale ⇒ fully plastic (1.0); zero surprise ⇒ the floor (rigid, but still learning)."""
        floor = self._g("context_learn_floor")
        ref = max(1e-6, self._g("context_surprise_ref"))
        drive = clamp01(float(surprise_rate) / ref)
        return floor + (1.0 - floor) * drive

    def observe_mode(self, bundle, surprise_rate: float) -> dict | None:
        """One tick of mode-learning: collapse the `learn` axis toward the plasticity the surprise rate
        implies. Returns a small report (or None if absent/membrane-pinned). The caller gates this on
        UMWELT_CONTEXT_LEARN, so it is inert by default.

        Membrane guard: if the operator has IMPOSED a freeze (learn pinned at or below the floor — a
        certain low value, e.g. forebrain's learn=0), do NOT touch it. Otherwise, the first time we run
        we RELAX the axis (hand it to the brain: a mixed state it can learn) — that relaxation IS the
        opt-in act. Thereafter we observe it from evidence via the one law."""
        self.bundle = bundle if bundle is not None else self.bundle
        if bundle is None:
            return None
        p = bundle.get_param(PARAM_LEARN)
        if p is None:
            return None
        floor = self._g("context_learn_floor")
        # imposed-freeze membrane: a confident (pure) low learn value = the operator froze us. Leave it.
        if float(p.value) <= floor and float(getattr(p, "purity_r", 1.0)) > 0.9:
            return {"axis": "learn", "skipped": "imposed-freeze",
                    "value": round(float(p.value), 4)}
        if not self._relaxed:
            # hand the axis to the brain: seed it mid-plastic + mixed so it can be learned (L4 — a
            # learnable coordinate must start uncertain). Done once; thereafter evidence drives it.
            p.value = 0.5 * (floor + 1.0)
            self._relaxed = True
        target = self.plasticity_target(surprise_rate)
        ContextState.observe(bundle, "learn", target, obs_sigma=self._g("context_learn_obs_sigma"))
        return {"axis": "learn", "surprise_rate": round(float(surprise_rate), 4),
                "target": round(target, 4), "value": round(float(p.value), 4),
                "confidence": round(float(getattr(p, "purity_r", 0.0)), 4)}


def _selftest() -> None:
    """Sustained surprise → plasticity belief rises toward 1 + grows confident; quiet → drifts to floor;
    never below floor; an imposed freeze is respected. No reservoir needed."""
    from umwelt.substrate.params import ParameterBundle
    from umwelt.substrate.product_cluster import ProductQubitCluster

    # a minimal root bundle whose context_learn is a qubit-backed param (mirror the live fiber)
    bundle = ParameterBundle.from_dict({PARAM_LEARN: (1.0, 0.05, 0.0, 1.0)})
    cl = ProductQubitCluster("ctx", [PARAM_LEARN])
    bundle.bind_qubit(PARAM_LEARN, cl, cl.role_index[PARAM_LEARN])

    learner = ContextLearner(bundle)
    floor = DEFAULTS["context_learn_floor"]

    # sustained high surprise → plasticity climbs toward 1
    rep = None
    for _ in range(30):
        rep = learner.observe_mode(bundle, surprise_rate=0.08)
    assert rep is not None
    high = float(bundle.get(PARAM_LEARN))
    print(f"high-surprise plasticity={high:.3f} conf={rep['confidence']:.3f}")
    assert high > 0.8, high
    assert high >= floor

    # then quiet → drifts back toward the floor (rigid, but still ≥ floor = learners stay on)
    for _ in range(60):
        learner.observe_mode(bundle, surprise_rate=0.0)
    low = float(bundle.get(PARAM_LEARN))
    print(f"quiet plasticity={low:.3f} (floor={floor})")
    assert low < high
    assert low >= floor - 1e-6, (low, floor)

    # imposed freeze (learn pinned 0, pure) is respected
    ContextState.shadow().write_to_bundle(bundle)   # learn stays 1 in shadow; use an explicit freeze
    bundle.get_param(PARAM_LEARN).value = 0.0        # operator pins a frozen learn (pure low)
    rep2 = ContextLearner(bundle).observe_mode(bundle, surprise_rate=0.08)
    assert rep2 and rep2.get("skipped") == "imposed-freeze", rep2
    print("imposed-freeze respected:", rep2)
    print("OK")


if __name__ == "__main__":
    _selftest()
