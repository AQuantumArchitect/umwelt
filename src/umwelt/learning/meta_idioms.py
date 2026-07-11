"""Shared meta-learning vocabulary for the φ-clocked learning tower.

The proportional-nudge update — `value · clip(ratio, lo, hi)` — is the core
idiom by which every meta-learner (FractalScale.self_tune, calibration's
dynamics/coupling/_meta_learn, the periodic-driver calibration) adjusts a parameter
toward a target. Centralizing it kills duplication and means there is exactly
ONE place that defines what "a gentle proportional step" is for the tower.

Bound defaults match the historical idiom (clip ratio to [0.8, 1.2], ±5% binary
step). Callers source the bounds from the root parameter bundle (`nudge_lo`,
`nudge_hi`, `step_down`, `step_up`, `step_down_bold`, `step_up_bold`) — the
defaults below are the standalone fallback when the bundle isn't configured.
Wrapping these on the fiber satisfies the "no magic numbers; everything on the
tower" principle: even the optimizer's own step sizes live on the parameter
fiber as named priors.
"""
from __future__ import annotations

# Standalone fallbacks (used when no root bundle is available, e.g. in unit
# tests). In production the same values are registered as fiber priors on the
# root bundle and live-read.
NUDGE_LO_DEFAULT, NUDGE_HI_DEFAULT = 0.8, 1.2
STEP_DOWN_DEFAULT, STEP_UP_DEFAULT = 0.95, 1.05
STEP_DOWN_BOLD_DEFAULT, STEP_UP_BOLD_DEFAULT = 0.9, 1.1


def proportional_nudge(
    current: float,
    ratio: float,
    lo: float = NUDGE_LO_DEFAULT,
    hi: float = NUDGE_HI_DEFAULT,
) -> float:
    """The tower's proportional update step: `value · clip(ratio, lo, hi)`.

    Adjust `current` toward a target signal that arrived as `ratio` (e.g.
    target/observed). The clamp keeps each step gentle so the level above
    sees a smooth response, not a spike.
    """
    return current * max(lo, min(hi, ratio))


def tower_steps(root_bundle) -> dict[str, float]:
    """Live-read the named step priors off the root bundle (with fallbacks).

    Returns {nudge_lo, nudge_hi, step_down, step_up, step_down_bold,
    step_up_bold}. Callers pass these into proportional_nudge / use the binary
    factors directly. Centralizes the "where do the step sizes come from"
    question so it has exactly one answer.
    """
    if root_bundle is None:
        return {
            "nudge_lo": NUDGE_LO_DEFAULT, "nudge_hi": NUDGE_HI_DEFAULT,
            "step_down": STEP_DOWN_DEFAULT, "step_up": STEP_UP_DEFAULT,
            "step_down_bold": STEP_DOWN_BOLD_DEFAULT,
            "step_up_bold": STEP_UP_BOLD_DEFAULT,
        }
    g = root_bundle.get
    return {
        "nudge_lo": g("nudge_lo", NUDGE_LO_DEFAULT),
        "nudge_hi": g("nudge_hi", NUDGE_HI_DEFAULT),
        "step_down": g("step_down", STEP_DOWN_DEFAULT),
        "step_up": g("step_up", STEP_UP_DEFAULT),
        "step_down_bold": g("step_down_bold", STEP_DOWN_BOLD_DEFAULT),
        "step_up_bold": g("step_up_bold", STEP_UP_BOLD_DEFAULT),
    }
