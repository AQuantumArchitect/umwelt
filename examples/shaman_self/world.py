"""The shaman-self world: the FIRST example of a belief-field that foresees ITSELF.

A minimal self-model of a mind doing work. One `self` node carries four facets it holds
beliefs about — and, via the new spec-declared `forecast_leaves`, learns to ANTICIPATE:

  comprehension  how well the task is understood (grounds up, dips when reality surprises)
  momentum       progress velocity (co-moves with comprehension, lags a little)
  surprise       how much reality is diverging from the current model (spikes, then decays)
  conviction     confidence in the current path (a slow EMA of comprehension)

Because these facets have STRUCTURE (they are autocorrelated and coupled), the forecast brain
can learn to predict each one's own future Bloch-z on a φ-ladder of horizons — a mind watching
itself think AND foretelling its own next state, then graded on whether it was right
(confidence = forecast skill × belief purity). This is the self-foreseeing cognifold: the
transparency thesis's forecast half, made concrete on the smallest honest self.

Compile/validate: `python -m umwelt.spec.validate examples.shaman_self.world:SHAMAN_SELF_SPEC`
Run the proof:      `python proofs/self_forecast_walk.py`
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from umwelt.spec.schema import BindingSpec, DomainSpec, DriverSpec, NodeSpec

FACETS = ("comprehension", "momentum", "surprise", "conviction")

SHAMAN_SELF_SPEC = DomainSpec(
    name="shaman_self",
    nodes=(
        NodeSpec("psyche", parent=None, kind="root", roles=()),
        NodeSpec(
            "self", parent="psyche", roles=FACETS,
            role_modes={f: "dissipative" for f in FACETS},
            params={"gamma_diss": (0.10, 0.02, 0.0, 1.0)},
        ),
    ),
    bindings=tuple(
        # force_observe: the belief SNAPS to each facet's reading (belief-z ≈ the signal),
        # so the forecast is genuinely about the facet's own trajectory — not a heavily-
        # smoothed dissipative echo that any persistence anchor foresees trivially. This is
        # what lets skill DISCRIMINATE the forecastable facets from the noisy ones.
        BindingSpec(f"self_{f}", zone="self", role=f,
                    normalizer={"type": "range", "lo": 0.0, "hi": 1.0},
                    force_observe=True)
        for f in FACETS
    ),
    drivers=(DriverSpec("day", period_s=86400.0),),
    # THE NEW CAPABILITY: the self declares the facets it watches itself along — the leaves
    # the forecast brain anticipates. build_engine attaches the organ; ingest steps it live.
    forecast_leaves=tuple(("self", f) for f in FACETS),
    # Horizons in the octave where the self actually MOVES (≈1–2.5 h). Below ~1 h the belief
    # barely changes, so "predict my current z" is trivially near-perfect and skill saturates —
    # the honest contrast (which facets I can foresee vs which I can't) only shows once the
    # forecast reaches past the belief's own persistence.
    forecast_horizons_min=(55.0, 89.0, 144.0),
)


def self_signal_stream(seed: int = 7, ticks: int = 720, dt_min: float = 2.0,
                       t0: datetime | None = None):
    """A structured, self-correlated replay of a mind at work — smooth enough to be
    forecastable, surprising enough to be worth foreseeing. Yields (now, {sensor: value}).

    Dynamics (all in [0,1]): comprehension climbs toward mastery but is knocked down by
    surprise spikes and recovers; momentum tracks the *change* in comprehension; surprise
    is a decaying pulse train (scripted shocks); conviction is a slow EMA of comprehension.
    """
    rng = np.random.default_rng(seed)
    t0 = t0 or datetime(2026, 7, 22, 9, 0, 0)
    comp = 0.35            # comprehension
    conv = 0.30           # conviction (slow EMA of comp)
    surprise = 0.05
    # scripted shocks: reality diverges hard at these ticks
    shocks = set(int(x) for x in rng.integers(40, ticks - 20, size=max(6, ticks // 60)))
    for k in range(ticks):
        now = t0 + timedelta(minutes=dt_min * k)
        shock = 0.6 + 0.35 * rng.random() if k in shocks else 0.0
        # surprise: decaying pulse train — mostly quiet, sharp shocks (moderately forecastable:
        # the decay is predictable, the shocks are not).
        surprise = float(np.clip(0.80 * surprise + shock + 0.02 * rng.random(), 0.0, 1.0))
        # comprehension: slow drift up toward mastery, knocked down by shocks, lightly noised
        # → smooth, highly forecastable.
        drift = 0.012 * (0.9 - comp)
        comp = float(np.clip(comp + drift - 0.5 * shock + 0.008 * (rng.random() - 0.5), 0.02, 0.99))
        # conviction: slow EMA of comprehension — the steadiest facet → MOST forecastable.
        conv = float(np.clip(0.94 * conv + 0.06 * comp, 0.02, 0.99))
        # momentum: moment-to-moment progress velocity — genuinely jittery, near-white →
        # the facet the self CANNOT foresee (skill floors near "predict the mean").
        momentum = float(np.clip(0.5 + 0.9 * (rng.random() - 0.5), 0.0, 1.0))
        yield now, {
            "self_comprehension": comp,
            "self_momentum": momentum,
            "self_surprise": surprise,
            "self_conviction": conv,
        }
