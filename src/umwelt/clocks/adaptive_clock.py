"""Adaptive clock — transition-intensity-driven compute compression.

Compute finely through transitions, COAST through calm. The step controller is the
*transition intensity*, blended from three signals the brain already computes:
  - imminence of the next input event (an event about to land → step finely),
  - the Berry-phase velocity (how fast the brain is moving) — berry_tape.ticker.speed,
  - the surprise EMA (how wrong it currently is) — fractal_stack.scales[0]._surprise_ema.

From intensity we derive (a) a smooth live `dt_factor` (calm → large → the φ-ladder
strides up, coarse; change → →1, fine) and (b) a COAST decision.

Coasting is EXACT only near a fixed point, so the guard coasts ONLY when the field is
genuinely parked (speed AND surprise below `coast_eps`) and no event is imminent — the
"don't skip if still moving" half of adaptive step-size integration. During a coast the
deterministic fibers (sun/clocks) are advanced analytically (now+Δt), so the clock stays
anchored at zero matrix cost.

Params live on a ParameterBundle so the outer meta-loop can tune them — especially
`coast_eps` and `dt_factor_max` — against skill-per-compute. See the plan and
[[project_phi_clock_console]], [[project_berry_tape]].

  python -m umwelt.clocks.adaptive_clock        # self-test (no engine needed)
"""
from __future__ import annotations
from umwelt._util import clamp01

from dataclasses import dataclass

# Tunable defaults — every one of these is a meta-param the outer PBT loop can perturb.
DEFAULTS: dict[str, float] = {
    "coast_eps": 0.05,            # speed/surprise floor below which the field is "parked"
    "event_horizon_s": 30.0,      # an input event within this many seconds blocks coasting
    "dt_factor_max": 8.0,         # max φ-ladder slide during deep calm
    "intensity_speed_ref": 0.10,  # berry speed that counts as "fully busy"
    "intensity_surprise_ref": 0.10,  # surprise EMA that counts as "fully busy"
    "w_event": 1.0,               # blend weights
    "w_speed": 0.6,
    "w_surprise": 0.6,
    # D.3 — positional certainty gate (only active when a position_certainty is passed):
    "position_certainty_relax": 0.85,  # above this (confidently settled/away) → tolerate a
                                       #   mild residual and still coast (interior can sleep).
                                       #   An UNCERTAIN/transitioning person needs no special
                                       #   case — their motion already wakes the clock via speed.
}


@dataclass
class ClockDecision:
    coast: bool          # skip the full reservoir step this tick (advance fibers analytically)
    dt_factor: float     # live φ-ladder slide (>=1.0); large = coarse during calm
    intensity: float     # [0,1] transition intensity (1 = changing fast / event imminent)


def transition_intensity(secs_to_event: float | None, speed: float,
                         surprise: float, p: dict) -> float:
    """Blend event-imminence + brain velocity + surprise into [0,1]. 1 = compute finely."""
    h = max(1e-6, p["event_horizon_s"])
    ev = 0.0 if secs_to_event is None else max(0.0, 1.0 - min(max(secs_to_event, 0.0), h) / h)
    sp = min(1.0, abs(speed) / max(1e-6, p["intensity_speed_ref"]))
    su = min(1.0, abs(surprise) / max(1e-6, p["intensity_surprise_ref"]))
    wsum = p["w_event"] + p["w_speed"] + p["w_surprise"]
    raw = p["w_event"] * ev + p["w_speed"] * sp + p["w_surprise"] * su
    return clamp01(raw / max(1e-6, wsum))


class AdaptiveClock:
    """Decides coast-vs-step and the live dt_factor each tick. Reads tunables from an
    optional ParameterBundle (so the meta-loop can learn them); falls back to DEFAULTS."""

    def __init__(self, params: dict | None = None, bundle=None):
        self.p = dict(DEFAULTS)
        if params:
            self.p.update(params)
        self.bundle = bundle

    def _g(self, key: str) -> float:
        if self.bundle is not None and key in getattr(self.bundle, "params", {}):
            return float(self.bundle.get(key))
        return float(self.p[key])

    def decide(self, secs_to_event: float | None, speed: float, surprise: float,
               position_certainty: float | None = None) -> ClockDecision:
        """`position_certainty` (optional, D.3): a [0,1] scalar the caller derives from the
        coarse positional qubit — HIGH = confidently settled/away (the interior can sleep),
        LOW = uncertain / a person in transition (fetch depth, track it). None → no effect,
        so every existing caller is unchanged."""
        p = {k: self._g(k) for k in DEFAULTS}
        intensity = transition_intensity(secs_to_event, speed, surprise, p)
        eps = p["coast_eps"]
        event_imminent = secs_to_event is not None and secs_to_event <= p["event_horizon_s"]
        coast = (abs(speed) < eps) and (abs(surprise) < eps) and not event_imminent
        # D.3 positional gate — probabilistic depth allocation: confidently settled/away lets
        # a mild interior residual still coast (the interior can sleep). Uncertain needs no
        # special case — a person in transition moves, and motion already wakes the clock.
        if position_certainty is not None and position_certainty >= p["position_certainty_relax"]:
            coast = coast or ((abs(speed) < 2 * eps) and
                              (abs(surprise) < 2 * eps) and not event_imminent)
        # smooth (geometric) ladder slide: deep calm → dt_factor_max, busy → 1.0
        dt_factor = max(1.0, p["dt_factor_max"] ** (1.0 - intensity))
        return ClockDecision(coast=coast, dt_factor=dt_factor, intensity=round(intensity, 4))


def _selftest():
    clk = AdaptiveClock()
    cases = [
        ("deep calm (parked, no event)",      None, 0.001, 0.002),
        ("calm but event in 10s",             10.0, 0.001, 0.002),
        ("event imminent (2s)",                2.0, 0.02,  0.03),
        ("moving fast (high berry speed)",    None, 0.30,  0.01),
        ("surprised (high surprise EMA)",     None, 0.01,  0.40),
        ("mild drift (just above eps)",       None, 0.06,  0.04),
    ]
    print("╔══ ADAPTIVE CLOCK self-test  (coast = skip full step; dt_factor = ladder slide)\n")
    print(f"   {'scenario':>34} {'→event':>7} {'speed':>6} {'surpr':>6} | "
          f"{'coast':>5} {'dt_f':>5} {'intens':>6}")
    for name, ev, sp, su in cases:
        d = clk.decide(ev, sp, su)
        evs = "  none" if ev is None else f"{ev:5.0f}s"
        print(f"   {name:>34} {evs:>7} {sp:>6.3f} {su:>6.3f} | "
              f"{str(d.coast):>5} {d.dt_factor:>5.2f} {d.intensity:>6.2f}")
    print("\n   → deep calm coasts at max dt_factor; any event/speed/surprise wakes it. ✓")

    print("\n╔══ D.3 positional gate  (same mild-drift field, different position certainty)\n")
    print(f"   {'position_certainty':>34} {'coast':>5}")
    for name, pc in [("uncertain (transitioning, 0.20)", 0.20),
                     ("middling (0.60, no override)", 0.60),
                     ("confidently away/settled (0.95)", 0.95)]:
        d = clk.decide(None, 0.06, 0.04, position_certainty=pc)   # mild drift just above eps
        print(f"   {name:>34} {str(d.coast):>5}")
    print("   → uncertain forces compute; confident lets a mild residual coast (interior sleeps). ✓")


if __name__ == "__main__":
    _selftest()
