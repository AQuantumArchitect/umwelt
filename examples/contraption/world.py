"""The contraption: a hand-wired act → collapse → surprise → forecast loop.

The ant-grounding run on the flagship recorded world found every belief frozen at
purity≈0.5, r_bloch≈0.02, yet forecast_skill≈0.999 — a persistence mirage, because the
beliefs never MOVE. This world is the cure: bots that "make things" by COLLAPSING belief
qubits, so the beliefs genuinely jump, and a forecast fed the cross-leaf collapse-surprise
learns to anticipate the next collapse.

Two named design lessons made concrete:
  glass-ball θ actuators   botA/botB/botC.progress — γ=0 unitary qubits, re-observed onto
                            the sphere every tick (purity stays high), θ ramping 0→π toward
                            each bot's commit. θ is the mixedness-independent "how close to
                            making the next thing" readout. (Demonstration + the visible
                            actuators; NOT scored forecast leaves, so attribution stays clean.)
  measurement collapse     thing_1/thing_2/thing_3.state — force_observe beliefs that a bot
                            SNAPS between ±1 at its commit. The flip is the collapse: |Δz|≈2,
                            the collapse-surprise the forecast reads as context.

The dependency graph is what makes the surprise EARN its keep (forecastable beyond persistence
AND beyond the clock):
  botA  toggles thing_1 every ~P_A ticks, with JITTER (not a pure clock function).
  botB  toggles thing_2 exactly H_DEP ticks after thing_1's ACTUAL flip → thing_2's timing
        carries thing_1's jitter, knowable only by WATCHING thing_1 collapse.
  botC  toggles thing_3 H_DEP ticks after thing_1 and thing_2 agree (an AND chain).

So a forecaster that sees "thing_1 just collapsed" (its displacement spike in the context
vector) can anticipate thing_2's collapse; the clock alone cannot (jitter), and persistence
cannot (it is a toggle). That is the falsifiable claim proofs/contraption_walk.py tests.

The meerkat de-confounding lesson (you cannot cleanly observe a system you are driving) is
honored structurally: the context is PAST/PRESENT displacement only — a leaf's own future
collapse is never a feature — so anticipating thing_2 from thing_1's spike is genuine foresight,
not leakage.

Compile/validate: python -m umwelt.spec.validate examples.contraption.world:CONTRAPTION_SPEC
Run the proof:      python proofs/contraption_walk.py
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from umwelt.spec.schema import BindingSpec, DomainSpec, DriverSpec, NodeSpec

# The three things being made (the scored forecast leaves — where the mirage-vs-real test lives).
THINGS = ("thing_1", "thing_2", "thing_3")
# The three glass-ball actuators (demonstration; driven every tick; NOT scored).
BOTS = ("botA", "botB", "botC")

FORECAST_LEAVES = tuple((t, "state") for t in THINGS)

# Timing (in ticks; dt below sets minutes/tick). H_DEP is the dependency lag that makes
# thing_2/thing_3 forecastable ONLY by watching thing_1/thing_2 collapse.
P_A = 20          # botA base period
JITTER = 5        # ± ticks of jitter on botA's period — the part the clock can't foresee
H_DEP = 6         # ticks between a parent collapse and the dependent collapse

CONTRAPTION_SPEC = DomainSpec(
    name="contraption",
    nodes=(
        NodeSpec("workshop", parent=None, kind="root", roles=()),
        # things: dissipative beliefs, but force_observe re-pins them each tick to their held
        # commanded state, so between collapses they HOLD (a clean step function → persistence is
        # a strong, honest baseline) and at a collapse they JUMP.
        *(NodeSpec(t, parent="workshop", roles=("state",)) for t in THINGS),
        # bots: unitary (glass-ball) progress qubits — θ is the actuator readout.
        *(NodeSpec(b, parent="workshop", roles=("progress",),
                   role_modes={"progress": "unitary"}) for b in BOTS),
    ),
    bindings=(
        *(BindingSpec(f"{t}_state", zone=t, role="state",
                      normalizer={"type": "range", "lo": -1.0, "hi": 1.0},
                      force_observe=True) for t in THINGS),
        *(BindingSpec(f"{b}_progress", zone=b, role="progress",
                      normalizer={"type": "range", "lo": -1.0, "hi": 1.0},
                      force_observe=True) for b in BOTS),
    ),
    drivers=(DriverSpec("day", period_s=86400.0),),
    forecast_leaves=FORECAST_LEAVES,
    forecast_horizons_min=(8.0, 16.0, 26.0),   # decisive ≈ H_DEP·dt (the dependent-collapse lag)
    forecast_context_surprise=True,             # feed each leaf the cross-leaf collapse-surprise
)


def bot_collapse_stream(seed: int = 1, ticks: int = 1500, dt_min: float = 2.0,
                        t0: datetime | None = None):
    """Replay the contraption: yields (now, {sensor: value}) each tick.

    things are fed their HELD ±1 state every tick (step function; flips at a bot commit); bots
    are fed (1+cosθ)/2·2−1 = cosθ so the belief-z traces the ramping actuator angle θ (glass ball).
    Distinct `seed` → distinct jitter → a genuinely held-out schedule for the train/test split.
    """
    rng = np.random.default_rng(seed)
    t0 = t0 or datetime(2026, 7, 22, 9, 0, 0)

    s = {t: 1.0 for t in THINGS}          # held state of each thing ∈ {+1,-1}
    # botA's jittered commit schedule
    last_A = 0
    next_A = P_A + int(rng.integers(-JITTER, JITTER + 1))
    # pending dependent commits: tick -> which thing to toggle
    pending: dict[int, str] = {}
    # per-bot commit anchor (for θ ramp) and target tick
    bot_last = {b: 0 for b in BOTS}
    bot_next = {"botA": next_A, "botB": None, "botC": None}
    # botC readiness (fires H_DEP after thing_1 and thing_2 last agreed at +1)
    agree_since: int | None = None

    for k in range(ticks):
        now = t0 + timedelta(minutes=dt_min * k)

        # botA: periodic + jitter → toggle thing_1, schedule the dependent thing_2 commit
        if k >= next_A:
            s["thing_1"] = -s["thing_1"]
            bot_last["botA"] = k
            last_A = k
            next_A = k + P_A + int(rng.integers(-JITTER, JITTER + 1))
            bot_next["botA"] = next_A
            pending[k + H_DEP] = "thing_2"
            bot_last["botB"] = k
            bot_next["botB"] = k + H_DEP

        # botB (and any dependent): fire scheduled toggles
        if k in pending:
            which = pending.pop(k)
            s[which] = -s[which]
            if which == "thing_2":
                bot_last["botB"] = k

        # botC: H_DEP after thing_1 and thing_2 have BOTH been +1 together
        if s["thing_1"] > 0 and s["thing_2"] > 0:
            if agree_since is None:
                agree_since = k
                bot_last["botC"] = k
                bot_next["botC"] = k + H_DEP
            elif k >= agree_since + H_DEP and bot_next["botC"] is not None and k >= bot_next["botC"]:
                s["thing_3"] = -s["thing_3"]
                bot_last["botC"] = k
                bot_next["botC"] = None
                agree_since = None
        else:
            agree_since = None
            bot_next["botC"] = None

        readings = {f"{t}_state": s[t] for t in THINGS}
        # glass-ball actuators: θ ramps 0→π across [bot_last, bot_next]; feed cosθ so belief-z ≈ cosθ
        for b in BOTS:
            nxt = bot_next.get(b)
            if nxt is not None and nxt > bot_last[b]:
                frac = min(1.0, max(0.0, (k - bot_last[b]) / (nxt - bot_last[b])))
            else:
                frac = 0.0
            theta = np.pi * frac
            readings[f"{b}_progress"] = float(np.cos(theta))
        yield now, readings
