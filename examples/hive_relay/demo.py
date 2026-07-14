#!/usr/bin/env python3
"""Hive relay demo — agent reports are sensors, and the trust web prices them.

The HIVE thesis: many LLMs coordinate on a complex task by sharing one world
state. The first live deployment (a relay of haiku-class playtester agents
supervised by one coordinating model, 2026-07-14) found the thesis's central
enemy on day one: **confabulation**. Agents reported checkpoint banks that
never happened and story flags that never fired — fluently, in perfect JSON.
One phantom bank sent the next agent into a wrong world state and burned two
relay legs.

The supervisor's manual fix was a referee: verify every claim against ground
truth (manifest diffs, directory listings) before acting on it. This demo
shows that fix is already structural in umwelt: agent claims ingest as
SENSOR READINGS at honest η, referees are just more sensors, and the trust
web assigns each reporter the reliability it earns — including the
leave-one-out form where NO privileged oracle is needed, only three
heterogeneous reporters.

Three acts over the real tape (relay_tape.json — nine legs, verified):

  1. NAIVE (uniform trust): the fused belief follows the confabulators
     whenever they outnumber or outshout the truth.
  2. SUPERVISED (label = the referee): reliabilities converge to each
     source's true skill; the agent's weight collapses to what it earned.
  3. LEAVE-ONE-OUT (no oracle): each source scored against its peers'
     consensus — the confabulating reporter isolates anyway.

Run from the repo root:  python3 examples/hive_relay/demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from umwelt.foresight.trust_web import TrustWeb

HERE = Path(__file__).resolve().parent

ETA = {"haiku_flags": 0.7, "manifest_flags": 1.0, "supervisor_flags": 0.9,
       "haiku_banks": 0.7, "dir_banks": 1.0}


def load_tape() -> dict:
    return json.loads((HERE / "relay_tape.json").read_text(encoding="utf-8"))


def inputs_for(readings: dict) -> dict:
    return {s: (float(z), ETA[s], True) for s, z in readings.items()}


def act_naive(ticks: list) -> tuple[int, float]:
    """The pre-referee condition, as it actually happened: the agent's report
    is the ONLY input, taken at face value. This is how one phantom bank sent
    the next relay leg into a wrong world state."""
    web = TrustWeb()  # never learns
    wrong = 0
    conf_when_wrong = 0.0
    for t in ticks:
        z, conf = web.fuse(inputs_for({"haiku_flags": t["flags"]["haiku_flags"]}))
        truth = t["flags"]["manifest_flags"]
        if z * truth <= 0.0:
            wrong += 1
            conf_when_wrong = max(conf_when_wrong, conf)
    return wrong, conf_when_wrong


def act_supervised(ticks: list) -> TrustWeb:
    """Referee as label: reliabilities converge to true per-source skill."""
    web = TrustWeb(lr=0.35)  # nine ticks only — learn fast, honestly noted
    for t in ticks:
        inp = inputs_for(t["flags"])
        web.fuse(inp)
        web.learn(inp, t["flags"]["manifest_flags"])
    return web


def act_loo(ticks: list) -> TrustWeb:
    """No oracle: three heterogeneous reporters, leave-one-out labels."""
    web = TrustWeb(lr=0.35)
    for t in ticks:
        inp = inputs_for(t["flags"])
        web.fuse(inp)
        labels = web.loo_labels(inp)
        if labels:
            web.learn(inp, labels)
    return web


def main() -> int:
    tape = load_tape()
    ticks = tape["ticks"]
    n = len(ticks)
    haiku_right = sum(1 for t in ticks
                      if t["flags"]["haiku_flags"] == t["flags"]["manifest_flags"])
    print(f"the tape: {n} real relay legs; the agent's flag reports were right "
          f"{haiku_right}/{n} (three over-claims, one under-claim)\n")

    wrong, conf = act_naive(ticks)
    print("ACT 1 — the pre-referee condition (agent report taken at face value):")
    print(f"  shared state wrong on {wrong}/{n} ticks, held at confidence {conf:.2f} —")
    print("  fluent, structured, and false. This is the m8 phantom bank, generalized.\n")

    web_s = act_supervised(ticks)
    print("ACT 2 — supervised by the referee (label = manifest diff):")
    for s in ("haiku_flags", "supervisor_flags", "manifest_flags"):
        print(f"  reliability[{s}] = {web_s.r.get(s, 1.0):.2f}")
    print("  (reliability is an EMA — recent skill; this tape ends on an accurate")
    print("  streak, so 0.80 is 'improving lately', not a career average of 5/9.)\n")

    web_l = act_loo(ticks)
    print("ACT 3 — leave-one-out (no privileged oracle, three reporters):")
    for s in ("haiku_flags", "supervisor_flags", "manifest_flags"):
        print(f"  reliability[{s}] = {web_l.r.get(s, 1.0):.2f}")
    print("  the confabulator isolates against its peers' consensus alone.\n")

    r = web_l.r
    assert r["haiku_flags"] < r["manifest_flags"], "the agent must price below the referee"
    assert r["haiku_flags"] < r["supervisor_flags"], "and below the verifying supervisor"
    print("hive law, demonstrated on real coordination data: an agent's report is a")
    print("sensor reading at honest η — never a write. Referees are cheap sensors.")
    print("With three heterogeneous reporters, no oracle is needed at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
