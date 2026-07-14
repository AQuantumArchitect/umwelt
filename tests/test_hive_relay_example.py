"""Pin the hive-relay example — real multi-LLM coordination data through the trust web.

The tape is nine verified relay legs (2026-07-14): agent self-reports vs manifest
referees. The pins: taken at face value the agent corrupts shared state on 4/9
ticks; priced by the web (supervised OR leave-one-out) the agent lands strictly
below both referees and the fused belief tracks truth.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from examples.hive_relay.demo import (  # noqa: E402
    act_loo, act_naive, act_supervised, inputs_for, load_tape)


def test_tape_shape_and_agent_skill():
    tape = load_tape()
    ticks = tape["ticks"]
    assert len(ticks) == 9
    right = sum(1 for t in ticks
                if t["flags"]["haiku_flags"] == t["flags"]["manifest_flags"])
    assert right == 5, "the tape's documented 5/9 agent accuracy must not drift"


def test_face_value_agent_corrupts_shared_state():
    wrong, conf = act_naive(load_tape()["ticks"])
    assert wrong == 4
    assert conf >= 0.5, "and it is confidently wrong — that is the danger"


def test_referee_prices_the_agent_below_ground_truth():
    web = act_supervised(load_tape()["ticks"])
    assert web.r["haiku_flags"] < web.r["manifest_flags"]
    assert web.r["haiku_flags"] < web.r["supervisor_flags"]


def test_leave_one_out_isolates_without_an_oracle():
    web = act_loo(load_tape()["ticks"])
    assert web.r["haiku_flags"] < web.r["manifest_flags"]
    assert web.r["haiku_flags"] < web.r["supervisor_flags"]


def test_fused_belief_tracks_truth_once_priced():
    from umwelt.foresight.trust_web import TrustWeb
    ticks = load_tape()["ticks"]
    web = act_supervised(ticks)  # priced reliabilities
    fresh = TrustWeb()
    fresh.r = dict(web.r)
    wrong = 0
    for t in ticks:
        z, _conf = fresh.fuse(inputs_for(t["flags"]))
        if z * t["flags"]["manifest_flags"] <= 0.0:
            wrong += 1
    assert wrong == 0, "with earned weights, the shared state never follows the confabulator"
