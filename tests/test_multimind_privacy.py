"""Phase 3 multi-mind privacy assertion suite."""
from __future__ import annotations

from examples.fledgeling_fog.world import FOG_SPEC, place_names
from umwelt.host import WorldSession
from umwelt.host.api import Intent
from umwelt.learning.confounding import (
    actor_confounded_now,
    actor_intent_log,
    confounded_now,
    record_actor_intent,
)


def _two_agent_session():
    places = place_names()
    # A can scout all places; B only first half
    half = places[: len(places) // 2]
    sess = WorldSession().register_world(FOG_SPEC)
    mask_a = {f"scout_{p}" for p in places}
    mask_b = {f"scout_{p}" for p in half}
    sess.add_mind("A", channel_mask=mask_a)
    sess.add_mind("B", channel_mask=mask_b)
    return sess, places, half


def test_asymmetric_observation_diverges_beliefs():
    sess, places, half = _two_agent_session()
    far = places[-1]
    near = places[0]
    # B's mask rejects far channel even if host tries to push it
    rejected = sess.observe_raw("B", f"scout_{far}", 1.0, confidence=1.0)
    assert rejected.get("accepted") is False
    assert rejected.get("reason") == "channel_masked"
    # Boot both with "empty" on channels they can sense
    for _ in range(4):
        for p in places:
            sess.observe_raw("A", f"scout_{p}", 0.0, confidence=1.0)
        for p in half:
            sess.observe_raw("B", f"scout_{p}", 0.0, confidence=1.0)
        sess.step()
    b_far_before = sess.beliefs("B")[f"{far}.agent_near"].value
    # Only A sees far place occupied
    for _ in range(12):
        sess.observe_raw("A", f"scout_{far}", 1.0, confidence=1.0)
        sess.observe_raw("A", f"scout_{near}", 0.0, confidence=1.0)
        sess.observe_raw("B", f"scout_{near}", 0.0, confidence=1.0)
        sess.step()
    a_far = sess.beliefs("A")[f"{far}.agent_near"].value
    b_far = sess.beliefs("B")[f"{far}.agent_near"].value
    # Private minds diverge: A has direct far evidence; B does not
    assert a_far > b_far + 0.05, f"A={a_far:.3f} B={b_far:.3f}"
    # B's far belief must not jump to "occupied" without a far observation path
    assert b_far < 0.85, f"B far inflated without path: {b_far:.3f}"
    assert abs(b_far - b_far_before) < 0.35


def test_action_without_observation_path_does_not_inflate_other_mind():
    sess, places, half = _two_agent_session()
    far = places[-1]
    b_before = sess.beliefs("B")[f"{far}.agent_near"].value
    # A intends claim (shadow) — no observation path into B
    sess.intend("A", Intent(actor_id="A", name="claim_safe", shadow=True))
    for _ in range(5):
        sess.step()
    b_after = sess.beliefs("B")[f"{far}.agent_near"].value
    assert abs(b_after - b_before) < 0.02, (
        f"B's belief moved without observation path: {b_before} → {b_after}"
    )


def test_shared_global_belief_cheat_loses_privacy_suite():
    """A single shared field (cheat) cannot pass privacy-of-mind assertions."""
    places = place_names()
    far = places[-1]
    # CHEAT: one mind used for both "A" and "B" views
    sess = WorldSession().register_world(FOG_SPEC)
    shared = sess.add_mind("shared")
    # Simulate cheat: both observers alias the same host
    sess.minds["A"] = shared
    sess.minds["B"] = shared
    for _ in range(10):
        sess.observe_raw("A", f"scout_{far}", 1.0, confidence=1.0)
        sess.step()
    a_far = sess.beliefs("A")[f"{far}.agent_near"].value
    b_far = sess.beliefs("B")[f"{far}.agent_near"].value
    # Cheat fails privacy: beliefs are identical (no private umwelten)
    privacy_holds = abs(a_far - b_far) > 0.05
    assert not privacy_holds, "shared-global cheat incorrectly looked private"
    # The suite marks cheat as loss:
    assert a_far == b_far


def test_actor_keyed_confounding_surface():
    sess, places, _ = _two_agent_session()
    eng = sess.mind("A").engine
    record_actor_intent(eng, "A", "claim_safe")
    log = actor_intent_log(eng)
    assert log and log[-1][0] == "A"
    # Graph surface still works; actor filter is additive
    surface = confounded_now(eng.graph, {})
    assert isinstance(surface, dict)
    keyed = actor_confounded_now(
        eng.graph,
        {},
        actor_id="A",
        actor_actuated={"A": {}, "B": {"place_0": {"agent_near"}}},
    )
    assert isinstance(keyed, dict)


def test_multi_engine_cost_probe_recorded():
    sess = WorldSession().register_world(FOG_SPEC)
    r8 = sess.measure_cost(8, ticks=3)
    r32 = sess.measure_cost(32, ticks=2)
    assert r8["n_agents"] == 8
    assert r32["n_agents"] == 32
    assert r8["boot_s"] >= 0
    assert len(sess.cost_notes) >= 2
