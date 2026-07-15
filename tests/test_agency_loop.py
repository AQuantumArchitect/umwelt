"""Phase 4 agency: sub-routines, attention, earned automation, no self-confound."""
from __future__ import annotations

from examples.fledgeling_fog.world import FOG_SPEC
from umwelt.host import GameHost
from umwelt.host.agency_loop import (
    AgencyLoop,
    AttentionBudget,
    PromotionGate,
    SubRoutine,
    TimeContraction,
)
from umwelt.learning.confounding import actor_intent_log


def _loop() -> AgencyLoop:
    h = GameHost()
    h.register_world(FOG_SPEC, population=False)
    loop = AgencyLoop(h, attention=AttentionBudget(capacity=5.0), promotion=PromotionGate(min_successes=3))
    loop.add_routine(
        SubRoutine(
            name="patrol",
            intent_name="claim_safe",
            period_turns=1,
            attention_cost=1.0,
            actor_id="player",
        )
    )
    return loop


def test_patrol_auto_intend_shadow_after_n_successes():
    loop = _loop()
    n = loop.promotion.min_successes
    assert n == 3
    # Before successes: no auto fire
    loop.host.step_turn(1)
    fired = loop.tick()
    assert fired == []
    # One success is not enough (spot-check the min_successes gate)
    loop.teach_success("patrol")
    loop.host.step_turn(1)
    fired = loop.tick()
    assert fired == [], "auto-intend must not fire after only 1 success when N=3"
    # N-1 still blocked
    loop.teach_success("patrol")  # total 2
    loop.host.step_turn(1)
    assert loop.tick() == []
    # Teach remaining to reach N
    loop.teach_success("patrol")  # total 3
    assert loop.routines["patrol"].successes >= n
    assert loop.promotion.can_auto_intend(loop.routines["patrol"])
    loop.host.step_turn(1)
    fired = loop.tick()
    assert len(fired) >= 1
    assert fired[0].mode == "shadow"
    assert fired[0].dispatched is False
    assert fired[0].actor_id == "player"
    assert fired[0].intent_name == "claim_safe"


def test_promotion_explicit_and_gated():
    loop = _loop()
    r = loop.routines["patrol"]
    assert loop.promote("patrol") is False  # not enough successes
    loop.patrol_demo_ready("patrol", 3)
    assert loop.promotion.can_promote(r)
    assert loop.promote("patrol") is True
    assert r.auto_live is True
    assert "patrol" in loop.promotion.promoted


def test_surprise_rest_pauses_ff():
    clock = TimeContraction()
    att = AttentionBudget(capacity=10.0, free=1.0)  # low → would FF
    clock.update(attention=att, surprise=0.0, rest=False)
    assert clock.ff_enabled is True
    clock.update(attention=att, surprise=0.9, rest=False)
    assert clock.paused is True
    assert clock.reason == "surprise"
    clock.update(attention=AttentionBudget(capacity=10.0, free=10.0), surprise=0.0, rest=True)
    assert clock.paused is True
    assert clock.reason == "rest"


def test_automation_does_not_reintroduce_self_confound():
    loop = _loop()
    loop.patrol_demo_ready("patrol", 3)
    loop.host.step_turn(1)
    fired = loop.tick()
    assert fired
    # Actor-tagged intents on the engine
    log = actor_intent_log(loop.host.engine)
    assert any(a == "player" for a, _, _ in log)
    # Guard list recorded
    assert loop._self_confound_guards
    # Shadow → no live world side effects
    assert loop.host.live_dispatches == []


def test_attention_budget_blocks_overspend():
    loop = _loop()
    # Earn auto-intend eligibility, then starve the budget
    loop.patrol_demo_ready("patrol", loop.promotion.min_successes)
    loop.attention = AttentionBudget(capacity=0.5, free=0.5)
    loop.routines["patrol"].attention_cost = 2.0
    loop.host.step_turn(1)
    fired = loop.tick()
    assert fired == []
