#!/usr/bin/env python3
"""Phase 4 agency demo — patrol sub-routine earns shadow auto-intend.

Teaches a patrol routine N successes, then shows:
  - no auto fire before N
  - shadow auto-intend after N (no live world side effects)
  - explicit promote is still required for live posture
  - surprise gate pauses FF / auto tick

Run:  python examples/fledgeling_fog/agency_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.fledgeling_fog.world import FOG_SPEC
from umwelt.host import GameHost
from umwelt.host.agency_loop import (
    AgencyLoop,
    AttentionBudget,
    PromotionGate,
    SubRoutine,
)


def main() -> None:
    host = GameHost()
    host.register_world(FOG_SPEC, population=False)
    n = 3
    loop = AgencyLoop(
        host,
        attention=AttentionBudget(capacity=5.0),
        promotion=PromotionGate(min_successes=n),
    )
    loop.add_routine(
        SubRoutine(
            name="patrol",
            intent_name="claim_safe",
            period_turns=1,
            attention_cost=1.0,
            actor_id="player",
        )
    )
    print(f"patrol sub-routine registered; min_successes={n}")
    print(f"live_dispatches_before={len(host.live_dispatches)}")

    # Before any success — must not auto-intend
    host.step_turn(1)
    fired = loop.tick()
    print(f"after 0 successes: fired={len(fired)} (expect 0)")
    assert fired == []

    # One success still blocked
    loop.teach_success("patrol")
    host.step_turn(1)
    fired = loop.tick()
    print(f"after 1 success: fired={len(fired)} (expect 0)")
    assert fired == []

    # Reach N
    loop.patrol_demo_ready("patrol", n - 1)  # was 1; add n-1 more → n
    assert loop.routines["patrol"].successes >= n
    host.step_turn(1)
    fired = loop.tick()
    print(
        f"after {n} successes: fired={len(fired)} mode={fired[0].mode if fired else None} "
        f"dispatched={fired[0].dispatched if fired else None}"
    )
    assert len(fired) >= 1 and fired[0].mode == "shadow" and not fired[0].dispatched
    assert len(host.live_dispatches) == 0
    print(f"live_dispatches_after_shadow_auto={len(host.live_dispatches)}")

    # Explicit promotion (still app-owned)
    ok = loop.promote("patrol")
    print(f"promote(patrol)={ok} auto_live={loop.routines['patrol'].auto_live}")

    # Surprise pauses FF / agency tick
    host.step_turn(1)
    paused = loop.tick(surprise=0.9)
    print(f"surprise tick fired={len(paused)} clock={loop.clock.reason} paused={loop.clock.paused}")
    assert paused == [] and loop.clock.paused

    print("agency demo complete: earned shadow auto-intend after N; no self-confound live dispatch")


if __name__ == "__main__":
    main()
