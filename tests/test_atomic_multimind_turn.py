"""FIELD_NOTES_SEPTACRYPT §7 item 1: no focus-selected physics.

A world turn advances EVERY mind's engine equally — there is no code path
where only the "focused"/"active" mind steps while the others freeze. The
septacrypt consumer hit this class of bug on its own composite worlds
(UI-selected zone accidentally deciding which physics ran); this test pins
the law on WorldSession so the host layer can rely on it.
"""
from examples.fledgeling_fog.world import FOG_SPEC

from umwelt.host.session import WorldSession


def _clocks(session):
    return {oid: host._t for oid, host in session.minds.items()}


def test_step_advances_all_minds_equally():
    session = WorldSession().register_world(FOG_SPEC)
    for oid in ("ada", "ben", "cal"):
        session.add_mind(oid)

    before = _clocks(session)
    assert len(set(before.values())) == 1  # boot in sync

    # 'ada' being the mind we interact with must not privilege her physics.
    session.observe_raw("ada", "scout_place_0", 1.0, confidence=0.9)
    results = session.step()

    after = _clocks(session)
    assert set(results) == {"ada", "ben", "cal"}, "step skipped a mind"
    assert len(set(after.values())) == 1, "minds' clocks diverged after step"
    assert all(after[o] > before[o] for o in after), "a mind failed to advance"


def test_step_turn_keeps_minds_in_lockstep():
    session = WorldSession().register_world(FOG_SPEC)
    session.add_mind("ada")
    session.add_mind("ben")
    session.step_turn(5)
    clocks = _clocks(session)
    assert len(set(clocks.values())) == 1
