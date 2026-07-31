"""P3 milestone: OutputSpec → live shadow tendrils that emit Actions; a simulated
operator override moves the tendril's learned rise/fall geometry."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from tests.test_spec_to_field import tiny_grid_spec
from umwelt.spec.schema import BindingSpec, OutputSpec


def grid_spec_with_outputs():
    spec = tiny_grid_spec()
    outputs = (
        OutputSpec("harvest", node="cell_1_1", role="agent_near", kind="binary",
                   decode="sticky", gates={"rate_limit_s": 0.0},
                   dispatch={"actuator_id": "harvester_1"}),
        OutputSpec("move_bias", node="cell_0_0", role="agent_near", kind="scalar",
                   decode="linear", codomain=(0.0, 100.0),
                   readback_sensor="move_bias_readback",
                   gates={"rate_limit_s": 0.0, "deadband": 2.0},
                   coupling={"coupling": 0.3, "decay": 0.05}),
    )
    bindings = spec.bindings + (
        BindingSpec("move_bias_readback", zone="cell_0_0", role="resource",
                    normalizer={"type": "range", "lo": 0.0, "hi": 100.0}),
    )
    return replace(spec, outputs=outputs, bindings=bindings)


def _boot():
    from umwelt.boot import build_engine
    return build_engine(spec=grid_spec_with_outputs())


def _drive(engine, occupied: str, ticks: int, t0, offset: int = 0):
    result = {}
    for i in range(ticks):
        readings = {f"sight_{occupied}": 1.0}
        for c in ["cell_0_0", "cell_0_1", "cell_1_0", "cell_1_1"]:
            if c != occupied:
                readings[f"sight_{c}"] = 0.0
        result = engine.ingest(sensor_readings=readings,
                               now=t0 + timedelta(seconds=10 * (offset + i)))
    return result


def test_spec_outputs_become_shadow_tendrils_and_emit():
    engine = _boot()
    assert [t.name for t in engine.tendrils] == ["harvest", "move_bias"]
    assert all(t.shadow for t in engine.tendrils)

    t0 = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    _drive(engine, "cell_1_1", 20, t0)

    # the harvest tendril committed ON (evidence pumped its slow belief past the band)
    harvest = engine.tendrils[0]
    assert harvest.commit.level > 0.5
    # decisions were emitted but NOTHING dispatched: shadow law + no dispatcher injected
    surf = engine.output_surface
    assert len(surf.recommendations) > 0
    assert all(not live for _, _, live in surf.history)
    acts = [a for _, a, _ in surf.history if a.actuator_id == "harvester_1"]
    assert acts and acts[-1].command["on"] is True
    assert acts[-1].reason == "harvest_auto"


def test_override_moves_the_coupling():
    engine = _boot()
    t0 = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    move = engine.tendrils[1]
    c0, d0 = move.coupling, move.decay

    # agent sits in cell_0_0 → the evidence pumps move_bias HIGH, but the operator's
    # readback persistently says ~10/100 — a genuine correction AGAINST the evidence.
    for i in range(15):
        readings = {"sight_cell_0_0": 1.0, "sight_cell_1_1": 0.0,
                    "move_bias_readback": 10.0}
        engine.ingest(sensor_readings=readings, now=t0 + timedelta(seconds=10 * i))

    # the override loop pulled the belief toward the revealed preference and nudged
    # the rise/fall geometry — the tendril LEARNED from being corrected
    assert (move.coupling, move.decay) != (c0, d0)
    assert move.commit.level < 0.6      # dragged well below the evidence-only level


def test_dispatcher_receives_non_shadow_auto_actions():
    spec = grid_spec_with_outputs()
    live_outputs = tuple(replace(o, shadow=False) for o in spec.outputs)
    spec = replace(spec, outputs=live_outputs)
    sent = []
    from umwelt.boot import build_engine
    engine = build_engine(spec=spec, dispatch=sent.append)
    t0 = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    _drive(engine, "cell_1_1", 20, t0)
    assert sent, "non-shadow auto tendrils must reach the injected dispatcher"
    assert any(a.actuator_id == "harvester_1" for a in sent)


def _refire_engine():
    spec = grid_spec_with_outputs()
    outputs = tuple(
        replace(o, shadow=False, gates={**o.gates, "refire_s": 5.0})
        if o.name == "harvest" else replace(o, shadow=False)
        for o in spec.outputs
    )
    sent = []
    from umwelt.boot import build_engine
    engine = build_engine(spec=replace(spec, outputs=outputs), dispatch=sent.append)
    return engine, sent


def test_refire_redispatches_unchanged_command_while_committed():
    """gates.refire_s: an actuator that consumes events (not states) gets an honest
    repeat of the unchanged command while the committed belief stays engaged — the
    alternative apps invented was forging cool/warm observation edges to re-arm."""
    engine, _ = _refire_engine()
    t0 = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    _drive(engine, "cell_1_1", 20, t0)

    harvest = engine.tendrils[0]
    assert harvest.commit.level > 0.5
    base = harvest.last_dispatch_ts()
    assert base is not None

    # inside the refire window the unchanged command stays suppressed
    assert harvest.step(now_ts=base + 1.0) is None
    # due: the SAME command re-dispatches, no observation forgery required
    again = harvest.step(now_ts=base + 5.1)
    assert again is not None and again.command["on"] is True
    assert again.reason == "harvest_auto"

    # refire_s=0 (the default) keeps the classic change-only law
    harvest.refire_s = 0.0
    assert harvest.step(now_ts=base + 3600.0) is None


def test_refire_stops_when_commitment_decays():
    engine, _ = _refire_engine()
    t0 = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    _drive(engine, "cell_1_1", 20, t0)
    _drive(engine, "cell_0_0", 60, t0, offset=20)

    harvest = engine.tendrils[0]
    assert harvest.commit.level < 0.5
    base = harvest.last_dispatch_ts() or 0.0
    # the first far-future step may legitimately dispatch the sticky OFF flip
    harvest.step(now_ts=base + 10.0)
    # after that the command is unchanged and the belief is disengaged: no refire
    assert harvest.step(now_ts=base + 20.0) is None
    assert harvest.step(now_ts=base + 3600.0) is None
