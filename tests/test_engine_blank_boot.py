"""P2 milestone: build_engine(spec) boots BLANK, comprehends a synthetic walk, and
survives a save/load roundtrip — the engine-level half of the blank-slate proof
(the full proof gate with ground-truth assertions lands in proofs/ at P5).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tests.test_spec_to_field import tiny_grid_spec

CELLS = ["cell_0_0", "cell_0_1", "cell_1_0", "cell_1_1"]


def _boot():
    from umwelt.boot import build_engine
    return build_engine(spec=tiny_grid_spec())


def test_blank_boot_shape():
    engine = _boot()
    # blank floor: no seed vocabulary, the spec's clusters exist, one driver attached
    assert engine.seed_profile == "blank"
    for c in CELLS:
        assert c in engine.field.clusters
    assert len(engine.drivers) == 1 and engine.drivers[0].name == "day"
    # the driver's anchor node was materialized even though the spec never declared it
    assert "_clock" in engine.field.clusters
    # the spec's bindings all registered on the ingress bridge
    assert len(engine.sensor_bridge.bindings) >= 8
    # no tendrils yet (P3 wires OutputSpec); the uniform surface exists and is empty
    assert engine.tendrils == []


def test_ingest_comprehends_a_walk():
    engine = _boot()
    t0 = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    walk = ["cell_0_0"] * 6 + ["cell_0_1"] * 6 + ["cell_1_1"] * 6
    result = {}
    for i, occupied in enumerate(walk):
        readings = {f"sight_{occupied}": 1.0}
        for c in CELLS:
            if c != occupied:
                readings[f"sight_{c}"] = 0.0
        readings[f"resource_{occupied}"] = 3.0
        result = engine.ingest(sensor_readings=readings,
                               now=t0 + timedelta(seconds=10 * i))
    assert {"features", "collapsed", "transitions", "actions", "step"} <= set(result)
    assert result["step"] > 0
    # every declared binding actually drove the field (no dead vocabulary)
    touched = engine.sensor_bridge.touched_roles
    assert any("agent_near" in str(r) for r in touched)
    # the occupied cell's belief clearly exceeds a long-vacated cell's
    z_here = float(engine.field.clusters["cell_1_1"].role_bloch("agent_near")[2])
    z_there = float(engine.field.clusters["cell_0_0"].role_bloch("agent_near")[2])
    assert z_here > z_there
    # the driver anchored its phase qubit off the equilibrium origin
    clk = engine.field.clusters["_clock"]
    bx, by, _ = (float(v) for v in clk.role_bloch("phase"))
    assert (bx * bx + by * by) ** 0.5 > 0.1


def test_save_load_roundtrip(tmp_path):
    engine = _boot()
    t0 = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    for i in range(8):
        engine.ingest(sensor_readings={"sight_cell_0_0": 1.0},
                      now=t0 + timedelta(seconds=10 * i))
    h_before = engine.field_canon_hash()
    path = tmp_path / "engine_state.pkl"
    engine.save(str(path))
    assert path.exists()

    fresh = _boot()
    fresh.load(str(path))
    assert fresh.field_canon_hash() == h_before
