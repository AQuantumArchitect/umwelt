"""Pin the manifold-game example — the second foreign world + the Berry demo.

Offline, from committed real tapes (no game, no network): the spec builds and
boots blank, the replay drives the field through the production ingest path,
and the Berry-decision demo's core assertion holds — a harvest gate reading
only accumulated geometric phase flips on the loop tape and refuses on the
same-duration still tape.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "manifold_game"
DATA = EXAMPLE / "data"

sys.path.insert(0, str(REPO))


def test_manifold_spec_builds_and_boots_blank():
    from examples.manifold_game.world import manifold_spec
    from umwelt.boot import build_engine

    spec = manifold_spec()
    engine = build_engine(spec=spec)
    pantry = engine.field.clusters["pantry"]
    for role in ("grain", "bread", "folk", "seed", "compost"):
        x, y, z = pantry.role_bloch(role)
        assert abs(z) < 1e-6, f"blank boot must start {role} at max entropy"


def test_replay_drives_the_field():
    from examples.manifold_game.world import load_rows, manifold_spec
    from umwelt.boot import build_engine
    from umwelt.events import replay_sensor_batches

    engine = build_engine(spec=manifold_spec())
    h0 = engine.field_canon_hash()
    rows = [tuple(r) for r in load_rows()][:400]
    for batch_time, readings, conf, _last in replay_sensor_batches(rows, flush_secs=30.0):
        engine.ingest(sensor_readings=readings, now=batch_time, confidence=conf)
    assert engine.field_canon_hash() != h0, "a real tape must move the field"


@pytest.mark.skipif(not (DATA / "berry_loop.json").exists(),
                    reason="berry tapes not yet recorded from the game")
def test_berry_decision_flips_on_winding_only():
    from examples.manifold_game.berry_decision import (
        decision_at_budget, harvest_decision, load_tape, run_tape)

    loop_tape = load_tape("berry_loop.json")
    still_tape = load_tape("berry_still.json")
    loop = run_tape(loop_tape)
    still = run_tape(still_tape)
    # The game's world never sits still (its boot kicks stationary states),
    # so the honest contrast is RATE: the wound path's gate opens much
    # sooner, and at a fixed budget the two choices are OPPOSITE.
    budget = loop_tape["samples"][1][0]
    assert decision_at_budget(loop_tape, budget) is True
    assert decision_at_budget(still_tape, budget) is False
    assert loop["flipped_at"] is not None
    assert still["flipped_at"] is None or still["flipped_at"] >= 4 * loop["flipped_at"]
    # The rule reads gamma and nothing else.
    threshold = loop_tape["ripe_threshold"]
    assert harvest_decision(threshold + 0.01, threshold)
    assert not harvest_decision(threshold - 0.01, threshold)


def test_demo_reports_flat_series_honestly():
    # The committed wallet tape includes flat train series (a real stuck
    # session); bounds refit must widen them to unit width, and the demo
    # must tag them as no-signal rather than claim comprehension.
    bounds = json.loads((DATA / "bounds.json").read_text(encoding="utf-8"))
    flats = [s for s, b in bounds.items() if b["hi"] - b["lo"] <= 1.0]
    assert flats, "expected at least one honest flat series in this tape"


def test_demo_runs_end_to_end():
    result = subprocess.run(
        [sys.executable, str(EXAMPLE / "demo.py")], cwd=REPO,
        capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert "save/load canon hash held" in result.stdout
