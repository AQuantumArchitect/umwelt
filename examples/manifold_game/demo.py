#!/usr/bin/env python3
"""Manifold-game demo — boot blank, replay a REAL game session, score honestly.

The second foreign world: a real quantum-farming-game session (recorded off
the game's player-parity LLM seat) replayed through the production ingest
path. Blank-slate style checks, then prequential comprehension scoring on a
held-out tail: predict each next wallet reading from the belief field
(denormalized z) vs the persistence baseline. Persistence is genuinely hard
to beat (CLAIMS.md, the ladder-walk verdict) — the honest deliverable is the
number, not a victory lap.

Run from the repo root:
  python3 examples/manifold_game/demo.py            # replay + score
  python3 examples/manifold_game/demo.py --refit    # recompute data/bounds.json
                                                    # from the train split
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HERE = Path(__file__).resolve().parent
TRAIN_FRACTION = 0.7


def _split(rows: list) -> tuple[list, list]:
    cut = int(len(rows) * TRAIN_FRACTION)
    return rows[:cut], rows[cut:]


def refit_bounds() -> dict:
    rows = json.loads((HERE / "data" / "rows.json").read_text(encoding="utf-8"))
    train, _ = _split(rows)
    per: dict[str, list[float]] = {}
    for _ts, sensor, value, _ in train:
        per.setdefault(sensor, []).append(float(value))
    bounds = {}
    for sensor, vals in sorted(per.items()):
        vals.sort()
        lo = vals[int(0.05 * (len(vals) - 1))]
        hi = vals[int(0.95 * (len(vals) - 1))]
        if hi - lo < 1e-9:
            hi = lo + 1.0  # flat train series: unit width, honest no-signal
        bounds[sensor] = {"lo": lo, "hi": hi}
    (HERE / "data" / "bounds.json").write_text(
        json.dumps(bounds, indent=1, sort_keys=True), encoding="utf-8")
    print(f"bounds refit from train split → {HERE / 'data' / 'bounds.json'}")
    return bounds


def main() -> int:
    if "--refit" in sys.argv:
        refit_bounds()

    # Import AFTER a possible refit so the spec picks the fresh bounds up.
    from examples.manifold_game.world import (
        PANTRY_ROLES, load_bounds, load_rows, manifold_spec)
    from umwelt.boot import build_engine
    from umwelt.events import replay_sensor_batches

    bounds = load_bounds()
    spec = manifold_spec(bounds)
    engine = build_engine(spec=spec)
    print(f"booted BLANK: {len(engine.field.clusters)} clusters, "
          f"{len(spec.bindings)} bindings, ingest_hold_s={spec.ingest_hold_s}")

    rows = load_rows()
    train, test = _split(rows)
    sensor_role = {sensor: role for role, sensor in PANTRY_ROLES.items()}

    def denorm(sensor: str, z: float) -> float:
        b = bounds[sensor]
        return b["lo"] + (z + 1.0) * 0.5 * (b["hi"] - b["lo"])

    def belief(sensor: str) -> float:
        cluster = engine.field.clusters["pantry"]
        return denorm(sensor, float(cluster.role_bloch(sensor_role[sensor])[2]))

    # Replay the train split through the production ingest path.
    for batch_time, readings, conf, _last in replay_sensor_batches(
            [tuple(r) for r in train], flush_secs=30.0):
        engine.ingest(sensor_readings=readings, now=batch_time, confidence=conf)

    # Prequential tail: before each test reading, record belief + persistence.
    err_belief: dict[str, list[float]] = {}
    err_persist: dict[str, list[float]] = {}
    last_seen: dict[str, float] = {}
    for _ts, sensor, value, _ in train:
        last_seen[sensor] = float(value)

    for batch_time, readings, conf, _last in replay_sensor_batches(
            [tuple(r) for r in test], flush_secs=30.0):
        for sensor, value in readings.items():
            if sensor not in sensor_role:
                continue
            actual = float(value)
            err_belief.setdefault(sensor, []).append(abs(belief(sensor) - actual))
            err_persist.setdefault(sensor, []).append(
                abs(last_seen.get(sensor, actual) - actual))
            last_seen[sensor] = actual
        engine.ingest(sensor_readings=readings, now=batch_time, confidence=conf)

    print("\nprequential MAE on the held-out tail (belief vs persistence):")
    informative = 0
    for sensor in sorted(err_belief):
        mb = sum(err_belief[sensor]) / len(err_belief[sensor])
        mp = sum(err_persist[sensor]) / len(err_persist[sensor])
        flat = bounds[sensor]["hi"] - bounds[sensor]["lo"] <= 1.0
        tag = " (flat in train — no signal to comprehend)" if flat else ""
        if not flat:
            informative += 1
        print(f"  {sensor:34s} belief {mb:7.3f}   persistence {mp:7.3f}{tag}")
    if informative == 0:
        print("  NOTE: this tape's economy barely moved — an honest null. "
              "Re-export a livelier session and --refit.")

    # Save/load canon: the field survives a round-trip.
    out = HERE / "data" / "_demo_state.pkl"
    h0 = engine.field_canon_hash()
    engine.save(str(out))
    engine.load(str(out))
    out.unlink(missing_ok=True)
    assert engine.field_canon_hash() == h0, "save/load canon hash drifted"
    print(f"\nsave/load canon hash held: {h0[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
