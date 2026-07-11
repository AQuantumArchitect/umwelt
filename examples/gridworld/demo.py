#!/usr/bin/env python3
"""Gridworld demo — boot blank, replay a synthetic day, watch belief ease.

A viewer over the proof harness (proofs/_gridworld.py — the single source of the spec
and the deterministic day). Prints the agent_near belief per cell as a text heatmap at
checkpoints: the occupied cell rises on a sighting; vacated cells EASE back toward
uncertainty instead of snapping — the belief carries how sure it is.

Run from the repo root:  python3 examples/gridworld/demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from proofs._gridworld import (agent_walk, grid_cells, gridworld_spec, occupied_cell,
                               runner_batches, synthesize_rows)
from umwelt.boot import build_engine

SHADES = " ░▒▓█"


def shade(z: float) -> str:
    level = (z + 1.0) / 2.0
    return SHADES[min(int(level * len(SHADES)), len(SHADES) - 1)] * 2


def heatmap(engine, rows: int, cols: int) -> str:
    lines = []
    for r in range(rows):
        cells = []
        for c in range(cols):
            cl = engine.field.clusters.get(f"cell_{r}_{c}")
            z = float(cl.role_bloch("agent_near")[2]) if cl is not None else 0.0
            cells.append(shade(z))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def main(rows: int = 3, cols: int = 3) -> None:
    spec = gridworld_spec(rows, cols)
    engine = build_engine(spec=spec)
    print(f"booted BLANK: seed_profile={engine.seed_profile!r}, "
          f"{len(engine.field.clusters)} clusters, {len(engine.drivers)} driver(s)\n")

    segments = agent_walk(rows, cols, seed=7, days=1.0)
    stream = synthesize_rows(spec, segments, seed=7)
    checkpoint_every = 60
    for i, (readings, batch_t, conf) in enumerate(runner_batches(stream)):
        engine.ingest(sensor_readings=readings, now=batch_t, confidence=conf)
        if i % checkpoint_every == 0:
            truth = occupied_cell(segments, batch_t)
            print(f"t={batch_t:%H:%M}  agent truly in {truth}")
            print(heatmap(engine, rows, cols))
            print()
    print("day replayed — the same walk the proof gate asserts on (proofs/blank_slate.py)")


if __name__ == "__main__":
    main()
