"""Blank-slate style proof for the Fledgeling fog corridor (Phase 1 + host path).

  (1) blank floor before evidence
  (2) every declared binding drove the field
  (3) beliefs track ground-truth walk at checkpoints
  (4) save/load round-trip keeps hash + still ingests
Happy path uses GameHost (not raw engine.ingest).
"""
from __future__ import annotations

import math
import random
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from examples.fledgeling_fog.world import (
    FOG_SPEC,
    N_PLACES,
    agent_walk,
    last_vacated,
    occupied_place,
    place_names,
    runner_batches,
    synthesize_rows,
)
from umwelt.host import GameHost

PLACES = place_names(N_PLACES)
VACATED_S = 600.0
Z_MARGIN = 0.08  # value-space margin (calibrated [0,1])


def _pin_rngs() -> None:
    random.seed(1234)
    np.random.seed(1234)


def _boot() -> GameHost:
    host = GameHost()
    host.register_world(FOG_SPEC, population=False)
    return host


def test_blank_fog_corridor_comprehends_via_host() -> None:
    _pin_rngs()
    host = _boot()
    eng = host.engine

    # (1) blank floor
    assert eng.seed_profile == "blank"
    blochs = {
        tuple(
            round(float(v), 9)
            for v in eng.field.clusters[p].role_bloch("agent_near")
        )
        for p in PLACES
    }
    assert len(blochs) == 1, f"places distinguishable before evidence: {blochs}"

    walk = agent_walk(seed=11, ticks=200)
    rows = synthesize_rows(FOG_SPEC, walk, seed=11)
    checkpoints: list[tuple[datetime, dict]] = []
    fresh = [0]

    for readings, now, conf in runner_batches(rows, flush_secs=30.0):
        host.observe_many("scout", readings, confidence=conf, t=now)
        here = occupied_place(walk, now)
        if readings.get(f"scout_{here}") == 1.0:
            fresh[0] += 1
            if fresh[0] % 15 == 0:
                beliefs = host.beliefs("scout")
                checkpoints.append(
                    (
                        now,
                        {
                            p: beliefs[f"{p}.agent_near"].value
                            for p in PLACES
                            if f"{p}.agent_near" in beliefs
                        },
                    )
                )

    # (2) every binding drove the field
    bridge = eng.sensor_bridge
    touched = {
        b.sensor_id
        for b in bridge.bindings.values()
        if (b.node, b.qubit_role) in bridge.touched_roles
    }
    assert touched == {b.sensor_id for b in FOG_SPEC.bindings}

    # (3) track ground truth
    assert len(checkpoints) >= 2, f"too few checkpoints: {len(checkpoints)}"
    wins = 0
    for now, vals in checkpoints:
        here = occupied_place(walk, now)
        vacated = [
            p
            for p, ago in last_vacated(walk, now, PLACES).items()
            if p != here and ago >= VACATED_S
        ]
        if not vacated:
            continue
        ok = all(vals[here] > vals[p] + Z_MARGIN for p in vacated if p in vals)
        if ok:
            wins += 1
    assert wins >= 1, f"no checkpoint clearly tracked occupancy (checked {len(checkpoints)})"

    # beliefs face is calibrated value+confidence, not Bloch-required
    b = host.belief_value(PLACES[0], "agent_near")
    assert 0.0 <= b.value <= 1.0
    assert 0.0 <= b.confidence <= 1.0 + 1e-9

    # (4) save/load
    h_before = host.field_canon_hash()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "fog_state.pkl"
        host.save(path)
        fresh_host = _boot()
        fresh_host.load(path)
        assert fresh_host.field_canon_hash() == h_before
        t1 = datetime.fromisoformat(rows[-1][0])
        walk2 = agent_walk(
            seed=12,
            ticks=20,
            start=(t1 + timedelta(seconds=60)).isoformat(),
        )
        more = synthesize_rows(FOG_SPEC, walk2, seed=12)
        n = 0
        for readings, now, conf in runner_batches(more, flush_secs=30.0):
            fresh_host.observe_many("scout", readings, confidence=conf, t=now)
            n += 1
        assert n > 0

    # finite field
    for cname, c in eng.field.clusters.items():
        for role, idx in getattr(c, "role_index", {}).items():
            bb = c.qubit_bloch(idx)
            assert all(math.isfinite(float(v)) for v in bb), f"{cname}:{role}"
