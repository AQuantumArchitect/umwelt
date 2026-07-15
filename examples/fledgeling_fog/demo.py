#!/usr/bin/env python3
"""Fog corridor demo — host API happy path (Phase 1 + 2).

Boots blank via GameHost, replays a synthetic scout walk, prints a JSON-ish
timeline of beliefs. Happy path uses GameHost.observe_many only (no raw ingest).

Run from repo root:  python examples/fledgeling_fog/demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.fledgeling_fog.world import (
    FOG_SPEC,
    N_PLACES,
    agent_walk,
    occupied_place,
    place_names,
    runner_batches,
    synthesize_rows,
)
from umwelt.host import GameHost


def main() -> None:
    places = place_names(N_PLACES)
    host = GameHost()
    host.register_world(FOG_SPEC, population=False)
    print(
        f"booted BLANK via GameHost: seed_profile={host.engine.seed_profile!r}, "
        f"{len(host.engine.field.clusters)} clusters\n"
    )

    segments = agent_walk(seed=11, ticks=180)
    stream = synthesize_rows(FOG_SPEC, segments, seed=11)
    timeline: list[dict] = []
    checkpoint_every = 40

    for i, (readings, batch_t, conf) in enumerate(runner_batches(stream)):
        # Host API path (observe_many), not raw engine ingest
        host.observe_many("scout", readings, confidence=conf, t=batch_t)
        if i % checkpoint_every == 0:
            truth = occupied_place(segments, batch_t)
            beliefs = host.beliefs("scout")
            row = {
                "t": batch_t.isoformat(),
                "truth": truth,
                "near": {
                    p: {
                        "value": round(beliefs[f"{p}.agent_near"].value, 3),
                        "conf": round(beliefs[f"{p}.agent_near"].confidence, 3),
                    }
                    for p in places
                    if f"{p}.agent_near" in beliefs
                },
            }
            timeline.append(row)
            cells = " ".join(
                f"{p[-1]}:{row['near'].get(p, {}).get('value', 0.5):.2f}"
                for p in places
            )
            print(f"t={batch_t:%H:%M} truth={truth}  {cells}")

    print(f"\ntimeline_events={len(timeline)}")
    if timeline:
        print(json.dumps(timeline[-1], indent=2))
    print("fog corridor demo complete (host API path)")


if __name__ == "__main__":
    main()
