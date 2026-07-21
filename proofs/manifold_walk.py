"""proofs/manifold_walk.py — the richest cognifold instrument: a LEARNED correlation web.

Companion to forecast_walk.py. The gridworld's clusters are product-state, so its cognifold
trace shows only structural `bridge` edges — no learned coupling. manifold_game is different:
its 5-role `pantry` node, run under the CUMULANT backend, learns a genuine fully-connected
`zz` web among semantically meaningful farm beliefs (grain / bread / folk / seed / compost) at
|J| up to ~0.22 — real "these beliefs move together because the field learned they're coupled".

Two levers matter (both set here): `UMWELT_CUMULANT=1` (the dense default never surfaces zz/xy)
and a multi-qubit node (pantry has 5 roles → one cumulant cluster with real intra-cluster pairs).

Exports the same `cognifold_trace_v1` document as forecast_walk (registers + edges + forecast +
actions), so SpaceWheat's CognifoldTraceView renders it UNCHANGED — same instrument, a world
whose beliefs genuinely correlate. Honest caveat: this recorded tape's economy "barely moved",
so forecast skill reads persistence-easy (~0.99); here the zz correlation web is the prize.

Run:
    cd /home/primearchitect/ws/umwelt
    PYTHONPATH=. python proofs/manifold_walk.py --out /tmp/manifold_trace.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# The two levers — MUST be set before build_engine imports the substrate.
os.environ.setdefault("UMWELT_CUMULANT", "1")
os.environ.setdefault("UMWELT_CUMULANT_MIN_QUBITS", "2")

from examples.manifold_game.world import (  # noqa: E402
    PANTRY_ROLES, load_bounds, load_rows, manifold_spec,
)
from umwelt.boot import build_engine  # noqa: E402
from umwelt.events import replay_sensor_batches  # noqa: E402
from umwelt.foresight.forecast_surface import ForecastSurface  # noqa: E402
from umwelt.learning.context import ContextState  # noqa: E402
from umwelt.learning.runner import BrainRunner  # noqa: E402
from umwelt.projection.cognifold_trace import cognifold_trace  # noqa: E402


def _setup():
    spec = manifold_spec(load_bounds())
    engine = build_engine(spec=spec, population=False, role=ContextState.replay(dt_factor=10.0))
    leaves = tuple(("pantry", r) for r in PANTRY_ROLES) + (("story", "progress"),)
    engine.forecast_surface = ForecastSurface(leaves=leaves, horizons_min=(55.0, 89.0, 144.0, 233.0))
    batches = [(rd, bt, cf) for bt, rd, cf, _ in
               replay_sensor_batches([tuple(r) for r in load_rows()], flush_secs=30.0)]
    return engine, batches


def build_manifold_trace():
    engine, batches = _setup()
    BrainRunner(engine).replay(
        iter(batches),
        on_batch=lambda _n, item, _r: engine.forecast_surface.step(item[1], engine.field))
    return cognifold_trace(engine, world="manifold_game"), len(batches)


def build_filmstrip(*, out_dir: str, frames: int = 12) -> int:
    engine, batches = _setup()
    total = len(batches)
    marks = sorted({min(total, int(round((k + 1) * total / frames))) for k in range(frames)})
    mark_set = set(marks)
    os.makedirs(out_dir, exist_ok=True)
    state = {"written": 0}

    def _on_batch(n, item, _r):
        engine.forecast_surface.step(item[1], engine.field)
        if n in mark_set:
            with open(os.path.join(out_dir, f"frame_{state['written']:03d}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(cognifold_trace(engine, world="manifold_game"), f, ensure_ascii=False)
            state["written"] += 1

    BrainRunner(engine).replay(iter(batches), on_batch=_on_batch)
    return state["written"]


def summarize(trace: dict) -> dict:
    edges = trace.get("edges", [])
    kinds: dict[str, int] = {}
    for e in edges:
        kinds[e.get("kind", "?")] = kinds.get(e.get("kind", "?"), 0) + 1
    zz = sorted((abs(float(e.get("weight", 0.0))) for e in edges if e.get("kind") == "zz"),
                reverse=True)
    return {
        "registers": trace.get("n_registers", 0),
        "edges": len(edges), "edge_kinds": kinds,
        "zz_max": round(zz[0], 4) if zz else 0.0,
        "with_forecast": sum(1 for r in trace.get("registers", []) if r.get("forecast")),
        "actions": len(trace.get("actions", [])),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--filmstrip", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args(argv)

    if args.filmstrip > 0:
        if not args.out_dir:
            ap.error("--filmstrip requires --out-dir")
        written = build_filmstrip(out_dir=args.out_dir, frames=args.filmstrip)
        print(f"[manifold_walk] wrote {written} filmstrip frames to {args.out_dir}", file=sys.stderr)
        return 0

    trace, n = build_manifold_trace()
    print(f"[manifold_walk] replayed {n} batches", file=sys.stderr)
    print(f"[manifold_walk] {json.dumps(summarize(trace))}", file=sys.stderr)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
        print(f"[manifold_walk] wrote {args.out}", file=sys.stderr)
    else:
        json.dump(trace, sys.stdout, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
