"""proofs/forecast_walk.py — the SELF-FORECAST gym walk.

The forecast subsystem (foresight/forecast_surface.py) is fully implemented but no
shipped run loop attaches or steps it (`DEFAULT_FORECAST_LEAVES` is empty). This walk
turns it on: boot a blank engine on the gridworld, attach a ForecastSurface over every
cell's `agent_near` + `resource` leaves, replay a multi-day synthetic walk while
stepping the surface each batch, then export a cognifold trace whose per-register
`forecast` ladder is POPULATED — the belief-field predicting its own near future at the
13 / 21 / 34 / 55-minute Fibonacci horizons.

The honest signal is the CONTRAST between the two roles:
  - `resource` rides the world's diurnal driver (world.driver_ambient) → forecastable
    → forecast skill climbs, the ladder is bold;
  - `agent_near` rides a seeded random walk → barely forecastable at 13-55 min
    → skill stays near zero, the ladder is faint.
So the field shows which of its own beliefs it can anticipate and which it cannot —
exactly the reasoning-transparency / self-forecast story, rendered as real numbers.

This is the producer half of the SpaceWheat cognifold transparency instrument: the
exported trace (`cognifold_trace_v1`) is what SpaceWheat's CognifoldForecastField draws
as a forward "ghost ladder" from each belief's live Bloch point.

Run:
    cd /home/primearchitect/ws/umwelt
    PYTHONPATH=. python proofs/forecast_walk.py --days 5 --out /tmp/forecast_trace.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from examples.gridworld.world import (
    agent_walk, gridworld_spec, runner_batches, synthesize_rows,
)
from umwelt.boot import build_engine
from umwelt.foresight.forecast_surface import ForecastSurface
from umwelt.learning.context import ContextState
from umwelt.learning.runner import BrainRunner
from umwelt.projection.cognifold_trace import cognifold_trace


def build_forecast_trace(*, days: float = 5.0, seed: int = 7, rows: int = 3, cols: int = 3,
                         horizons_min=(55.0, 89.0, 144.0, 233.0)):
    """Run the walk and return (trace_dict, n_batches, engine). Deterministic in the args.

    Default horizons are the 55/89/144/233-min Fibonacci octave (≈1-4 h), where the world's
    beliefs genuinely move — the diurnal `resource` cycle turns and `agent_near` regresses
    toward its long-run mean — so the forecast has visible reach and skill decays honestly
    with horizon. (The engine's own DEFAULT_HORIZONS_MIN, 13-55 min, is the actionable
    near-ladder; the belief barely moves in <1 h, so the forward trail is a stub.)"""
    spec = gridworld_spec(rows, cols)
    engine = build_engine(spec=spec, population=False, role=ContextState.replay(dt_factor=10.0))

    # Attach the forecast surface over BOTH roles of every cell (the dormant seam:
    # nothing in build_engine/BrainRunner does this — DEFAULT_FORECAST_LEAVES is empty).
    cells = [n.name for n in spec.nodes if n.name.startswith("cell_")]
    leaves = tuple((c, "agent_near") for c in cells) + tuple((c, "resource") for c in cells)
    engine.forecast_surface = ForecastSurface(leaves=leaves, horizons_min=tuple(horizons_min))

    walk = agent_walk(rows, cols, seed=seed, days=days)
    stream = synthesize_rows(spec, walk, seed=seed)

    # Step the forecast surface with the SAME sim clock after each ingest (item[1] is `now`).
    def _on_batch(_n, item, _result):
        engine.forecast_surface.step(item[1], engine.field)

    n = BrainRunner(engine).replay(runner_batches(stream), on_batch=_on_batch)
    trace = cognifold_trace(engine, world="gridworld")
    return trace, n, engine


def build_filmstrip(*, out_dir: str, frames: int = 24, days: float = 6.0, seed: int = 7,
                    rows: int = 3, cols: int = 3,
                    horizons_min=(55.0, 89.0, 144.0, 233.0)) -> int:
    """Replay the same walk but export `frames` cognifold-trace SNAPSHOTS taken DURING the
    replay (evenly spaced by batch), as frame_000.json … — the belief-field evolving over
    time. Early frames catch beliefs still forming and forecasts not yet matured (empty
    ladders); later frames show the field settling and the self-forecast ladders appearing
    and sharpening. SpaceWheat's CognifoldTraceView plays these back (SW_COGNIFOLD_TRACE_DIR)
    so the instrument becomes LIVE — you watch a mind think and anticipate itself. Returns the
    number of frames written."""
    spec = gridworld_spec(rows, cols)
    engine = build_engine(spec=spec, population=False, role=ContextState.replay(dt_factor=10.0))
    cells = [n.name for n in spec.nodes if n.name.startswith("cell_")]
    leaves = tuple((c, "agent_near") for c in cells) + tuple((c, "resource") for c in cells)
    engine.forecast_surface = ForecastSurface(leaves=leaves, horizons_min=tuple(horizons_min))

    walk = agent_walk(rows, cols, seed=seed, days=days)
    batches = list(runner_batches(synthesize_rows(spec, walk, seed=seed)))  # materialize to count
    total = len(batches)
    marks = sorted({min(total, int(round((k + 1) * total / frames))) for k in range(frames)})
    mark_set = set(marks)

    os.makedirs(out_dir, exist_ok=True)
    state = {"written": 0}

    def _on_batch(n, item, _result):
        engine.forecast_surface.step(item[1], engine.field)
        if n in mark_set:
            trace = cognifold_trace(engine, world="gridworld")
            with open(os.path.join(out_dir, f"frame_{state['written']:03d}.json"),
                      "w", encoding="utf-8") as f:
                json.dump(trace, f, ensure_ascii=False)
            state["written"] += 1

    BrainRunner(engine).replay(iter(batches), on_batch=_on_batch)
    return state["written"]


def summarize(trace: dict) -> dict:
    """Honest readout: how much of the ladder actually populated, and the skill spread."""
    regs = trace.get("registers", [])
    with_fc = [r for r in regs if r.get("forecast")]
    rungs = sum(len(r.get("forecast", [])) for r in regs)
    skills = [float(e.get("skill", 0.0))
              for r in regs for e in r.get("forecast", []) if e.get("skill") is not None]
    by_role: dict[str, list[float]] = {}
    for r in regs:
        if not r.get("forecast"):
            continue
        role = r.get("role", "?")
        best = max((float(e.get("skill", 0.0)) for e in r["forecast"]), default=0.0)
        by_role.setdefault(role, []).append(best)
    return {
        "registers": len(regs),
        "registers_with_forecast": len(with_fc),
        "total_rungs": rungs,
        "skill_max": round(max(skills), 4) if skills else 0.0,
        "skill_mean": round(sum(skills) / len(skills), 4) if skills else 0.0,
        "best_skill_by_role": {k: round(max(v), 4) for k, v in by_role.items()},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=float, default=5.0, help="synthetic days to replay")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--horizons", type=str, default="55,89,144,233",
                    help="comma-separated forecast horizons in minutes (Fibonacci by default)")
    ap.add_argument("--out", type=str, default="", help="write the (single) trace JSON here")
    ap.add_argument("--filmstrip", type=int, default=0,
                    help="if >0, export this many trace snapshots DURING the replay")
    ap.add_argument("--out-dir", type=str, default="",
                    help="directory for --filmstrip frames (frame_000.json …)")
    args = ap.parse_args(argv)

    horizons = tuple(float(x) for x in args.horizons.split(",") if x.strip())

    if args.filmstrip > 0:
        if not args.out_dir:
            ap.error("--filmstrip requires --out-dir")
        written = build_filmstrip(out_dir=args.out_dir, frames=args.filmstrip,
                                  days=args.days, seed=args.seed, horizons_min=horizons)
        print(f"[forecast_walk] wrote {written} filmstrip frames to {args.out_dir}", file=sys.stderr)
        return 0

    trace, n, _engine = build_forecast_trace(days=args.days, seed=args.seed, horizons_min=horizons)
    stats = summarize(trace)
    print(f"[forecast_walk] replayed {n} batches over {args.days} days", file=sys.stderr)
    print(f"[forecast_walk] {json.dumps(stats)}", file=sys.stderr)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
        print(f"[forecast_walk] wrote {args.out}", file=sys.stderr)
    else:
        json.dump(trace, sys.stdout, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
