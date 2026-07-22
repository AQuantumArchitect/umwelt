"""examples/contraption/demo.py — drive the contraption and watch it foresee its own collapses.

A single-engine run (the spec's own forecast organ, surprise-context ON) that prints, per thing,
the live belief-z, its realized displacement (the collapse-surprise), and the decisive-horizon
forecast with both skill and skill_vs_persistence — so you can see the forecast anticipate the
next collapse rather than merely persisting a frozen belief.

Run:  cd /home/primearchitect/ws/umwelt && PYTHONPATH=. python examples/contraption/demo.py
"""
from __future__ import annotations

import sys

from examples.contraption.world import CONTRAPTION_SPEC, FORECAST_LEAVES, bot_collapse_stream
from umwelt.boot import build_engine
from umwelt.projection.cognifold_trace import cognifold_trace

DECISIVE_H = 16.0


def main(argv=None) -> int:
    eng = build_engine(spec=CONTRAPTION_SPEC, population=False, calibration=True, fractal=True)
    assert eng.forecast_surface is not None and eng.forecast_surface.n_context > 0, \
        "the contraption spec should attach a surprise-context forecast organ"

    for now, readings in bot_collapse_stream(seed=1, ticks=1500):
        eng.ingest(sensor_readings=readings, now=now)   # ingest auto-steps the organ

    trace = cognifold_trace(eng, world="contraption")
    print(f"[contraption] {len(trace['registers'])} registers, "
          f"{len(trace.get('edges', []))} edges", file=sys.stderr)

    print(f"\n{'leaf':10}{'value':>8}{'purity':>8}{'skill':>8}{'skill_vs_persist':>18}")
    fc_leaf_nodes = {n for (n, _r) in FORECAST_LEAVES}
    for r in trace["registers"]:
        if r["node"] not in fc_leaf_nodes or not r.get("forecast"):
            continue
        best = max(r["forecast"], key=lambda e: e.get("skill", 0.0))
        svp = max((e.get("skill_vs_persistence", 0.0) for e in r["forecast"]), default=0.0)
        print(f"{r['node']:10}{r['value']:8.3f}{r['purity']:8.3f}"
              f"{best['skill']:8.3f}{svp:18.3f}")
    print("\n  A frozen belief would show skill≈1 but skill_vs_persistence≈0; here the beliefs\n"
          "  genuinely collapse, so a positive skill_vs_persistence is real anticipation.\n"
          "  NOTE: these are IN-SAMPLE (one seed, trained online) — the honest HELD-OUT\n"
          "  generalization verdict (train/test on disjoint seeds) is proofs/contraption_walk.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
