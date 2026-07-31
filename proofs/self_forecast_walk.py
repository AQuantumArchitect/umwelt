"""proofs/self_forecast_walk.py — THE FIRST SELF-FORESEEING COGNIFOLD.

The counterpart to `forecast_walk.py`, but proving the SHIPPED seam rather than a manual bolt-on.
`forecast_walk.py` attaches a ForecastSurface by hand (`engine.forecast_surface = ...`). Here the
surface is attached AUTOMATICALLY by `build_engine` because the spec DECLARES `forecast_leaves`,
and it is stepped AUTOMATICALLY by `engine.ingest` every full tick. Nothing bolts anything on:
the world simply says "these are the facets I watch myself along," and the field foresees itself.

The subject is the smallest honest self: a mind at work, with four facets (comprehension, momentum,
surprise, conviction). We replay a structured self-signal stream and read back, from the field's own
forecast organ, how well it learned to anticipate each of its OWN facets — a mind watching itself
think AND foretelling its next state, graded on whether it was right (skill = anticipation quality).

The honest signal is the CONTRAST across facets:
  - conviction (a slow EMA — highly autocorrelated)      → most forecastable, skill climbs;
  - comprehension (smooth drift, shocked)                → forecastable;
  - surprise (spiky decaying pulses)                     → harder;
  - momentum (a derivative — near-white)                 → least forecastable, skill ~0.
So the field reports which of its own facets it can foresee and which it cannot — the
self-forecast half of the transparency thesis, on the self.

Run:
    cd /home/primearchitect/ws/umwelt
    PYTHONPATH=. python proofs/self_forecast_walk.py
"""
from __future__ import annotations

import argparse
import json
import sys

from examples.shaman_self.world import FACETS, SHAMAN_SELF_SPEC, self_signal_stream
from umwelt.boot import build_engine
from umwelt.projection.cognifold_trace import cognifold_trace
from proofs.forecast_walk import summarize


def run_self_forecast(*, seed: int = 7, ticks: int = 900, dt_min: float = 2.0):
    """Boot the shaman-self world and replay it. The forecast organ is spec-declared, so it
    attaches + steps itself — we only ingest. Returns (trace, engine, n_ticks)."""
    engine = build_engine(spec=SHAMAN_SELF_SPEC, population=False,
                          calibration=True, fractal=True)
    # THE PROOF THAT THE SEAM IS LIVE: nothing bolted the organ on — the spec did.
    assert engine.forecast_surface is not None, "spec.forecast_leaves did not auto-attach the organ"

    n = 0
    for now, readings in self_signal_stream(seed=seed, ticks=ticks, dt_min=dt_min):
        engine.ingest(sensor_readings=readings, now=now)   # ingest AUTO-steps the forecast organ
        n += 1
    trace = cognifold_trace(engine, world="shaman_self")
    return trace, engine, n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ticks", type=int, default=900)
    ap.add_argument("--dt-min", type=float, default=2.0)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args(argv)

    trace, engine, n = run_self_forecast(seed=args.seed, ticks=args.ticks, dt_min=args.dt_min)
    stats = summarize(trace)
    preds = engine.forecast_surface.predictions()

    print(f"[self_forecast] replayed {n} self-ticks · organ auto-attached from spec.forecast_leaves",
          file=sys.stderr)
    print(f"[self_forecast] {json.dumps(stats)}", file=sys.stderr)
    print("[self_forecast] the self foreseeing itself — best skill per facet "
          "(conviction>comprehension>surprise>momentum expected):", file=sys.stderr)
    by_role = stats.get("best_skill_by_role", {})
    for facet in FACETS:
        bar = "#" * int(max(0.0, by_role.get(facet, 0.0)) * 40)
        print(f"    self.{facet:14s} skill={by_role.get(facet, 0.0):.4f} {bar}", file=sys.stderr)
    print(f"[self_forecast] live forecast rungs: {len(preds)} "
          f"(registers_with_forecast={stats['registers_with_forecast']}/{stats['registers']})",
          file=sys.stderr)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
        print(f"[self_forecast] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
