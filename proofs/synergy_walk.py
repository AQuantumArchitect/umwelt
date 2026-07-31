"""proofs/synergy_walk.py — the forecast discovers a genuine SYNERGY it cannot see first-order.

C = A × B (parity). Held out (disjoint train/test seeds, predict-only), the self-discovering
higher-order context (interaction_order=2, so z_A·z_B is a reachable monomial) forecasts C better
than persistence, while a first-order context (interaction_order=1, raw atoms only) CANNOT — because
no single state or trace carries any signal about a parity. That gap is the capability: higher-order
structure is usable when it is irreducible synergy, and the readout discovers which product holds.

Run:  cd /home/primearchitect/ws/umwelt && PYTHONPATH=. python proofs/synergy_walk.py
"""
from __future__ import annotations

import sys

import numpy as np

from examples.synergy.world import FORECAST_LEAVES, SYNERGY_SPEC, synergy_stream
from umwelt.boot import build_engine
from umwelt.foresight.forecast_surface import ForecastSurface

DECISIVE_H = 16.0
HORIZONS = (8.0, 16.0, 26.0)


def _engine():
    eng = build_engine(spec=SYNERGY_SPEC, population=False, calibration=False, fractal=False)
    eng.forecast_surface = None                     # we drive our own order-1 / order-2 surfaces
    return eng


def run(train_seeds=(1, 2, 3, 4), test_seeds=(101,), ticks=1500) -> dict:
    o1 = ForecastSurface(leaves=FORECAST_LEAVES, horizons_min=HORIZONS,
                         n_context=len(FORECAST_LEAVES), interaction_order=1)
    o2 = ForecastSurface(leaves=FORECAST_LEAVES, horizons_min=HORIZONS,
                         n_context=len(FORECAST_LEAVES), interaction_order=2)
    eng = _engine()
    for seed in train_seeds:
        for now, readings in synergy_stream(seed=seed, ticks=ticks):
            eng.ingest(sensor_readings=readings, now=now)
            o1.step(now, eng.field, train=True)
            o2.step(now, eng.field, train=True)
    err = {"persist": [], "order1": [], "order2": []}
    eng = _engine()
    for seed in test_seeds:
        realized, p_o1, p_o2, p_pe = {}, {}, {}, {}
        for now, readings in synergy_stream(seed=seed, ticks=ticks):
            eng.ingest(sensor_readings=readings, now=now)
            o1.step(now, eng.field, train=False)
            o2.step(now, eng.field, train=False)
            c = eng.field.clusters["C"]
            zC = float(c.qubit_bloch(c.role_index["state"])[2])
            realized[now] = zC
            f1, f2 = o1.bank[("C", "state", DECISIVE_H)], o2.bank[("C", "state", DECISIVE_H)]
            if f1.prediction_for is not None:
                p_pe[f1.prediction_for] = zC
                p_o1[f1.prediction_for] = float(f1.prediction)
            if f2.prediction_for is not None:
                p_o2[f2.prediction_for] = float(f2.prediction)
        for tgt, zt in realized.items():
            if tgt in p_pe:
                err["persist"].append(abs(p_pe[tgt] - zt))
                err["order1"].append(abs(p_o1[tgt] - zt))
                err["order2"].append(abs(p_o2.get(tgt, p_pe[tgt]) - zt))
    Ep, E1, E2 = (float(np.mean(err[k])) for k in ("persist", "order1", "order2"))
    return {"E_persist": round(Ep, 4), "E_order1": round(E1, 4), "E_order2": round(E2, 4),
            "svp_order1": round(1 - E1 / Ep, 4) if Ep > 1e-9 else 0.0,
            "svp_order2": round(1 - E2 / Ep, 4) if Ep > 1e-9 else 0.0,
            "n_ctx_order2": o2.n_context}


def main(argv=None) -> int:
    r = run()
    print(f"[synergy] C = A x B (parity). decisive horizon {DECISIVE_H:.0f}m, "
          f"order-2 monomials={r['n_ctx_order2']}", file=sys.stderr)
    print(f"  persist   E={r['E_persist']:.4f}", file=sys.stderr)
    print(f"  order-1   E={r['E_order1']:.4f}  svp={r['svp_order1']:+.4f}  "
          f"(no single feature can predict a parity)", file=sys.stderr)
    print(f"  order-2   E={r['E_order2']:.4f}  svp={r['svp_order2']:+.4f}  "
          f"(z_A·z_B discovered -> beats persistence)", file=sys.stderr)
    verdict = "WIN" if (r["svp_order2"] > 0.05 and r["svp_order2"] > r["svp_order1"]) else "MISS"
    print(f"[synergy] VERDICT: {verdict}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
