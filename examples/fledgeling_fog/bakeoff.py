#!/usr/bin/env python3
"""Belief vs freeze baseline on held-out 'is the agent near?' labels.

Public synthetic only. Prints MAE for engine-on vs persistence-of-last-input.
Does not invent a win: if freeze wins, metrics still print honestly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from examples.fledgeling_fog.world import (
    FOG_SPEC,
    N_PLACES,
    agent_walk,
    freeze_baseline_scores,
    occupied_place,
    place_names,
    runner_batches,
    sparse_scout_reports,
    synthesize_rows,
    binned_truth,
)
from umwelt.host import GameHost


def engine_scores(segments, places, *, seed: int = 11) -> np.ndarray:
    host = GameHost()
    host.register_world(FOG_SPEC, population=False)
    rows = synthesize_rows(FOG_SPEC, segments, seed=seed)
    # Collect belief snapshots aligned to bins
    t0, t_end = segments[0][0], segments[-1][1]
    bin_s = 60.0
    T = int((t_end - t0).total_seconds() // bin_s)
    scores = np.full((T, len(places)), 0.5)
    idx = {p: k for k, p in enumerate(places)}
    last_bin = -1
    for readings, batch_t, conf in runner_batches(rows, flush_secs=30.0):
        host.observe_many("scout", readings, confidence=conf, t=batch_t)
        b = int((batch_t - t0).total_seconds() // bin_s)
        if b < 0 or b >= T:
            continue
        if b != last_bin:
            beliefs = host.beliefs("scout")
            for p in places:
                key = f"{p}.agent_near"
                if key in beliefs:
                    scores[b, idx[p]] = beliefs[key].value
            last_bin = b
    # forward-fill empty bins
    last = scores[0].copy()
    for t in range(T):
        if np.allclose(scores[t], 0.5) and t > 0:
            scores[t] = last
        else:
            last = scores[t].copy()
    return scores


def mae_near(scores: np.ndarray, truth: np.ndarray) -> float:
    """MAE of score vs binary near (truth +1 → 1.0, -1 → 0.0)."""
    target = (truth + 1.0) / 2.0
    return float(np.mean(np.abs(scores - target)))


def main() -> None:
    places = place_names(N_PLACES)
    segments = agent_walk(seed=11, ticks=200)
    truth = binned_truth(segments, places, bin_s=60.0)
    reports = sparse_scout_reports(truth, report_p=0.35, seed=3)
    freeze = freeze_baseline_scores(truth, reports)
    # Map freeze ±1 to [0,1]
    freeze01 = (freeze + 1.0) / 2.0
    eng = engine_scores(segments, places, seed=11)

    mae_e = mae_near(eng, truth)
    mae_f = mae_near(freeze01, truth)
    # Accuracy: argmax place == truth place
    pred_e = eng.argmax(axis=1)
    pred_f = freeze01.argmax(axis=1)
    true_i = truth.argmax(axis=1)
    acc_e = float((pred_e == true_i).mean())
    acc_f = float((pred_f == true_i).mean())

    print("=== Fog corridor bake-off (public synthetic) ===")
    print(f"bins={truth.shape[0]} places={len(places)}")
    print(f"engine MAE={mae_e:.4f}  acc={acc_e:.4f}")
    print(f"freeze MAE={mae_f:.4f}  acc={acc_f:.4f}")
    if acc_e > acc_f or mae_e < mae_f:
        print("RESULT: engine-on beats freeze baseline")
    else:
        print("RESULT: engine-on does NOT beat freeze (honest failure — do not paper over)")
    # also score held-out last 25% for honesty
    cut = int(truth.shape[0] * 0.75)
    mae_e_h = mae_near(eng[cut:], truth[cut:])
    mae_f_h = mae_near(freeze01[cut:], truth[cut:])
    print(f"held-out last 25%: engine MAE={mae_e_h:.4f} freeze MAE={mae_f_h:.4f}")
    return 0 if (acc_e > acc_f or mae_e < mae_f) else 1


if __name__ == "__main__":
    raise SystemExit(main())
