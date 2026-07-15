"""Public synthetic fog cassette + baseline harness."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from examples.fledgeling_fog.world import (
    FOG_SPEC,
    agent_walk,
    binned_truth,
    freeze_baseline_scores,
    place_names,
    runner_batches,
    sparse_scout_reports,
    synthesize_rows,
)
from umwelt.host import GameHost

FOG_KIT_SPEC = FOG_SPEC
HONESTY_TIER = "synthetic CI cassette — scout partial observation"


@dataclass
class BaselineReport:
    kit: str
    engine_mae: float
    freeze_mae: float
    engine_acc: float
    freeze_acc: float
    beats_baseline: bool
    honesty: str

    def summary(self) -> str:
        verdict = "BEATS" if self.beats_baseline else "DOES_NOT_BEAT"
        return (
            f"[{self.kit}] engine MAE={self.engine_mae:.4f} acc={self.engine_acc:.4f} | "
            f"freeze MAE={self.freeze_mae:.4f} acc={self.freeze_acc:.4f} → {verdict} "
            f"({self.honesty})"
        )


def run_fog_baseline(*, seed: int = 11, ticks: int = 160) -> BaselineReport:
    places = place_names()
    segments = agent_walk(seed=seed, ticks=ticks)
    truth = binned_truth(segments, places, bin_s=60.0)
    reports = sparse_scout_reports(truth, report_p=0.35, seed=seed)
    freeze = (freeze_baseline_scores(truth, reports) + 1.0) / 2.0

    host = GameHost()
    host.register_world(FOG_KIT_SPEC, population=False)
    rows = synthesize_rows(FOG_KIT_SPEC, segments, seed=seed)
    t0 = segments[0][0]
    T, n = truth.shape
    eng = np.full((T, n), 0.5)
    idx = {p: k for k, p in enumerate(places)}
    last_b = -1
    for readings, bt, conf in runner_batches(rows, flush_secs=30.0):
        host.observe_many("scout", readings, confidence=conf, t=bt)
        b = int((bt - t0).total_seconds() // 60.0)
        if 0 <= b < T and b != last_b:
            beliefs = host.beliefs("scout")
            for p in places:
                key = f"{p}.agent_near"
                if key in beliefs:
                    eng[b, idx[p]] = beliefs[key].value
            last_b = b
    last = eng[0].copy()
    for t in range(T):
        if np.allclose(eng[t], 0.5) and t > 0:
            eng[t] = last
        else:
            last = eng[t].copy()

    target = (truth + 1.0) / 2.0
    mae_e = float(np.mean(np.abs(eng - target)))
    mae_f = float(np.mean(np.abs(freeze - target)))
    true_i = truth.argmax(axis=1)
    acc_e = float((eng.argmax(axis=1) == true_i).mean())
    acc_f = float((freeze.argmax(axis=1) == true_i).mean())
    beats = acc_e > acc_f or mae_e < mae_f
    return BaselineReport(
        kit="fog",
        engine_mae=mae_e,
        freeze_mae=mae_f,
        engine_acc=acc_e,
        freeze_acc=acc_f,
        beats_baseline=beats,
        honesty=HONESTY_TIER,
    )
