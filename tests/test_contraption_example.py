"""The contraption: bots collapse beliefs → surprise → forecast, tested falsifiably.

Two pins:
  (1) the world cures the persistence mirage — the beliefs genuinely MOVE (collapse), so purity is
      not frozen at 0.5 and the forecast has something real to anticipate;
  (2) on HELD-OUT bot runs, collapse-surprise forecasting BEATS both persistence and clock-only on
      the decisive single-parent-dependent leaf (thing_2) and the periodic control (thing_1) — while
      the conjunctive AND-chained thing_3 is NOT beaten (a pairwise state×surprise interaction cannot
      encode a two-parent AND; the documented higher-order limit).

The margins pinned are generous (observed thing_2 svp≈+0.32, thing_1≈+0.26) so the test tracks the
real effect, not noise. If this ever flips (surprise stops beating persistence on thing_2), the
CLAIMS row for the surprise→forecast wire is falsified.
"""
import numpy as np

from examples.contraption.world import CONTRAPTION_SPEC, THINGS, bot_collapse_stream
from proofs.contraption_walk import run
from umwelt.boot import build_engine


def test_beliefs_move_no_persistence_mirage():
    """The cure for the frozen-belief mirage: driven by collapses, each thing swings the full ±1
    and holds high purity — never the purity≈0.5 / z≈0 frozen point that inflates absolute skill."""
    eng = build_engine(spec=CONTRAPTION_SPEC, population=False, calibration=False, fractal=False)
    assert eng.forecast_surface is not None
    assert eng.forecast_surface.n_context == len(THINGS)      # surprise-context channel is on
    z = {t: [] for t in THINGS}
    p = {t: [] for t in THINGS}
    for now, readings in bot_collapse_stream(seed=1, ticks=400):
        eng.ingest(sensor_readings=readings, now=now)
        for t in THINGS:
            c = eng.field.clusters[t]
            idx = c.role_index["state"]
            z[t].append(float(c.qubit_bloch(idx)[2]))
            p[t].append(float(c.purity))
    for t in THINGS:
        assert np.std(z[t]) > 0.3, f"{t} barely moved — mirage risk"      # genuinely moving
        assert np.median(p[t]) > 0.7, f"{t} decohered to mixed"           # not the frozen point


def test_collapse_surprise_beats_persistence_and_clock_held_out():
    """The falsifiable core: on held-out seeds the surprise-fed forecast beats BOTH persistence and
    clock on the decisive dependent leaf and the periodic control; the AND-chain leaf is unbeaten."""
    summary = run(train_seeds=(1, 2, 3), test_seeds=(101,), ticks=900)
    rows = summary["rows"]

    # decisive leaf: thing_2's collapse tracks thing_1's jittered flip — only watching thing_1 helps.
    t2 = rows["thing_2"]
    assert t2["surprise_beats_clock"], t2
    assert t2["skill_vs_persist_surprise"] > 0.10, t2          # observed ≈ +0.32

    # periodic control: thing_1 also beats both baselines.
    t1 = rows["thing_1"]
    assert t1["surprise_beats_clock"], t1
    assert t1["skill_vs_persist_surprise"] > 0.10, t1          # observed ≈ +0.26

    # documented higher-order limit: the conjunctive AND-chained thing_3 does NOT beat persistence
    # (a pairwise state×surprise interaction cannot encode a two-parent AND).
    t3 = rows["thing_3"]
    assert t3["skill_vs_persist_surprise"] < 0.0, t3

    assert summary["decisive_win"] and summary["control_win"]
    assert summary["verdict"].startswith("WIN")
