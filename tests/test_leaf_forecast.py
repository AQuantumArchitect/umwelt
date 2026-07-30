"""LeafForecaster — the engine's native "learn by forecasting" leaf (was untested).

An online forward model over one leaf: it predicts the leaf's value one horizon ahead
and self-scores skill = 1 - normalized_error. This is the engine version of the gym's
forecast-skill. Deterministic (np.zeros init, clock features from the stored ts), so the
same stream replays bit-identically — the property the persistence-without-checkpointing
readout relies on.
"""
from datetime import datetime, timedelta

from umwelt.foresight.leaf_forecast import LeafForecaster


def _run(values, *, horizon_min=10.0, step_min=5.0):
    fc = LeafForecaster(leaf_id="a.level", sensor_id="a.level_forecast",
                        node="a", role="level", binary=False, center=0.0, scale=1.0,
                        horizon_minutes=horizon_min)
    t0 = datetime(2026, 7, 20, 9, 0)
    for i, v in enumerate(values):
        fc.step(now=t0 + timedelta(minutes=step_min * i), current_value=float(v))
    return fc.snapshot()


def test_labels_mature_after_spanning_horizon():
    # a tape spanning many horizons matures delayed labels -> the regressor updates.
    snap = _run([(i % 4) / 3.0 for i in range(60)])
    assert snap["n_updates"] > 0, snap
    assert 0.0 <= snap["skill"] <= 1.0, snap


def test_no_updates_before_first_horizon():
    # too short to span even one horizon -> no matured labels yet, skill stays 0.
    snap = _run([0.5, 0.6], horizon_min=60.0, step_min=5.0)
    assert snap["n_updates"] == 0 and snap["skill"] == 0.0, snap


def test_deterministic_replay():
    vals = [(i % 3) / 2.0 for i in range(40)]
    a = _run(vals)
    b = _run(vals)
    assert a["skill"] == b["skill"] and a["n_updates"] == b["n_updates"]
    assert a["prediction"] == b["prediction"]


def test_predictable_beats_flat_noise_on_skill():
    # a clean cyclic signal should be at least as forecastable as an erratic one;
    # both are valid skills in [0,1] and the cyclic one should not be worse.
    cyclic = _run([(i % 6) / 5.0 for i in range(72)])
    erratic = _run([((i * 37) % 7) / 6.0 for i in range(72)])
    assert cyclic["n_updates"] > 0 and erratic["n_updates"] > 0
    assert cyclic["skill"] >= erratic["skill"] - 1e-6
