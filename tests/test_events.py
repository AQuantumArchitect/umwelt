"""Contract tests for the vendored event + replay module (the P0 floor)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from umwelt.events import Event, replay_sensor_batches


def _rows(t0: datetime, spec: list[tuple[float, str, float, str | None]]):
    """(offset_s, signal_id, value, metadata_json) -> replay rows."""
    return [
        ((t0 + timedelta(seconds=dt)).isoformat(), sid, str(val), meta)
        for dt, sid, val, meta in spec
    ]


def test_event_defaults():
    e = Event(timestamp=datetime.now(timezone.utc), source_device="s1",
              event_type="reading", value=0.5)
    assert e.metadata == {} and e.synthetic is False


def test_replay_buckets_by_flush_window_last_wins():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = _rows(t0, [
        (0.0, "a", 1.0, None),
        (0.5, "a", 0.25, None),       # same bucket: last wins
        (1.0, "b", 2.0, None),
        (5.0, "a", 9.0, None),        # >= flush_secs later: new bucket
    ])
    batches = list(replay_sensor_batches(rows, flush_secs=2.0))
    assert len(batches) == 2
    (bt1, readings1, conf1, _), (bt2, readings2, conf2, _) = batches
    assert bt1 == t0 and readings1 == {"a": 0.25, "b": 2.0} and conf1 is None
    assert readings2 == {"a": 9.0}


def test_replay_carries_forecast_confidence_not_ground_truth():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = _rows(t0, [
        (0.0, "sensor_x", 1.0, None),
        (0.2, "forecast_x", 0.7, '{"bridge": "forecast", "confidence": 0.42}'),
    ])
    [(_, readings, conf, _)] = list(replay_sensor_batches(rows, flush_secs=2.0))
    assert readings == {"sensor_x": 1.0, "forecast_x": 0.7}
    assert conf == {"forecast_x": 0.42}   # forecasts replay at recorded confidence only


def test_replay_skips_unparseable_and_pre_floor_rows():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = _rows(t0, [(0.0, "old", 1.0, None), (10.0, "new", 2.0, None)])
    rows.insert(0, ("not-a-timestamp", "junk", "1.0", None))
    rows.append(((t0 + timedelta(seconds=11)).isoformat(), "bad", "not-a-float", None))
    batches = list(replay_sensor_batches(
        rows, flush_secs=2.0, floor_ts=(t0 + timedelta(seconds=5)).timestamp()))
    assert len(batches) == 1 and batches[0][1] == {"new": 2.0}
