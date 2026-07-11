"""The engine's one event type + sync event-backlog replay.

`Event` is the neutral wire format every ingress adapter produces: a timestamped scalar
(or payload) from a named source. The replay half reads a logged-events SQLite store and
buckets rows back into the same `{signal_id: float}` batches a live ingest flush produces,
so an offline learner replays the backlog exactly the way the deployment lived it. One
home for "logged events → ingestable batches" instead of a hand-rolled sqlite query +
bucketing loop per caller.

The replay contract preserves per-event confidence: a forecast event recorded with
`bridge:"forecast"` + `confidence` metadata replays at that SAME confidence rather than as
a full-confidence signal — replaying a forecast as ground truth would be a gauge mismatch
between training and deployment.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator


@dataclass
class Event:
    timestamp: datetime
    source_device: str
    event_type: str
    value: Any
    metadata: dict = field(default_factory=dict)
    synthetic: bool = False  # True = derived/computed, not raw hardware


def read_events_since(db_path, since: str, until: str | None = None,
                      limit: int | None = None) -> list[tuple[str, str, str, str]]:
    """Rows `(timestamp_iso, source_device, value, metadata_json)` with timestamp >
    `since` (empty string → from the beginning), ordered by time. `until`/`limit`
    optional. metadata is included so the replay can reconstruct per-event confidence
    (see the module docstring's forecast-replay contract)."""
    query = "SELECT timestamp, source_device, value, metadata FROM events WHERE timestamp > ?"
    params: list = [since or ""]
    if until is not None:
        query += " AND timestamp <= ?"
        params.append(until)
    query += " ORDER BY timestamp"
    if limit:
        query += f" LIMIT {int(limit)}"
    con = sqlite3.connect(str(db_path))
    try:
        return con.execute(query, params).fetchall()
    finally:
        con.close()


def latest_event_ts(db_path) -> str | None:
    """ISO timestamp of the most recent logged event, or None if the store is empty."""
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute("SELECT MAX(timestamp) FROM events").fetchone()
        return row[0] if row and row[0] else None
    except sqlite3.Error:
        return None
    finally:
        con.close()


def replay_lag_seconds(db_path, cursor_ts: str) -> float:
    """How far a replay consumer is BEHIND: wall-seconds between the most recent logged
    event and the replay cursor (the last event it digested). 0 when caught up (cursor ==
    latest) or the store is empty — nothing to catch up. This is the backlog depth an
    offline learner reads to coarsen its replay geometry, and a live engine reads to
    throttle sampling during lulls. Raw seconds."""
    latest = latest_event_ts(db_path)
    if not latest or not cursor_ts:
        return 0.0
    try:
        return max(0.0, (datetime.fromisoformat(latest)
                         - datetime.fromisoformat(cursor_ts)).total_seconds())
    except (ValueError, TypeError):
        return 0.0


def _forecast_confidence(meta_json) -> float | None:
    """Extract a forecast event's confidence from its metadata json, or None if the
    event is an ordinary signal (→ full confidence downstream, the contract default)."""
    if not meta_json:
        return None
    try:
        meta = json.loads(meta_json) if isinstance(meta_json, (str, bytes)) else meta_json
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if isinstance(meta, dict) and meta.get("bridge") == "forecast" and "confidence" in meta:
        try:
            return float(meta["confidence"])
        except (TypeError, ValueError):
            return None
    return None


def replay_sensor_batches(
    rows, flush_secs: float = 2.0, floor_ts: float | None = None,
) -> Iterator[tuple[datetime, dict[str, float], dict[str, float] | None, str]]:
    """Bucket `(ts_iso, source_device, value[, metadata_json])` rows into signal-reading
    batches: consecutive events within `flush_secs` of the batch start share one batch
    (last value wins per signal), mirroring a live flush. Rows older than `floor_ts`
    (epoch seconds) and non-float values are skipped. Yields `(batch_time, {sid: float},
    confidence | None, last_row_ts_iso)` where `confidence` carries the forecast events'
    recorded confidence (None when the batch has no forecasts) — feed batch_time +
    readings + confidence straight to `engine.ingest` so a replay consumes forecasts at
    the SAME confidence the live engine consumed them at. Rows may be 3-tuples (legacy,
    no metadata) — then confidence is always None."""
    batch: dict[str, float] = {}
    conf: dict[str, float] = {}
    batch_t: datetime | None = None
    batch_last_ts: str = ""
    for row in rows:
        ts, sid, val = row[0], row[1], row[2]
        meta_json = row[3] if len(row) > 3 else None
        try:
            t = datetime.fromisoformat(ts)
        except Exception:
            continue
        if floor_ts is not None and t.timestamp() < floor_ts:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if batch_t is not None and (t - batch_t).total_seconds() >= flush_secs:
            yield batch_t, dict(batch), (dict(conf) or None), batch_last_ts
            batch = {}
            conf = {}
            batch_t = t
        if batch_t is None:
            batch_t = t
        batch[sid] = fval  # last-wins within a bucket
        c = _forecast_confidence(meta_json)
        if c is not None:
            conf[sid] = c
        batch_last_ts = ts
    if batch and batch_t is not None:
        yield batch_t, dict(batch), (dict(conf) or None), batch_last_ts
