"""Stream Tape — a bounded, gauge-pruned per-stream VALUE history (the datastream fiber's history layer).

The gauge spine (datastream_health / sensor_health / sound_fiber) answers "is this stream healthy *now*?" from
the live field — current-state only. A LEARNER needs the other thing: each stream's recent VALUE TRAJECTORY, to
compute lagged co-occurrence / contrast. Today the only source of trajectories is the raw 8.3 GB
`meerkat_events.db` — an unbounded firehose that scans for >100s per pull on the A55. That raw store was
scaffolding; this is the fiber layer that subsumes it as the learner's source.

The design mirrors surprise_tape.py (the proven low-pressure pattern):
    • a bounded in-memory ring per stream, fed cheaply from the ingest path (no extra reads);
    • a batched WAL flush to a COMPACT `meerkat_streams.db` — so an OFFLINE learner (a separate process) reads
      a small indexed table in ~ms instead of scanning the firehose;
    • CONTRIBUTION-GAUGED pruning: each stream's retention window scales with its contribution (informative,
      live streams keep a dense recent history; saturated/constant or stale streams shrink toward nothing).
      The gauge IS the retention policy — algorithmic pruning, not a fixed cap.

A stream is identified by `stream_id = "{event_type}\x1f{source_device}"` — the same (event_type, device) the
learner resolves a role to, so the read path is a drop-in for the raw pull, just bounded and fast.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from collections import deque
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SEP = "\x1f"


def stream_id(event_type: str, device: str | None) -> str:
    """Canonical stream identity — matches how a learner resolves a role to (event_type, device)."""
    return f"{event_type}{SEP}{device or ''}"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS stream_samples (
  stream_id TEXT NOT NULL,
  t         REAL NOT NULL,   -- epoch seconds (no ISO/JSON parse on read — the A55 win)
  value     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_stream_t ON stream_samples(stream_id, t);
"""


class StreamTape:
    """Per-stream bounded value history with batched SQLite persistence + contribution-gauged pruning."""

    def __init__(self, db_path: str | Path, *, recent_per_stream: int = 4000, flush_interval: float = 300.0,
                 base_retention_s: float = 14 * 86400.0, prune_interval: float = 6 * 3600.0):
        self.db_path = Path(db_path)
        self.recent_per_stream = recent_per_stream
        self.flush_interval = flush_interval
        self.base_retention_s = base_retention_s
        self.prune_interval = prune_interval
        self._rings: dict[str, deque] = {}          # stream_id -> deque[(t, value)], live this session
        self._pending: list[tuple] = []             # (stream_id, t, value) awaiting flush
        self._seen: set[str] = set()
        self._lock = asyncio.Lock()
        self._total_written = 0
        self._last_prune = time.time()
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    # ── hot path ────────────────────────────────────────────────────────────────────────────────────────
    def observe(self, event_type: str, device: str | None, value, t: float | None = None) -> None:
        """Record one sample. Cheap + non-blocking (ring + pending); persistence is deferred to flush()."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(v):
            return
        sid = stream_id(event_type, device)
        tt = float(t) if t is not None else time.time()
        ring = self._rings.get(sid)
        if ring is None:
            ring = self._rings[sid] = deque(maxlen=self.recent_per_stream)
            self._seen.add(sid)
        ring.append((tt, v))
        self._pending.append((sid, tt, v))

    # ── persistence ─────────────────────────────────────────────────────────────────────────────────────
    async def flush(self) -> int:
        async with self._lock:
            if not self._pending:
                return 0
            batch = self._pending
            self._pending = []

        def _do_write() -> int:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA busy_timeout=5000")
                conn.executemany("INSERT INTO stream_samples (stream_id, t, value) VALUES (?, ?, ?)", batch)
                conn.commit()
            finally:
                conn.close()
            return len(batch)

        n = await asyncio.to_thread(_do_write)
        self._total_written += n
        return n

    async def flush_loop(self) -> None:
        logger.info("StreamTape flush loop started (every %.0fs → %s)", self.flush_interval, self.db_path)
        while True:
            await asyncio.sleep(self.flush_interval)
            try:
                await self.flush()
                if time.time() - self._last_prune >= self.prune_interval:
                    await asyncio.to_thread(self.prune)
                    self._last_prune = time.time()
            except Exception:
                logger.exception("StreamTape flush/prune failed")

    def prune(self, contribution: dict[str, float] | None = None, *, floor: float = 0.05,
              now: float | None = None) -> dict:
        """CONTRIBUTION-GAUGED pruning — the retention policy IS the gauge. Each stream's retention window =
        base_retention_s × contribution(stream); streams below `floor` are dropped entirely. When no external
        contribution is supplied, a self-contained signal is used: informativeness × liveness, where
        informativeness = (a recently-varying stream carries info; a constant/saturated one ~0) and liveness =
        (still arriving). So saturated always-asserted / dead streams shrink toward nothing on their own,
        while a richer field-gauge contribution (purity·surprise·Δinfo) can be passed in to refine it."""
        now = now or time.time()
        contribution = contribution or self._self_contribution(now)
        conn = sqlite3.connect(self.db_path)
        dropped, shrunk = 0, 0
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            sids = [r[0] for r in conn.execute("SELECT DISTINCT stream_id FROM stream_samples")]
            for sid in sids:
                c = float(contribution.get(sid, 1.0))
                if c < floor:
                    conn.execute("DELETE FROM stream_samples WHERE stream_id=?", (sid,))
                    dropped += 1
                else:
                    keep_s = self.base_retention_s * min(max(c, floor), 1.0)
                    conn.execute("DELETE FROM stream_samples WHERE stream_id=? AND t < ?", (sid, now - keep_s))
                    shrunk += 1
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
        logger.info("StreamTape prune: %d streams dropped (<%.2f), %d retention-bounded", dropped, floor, shrunk)
        return {"dropped": dropped, "kept": shrunk}

    def _self_contribution(self, now: float) -> dict[str, float]:
        """Self-contained per-stream contribution in [0,1] from the live rings: informativeness (recent value
        spread, saturating) × liveness (decays with staleness). A constant or long-stale stream → ~0."""
        out: dict[str, float] = {}
        for sid, ring in self._rings.items():
            if not ring:
                out[sid] = 0.0
                continue
            vals = np.fromiter((v for _t, v in ring), float)
            spread = float(np.std(vals)) if vals.size > 2 else 0.0
            info = spread / (spread + 1e-3)                          # 0 (constant) → ~1 (varying)
            age = now - ring[-1][0]
            live = float(np.exp(-age / (3 * 86400.0)))               # ~1 fresh → decays over days
            out[sid] = info * live
        return out

    # ── read path (the learner's source — replaces a raw events.db pull) ─────────────────────────────────
    def recent(self, event_type: str, device: str | None, since: float | None = None,
               limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Bounded (ts, values) for one stream from the compact db — the drop-in for coupling_learn.pull_stream,
        but ~ms (small indexed table, epoch compare, float values — no firehose scan, no ISO/JSON parse)."""
        return pull(self.db_path, stream_id(event_type, device), since=since, limit=limit)

    def stats(self) -> dict:
        return {"db_path": str(self.db_path), "streams": len(self._seen),
                "pending": len(self._pending), "total_written": self._total_written}

    def last_seen_by_device(self) -> dict[str, float]:
        """{source_device: last sample epoch} across all streams — the restart-durable
        last-seen the health spine hydrates from at boot (b9.52). Index-assisted (ms)."""
        try:
            con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0)
        except sqlite3.Error:
            return {}
        try:
            rows = con.execute("SELECT stream_id, MAX(t) FROM stream_samples GROUP BY stream_id").fetchall()
        except sqlite3.Error:
            return {}
        finally:
            con.close()
        out: dict[str, float] = {}
        for sid, t in rows:
            device = sid.split(SEP, 1)[1] if SEP in sid else sid
            if device and t is not None:
                out[device] = max(out.get(device, 0.0), float(t))
        return out


def pull(db_path: str | Path, sid: str, *, since: float | None = None,
         limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Open the compact streams db read-only and return (epoch_ts, values) for one stream, ascending in time."""
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return np.empty(0), np.empty(0)
    try:
        clauses, params = ["stream_id = ?"], [sid]
        if since is not None:
            clauses.append("t >= ?")
            params.append(float(since))
        where = " AND ".join(clauses)
        if limit:
            sql = (f"SELECT t, value FROM (SELECT t, value FROM stream_samples WHERE {where} "
                   f"ORDER BY t DESC LIMIT {int(limit)}) ORDER BY t")
        else:
            sql = f"SELECT t, value FROM stream_samples WHERE {where} ORDER BY t"
        rows = con.execute(sql, params).fetchall()
    except sqlite3.Error:
        return np.empty(0), np.empty(0)
    finally:
        con.close()
    if not rows:
        return np.empty(0), np.empty(0)
    arr = np.asarray(rows, float)
    return arr[:, 0], arr[:, 1]
