"""
Surprise Tape — phase-indexed record of "when the model's beliefs got broken."

The quantum reservoir is unitary: one field, one set of beliefs. But several
*learners* read it from different angles — each fractal scale, each calibration
channel, the training runner. Every one of them computes its own surprise
signal as an EMA that lives only in RAM, never durably recorded.

This module records those surprise events on a persistent timeline keyed by
Berry phase (so the tape reflects where in learning-space each event
happened, not just when on the wall clock).

Two composable tricks keep write pressure negligible on the RDK:

    1. Shannon-rate-matched gating
       ───────────────────────────
       Each new surprise value s is recorded with probability

           p = 1 − exp(−|s| / τ)

       where τ is a per-source EMA of past surprise magnitudes. This is the
       information-theoretic sampling probability: in expectation the write
       rate equals the entropy rate of the signal. Calm periods produce
       almost no writes; novel moments produce guaranteed writes.

    2. Weighted reservoir sampling (Efraimidis-Spirakis 2006)
       ────────────────────────────────────────────────────
       A bounded in-memory pool of K "most-surprising events ever" per
       source. Each observation gets a key `u ** (1/w)` with u ~ U(0,1) and
       w = |surprise|. Keep top-K by key. The pool is a mathematically exact
       weighted sample of all history — zero write amplification beyond K.

Persistence uses one batched SQLite transaction per flush interval
(default 5 min), WAL mode, synchronous=NORMAL. One fsync per 5 min is
effectively zero NAND pressure.
"""
from __future__ import annotations

import asyncio
import heapq
import json
import logging
import math
import random
import sqlite3
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ================================================================
# Stamp — one surprise event
# ================================================================

def stamp_category(source: str, metadata: dict | None) -> str:
    """Classify a stamp as 'grounded' (the world model learning) or
    'substrate' (the compute system learning).

    Prefers an explicit metadata["category"]; falls back to the source-name
    convention (fractal_scale_* / training are substrate, everything else is
    grounded) so old or untagged stamps still classify the same way the UI does.
    """
    cat = (metadata or {}).get("category")
    if cat in ("grounded", "substrate"):
        return cat
    return "substrate" if (source.startswith("fractal_scale_") or source == "training") else "grounded"


@dataclass(frozen=True)
class SurpriseStamp:
    wall_clock: float
    berry_phase: float
    source: str
    surprise: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> tuple:
        return (
            self.wall_clock,
            self.berry_phase,
            self.source,
            self.surprise,
            json.dumps(self.metadata) if self.metadata else None,
        )


# ================================================================
# Weighted reservoir — Efraimidis-Spirakis 2006
# ================================================================

class WeightedReservoir:
    """Bounded sample of the top-K most-surprising events, exact by weight.

    Uses a min-heap of (key, counter, stamp) where key = u**(1/w). Keeping
    the top-K largest keys == a weighted sample without replacement with
    inclusion probability proportional to weight.
    """

    def __init__(self, k: int = 10_000):
        self.k = k
        # heap of (key, tie_breaker, stamp) — min-heap, so .heap[0] is the
        # smallest key = the one to evict when a larger one shows up.
        self.heap: list[tuple[float, int, SurpriseStamp]] = []
        self._counter = 0  # tie-breaker for heap stability

    def offer(self, stamp: SurpriseStamp, weight: float) -> None:
        if weight <= 0.0 or not math.isfinite(weight):
            return
        u = random.random()
        if u <= 0.0:
            u = 1e-18
        key = u ** (1.0 / weight)
        self._counter += 1
        entry = (key, self._counter, stamp)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, entry)
        elif key > self.heap[0][0]:
            heapq.heapreplace(self.heap, entry)

    def __len__(self) -> int:
        return len(self.heap)

    def dump(self) -> list[SurpriseStamp]:
        """Current reservoir contents, highest-weight first."""
        return [s for _, _, s in sorted(self.heap, key=lambda e: -e[0])]


# ================================================================
# Per-source gate
# ================================================================

class SourceGate:
    """Information-theoretic gate + weighted reservoir, per source.

    Bootstrap: first BOOTSTRAP_N observations are all accepted and buffered
    to seed τ = median(|surprise|). τ is FROZEN after bootstrap so the gate
    stays calibrated against the source's baseline noise, not its current
    value (otherwise a steady low signal would keep being recorded after
    τ chases it downward).
    """

    BOOTSTRAP_N = 100
    MIN_TAU = 1e-6

    def __init__(self, source: str, reservoir_size: int = 10_000):
        self.source = source
        self.tau = 0.0
        self.reservoir = WeightedReservoir(k=reservoir_size)
        self._bootstrap: list[float] = []
        self._bootstrapped = False
        self.observed = 0
        self.kept = 0

    def _maybe_finalize_bootstrap(self) -> None:
        if self._bootstrapped:
            return
        if len(self._bootstrap) < self.BOOTSTRAP_N:
            return
        mag = [abs(s) for s in self._bootstrap]
        self.tau = max(statistics.median(mag), self.MIN_TAU)
        self._bootstrap = []
        self._bootstrapped = True
        logger.info(
            "SurpriseTape source=%s bootstrapped: τ=%.6f from %d obs",
            self.source, self.tau, self.BOOTSTRAP_N,
        )

    def observe(self, stamp: SurpriseStamp) -> bool:
        """Return True if stamp was kept (passed gate)."""
        self.observed += 1
        s = abs(stamp.surprise)

        if not self._bootstrapped:
            self._bootstrap.append(stamp.surprise)
            self._maybe_finalize_bootstrap()
            self.reservoir.offer(stamp, weight=max(s, self.MIN_TAU))
            self.kept += 1
            return True

        # τ is frozen post-bootstrap — p is a pure function of |s| / τ_baseline.
        tau_eff = max(self.tau, self.MIN_TAU)
        p = 1.0 - math.exp(-s / tau_eff)
        if random.random() < p:
            self.reservoir.offer(stamp, weight=max(s, self.MIN_TAU))
            self.kept += 1
            return True
        return False

    def stats(self) -> dict:
        return {
            "source": self.source,
            "tau": round(self.tau, 6),
            "bootstrapped": self._bootstrapped,
            "observed": self.observed,
            "kept": self.kept,
            "keep_rate": round(self.kept / max(self.observed, 1), 4),
            "reservoir_size": len(self.reservoir),
        }


# ================================================================
# Surprise tape — registry + persistence
# ================================================================

_SCHEMA = """
CREATE TABLE IF NOT EXISTS surprise_stamps (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  wall_clock  REAL    NOT NULL,
  berry_phase REAL    NOT NULL,
  source      TEXT    NOT NULL,
  surprise    REAL    NOT NULL,
  metadata    TEXT
);
CREATE INDEX IF NOT EXISTS ix_stamps_source_time ON surprise_stamps(source, wall_clock);
CREATE INDEX IF NOT EXISTS ix_stamps_phase       ON surprise_stamps(berry_phase);
"""


class SurpriseTape:
    """Per-source gated recording with batched SQLite persistence."""

    def __init__(
        self,
        db_path: str | Path,
        reservoir_size: int = 10_000,
        flush_interval: float = 300.0,
        recent_size: int = 500,
    ):
        self.db_path = Path(db_path)
        self.reservoir_size = reservoir_size
        self.flush_interval = flush_interval
        self.gates: dict[str, SourceGate] = {}
        self._pending: list[SurpriseStamp] = []
        # Time-ordered ring of stamps kept *this session*. The weighted
        # reservoir is "most surprising ever" and SQLite is 5-min-stale, so
        # neither reflects "what just happened." This buffer does — it backs
        # the live /api/surprise/recent feed without waiting for a flush.
        self._recent: deque[SurpriseStamp] = deque(maxlen=recent_size)
        self.process_start = time.time()
        self._total_written = 0
        self._lock = asyncio.Lock()
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")  # retry on lock, don't drop (b9.7.3 hardening)
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _gate(self, source: str) -> SourceGate:
        g = self.gates.get(source)
        if g is None:
            g = SourceGate(source, reservoir_size=self.reservoir_size)
            self.gates[source] = g
        return g

    def observe(
        self,
        source: str,
        surprise: float,
        berry_phase: float,
        metadata: dict | None = None,
        wall_clock: float | None = None,
    ) -> bool:
        """Record a surprise observation; returns True if it passed the gate.

        Hot path — called per step from inside the quantum loop. Must be
        non-blocking and cheap: no I/O, no waits. Persistence is deferred
        to flush_loop().
        """
        if not math.isfinite(surprise) or abs(surprise) < 1e-10:
            return False
        stamp = SurpriseStamp(
            wall_clock=wall_clock if wall_clock is not None else time.time(),
            berry_phase=float(berry_phase),
            source=source,
            surprise=float(surprise),
            metadata=metadata or {},
        )
        gate = self._gate(source)
        if gate.observe(stamp):
            self._pending.append(stamp)
            self._recent.append(stamp)
            return True
        return False

    async def flush(self) -> int:
        """Write pending stamps to SQLite in a single transaction."""
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
                conn.executemany(
                    "INSERT INTO surprise_stamps "
                    "(wall_clock, berry_phase, source, surprise, metadata) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [s.to_row() for s in batch],
                )
                conn.commit()
            finally:
                conn.close()
            return len(batch)

        n = await asyncio.to_thread(_do_write)
        self._total_written += n
        logger.info("SurpriseTape flushed %d stamps (total=%d)", n, self._total_written)
        return n

    async def flush_loop(self) -> None:
        """Periodic flush task for asyncio.gather."""
        logger.info(
            "SurpriseTape flush loop started (every %.0fs → %s)",
            self.flush_interval, self.db_path,
        )
        while True:
            await asyncio.sleep(self.flush_interval)
            try:
                await self.flush()
            except Exception:
                logger.exception("SurpriseTape flush failed")

    # ────────────────────────────────────────────────────────────
    # Read API — for /api/surprise/* endpoints
    # ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        live_grounded = sum(
            1 for s in self._recent
            if stamp_category(s.source, s.metadata) == "grounded"
        )
        return {
            "db_path": str(self.db_path),
            "pending": len(self._pending),
            "total_written": self._total_written,
            "process_start": self.process_start,
            "recent_live": len(self._recent),
            "live_grounded": live_grounded,
            "live_substrate": len(self._recent) - live_grounded,
            "sources": [g.stats() for g in self.gates.values()],
        }

    def reservoir_dump(self, source: str) -> list[SurpriseStamp]:
        g = self.gates.get(source)
        return g.reservoir.dump() if g else []

    def recent_live(
        self,
        source: str | None = None,
        limit: int = 100,
        category: str | None = None,
    ) -> list[dict]:
        """Most-recent stamps kept *this session*, newest first.

        Reads the in-memory ring — reflects what the running process just
        experienced, with no flush latency and no pre-restart history. This
        is what the live dashboard feed should show; query_recent() (SQLite)
        is for historical/archival queries.

        `category` ('grounded' | 'substrate') filters the stream so the feed
        can surface rare grounded events without substrate ticks drowning them.
        """
        stamps = reversed(self._recent)  # deque is oldest→newest
        out: list[dict] = []
        for s in stamps:
            if source and s.source != source:
                continue
            if category and stamp_category(s.source, s.metadata) != category:
                continue
            out.append({
                "wall_clock": s.wall_clock,
                "berry_phase": s.berry_phase,
                "source": s.source,
                "surprise": s.surprise,
                "metadata": s.metadata,
            })
            if len(out) >= limit:
                break
        return out

    def query_recent(
        self,
        source: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        try:
            if source:
                cur = conn.execute(
                    "SELECT wall_clock, berry_phase, source, surprise, metadata "
                    "FROM surprise_stamps WHERE source=? "
                    "ORDER BY id DESC LIMIT ?",
                    (source, limit),
                )
            else:
                cur = conn.execute(
                    "SELECT wall_clock, berry_phase, source, surprise, metadata "
                    "FROM surprise_stamps ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            rows = cur.fetchall()
        finally:
            conn.close()
        return [
            {
                "wall_clock": r[0],
                "berry_phase": r[1],
                "source": r[2],
                "surprise": r[3],
                "metadata": json.loads(r[4]) if r[4] else {},
            }
            for r in rows
        ]

    def query_range(
        self,
        start: float | datetime,
        end: float | datetime,
        source: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """Query persisted stamps in a wall-clock range."""
        if isinstance(start, datetime):
            start_ts = start.timestamp()
        else:
            start_ts = float(start)
        if isinstance(end, datetime):
            end_ts = end.timestamp()
        else:
            end_ts = float(end)

        conn = sqlite3.connect(self.db_path)
        try:
            if source:
                cur = conn.execute(
                    "SELECT wall_clock, berry_phase, source, surprise, metadata "
                    "FROM surprise_stamps WHERE source=? AND wall_clock >= ? AND wall_clock <= ? "
                    "ORDER BY wall_clock DESC LIMIT ?",
                    (source, start_ts, end_ts, limit),
                )
            else:
                cur = conn.execute(
                    "SELECT wall_clock, berry_phase, source, surprise, metadata "
                    "FROM surprise_stamps WHERE wall_clock >= ? AND wall_clock <= ? "
                    "ORDER BY wall_clock DESC LIMIT ?",
                    (start_ts, end_ts, limit),
                )
            rows = cur.fetchall()
        finally:
            conn.close()
        return [
            {
                "wall_clock": r[0],
                "berry_phase": r[1],
                "source": r[2],
                "surprise": r[3],
                "metadata": json.loads(r[4]) if r[4] else {},
            }
            for r in rows
        ]

    def query_cosurprise(
        self,
        window_phase: float = 0.05,
        window_seconds: float = 60.0,
        limit: int = 50,
    ) -> list[dict]:
        """Find phase-adjacent surprise events across different sources.

        The Hamiltonian-comparison query: moments when multiple learners
        got surprised at the same point in learning-space.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute(
                """
                SELECT a.wall_clock, a.berry_phase, a.source, a.surprise,
                       b.source, b.surprise
                FROM surprise_stamps a
                JOIN surprise_stamps b
                  ON ABS(a.berry_phase - b.berry_phase) < ?
                 AND ABS(a.wall_clock - b.wall_clock) < ?
                 AND a.source < b.source
                ORDER BY (ABS(a.surprise) + ABS(b.surprise)) DESC
                LIMIT ?
                """,
                (window_phase, window_seconds, limit),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        return [
            {
                "wall_clock": r[0],
                "berry_phase": r[1],
                "source_a": r[2],
                "surprise_a": r[3],
                "source_b": r[4],
                "surprise_b": r[5],
            }
            for r in rows
        ]
