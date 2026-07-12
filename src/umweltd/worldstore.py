"""The world directory — one world's durable state on disk.

    <home>/worlds/<name>/
        world.json      the world's manifest: spec ref, vocabulary ref, knobs
        events.db       the write-ahead log (umwelt.events schema, appended forever)
        snapshot.pkl    the engine state at the last snapshot (canonical save)
        cursor.txt      the last event timestamp INCLUDED in snapshot.pkl
        worker.port     the live worker's TCP port (written at bind, removed at exit)

Recovery contract: boot = load snapshot.pkl (if any) + replay events.db rows with
timestamp > cursor.txt through the production ingest path. The cursor is written only
at snapshot time, never on ingest — the log is the truth, the snapshot is a cache.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

DDL = """CREATE TABLE IF NOT EXISTS events (
    timestamp TEXT NOT NULL,
    source_device TEXT NOT NULL,
    value TEXT NOT NULL,
    metadata TEXT
)"""


@dataclass
class WorldDir:
    root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / "world.json"

    @property
    def events_db(self) -> Path:
        return self.root / "events.db"

    @property
    def snapshot_path(self) -> Path:
        return self.root / "snapshot.pkl"

    @property
    def cursor_path(self) -> Path:
        return self.root / "cursor.txt"

    @property
    def port_path(self) -> Path:
        return self.root / "worker.port"

    def manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text())

    def write_manifest(self, manifest: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))

    def append_events(self, rows: list[tuple]) -> int:
        """Append (ts_iso, sensor_id, value_str, metadata_json|None) rows to the log.
        The append happens BEFORE ingest — the log is the write-ahead truth."""
        con = sqlite3.connect(str(self.events_db))
        try:
            con.execute(DDL)
            con.executemany("INSERT INTO events VALUES (?,?,?,?)", rows)
            con.commit()
            return len(rows)
        finally:
            con.close()

    def cursor(self) -> str:
        return self.cursor_path.read_text().strip() if self.cursor_path.exists() else ""

    def write_cursor(self, ts_iso: str) -> None:
        self.cursor_path.write_text(ts_iso)
