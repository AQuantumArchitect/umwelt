"""Offline training backbone: a SQLite-backed corpus registry + deterministic replay
session over the engine's learning stack, with versioned, promotable artifacts.

Domain-agnostic by construction: bindings ALWAYS come from a DomainSpec (there is no
built-in signal catalog), and periodic-driver payloads come from the engine's attached
drivers (`d.target_bloch(now)`) — never from any particular sky.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses as dc
import hashlib
import json
import logging
import random
import sqlite3
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy as np

from umwelt.substrate.fractal_stack import FractalStackConfig, ScaleConfig
from umwelt.substrate.param_bundles import configure_param_bundles
from umwelt.learning.training import TrainingBurstProfile
from umwelt.learning.calibration import CalibrationConfig

if TYPE_CHECKING:  # the engine import stays lazy — see _build_smoke_engine
    from umwelt.engine import BeliefEngine

logger = logging.getLogger(__name__)

RAW_EVENT_KIND = "raw_event"
DERIVED_LABEL_KIND = "derived_label"
DEMO_WINDOW_KIND = "demo_window"
OPERATOR_FEEDBACK_KIND = "operator_feedback"
SNAPSHOT_KIND = "snapshot"


# the UTC-now helper lives in _util; local name kept for the call sites
from umwelt._util import utcnow as _utcnow  # noqa: E402


def _jsonify(value: Any) -> Any:
    # NOT _util.jsonable: this variant additionally handles dataclasses, Path and numpy
    # types (the training registry persists metric records) — visibly different.
    if dc.is_dataclass(value):
        return {k: _jsonify(v) for k, v in dc.asdict(value).items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_jsonify(value), sort_keys=True, separators=(",", ":"))


def _sha256_bytes(*chunks: bytes) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _directory_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(item.read_bytes())
    return digest.hexdigest()


@contextlib.contextmanager
def _seed_scope(seed: int | None):
    if seed is None:
        yield
        return

    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(int(seed) % (2**32 - 1))
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _apply_spec_bindings(reservoir: "BeliefEngine", spec) -> None:
    """Apply a DomainSpec's declarative bindings + ignored set onto the engine's ingress
    bridge. This is THE binding seam: in umwelt bindings always come from a spec — the
    origin deployment's literal catalog became spec data, so there is nothing built in
    to fall back to. Each BindingSpec's declarative normalizer resolves through the
    registry. Membrane-guarded — a bad spec binding must never break the bindings
    already registered."""
    if spec is None:
        return
    bridge = reservoir.sensor_bridge
    for b in (spec.bindings or ()):
        try:
            # measurement_alpha() = k·η when the binding declares a weak-measurement
            # model, else its collapse_alpha, else None → the bridge default. Bindings
            # without the measurement fields register byte-identically to before.
            _alpha = b.measurement_alpha() if hasattr(b, "measurement_alpha") else b.collapse_alpha
            bridge.register(
                b.sensor_id, zone=b.zone, qubit_role=b.role,
                normalize=b.build_normalizer(), weight=b.weight,
                event_type=(b.event_type or None),
                **({"collapse_alpha": _alpha} if _alpha is not None else {}),
                **({"force_observe": True} if b.force_observe else {}),
            )
        except Exception as exc:
            logger.warning("spec binding %s skipped: %s", b.sensor_id, exc)
    try:
        bridge.register_ignored(spec.ignored)
    except Exception:
        pass


@dataclass(frozen=True)
class TrainingRecord:
    record_id: str
    run_id: str
    ordinal: int
    kind: str
    source: str
    timestamp: datetime | None
    payload: dict[str, Any]
    parents: tuple[str, ...] = ()
    lineage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "run_id": self.run_id,
            "ordinal": self.ordinal,
            "kind": self.kind,
            "source": self.source,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "payload": _jsonify(self.payload),
            "parents": list(self.parents),
            "lineage": _jsonify(self.lineage),
        }


@dataclass(frozen=True)
class TrainingArtifact:
    artifact_id: str
    run_id: str
    kind: str
    version: int
    score: float
    baseline_score: float
    path: Path
    checksum: str
    status: str
    created_at: datetime
    promoted_at: datetime | None
    metrics: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "run_id": self.run_id,
            "kind": self.kind,
            "version": self.version,
            "score": round(self.score, 6),
            "baseline_score": round(self.baseline_score, 6),
            "path": str(self.path),
            "checksum": self.checksum,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "metrics": _jsonify(self.metrics),
            "metadata": _jsonify(self.metadata),
        }


@dataclass(frozen=True)
class TrainingReplayResult:
    run_id: str
    artifact_id: str
    artifact_path: Path
    score: float
    baseline_score: float
    promoted: bool
    replay_digest: str
    metrics: dict[str, Any]
    record_count: int
    counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "artifact_id": self.artifact_id,
            "artifact_path": str(self.artifact_path),
            "score": round(self.score, 6),
            "baseline_score": round(self.baseline_score, 6),
            "promoted": self.promoted,
            "replay_digest": self.replay_digest,
            "metrics": _jsonify(self.metrics),
            "record_count": self.record_count,
            "counts": dict(self.counts),
        }


class TrainingRegistry:
    """SQLite-backed registry for training runs, corpus records, and artifacts."""

    def __init__(self, db_path: str | Path, artifact_root: str | Path | None = None):
        self.db_path = Path(db_path)
        self.artifact_root = Path(artifact_root) if artifact_root is not None else self.db_path.parent / "training_artifacts"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.db_path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA busy_timeout=5000")  # retry on lock, don't drop
        self._init_schema()

    def close(self) -> None:
        self._db.close()

    def _init_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id TEXT PRIMARY KEY,
                corpus_name TEXT NOT NULL,
                seed INTEGER,
                status TEXT NOT NULL,
                baseline_artifact_id TEXT,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS training_records (
                record_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                kind TEXT NOT NULL,
                source TEXT NOT NULL,
                timestamp TEXT,
                payload_json TEXT NOT NULL,
                parents_json TEXT NOT NULL,
                lineage_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_training_records_run_ordinal
                ON training_records(run_id, ordinal);
            CREATE INDEX IF NOT EXISTS ix_training_records_run_kind
                ON training_records(run_id, kind);
            CREATE TABLE IF NOT EXISTS training_artifacts (
                artifact_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                version INTEGER NOT NULL,
                score REAL NOT NULL,
                baseline_score REAL NOT NULL,
                path TEXT NOT NULL,
                checksum TEXT NOT NULL,
                status TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                promoted_at TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_training_artifacts_kind_version
                ON training_artifacts(kind, version);
            CREATE INDEX IF NOT EXISTS ix_training_artifacts_run_kind
                ON training_artifacts(run_id, kind);
            CREATE TABLE IF NOT EXISTS training_active_artifacts (
                kind TEXT PRIMARY KEY,
                artifact_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS training_artifact_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                previous_artifact_id TEXT,
                artifact_id TEXT NOT NULL,
                action TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self._db.commit()

    def create_run(
        self,
        corpus_name: str,
        *,
        run_id: str | None = None,
        seed: int | None = None,
        metadata: dict[str, Any] | None = None,
        baseline_artifact_id: str | None = None,
    ) -> str:
        run_id = run_id or uuid.uuid4().hex[:16]
        now = _utcnow().isoformat()
        self._db.execute(
            """
            INSERT INTO training_runs (
                run_id, corpus_name, seed, status, baseline_artifact_id,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                corpus_name,
                seed,
                "pending",
                baseline_artifact_id,
                _json_dumps(metadata or {}),
                now,
                now,
            ),
        )
        self._db.commit()
        return run_id

    def update_run_status(
        self,
        run_id: str,
        *,
        status: str,
        baseline_artifact_id: str | None = None,
    ) -> None:
        self._db.execute(
            """
            UPDATE training_runs
               SET status = ?, baseline_artifact_id = COALESCE(?, baseline_artifact_id),
                   updated_at = ?
             WHERE run_id = ?
            """,
            (status, baseline_artifact_id, _utcnow().isoformat(), run_id),
        )
        self._db.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT * FROM training_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "corpus_name": row["corpus_name"],
            "seed": row["seed"],
            "status": row["status"],
            "baseline_artifact_id": row["baseline_artifact_id"],
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def append_record(
        self,
        run_id: str,
        kind: str,
        payload: dict[str, Any],
        *,
        source: str = "",
        timestamp: datetime | None = None,
        parents: Iterable[str] = (),
        lineage: dict[str, Any] | None = None,
    ) -> TrainingRecord:
        ordinal_row = self._db.execute(
            "SELECT COALESCE(MAX(ordinal), -1) + 1 AS next_ordinal FROM training_records WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        ordinal = int(ordinal_row["next_ordinal"])
        record = TrainingRecord(
            record_id=uuid.uuid4().hex[:16],
            run_id=run_id,
            ordinal=ordinal,
            kind=kind,
            source=source,
            timestamp=timestamp,
            payload=payload,
            parents=tuple(parents),
            lineage=lineage or {"kind": kind, "source": source, "parents": list(parents)},
        )
        self._db.execute(
            """
            INSERT INTO training_records (
                record_id, run_id, ordinal, kind, source, timestamp,
                payload_json, parents_json, lineage_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.record_id,
                record.run_id,
                record.ordinal,
                record.kind,
                record.source,
                record.timestamp.isoformat() if record.timestamp else None,
                _json_dumps(record.payload),
                _json_dumps(record.parents),
                _json_dumps(record.lineage),
            ),
        )
        self._db.commit()
        return record

    def record_raw_event(
        self,
        run_id: str,
        payload: dict[str, Any],
        *,
        source: str = "event",
        timestamp: datetime | None = None,
        parents: Iterable[str] = (),
    ) -> TrainingRecord:
        return self.append_record(
            run_id,
            RAW_EVENT_KIND,
            payload,
            source=source,
            timestamp=timestamp,
            parents=parents,
        )

    def record_derived_label(
        self,
        run_id: str,
        payload: dict[str, Any],
        *,
        source: str = "label",
        timestamp: datetime | None = None,
        parents: Iterable[str] = (),
    ) -> TrainingRecord:
        return self.append_record(
            run_id,
            DERIVED_LABEL_KIND,
            payload,
            source=source,
            timestamp=timestamp,
            parents=parents,
        )

    def record_demo_window(
        self,
        run_id: str,
        payload: dict[str, Any],
        *,
        source: str = "demo_window",
        timestamp: datetime | None = None,
        parents: Iterable[str] = (),
    ) -> TrainingRecord:
        return self.append_record(
            run_id,
            DEMO_WINDOW_KIND,
            payload,
            source=source,
            timestamp=timestamp,
            parents=parents,
        )

    def record_operator_feedback(
        self,
        run_id: str,
        payload: dict[str, Any],
        *,
        source: str = "operator_feedback",
        timestamp: datetime | None = None,
        parents: Iterable[str] = (),
    ) -> TrainingRecord:
        return self.append_record(
            run_id,
            OPERATOR_FEEDBACK_KIND,
            payload,
            source=source,
            timestamp=timestamp,
            parents=parents,
        )

    def record_snapshot(
        self,
        run_id: str,
        payload: dict[str, Any],
        *,
        source: str = "snapshot",
        timestamp: datetime | None = None,
        parents: Iterable[str] = (),
    ) -> TrainingRecord:
        return self.append_record(
            run_id,
            SNAPSHOT_KIND,
            payload,
            source=source,
            timestamp=timestamp,
            parents=parents,
        )

    def records_for_run(self, run_id: str) -> list[TrainingRecord]:
        rows = self._db.execute(
            """
            SELECT record_id, run_id, ordinal, kind, source, timestamp,
                   payload_json, parents_json, lineage_json
              FROM training_records
             WHERE run_id = ?
             ORDER BY ordinal ASC, record_id ASC
            """,
            (run_id,),
        ).fetchall()
        result: list[TrainingRecord] = []
        for row in rows:
            result.append(
                TrainingRecord(
                    record_id=row["record_id"],
                    run_id=row["run_id"],
                    ordinal=int(row["ordinal"]),
                    kind=row["kind"],
                    source=row["source"],
                    timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else None,
                    payload=json.loads(row["payload_json"]),
                    parents=tuple(json.loads(row["parents_json"])),
                    lineage=json.loads(row["lineage_json"]),
                )
            )
        return result

    def artifact_dir(self, artifact_id: str, kind: str, version: int) -> Path:
        return self.artifact_root / kind / f"v{version:04d}-{artifact_id}"

    def _next_version(self, kind: str) -> int:
        row = self._db.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 AS next_version FROM training_artifacts WHERE kind = ?",
            (kind,),
        ).fetchone()
        return int(row["next_version"])

    def _artifact_row(self, row: sqlite3.Row) -> TrainingArtifact:
        return TrainingArtifact(
            artifact_id=row["artifact_id"],
            run_id=row["run_id"],
            kind=row["kind"],
            version=int(row["version"]),
            score=float(row["score"]),
            baseline_score=float(row["baseline_score"]),
            path=Path(row["path"]),
            checksum=row["checksum"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            promoted_at=datetime.fromisoformat(row["promoted_at"]) if row["promoted_at"] else None,
            metrics=json.loads(row["metrics_json"]),
            metadata=json.loads(row["metadata_json"]),
        )

    def get_artifact(self, artifact_id: str) -> TrainingArtifact | None:
        row = self._db.execute(
            "SELECT * FROM training_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if row is None:
            return None
        return self._artifact_row(row)

    def get_active_artifact(self, kind: str) -> TrainingArtifact | None:
        row = self._db.execute(
            """
            SELECT a.*
              FROM training_artifacts a
              JOIN training_active_artifacts active ON active.artifact_id = a.artifact_id
             WHERE active.kind = ?
            """,
            (kind,),
        ).fetchone()
        if row is None:
            return None
        return self._artifact_row(row)

    def store_artifact(
        self,
        *,
        run_id: str,
        kind: str,
        path: str | Path,
        score: float,
        baseline_score: float,
        metrics: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        status: str = "candidate",
        artifact_id: str | None = None,
        version: int | None = None,
    ) -> TrainingArtifact:
        artifact_id = artifact_id or uuid.uuid4().hex[:16]
        version = version if version is not None else self._next_version(kind)
        path = Path(path)
        checksum = _directory_checksum(path)
        created_at = _utcnow()
        self._db.execute(
            """
            INSERT INTO training_artifacts (
                artifact_id, run_id, kind, version, score, baseline_score,
                path, checksum, status, metrics_json, metadata_json,
                created_at, promoted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                run_id,
                kind,
                version,
                score,
                baseline_score,
                str(path),
                checksum,
                status,
                _json_dumps(metrics),
                _json_dumps(metadata or {}),
                created_at.isoformat(),
                created_at.isoformat() if status == "active" else None,
            ),
        )
        self._db.commit()
        return self.get_artifact(artifact_id)  # type: ignore[return-value]

    def mark_artifact_status(self, artifact_id: str, status: str) -> None:
        self._db.execute(
            """
            UPDATE training_artifacts
               SET status = ?,
                   promoted_at = COALESCE(promoted_at, ?)
             WHERE artifact_id = ?
            """,
            (status, _utcnow().isoformat(), artifact_id),
        )
        self._db.commit()

    def promote_artifact(self, artifact_id: str) -> TrainingArtifact:
        artifact = self.get_artifact(artifact_id)
        if artifact is None:
            raise KeyError(f"unknown artifact: {artifact_id}")
        previous = self.get_active_artifact(artifact.kind)
        now = _utcnow().isoformat()
        if previous is not None and previous.artifact_id != artifact.artifact_id:
            self.mark_artifact_status(previous.artifact_id, "superseded")
        self._db.execute(
            """
            INSERT OR REPLACE INTO training_active_artifacts (kind, artifact_id, updated_at)
            VALUES (?, ?, ?)
            """,
            (artifact.kind, artifact.artifact_id, now),
        )
        self._db.execute(
            """
            INSERT INTO training_artifact_history (kind, previous_artifact_id, artifact_id, action, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (artifact.kind, previous.artifact_id if previous is not None else None, artifact.artifact_id, "promote", now),
        )
        self.mark_artifact_status(artifact.artifact_id, "active")
        return self.get_artifact(artifact.artifact_id)  # type: ignore[return-value]

    def rollback_artifact(self, kind: str) -> TrainingArtifact | None:
        active = self.get_active_artifact(kind)
        if active is None:
            return None
        row = self._db.execute(
            """
            SELECT previous_artifact_id
              FROM training_artifact_history
             WHERE kind = ? AND action = 'promote'
               AND previous_artifact_id IS NOT NULL
             ORDER BY id DESC
             LIMIT 1
            """,
            (kind,),
        ).fetchone()
        if row is None:
            return active
        previous_id = row["previous_artifact_id"]
        if previous_id is None:
            return active
        previous = self.get_artifact(previous_id)
        if previous is None:
            return active
        now = _utcnow().isoformat()
        self.mark_artifact_status(active.artifact_id, "superseded")
        self._db.execute(
            "UPDATE training_active_artifacts SET artifact_id = ?, updated_at = ? WHERE kind = ?",
            (previous.artifact_id, now, kind),
        )
        self._db.execute(
            """
            INSERT INTO training_artifact_history (kind, previous_artifact_id, artifact_id, action, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (kind, active.artifact_id, previous.artifact_id, "rollback", now),
        )
        self.mark_artifact_status(previous.artifact_id, "active")
        return self.get_artifact(previous.artifact_id)

    def list_artifacts(self, kind: str | None = None) -> list[TrainingArtifact]:
        if kind is None:
            rows = self._db.execute("SELECT * FROM training_artifacts ORDER BY kind, version").fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM training_artifacts WHERE kind = ? ORDER BY version",
                (kind,),
            ).fetchall()
        return [self._artifact_row(row) for row in rows]


class TrainingSession:
    """Deterministic offline replay over the existing engine learning stack."""

    def __init__(
        self,
        registry: TrainingRegistry,
        reservoir_factory: Callable[[], "BeliefEngine"],
        *,
        seed: int | None = 0,
        min_improvement: float = 1e-6,
    ):
        self.registry = registry
        self.reservoir_factory = reservoir_factory
        self.seed = seed
        self.min_improvement = min_improvement

    def _apply_record(self, reservoir: "BeliefEngine", record: TrainingRecord) -> dict[str, Any]:
        payload = record.payload
        if record.kind == RAW_EVENT_KIND:
            sensor_readings = payload.get("sensor_readings")
            raw_inputs = payload.get("raw_inputs")
            driver_observations = payload.get("driver_observations")
            result = reservoir.ingest(
                sensor_readings=sensor_readings,
                raw_inputs=raw_inputs,
                driver_observations=driver_observations,
                now=record.timestamp,
            )
            return {
                "kind": record.kind,
                "step": result["step"],
                "collapsed": result["collapsed"],
                "forecast_skill": (reservoir.driver_forecast.skill
                                   if getattr(reservoir, "driver_forecast", None) is not None else None),
            }
        if record.kind == DERIVED_LABEL_KIND:
            targets = payload.get("targets")
            if targets is not None:
                reservoir.set_driver_targets(
                    targets,
                    forecast_labels=payload.get("forecast_labels"),
                )
            return {
                "kind": record.kind,
                "targets": _jsonify(targets),
                "forecast_labels": _jsonify(payload.get("forecast_labels")),
            }
        if record.kind == DEMO_WINDOW_KIND:
            phase = payload.get("phase", "start")
            burst_profile = payload.get("burst_profile")
            if reservoir.training is not None:
                if phase == "start" and burst_profile is not None:
                    reservoir.training.start_demo_burst(TrainingBurstProfile(**burst_profile))
                elif phase in {"stop", "end"}:
                    reservoir.training.stop_demo_burst()
            return {
                "kind": record.kind,
                "phase": phase,
                "demo_window": payload.get("window_id"),
            }
        if record.kind == OPERATOR_FEEDBACK_KIND:
            mag = reservoir.observe_feedback(
                payload["node"],
                payload["role"],
                int(payload["value"]),
                alpha=float(payload.get("alpha", 1.0)),
                confidence=float(payload.get("confidence", 1.0)),
                decision=str(payload.get("decision", "confirm")),
            )
            return {
                "kind": record.kind,
                "magnitude": mag,
                "node": payload["node"],
                "role": payload["role"],
            }
        if record.kind == SNAPSHOT_KIND:
            return {
                "kind": record.kind,
                "keys": sorted(payload.keys()),
            }
        return {"kind": record.kind}

    def _collect_metrics(
        self,
        reservoir: "BeliefEngine",
        counts: Counter[str],
        records: list[TrainingRecord],
    ) -> dict[str, Any]:
        training = reservoir.training.snapshot() if reservoir.training is not None else {}
        calibration = reservoir.calibration.stats() if reservoir.calibration is not None else {}
        fractal = reservoir.fractal_stack.stats() if reservoir.fractal_stack is not None else {}
        df = getattr(reservoir, "driver_forecast", None)
        forecast = df.snapshot() if df is not None else {}

        scale_surprise = []
        for scale in fractal.get("scales", []):
            if isinstance(scale, dict) and "surprise_ema" in scale:
                scale_surprise.append(float(scale["surprise_ema"]))

        forecast_skill = float(forecast.get("skill") or 0.0)
        training_surprise = float(training.get("surprise_ema") or 0.0)
        calibration_surprise = 0.0
        if calibration:
            surprise_series = calibration.get("surprise_ema", {})
            if isinstance(surprise_series, dict) and surprise_series:
                calibration_surprise = float(np.mean(list(surprise_series.values())))
            tracking_series = calibration.get("tracking_ema", {})
            if isinstance(tracking_series, dict) and tracking_series:
                calibration_surprise = float(
                    np.mean([calibration_surprise, float(np.mean(list(tracking_series.values())))])
                )
        fractal_surprise = float(np.mean(scale_surprise)) if scale_surprise else 0.0

        score = (
            0.70 * forecast_skill
            + 0.10 * (1.0 / (1.0 + training_surprise))
            + 0.10 * (1.0 / (1.0 + calibration_surprise))
            + 0.10 * (1.0 / (1.0 + fractal_surprise))
        )
        return {
            "forecast": forecast,
            "training": training,
            "calibration": calibration,
            "fractal": fractal,
            "forecast_skill": round(forecast_skill, 6),
            "training_surprise_ema": round(training_surprise, 6),
            "calibration_surprise_ema": round(calibration_surprise, 6),
            "fractal_surprise_ema": round(fractal_surprise, 6),
            "score": round(float(score), 6),
            "record_counts": dict(counts),
            "record_count": len(records),
        }

    def replay(
        self,
        run_id: str,
        *,
        artifact_kind: str = "reservoir",
        promote: bool = True,
        baseline_artifact_id: str | None = None,
        artifact_metadata: dict[str, Any] | None = None,
    ) -> TrainingReplayResult:
        records = self.registry.records_for_run(run_id)
        counts: Counter[str] = Counter()
        trace: list[dict[str, Any]] = []
        active_before = self.registry.get_active_artifact(artifact_kind)
        baseline = self.registry.get_artifact(baseline_artifact_id) if baseline_artifact_id else active_before
        baseline_score = baseline.score if baseline is not None else 0.0

        with _seed_scope(self.seed):
            reservoir = self.reservoir_factory()
            if reservoir.training is not None:
                reservoir.training.stop_demo_burst()

            for record in records:
                counts[record.kind] += 1
                trace.append(
                    {
                        "ordinal": record.ordinal,
                        "record_id": record.record_id,
                        "event": self._apply_record(reservoir, record),
                    }
                )

            metrics = self._collect_metrics(reservoir, counts, records)
            replay_digest = _sha256_bytes(_json_dumps(trace).encode("utf-8"))

            artifact_id = uuid.uuid4().hex[:16]
            version = self.registry._next_version(artifact_kind)
            artifact_path = self.registry.artifact_dir(artifact_id, artifact_kind, version)
            artifact_path.mkdir(parents=True, exist_ok=True)
            checkpoint_path = artifact_path / "reservoir.pkl"
            summary_path = artifact_path / "summary.json"
            reservoir.save(checkpoint_path)
            summary = {
                "run_id": run_id,
                "artifact_kind": artifact_kind,
                "artifact_id": artifact_id,
                "version": version,
                "seed": self.seed,
                "replay_digest": replay_digest,
                "metrics": metrics,
                "trace": trace,
                "records": [record.to_dict() for record in records],
                "metadata": artifact_metadata or {},
            }
            summary_path.write_text(_json_dumps(summary), encoding="utf-8")
            artifact = self.registry.store_artifact(
                run_id=run_id,
                kind=artifact_kind,
                path=artifact_path,
                score=float(metrics["score"]),
                baseline_score=float(baseline_score),
                metrics=metrics,
                metadata={
                    "seed": self.seed,
                    "replay_digest": replay_digest,
                    "summary_path": str(summary_path),
                    "baseline_artifact_id": baseline.artifact_id if baseline is not None else None,
                    **(artifact_metadata or {}),
                },
                artifact_id=artifact_id,
                version=version,
            )

        if artifact.score <= baseline_score:
            self.registry.mark_artifact_status(artifact.artifact_id, "candidate")
            promoted = False
        else:
            promoted = False
            if promote:
                self.registry.promote_artifact(artifact.artifact_id)
                promoted = True
        self.registry.update_run_status(
            run_id,
            status="completed" if promoted else "evaluated",
            baseline_artifact_id=baseline.artifact_id if baseline is not None else None,
        )
        artifact = self.registry.get_artifact(artifact.artifact_id) or artifact
        return TrainingReplayResult(
            run_id=run_id,
            artifact_id=artifact.artifact_id,
            artifact_path=artifact.path,
            score=artifact.score,
            baseline_score=baseline_score,
            promoted=promoted,
            replay_digest=replay_digest,
            metrics=metrics,
            record_count=len(records),
            counts=dict(counts),
        )

    def replay_twice(self, run_id: str, *, artifact_kind: str = "reservoir") -> tuple[TrainingReplayResult, TrainingReplayResult]:
        first = self.replay(run_id, artifact_kind=artifact_kind, promote=False)
        second = self.replay(run_id, artifact_kind=artifact_kind, promote=False)
        return first, second


def _build_smoke_engine(spec, seed: int = 20260529) -> "BeliefEngine":
    # lazy: the engine module lands in P2; this keeps the backbone importable before it.
    from umwelt.engine import BeliefEngine
    from umwelt.spec.build import build_graph_from_spec

    reservoir = BeliefEngine(
        graph=build_graph_from_spec(spec),
        calibration=CalibrationConfig(hamiltonian_enabled=True),
        fractal_stack=FractalStackConfig(
            enabled=True,
            scales=[ScaleConfig(stride=2, h_scale=0.03)],
            rollout_horizon=2,
        ),
        seed=seed,
    )
    configure_param_bundles(reservoir.graph, spec)
    reservoir.sensor_bridge.refresh_node_params()
    # bindings ALWAYS come from the spec — the origin deployment's built-in catalog
    # (configure_sensors) has no umwelt equivalent; the spec seam replaces it.
    _apply_spec_bindings(reservoir, spec)
    reservoir.sensor_bridge.upgrade_weights()
    return reservoir


def run_training_smoke(spec, base_dir: Path | None = None, seed: int = 20260529) -> dict[str, Any]:
    """Corpus → replay → promote → rollback, end to end, against the given DomainSpec.

    Fully domain-agnostic: signal ids come from the spec's bindings, and periodic
    observations come from the engine's attached drivers (`d.target_bloch(t)`). The
    origin deployment's ephemeris-labelled corpus moved to its domain example.
    """
    root = Path(base_dir) if base_dir is not None else Path(tempfile.mkdtemp(prefix="umwelt-training-smoke-"))
    registry = TrainingRegistry(root / "training_registry.sqlite", artifact_root=root / "artifacts")

    # A probe engine gives the corpus its vocabulary: the first two bound signal ids
    # carry the raw readings; the attached drivers synthesize periodic observations.
    probe = _build_smoke_engine(spec, seed=seed)
    drivers = list(getattr(probe, "drivers", []) or [])
    binding_ids = [b.sensor_id for b in (spec.bindings or ())]
    if not binding_ids:
        raise ValueError("run_training_smoke needs a spec with at least one binding")
    sid_a = binding_ids[0]
    sid_b = binding_ids[1] if len(binding_ids) > 1 else binding_ids[0]

    def _driver_obs(t: datetime) -> dict[str, Any]:
        return {d.name: d.target_bloch(t) for d in drivers}

    # feedback lands on the first driver's anchor when one exists, else the first
    # binding's target leaf — both are guaranteed nodes of the spec-built graph.
    if drivers:
        fb_node, fb_role = drivers[0].node, drivers[0].role
    else:
        fb_node, fb_role = spec.bindings[0].zone, spec.bindings[0].role

    t0 = datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 5, 29, 12, 5, tzinfo=timezone.utc)

    try:
        baseline_run = registry.create_run(
            "smoke_baseline",
            seed=seed,
            metadata={"role": "baseline"},
        )
        baseline_first = registry.record_raw_event(
            baseline_run,
            {"sensor_readings": {sid_a: -0.4, sid_b: 0.2}, "driver_observations": _driver_obs(t0)},
            timestamp=t0,
        )
        registry.record_raw_event(
            baseline_run,
            {"sensor_readings": {sid_a: 0.1, sid_b: 0.25}, "driver_observations": _driver_obs(t1)},
            timestamp=t1,
            parents=[baseline_first.record_id],
        )
        registry.record_snapshot(
            baseline_run,
            {"context": {"notes": "baseline corpus", "step": 0}},
            parents=[baseline_first.record_id],
        )

        candidate_run = registry.create_run(
            "smoke_candidate",
            seed=seed,
            metadata={"role": "candidate"},
        )
        candidate_first = registry.record_raw_event(
            candidate_run,
            {"sensor_readings": {sid_a: -0.4, sid_b: 0.2}, "driver_observations": _driver_obs(t0)},
            timestamp=t0,
        )
        candidate_second = registry.record_raw_event(
            candidate_run,
            {"sensor_readings": {sid_a: 0.1, sid_b: 0.25}, "driver_observations": _driver_obs(t1)},
            timestamp=t1,
            parents=[candidate_first.record_id],
        )
        registry.record_demo_window(
            candidate_run,
            {
                "window_id": "demo-smoke-1",
                "phase": "start",
                "burst_profile": TrainingBurstProfile(
                    name="smoke",
                    lr_multiplier=1.5,
                    surprise_multiplier=1.25,
                    phase_align_interval=1,
                ).snapshot(),
            },
            parents=[candidate_second.record_id],
        )
        registry.record_operator_feedback(
            candidate_run,
            {
                "node": fb_node,
                "role": fb_role,
                "value": 1,
                "alpha": 1.0,
                "confidence": 0.95,
                "decision": "confirm",
            },
            parents=[candidate_second.record_id],
        )
        registry.record_demo_window(
            candidate_run,
            {
                "window_id": "demo-smoke-1",
                "phase": "stop",
            },
            parents=[candidate_second.record_id],
        )
        registry.record_snapshot(
            candidate_run,
            {"context": {"notes": "candidate corpus", "step": 1}},
            parents=[candidate_second.record_id],
        )

        session = TrainingSession(
            registry,
            lambda: _build_smoke_engine(spec, seed=seed),
            seed=seed,
        )

        baseline_result = session.replay(
            baseline_run,
            artifact_kind="reservoir",
            promote=True,
            artifact_metadata={"corpus": "baseline"},
        )
        candidate_result = session.replay(
            candidate_run,
            artifact_kind="reservoir",
            promote=True,
            artifact_metadata={"corpus": "candidate"},
        )
        repeated_result = session.replay(
            candidate_run,
            artifact_kind="reservoir",
            promote=False,
            artifact_metadata={"corpus": "candidate_replay"},
        )

        active_before_rollback = registry.get_active_artifact("reservoir")
        rolled_back = registry.rollback_artifact("reservoir")
        active_after_rollback = registry.get_active_artifact("reservoir")

        output = {
            "root": str(root),
            "baseline": baseline_result.to_dict(),
            "candidate": candidate_result.to_dict(),
            "candidate_repeat": repeated_result.to_dict(),
            "active_before_rollback": active_before_rollback.to_dict() if active_before_rollback else None,
            "rolled_back": rolled_back.to_dict() if rolled_back else None,
            "active_after_rollback": active_after_rollback.to_dict() if active_after_rollback else None,
            "reproducible": candidate_result.replay_digest == repeated_result.replay_digest,
            "promoted": candidate_result.promoted,
            "rollback_restored_previous": (
                rolled_back is not None
                and active_after_rollback is not None
                and rolled_back.artifact_id == active_after_rollback.artifact_id
                and active_before_rollback is not None
                and active_before_rollback.artifact_id != active_after_rollback.artifact_id
            ),
        }
        print(_json_dumps(output))
        return output
    finally:
        registry.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Offline training backbone smoke harness")
    parser.add_argument("--smoke", action="store_true", help="Run the built-in smoke harness")
    parser.add_argument("--spec", type=str, default=None,
                        help="DomainSpec ref as 'module:ATTR' (required for --smoke)")
    parser.add_argument("--work-dir", type=Path, default=None, help="Directory for registry and artifacts")
    parser.add_argument("--seed", type=int, default=20260529, help="Deterministic seed for replay")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.smoke:
        if not args.spec:
            parser.error("--smoke needs --spec module:ATTR (bindings always come from a spec)")
        from umwelt.spec.schema import load_spec
        run_training_smoke(load_spec(args.spec), args.work_dir, seed=args.seed)
        return
    parser.error("pass --smoke to run the harness")


if __name__ == "__main__":
    main()
