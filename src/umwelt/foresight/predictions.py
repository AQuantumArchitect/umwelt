from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

import aiosqlite

from umwelt.events import Event

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    prediction_id: str
    pattern_id: str
    description: str
    predicted_event_type: str
    predicted_time: Optional[datetime]
    confidence: float
    status: str = "active"  # active, correct, missed, expired
    created_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    # Structured forecast target (populated by the quantum brain; NULL for
    # legacy rows). See docs/SURFACES.md.
    node: Optional[str] = None
    role: Optional[str] = None
    target_value: Optional[float] = None
    target_sigma: Optional[float] = None
    kind: str = "forecast"  # 'forecast' (scorable) | 'observation' (log-only)

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]


class RecommendationStatus(str, Enum):
    SUGGESTED = "suggested"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"   # dispatched to the device by the actuator dispatcher


@dataclass
class Recommendation:
    recommendation_id: str
    pattern_id: str
    description: str
    trigger: str
    action: str
    confidence: float
    status: RecommendationStatus = RecommendationStatus.SUGGESTED
    created_at: Optional[datetime] = None
    # Structured actuator command (populated by the quantum brain; NULL for
    # legacy human-only rows). Brandon's actuator path reads these.
    actuator_id: Optional[str] = None
    command_json: Optional[str] = None
    node: Optional[str] = None
    role: Optional[str] = None
    value: Optional[int] = None
    reason: Optional[str] = None
    expires_at: Optional[datetime] = None

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]


_PREDICTION_NEW_COLS = [
    ("node",         "TEXT"),
    ("role",         "TEXT"),
    ("target_value", "REAL"),
    ("target_sigma", "REAL"),
    ("kind",         "TEXT DEFAULT 'forecast'"),
]

_RECOMMENDATION_NEW_COLS = [
    ("actuator_id",  "TEXT"),
    ("command_json", "TEXT"),
    ("node",         "TEXT"),
    ("role",         "TEXT"),
    ("value",        "INTEGER"),
    ("reason",       "TEXT"),
    ("expires_at",   "TEXT"),
]


async def _existing_columns(db: aiosqlite.Connection, table: str) -> set[str]:
    cursor = await db.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    return {row[1] for row in rows}


async def _add_missing_columns(
    db: aiosqlite.Connection, table: str, new_cols: list[tuple[str, str]]
) -> None:
    have = await _existing_columns(db, table)
    for name, decl in new_cols:
        if name not in have:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


class PredictionStore:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def init(self) -> None:
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                pattern_id TEXT NOT NULL,
                description TEXT NOT NULL,
                predicted_event_type TEXT NOT NULL,
                predicted_time TEXT,
                confidence REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT,
                resolved_at TEXT
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendations (
                recommendation_id TEXT PRIMARY KEY,
                pattern_id TEXT NOT NULL,
                description TEXT NOT NULL,
                trigger_desc TEXT NOT NULL,
                action_desc TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'suggested',
                created_at TEXT
            )
            """
        )
        # Additive migration for the quantum-brain output surface. Idempotent:
        # only ADDs columns that aren't already there. See SURFACES.md.
        await _add_missing_columns(self._db, "predictions", _PREDICTION_NEW_COLS)
        await _add_missing_columns(self._db, "recommendations", _RECOMMENDATION_NEW_COLS)
        await self._db.commit()

    async def store_prediction(self, p: Prediction) -> None:
        await self._db.execute(
            """
            INSERT OR REPLACE INTO predictions (
                prediction_id, pattern_id, description, predicted_event_type,
                predicted_time, confidence, status, created_at, resolved_at,
                node, role, target_value, target_sigma, kind
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                p.prediction_id, p.pattern_id, p.description,
                p.predicted_event_type,
                p.predicted_time.isoformat() if p.predicted_time else None,
                p.confidence, p.status,
                p.created_at.isoformat() if p.created_at else None,
                p.resolved_at.isoformat() if p.resolved_at else None,
                p.node, p.role, p.target_value, p.target_sigma, p.kind,
            ),
        )
        await self._db.commit()

    async def get_predictions(self, status: Optional[str] = None) -> list[Prediction]:
        sql = (
            "SELECT prediction_id, pattern_id, description, predicted_event_type, "
            "predicted_time, confidence, status, created_at, resolved_at, "
            "node, role, target_value, target_sigma, kind FROM predictions"
        )
        if status:
            cursor = await self._db.execute(
                sql + " WHERE status = ? ORDER BY created_at DESC", (status,)
            )
        else:
            cursor = await self._db.execute(sql + " ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [self._row_to_prediction(r) for r in rows]

    async def update_prediction_status(self, prediction_id: str, status: str) -> None:
        await self._db.execute(
            "UPDATE predictions SET status = ?, resolved_at = ? WHERE prediction_id = ?",
            (status, datetime.now(timezone.utc).isoformat(), prediction_id),
        )
        await self._db.commit()

    async def store_recommendation(self, r: Recommendation) -> None:
        await self._db.execute(
            """
            INSERT OR REPLACE INTO recommendations (
                recommendation_id, pattern_id, description, trigger_desc,
                action_desc, confidence, status, created_at,
                actuator_id, command_json, node, role, value, reason, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r.recommendation_id, r.pattern_id, r.description,
                r.trigger, r.action, r.confidence, r.status.value,
                r.created_at.isoformat() if r.created_at else None,
                r.actuator_id, r.command_json, r.node, r.role, r.value,
                r.reason,
                r.expires_at.isoformat() if r.expires_at else None,
            ),
        )
        await self._db.commit()

    async def get_recommendations(self, status: Optional[str] = None) -> list[Recommendation]:
        sql = (
            "SELECT recommendation_id, pattern_id, description, trigger_desc, "
            "action_desc, confidence, status, created_at, "
            "actuator_id, command_json, node, role, value, reason, expires_at "
            "FROM recommendations"
        )
        if status:
            cursor = await self._db.execute(
                sql + " WHERE status = ? ORDER BY created_at DESC", (status,)
            )
        else:
            cursor = await self._db.execute(sql + " ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [self._row_to_recommendation(r) for r in rows]

    async def update_recommendation_status(self, rec_id: str, status: RecommendationStatus) -> None:
        await self._db.execute(
            "UPDATE recommendations SET status = ? WHERE recommendation_id = ?",
            (status.value, rec_id),
        )
        await self._db.commit()

    async def get_recommendation(self, rec_id: str) -> Optional[Recommendation]:
        cursor = await self._db.execute(
            "SELECT recommendation_id, pattern_id, description, trigger_desc, "
            "action_desc, confidence, status, created_at, "
            "actuator_id, command_json, node, role, value, reason, expires_at "
            "FROM recommendations WHERE recommendation_id = ?",
            (rec_id,),
        )
        row = await cursor.fetchone()
        return self._row_to_recommendation(row) if row else None

    async def get_recent_commands(self, since_iso: str) -> list[Recommendation]:
        """Outstanding / just-dispatched commands (APPROVED or EXECUTED) created
        at or after `since_iso`. Used to derive the actuator 'in-transit' state
        for the control panel + console snapshot — a command whose target the
        device hasn't echoed yet. ISO8601-UTC timestamps sort lexically, so a
        string comparison is a valid time filter."""
        cursor = await self._db.execute(
            "SELECT recommendation_id, pattern_id, description, trigger_desc, "
            "action_desc, confidence, status, created_at, "
            "actuator_id, command_json, node, role, value, reason, expires_at "
            "FROM recommendations "
            "WHERE status IN ('approved','executed') AND created_at >= ? "
            "ORDER BY created_at DESC",
            (since_iso,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_recommendation(r) for r in rows]

    def _row_to_prediction(self, row: tuple) -> Prediction:
        return Prediction(
            prediction_id=row[0], pattern_id=row[1], description=row[2],
            predicted_event_type=row[3],
            predicted_time=datetime.fromisoformat(row[4]) if row[4] else None,
            confidence=row[5], status=row[6],
            created_at=datetime.fromisoformat(row[7]) if row[7] else None,
            resolved_at=datetime.fromisoformat(row[8]) if row[8] else None,
            node=row[9], role=row[10],
            target_value=row[11], target_sigma=row[12],
            kind=row[13] or "forecast",
        )

    def _row_to_recommendation(self, row: tuple) -> Recommendation:
        return Recommendation(
            recommendation_id=row[0], pattern_id=row[1], description=row[2],
            trigger=row[3], action=row[4], confidence=row[5],
            status=RecommendationStatus(row[6]),
            created_at=datetime.fromisoformat(row[7]) if row[7] else None,
            actuator_id=row[8], command_json=row[9],
            node=row[10], role=row[11], value=row[12], reason=row[13],
            expires_at=datetime.fromisoformat(row[14]) if row[14] else None,
        )


