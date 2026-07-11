"""Tiny shared helpers with NO project dependencies — the canonical home for idioms that
would otherwise be copy-pasted across modules. Importing this must stay cycle-free (stdlib
only), so it is safe to import from anywhere, including the foundational substrate modules.

Vendored from the meerkat deployment (meerkat/core/util.py) at extraction, unchanged in
behavior — these seven functions are the engine's entire "shared util" surface on purpose.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def env_flag(name: str, default: bool = False) -> bool:
    """A boolean environment flag: ON iff the var is set to "1". Unset → `default`. Replaces the
    scattered `os.environ.get(name) == "1"` / `os.environ.get(name, "0"|"1") == "1"` idiom."""
    v = os.environ.get(name)
    return default if v is None else v == "1"


def clamp01(value: float) -> float:
    """Clamp to the unit interval [0, 1] — the `max(0.0, min(1.0, x))` (and `min(1.0, max(0.0, x))`)
    idiom, one home. NOTE: only for [0,1]; range clamps to other bounds use `clamp`."""
    return max(0.0, min(1.0, float(value)))


def round_or_none(value, digits: int = 4):
    """`round(value, digits)` but None-safe — the `round(x, n) if x is not None else None` idiom
    that recurs across JSON serialization. Casts to float so numpy scalars round cleanly. NOTE: not
    for guard-then-compute sites (e.g. `None if ts is None else round(now - ts, 1)`), which must
    short-circuit BEFORE the arithmetic."""
    return None if value is None else round(float(value), digits)


def clamp(value: float, lo: float, hi: float) -> float:
    """General range clamp. (For the unit interval prefer `clamp01`.)"""
    return max(lo, min(hi, float(value)))


def ema(prev: float, value: float, alpha: float) -> float:
    """One exponential-moving-average step, new-sample weight `alpha`:
    `alpha·value + (1−alpha)·prev` — the inline smoothing idiom, one home. Stateful
    EMA banks and learners keep their own state machinery; this is just the arithmetic."""
    a = float(alpha)
    return a * float(value) + (1.0 - a) * float(prev)


def utcnow() -> datetime:
    """timezone-aware UTC now — one home."""
    return datetime.now(timezone.utc)


def iso_or_none(ts: datetime | None) -> str | None:
    """ISO-format a datetime, passing None through."""
    return ts.isoformat() if ts is not None else None


def jsonable(value: Any) -> Any:
    """Recursively convert datetimes/Enums/dicts/sequences to JSON-safe values."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value
