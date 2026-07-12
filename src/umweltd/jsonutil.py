"""JSON-safe conversion for engine projections (numpy scalars/arrays, tuples, sets)."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime


def jsonable(obj):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in obj]
    # numpy scalars/arrays without importing numpy here
    if hasattr(obj, "tolist"):
        return jsonable(obj.tolist())
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)
