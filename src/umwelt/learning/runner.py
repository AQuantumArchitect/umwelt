"""BrainRunner — the one driver for OFFLINE, sensor-batch learners.

The live forebrain is the app's async transport→actuator loop — NOT this. The offline
learners that replay a stream of `(sensor_readings, now)` batches through a reservoir under
a context gauge are this: build_reservoir (DRY-1) → set_role gauge (DRY-2) → replay the
batches → save. The HINDBRAIN is the first user; the picker's brain-eval and any
sensor-replay gym variant are the next — they instantiate BrainRunner instead of hand-
rolling a build + gauge + ingest-loop + save each time.

NOT forced into this mold (on purpose): training_backbone replays typed RECORDS via
_apply_record, and the rig replays a Tape — richer per-step application models. They already
share build_reservoir + set_role + save/_restore_learned from DRY-1/2/3; their replay loops
legitimately differ, so wrapping them here would be indirection, not DRY.
"""
from __future__ import annotations

import logging
from pathlib import Path

from umwelt.boot import build_reservoir, set_role

logger = logging.getLogger(__name__)


class BrainRunner:
    """Run a reservoir over a sequence of `(sensor_readings, now, *extra)` batches under a
    context gauge. Wrap an existing reservoir or build one; the gauge (a ContextState) is
    stamped once via set_role."""

    def __init__(self, reservoir=None, *, gauge=None, build_kwargs: dict | None = None):
        self.reservoir = reservoir if reservoir is not None else build_reservoir(**(build_kwargs or {}))
        self.gauge = gauge
        if gauge is not None:
            set_role(self.reservoir, gauge)

    def replay(self, batches, *, on_batch=None, max_batches: int | None = None) -> int:
        """Ingest each batch's `sensor_readings` at `now` under the gauge. `batches` is an
        iterable whose items are `(sensor_readings, now, ...)` — extra fields (e.g. a replay
        cursor) ride along untouched. `on_batch(n, item, result)` fires after each successful
        ingest (cursor tracking / scoring). Stops at `max_batches`. A batch that raises is
        logged + skipped — one bad reading never kills a replay. Returns the count ingested."""
        n = 0
        for item in batches:
            readings, now = item[0], item[1]
            # Optional per-batch confidence at item[2] (a dict) — the forecast tape, so a
            # replayed forecast lands at its recorded confidence, not as a full-confidence
            # sensor (train≡deploy gauge, FORESIGHT.md §4). A non-dict item[2] (e.g. a
            # cursor string) is ignored, so other callers are unaffected.
            conf = item[2] if (len(item) > 2 and isinstance(item[2], dict) and item[2]) else None
            extra = {"confidence": conf} if conf else {}
            try:
                result = self.reservoir.ingest(sensor_readings=readings, now=now, **extra)
            except Exception as e:
                logger.warning("BrainRunner: ingest failed @ batch %d: %s", n + 1, e)
                continue
            n += 1
            if on_batch is not None:
                on_batch(n, item, result)
            if max_batches is not None and n >= max_batches:
                break
        return n

    def save(self, path: str | Path) -> None:
        self.reservoir.save(path)

    def load(self, path: str | Path) -> None:
        self.reservoir.load(path)
