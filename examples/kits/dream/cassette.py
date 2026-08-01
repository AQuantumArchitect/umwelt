"""Counterfactual dream cassette: mutates a walk, replays on a clone, never actuates."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from examples.fledgeling_fog.world import (
    FOG_SPEC,
    agent_walk,
    runner_batches,
    synthesize_rows,
)
from umwelt.host import GameHost
from umwelt.host.api import Intent

HONESTY_TIER = "synthetic CI — counterfactual cassette, zero live dispatch"
DREAM_SPEC = FOG_SPEC
START = datetime(2026, 6, 1, tzinfo=timezone.utc)


@dataclass
class BaselineReport:
    kit: str
    live_dispatches: int
    counterfactual_batches: int
    field_changed: bool
    beats_baseline: bool
    honesty: str

    def summary(self) -> str:
        v = "BEATS" if self.beats_baseline else "DOES_NOT_BEAT"
        return (
            f"[{self.kit}] dispatches={self.live_dispatches} "
            f"cf_batches={self.counterfactual_batches} "
            f"field_changed={self.field_changed} → {v} ({self.honesty})"
        )


def _mutate_readings(readings: dict, flip: str) -> dict:
    """Counterfactual: flip one scout channel's polarity."""
    out = dict(readings)
    if flip in out:
        out[flip] = 1.0 - float(out[flip])
    return out


def run_dream_baseline(*, ticks: int = 80, seed: int = 4) -> BaselineReport:
    # Live host — must stay side-effect free during dream
    live = GameHost()
    live.register_world(DREAM_SPEC, population=False, start=START)
    hash_before = live.field_canon_hash()

    segments = agent_walk(seed=seed, ticks=ticks, start=START.isoformat())
    rows = synthesize_rows(DREAM_SPEC, segments, seed=seed)
    batches = list(runner_batches(rows, flush_secs=30.0))

    # Dream on a CLONE path: separate host, mutated cassette, shadow only
    dream = GameHost()
    dream.register_world(DREAM_SPEC, population=False, start=START)
    places = [n.name for n in DREAM_SPEC.nodes if n.name.startswith("place_")]
    flip = f"scout_{places[0]}"
    n_cf = 0
    for readings, bt, conf in batches:
        mutated = _mutate_readings(readings, flip)
        dream.observe_many("dreamer", mutated, confidence=conf, t=bt)
        # may intend in shadow only
        dream.intend(
            "dreamer",
            Intent(actor_id="dreamer", name="claim_safe", shadow=True),
        )
        n_cf += 1

    live_dispatches = len(dream.live_dispatches) + len(live.live_dispatches)
    # Live field untouched by dream clone
    field_changed = live.field_canon_hash() != hash_before
    # Baseline to beat: a "dream" that actuates live would dispatch > 0
    beats = live_dispatches == 0 and n_cf > 0 and not field_changed
    return BaselineReport(
        kit="dream",
        live_dispatches=live_dispatches,
        counterfactual_batches=n_cf,
        field_changed=field_changed,
        beats_baseline=beats,
        honesty=HONESTY_TIER,
    )
