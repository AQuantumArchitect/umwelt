"""examples/fledgeling_fog/world — Fledgeling-shaped fog corridor (Phase 1).

Public synthetic domain only: place nodes on a corridor graph, scout/probe
observations with η, unobserved beliefs relax (dissipative-friendly roles where
needed), tick-driven timestamps (not solar). Optional shadow claim/mark.

Deterministic: seeded RNG + fixed START. No private HA/meerkat data.
"""
from __future__ import annotations

import math
import random
from bisect import bisect_right
from datetime import datetime, timedelta, timezone

import numpy as np

from umwelt.spec.schema import (
    BindingSpec,
    BridgeSpec,
    DomainSpec,
    DriverSpec,
    NodeSpec,
    OutputSpec,
)

START = "2026-03-01T00:00:00+00:00"
TICK_S = 60.0  # game tick period — not a solar day
N_PLACES = 6


def place_names(n: int = N_PLACES) -> list[str]:
    return [f"place_{i}" for i in range(n)]


def corridor_edges(n: int = N_PLACES) -> list[tuple[str, str]]:
    """Linear corridor: place_0 — place_1 — … — place_{n-1}."""
    names = place_names(n)
    return [(names[i], names[i + 1]) for i in range(n - 1)]


def fog_corridor_spec(n_places: int = N_PLACES) -> DomainSpec:
    """Corridor DomainSpec: agent_near (unitary) + safe (dissipative) per place.

    Observation bindings: scout_{place} → agent_near with force_observe.
    Action: claim_safe on the far end in SHADOW only.
    Driver: harmonic tick at TICK_S (game cadence, not solar).
    """
    places = place_names(n_places)
    nodes = [NodeSpec("corridor", parent=None, kind="root", roles=("agent_near",))]
    for p in places:
        nodes.append(
            NodeSpec(
                p,
                parent="corridor",
                kind="region",
                roles=("agent_near", "safe"),
                role_modes={"agent_near": "unitary", "safe": "dissipative"},
                params={"gamma_diss": (0.04, 0.01, 0.001, 0.2)},
            )
        )
    bridges = tuple(
        BridgeSpec(a, b, shared_roles=("agent_near",), kind="open")
        for a, b in corridor_edges(n_places)
    )
    bindings = tuple(
        BindingSpec(
            f"scout_{p}",
            zone=p,
            role="agent_near",
            normalizer="binary",
            force_observe=True,
            efficiency=1.0,
            strength=0.35,
        )
        for p in places
    )
    far = places[-1]
    outputs = (
        OutputSpec(
            "claim_safe",
            node=far,
            role="safe",
            kind="binary",
            decode="sticky",
            gates={"rate_limit_s": 0.0},
            dispatch={"actuator_id": "claim_mark_1"},
            shadow=True,
        ),
    )
    return DomainSpec(
        name=f"fledgeling-fog-{n_places}",
        nodes=tuple(nodes),
        bridges=bridges,
        bindings=bindings,
        outputs=outputs,
        drivers=(DriverSpec("tick", node="_clock", role="phase", period_s=TICK_S),),
        anchors={},
    )


# Concrete instance for `python -m umwelt.spec.validate examples.fledgeling_fog.world:FOG_SPEC`
FOG_SPEC = fog_corridor_spec()


def agent_walk(
    n_places: int = N_PLACES,
    *,
    seed: int = 11,
    ticks: int = 240,
    start: str = START,
    mean_dwell_ticks: float = 8.0,
) -> list[tuple[datetime, datetime, str]]:
    """Seeded walk over the corridor as dwell segments (t_start, t_end, place)."""
    rng = random.Random(seed)
    places = place_names(n_places)
    adj: dict[str, list[str]] = {p: [] for p in places}
    for a, b in corridor_edges(n_places):
        adj[a].append(b)
        adj[b].append(a)
    t0 = datetime.fromisoformat(start)
    t_end = t0 + timedelta(seconds=ticks * TICK_S)
    t, cell = t0, places[0]
    segments: list[tuple[datetime, datetime, str]] = []
    while t < t_end:
        dwell = timedelta(seconds=rng.uniform(0.5, 1.5) * mean_dwell_ticks * TICK_S)
        seg_end = min(t + dwell, t_end)
        segments.append((t, seg_end, cell))
        cell = rng.choice(adj[cell])
        t = seg_end
    return segments


def occupied_place(segments, t: datetime) -> str:
    starts = [s for s, _, _ in segments]
    i = max(0, bisect_right(starts, t) - 1)
    return segments[i][2]


def last_vacated(
    segments, t: datetime, places: list[str] | None = None
) -> dict[str, float]:
    out: dict[str, float] = {p: math.inf for p in (places or ())}
    for s, e, place in segments:
        if s > t:
            break
        out[place] = max(0.0, (t - min(e, t)).total_seconds())
    return out


def synthesize_rows(
    spec: DomainSpec,
    segments,
    *,
    seed: int = 11,
    step_s: float | None = None,
    scout_eta: float = 1.0,
    heartbeat_p: float = 0.20,
) -> list[tuple[str, str, str, None]]:
    """Wire-shaped stream: (ts_iso, sensor_id, value_str, None).

    Sightings: TRUE on entry, FALSE on vacate, plus heartbeat republish.
    scout_eta is carried as confidence by the host path; raw rows stay value-only
    (confidence applied at ingest by the host/demo).
    """
    del scout_eta  # documented for callers; host maps it to confidence
    rng = random.Random(seed)
    step = float(step_s if step_s is not None else TICK_S)
    bound = {b.sensor_id for b in (spec.bindings or ())}
    places = [n.name for n in spec.nodes if n.name.startswith("place_")]
    rows: list[tuple[str, str, str, None]] = []

    def emit(t: datetime, sid: str, value: float) -> None:
        if sid in bound:
            rows.append((t.isoformat(), sid, f"{value:.4f}", None))

    t0, t_end = segments[0][0], segments[-1][1]
    first = occupied_place(segments, t0)
    for p in places:
        emit(t0, f"scout_{p}", 1.0 if p == first else 0.0)

    prev = first
    n_steps = int((t_end - t0).total_seconds() // step)
    for i in range(n_steps):
        t = t0 + timedelta(seconds=i * step)
        here = occupied_place(segments, t)
        if here != prev:
            emit(t, f"scout_{prev}", 0.0)
            emit(t, f"scout_{here}", 1.0)
        prev = here
        for p in places:
            if rng.random() < heartbeat_p:
                emit(t, f"scout_{p}", 1.0 if p == here else 0.0)
    return rows


def runner_batches(rows, flush_secs: float = 30.0):
    from umwelt.events import replay_sensor_batches

    return (
        (readings, bt, conf)
        for bt, readings, conf, _last in replay_sensor_batches(rows, flush_secs=flush_secs)
    )


def binned_truth(segments, places: list[str], *, bin_s: float = 60.0) -> np.ndarray:
    t0, t_end = segments[0][0], segments[-1][1]
    T = int((t_end - t0).total_seconds() // bin_s)
    truth = np.full((T, len(places)), -1.0)
    idx = {p: k for k, p in enumerate(places)}
    for t in range(T):
        place = occupied_place(segments, t0 + timedelta(seconds=t * bin_s))
        if place in idx:
            truth[t, idx[place]] = 1.0
    return truth


def freeze_baseline_scores(
    truth: np.ndarray, reports: np.ndarray
) -> np.ndarray:
    """Persistence-of-last-input: last non-NaN report freezes forever (or -1)."""
    out = np.full_like(truth, -1.0)
    last = np.full(truth.shape[1], -1.0)
    for t in range(truth.shape[0]):
        for j in range(truth.shape[1]):
            v = reports[t, j]
            if not np.isnan(v):
                last[j] = v
        out[t] = last
    return out


def sparse_scout_reports(
    truth: np.ndarray, *, report_p: float = 0.35, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    reports = np.full_like(truth, np.nan)
    mask = rng.random(truth.shape) < report_p
    reports[mask] = truth[mask]
    return reports
