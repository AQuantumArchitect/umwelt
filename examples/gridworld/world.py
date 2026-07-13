"""examples/gridworld/world — a self-contained gridworld domain + a deterministic
synthetic day.

This is the canonical new-domain template (see docs/NEW_DOMAIN.md) AND the proof
gate's own fixture: `proofs/blank_slate.py` and friends import this module rather
than owning a copy, so the domain you'd copy to start your own world is exactly the
one the gate proves comprehension on — no separate "example" vs "test fixture" drift.
Generalized from the origin proof's simulator (meerkat/brain/house_sim.py): there the
domain was a synthetic foreign house and the one domain coupling was a celestial
ephemeris import; here the domain is an N×M grid of cells walked by one agent, and
every ambient signal rides the spec's own harmonic driver phase — no astronomy, no
geography, no vendor.

Three exports matter:

    gridworld_spec()      — the DomainSpec: 3×3 cells (agent_near unitary + resource
                            dissipative), orthogonal-adjacency bridges, a few shadow
                            OutputSpecs, one harmonic DriverSpec, and an anchor qubit
                            node (`_geo`) the gauge-honesty assertions exercise.
    agent_walk()          — the ground truth: a seeded random walk over the grid
                            adjacency as (t_start, t_end, cell) dwell segments.
    synthesize_rows()     — the stream: event rows in the exact shape
                            events.replay_sensor_batches buckets
                            ((ts_iso, sensor_id, value_str, None)), emitting ONLY the
                            spec's declared sensor vocabulary.

Deterministic by construction: seeded random.Random, explicit start timestamp — same
(grid, seed, days) → byte-identical rows. No wall clock anywhere.
"""
from __future__ import annotations

import math
import random
from bisect import bisect_right
from datetime import datetime, timedelta, timezone

import numpy as np

from umwelt.spec.schema import (
    BindingSpec, BridgeSpec, DomainSpec, DriverSpec, NodeSpec, OutputSpec,
)

START = "2026-01-05T00:00:00+00:00"
DAY_S = 86400.0                     # the harmonic driver's period (one synthetic day)

# The synthetic site the anchor assertions ground to — an arbitrary coordinate pair on
# the anchor codec's sphere. It means nothing; that is the point (gauge honesty is
# about never minting a coordinate that wasn't given).
SITE = (25.0, 55.0)


# ── the grid ────────────────────────────────────────────────────────────────────────

def grid_cells(rows: int = 3, cols: int = 3) -> list[str]:
    return [f"cell_{r}_{c}" for r in range(rows) for c in range(cols)]


def grid_adjacency(rows: int = 3, cols: int = 3) -> list[tuple[str, str]]:
    """Orthogonal-neighbor pairs — the bridge topology and the walk graph."""
    pairs = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                pairs.append((f"cell_{r}_{c}", f"cell_{r}_{c + 1}"))
            if r + 1 < rows:
                pairs.append((f"cell_{r}_{c}", f"cell_{r + 1}_{c}"))
    return pairs


def gridworld_spec(rows: int = 3, cols: int = 3, *, with_anchor: bool = True) -> DomainSpec:
    """The proof-gate domain. Each cell carries `agent_near` (unitary, event-driven
    sightings) and `resource` (dissipative, continuous level). Outputs stay in SHADOW
    (the law: a new world decides visibly and dispatches nothing). One harmonic driver
    is the world's day. `with_anchor` adds the `_geo` anchor qubit node the
    gauge-honesty assertions ground; a spec without it is honestly UN-GROUNDED."""
    cells = grid_cells(rows, cols)
    nodes = [NodeSpec("grid", parent=None, kind="root", roles=("agent_near",))]
    for cell in cells:
        nodes.append(NodeSpec(
            cell, parent="grid", roles=("agent_near", "resource"),
            role_modes={"agent_near": "unitary", "resource": "dissipative"},
            params={"gamma": (0.04, 0.01, 0.001, 0.2)},
        ))
    if with_anchor:
        # The anchor qubit: one (node, role) holding a slowly-grounded coordinate.
        # Declared in the topology; the blank build de-locates it (a fresh qubit's
        # pure pole would be a definite-but-meaningless coordinate — see
        # engine.delocate_anchor).
        nodes.append(NodeSpec("_geo", parent="grid", kind="clock", roles=("geo",),
                              role_modes={"geo": "unitary"}))
    bridges = tuple(BridgeSpec(a, b, shared_roles=("agent_near",), kind="open")
                    for a, b in grid_adjacency(rows, cols))
    bindings = tuple(
        BindingSpec(f"sight_{cell}", zone=cell, role="agent_near", normalizer="binary",
                    force_observe=True)
        for cell in cells
    ) + tuple(
        BindingSpec(f"resource_{cell}", zone=cell, role="resource",
                    normalizer={"type": "range", "lo": 0.0, "hi": 10.0})
        for cell in cells
    )
    center = cells[len(cells) // 2]
    outputs = (
        OutputSpec("harvest", node=center, role="resource", kind="binary",
                   decode="sticky", gates={"rate_limit_s": 0.0},
                   dispatch={"actuator_id": "harvester_1"}),          # shadow (default)
        OutputSpec("beacon", node=cells[0], role="agent_near", kind="scalar",
                   decode="linear", codomain=(0.0, 100.0),
                   gates={"rate_limit_s": 0.0, "deadband": 2.0}),     # shadow (default)
    )
    return DomainSpec(
        name=f"gridworld-{rows}x{cols}",
        nodes=tuple(nodes),
        bridges=bridges,
        bindings=bindings,
        outputs=outputs,
        drivers=(DriverSpec("day", node="_clock", role="phase", period_s=DAY_S),),
        anchors={"geo": {"note": "synthetic site; grounded by the proof, never assumed"}}
                if with_anchor else {},
    )


# ── the ground truth: a seeded agent walk ───────────────────────────────────────────

def agent_walk(rows: int = 3, cols: int = 3, *, seed: int = 7, days: float = 1.0,
               start: str = START, mean_dwell_s: float = 2400.0,
               ) -> list[tuple[datetime, datetime, str]]:
    """One agent's seeded random walk over the grid adjacency, as dwell segments
    (t_start, t_end, cell), covering [start, start + days). Deterministic in
    (rows, cols, seed, days, start, mean_dwell_s)."""
    rng = random.Random(seed)
    adj: dict[str, list[str]] = {}
    for a, b in grid_adjacency(rows, cols):
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    t0 = datetime.fromisoformat(start)
    t_end = t0 + timedelta(seconds=days * DAY_S)
    t, cell = t0, grid_cells(rows, cols)[0]
    segments: list[tuple[datetime, datetime, str]] = []
    while t < t_end:
        dwell = timedelta(seconds=rng.uniform(0.5, 1.5) * mean_dwell_s)
        seg_end = min(t + dwell, t_end)
        segments.append((t, seg_end, cell))
        cell = rng.choice(adj[cell])
        t = seg_end
    return segments


def occupied_cell(segments, t: datetime) -> str:
    """The ground-truth occupied cell at time t (segments are contiguous + sorted)."""
    starts = [s for s, _, _ in segments]
    i = max(0, bisect_right(starts, t) - 1)
    return segments[i][2]


def last_vacated(segments, t: datetime, cells: list[str] | None = None) -> dict[str, float]:
    """Per cell: seconds since the agent last LEFT it, as of t. Cells never visited by
    t map to +inf; the currently-occupied cell maps to 0."""
    out: dict[str, float] = {c: math.inf for c in (cells or ())}
    for s, e, cell in segments:
        if s > t:
            break
        out[cell] = max(0.0, (t - min(e, t)).total_seconds())
    return out


# ── the ambient signal: the driver's own phase (no ephemeris import) ────────────────

def driver_ambient(t: datetime, period_s: float = DAY_S) -> float:
    """The world's 'daylight' ∈ [0, 1] — computed from the SAME harmonic phase the
    spec's driver anchors (clocks/drivers.HarmonicDriver: epoch 2000-01-01, phase =
    elapsed/period mod 1). The origin sim imported an ephemeris here; the gridworld's
    only sky is its own declared driver."""
    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    phase = ((t - epoch).total_seconds() / period_s) % 1.0
    return max(0.0, math.sin(2.0 * math.pi * (phase - 0.25)))


# ── the stream ──────────────────────────────────────────────────────────────────────

def synthesize_rows(spec: DomainSpec, segments, *, seed: int = 7, step_s: int = 120,
                    ) -> list[tuple[str, str, str, None]]:
    """The synthetic stream: rows (ts_iso, sensor_id, value_str, None) sorted by time,
    ready for events.replay_sensor_batches. Values ride the spec's declared sensor
    vocabulary; anything the spec doesn't bind is simply never emitted.

    Wire realism follows the origin sim: sightings have the event shape — TRUE at
    entry, FALSE at vacate — plus a heartbeat: each sighting sensor occasionally
    republishes its CURRENT state (a polled detector bank), and each bank reports its
    state once at boot, so every declared signal exists on the wire from t0. The
    heartbeat keeps the comprehension readout near fresh observations — an offline
    replay has no live tick between batches, so beliefs relax toward the field's own
    equilibrium in silence; reading hours after the last event would test the decay
    physics, not the comprehension (the origin proof's note, kept)."""
    rng = random.Random(seed)
    bound = {b.sensor_id for b in (spec.bindings or ())}
    cells = [n.name for n in spec.nodes if n.name.startswith("cell_")]
    rows: list[tuple[str, str, str, None]] = []

    def emit(t: datetime, sid: str, value: float) -> None:
        if sid in bound:
            rows.append((t.isoformat(), sid, f"{value:.4f}", None))

    t0, t_end = segments[0][0], segments[-1][1]
    # boot report: every sighting sensor announces its state once (real detector banks
    # report on join) — the walk then narrates every change.
    first = occupied_cell(segments, t0)
    for cell in cells:
        emit(t0, f"sight_{cell}", 1.0 if cell == first else 0.0)

    prev = first
    n_steps = int((t_end - t0).total_seconds() // step_s)
    for i in range(n_steps):
        t = t0 + timedelta(seconds=i * step_s)
        here = occupied_cell(segments, t)
        if here != prev:
            emit(t, f"sight_{prev}", 0.0)               # vacated: sighting falls
            emit(t, f"sight_{here}", 1.0)               # entered: sighting rises
        prev = here
        for cell in cells:                              # heartbeat state republish
            if rng.random() < 0.15:
                emit(t, f"sight_{cell}", 1.0 if cell == here else 0.0)

        if i % 5 == 0:                                  # 10-min continuous cadence
            ambient = driver_ambient(t)
            for cell in cells:
                level = (1.0 + 6.0 * ambient
                         + (2.0 if cell == here else 0.0)
                         + rng.gauss(0.0, 0.15))
                emit(t, f"resource_{cell}", min(10.0, max(0.0, level)))
    return rows


def runner_batches(rows, flush_secs: float = 30.0):
    """Synthetic rows → the `(readings, now, conf)` batches BrainRunner.replay consumes
    — the SAME bucketing a recorded deployment replays through (events.
    replay_sensor_batches), so the proof exercises the production ingest path."""
    from umwelt.events import replay_sensor_batches
    return ((readings, bt, conf)
            for bt, readings, conf, _last in replay_sensor_batches(rows, flush_secs=flush_secs))


# ── binned matrices (for the estimator-ladder / deconfound harnesses) ───────────────

def binned_truth(segments, cells: list[str], *, bin_s: float = 300.0) -> np.ndarray:
    """Ground truth as a (T, n_cells) ±1 matrix at a fixed bin cadence — the shape the
    ported evidence harnesses score against."""
    t0, t_end = segments[0][0], segments[-1][1]
    T = int((t_end - t0).total_seconds() // bin_s)
    truth = np.full((T, len(cells)), -1.0)
    idx = {c: k for k, c in enumerate(cells)}
    for t in range(T):
        cell = occupied_cell(segments, t0 + timedelta(seconds=t * bin_s))
        if cell in idx:
            truth[t, idx[cell]] = 1.0
    return truth


def sparse_reports(truth: np.ndarray, *, report_p: float = 0.35, seed: int = 0) -> np.ndarray:
    """Sparse sighting reports off the truth: each cell reports its ±1 state with
    probability report_p per bin, NaN otherwise — the sparse reality of a wire."""
    rng = np.random.default_rng(seed)
    reports = np.full_like(truth, np.nan)
    mask = rng.random(truth.shape) < report_p
    reports[mask] = truth[mask]
    return reports
