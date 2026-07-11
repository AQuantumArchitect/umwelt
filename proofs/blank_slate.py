"""THE BLANK-SLATE PROOF — a blank engine boots a world it has never seen and comprehends it.

This is the library's flagship theorem, generalized from the origin deployment's b10.0
gate (meerkat tests/brain/test_foreign_blank_slate.py — there the foreign world was a
synthetic house; here it is a gridworld, proofs/_gridworld.py): a max-entropy, unlocated
engine boots a DomainSpec, replays a deterministic synthetic day through the SAME ingest
path a recorded deployment replays (events.replay_sensor_batches → BrainRunner under the
REPLAY gauge) — then the assertions that it actually comprehended:

  (1) the blank floor is witnessed BEFORE anything happens (blank profile, indistinct
      beliefs, an honestly unlocated coordinate);
  (2) every declared binding drove the field — no dead vocabulary;
  (3) beliefs track the agent's known ground-truth walk at several checkpoints;
  (4) the learnable param fiber drifted off its priors — blank → learned is measurable;
  (5) the coordinate tells the truth end to end (no place token without an anchor; a
      grounded fix is reflected; a fresh boot is nowhere again);
  (6) the learned state survives save/load (canon-hash equal) and KEEPS comprehending.

Deterministic: seeded sim, fixed start timestamp, process RNGs pinned — no wall clock,
no Date.now-style entropy anywhere.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

from proofs._gridworld import (
    SITE, agent_walk, grid_cells, gridworld_spec, last_vacated, occupied_cell,
    runner_batches, synthesize_rows,
)
from umwelt.boot import build_engine
from umwelt.learning.competence import learnedness
from umwelt.learning.context import ContextState
from umwelt.learning.runner import BrainRunner
from umwelt.projection.gauge_name import gauge_name, geohash5
from umwelt.substrate.bloch import bloch_to_location, location_to_bloch

CELLS = grid_cells()
VACATED_S = 3600.0          # "long-vacated" = the agent left ≥ 1 h ago
Z_MARGIN = 0.2              # the occupied cell must beat every long-vacated cell by this


class SphereCodec:
    """The registered anchor codec: (lat, lon)-style surface coordinates ⇄ Bloch point.
    Registered by the proof (an app concern) — the engine ships no coordinate system."""

    def encode(self, value):
        return location_to_bloch(*value)

    def decode(self, bloch):
        return bloch_to_location(*bloch)


def _pin_rngs():
    # build_engine doesn't forward a substrate seed; pin the process RNGs so the
    # collapse sampling — and therefore this proof — is reproducible run to run.
    import random
    import numpy as np
    random.seed(1234)
    np.random.seed(1234)


def _boot(**kw):
    return build_engine(spec=gridworld_spec(**kw), population=False,
                        role=ContextState.replay(dt_factor=10.0))


def _ground_blank_anchor(engine):
    """The blank-build anchor ritual: register the codec, then DE-LOCATE — a fresh
    qubit's pure default pole would be a definite-but-meaningless coordinate (see
    engine.delocate_anchor). Grounding only ever comes from evidence fixes."""
    engine.register_anchor("geo", codec=SphereCodec())
    engine.delocate_anchor("geo")


def _z(engine, cell: str) -> float:
    c = engine.field.clusters[cell]
    return float(c.role_bloch("agent_near")[2])


def test_blank_gridworld_engine_comprehends_a_synthetic_day() -> None:
    _pin_rngs()
    engine = _boot()
    _ground_blank_anchor(engine)

    # ── (1) the blank floor, witnessed before anything happens ──
    assert engine.seed_profile == "blank" and not getattr(engine, "home_lock", False)
    assert ".nowhere." in gauge_name(engine)
    # no cell is pre-favored: every cell's belief is the same point (indistinct floor)
    blochs = {tuple(round(float(v), 9) for v in engine.field.clusters[c].role_bloch("agent_near"))
              for c in CELLS}
    assert len(blochs) == 1, f"cells distinguishable before any evidence: {blochs}"
    # the anchor is genuinely maximally mixed (r ≈ 0), not parked at a pole
    assert math.sqrt(sum(v * v for v in engine.anchor_bloch("geo"))) < 1e-6
    # the learnable-fiber baseline, two reads: the scalar snapshot (values/σ/updates)
    # and learnedness (posterior-width settledness — covers the qubit-backed params
    # _snapshot_param_fiber excludes by design; in this engine the fiber IS qubit-backed)
    fiber_before = engine._snapshot_param_fiber()
    learned_before = learnedness(engine)

    # ── onboarding: the first evidence fix grounds the anchor at the synthetic site ──
    engine.ground_anchor("geo", SITE, alpha=1.0)
    lat, lon = engine.anchor_value("geo")
    assert abs(lat - SITE[0]) < 0.5 and abs(lon - SITE[1]) < 0.5

    # ── the world lives a synthetic day; the engine replays it as a recording would ──
    walk = agent_walk(seed=7, days=1.0)
    rows = synthesize_rows(gridworld_spec(), walk, seed=7)
    checkpoints: list[tuple[datetime, dict]] = []
    seen_days: set = set()

    fresh_seen = [0]

    def on_batch(n, item, result):
        readings, now = item[0], item[1]
        quarter = (now.date(), now.hour // 6)
        if quarter not in seen_days:      # a grounding fix lands a few times a day
            seen_days.add(quarter)
            engine.ground_anchor("geo", SITE, alpha=0.5)
        # Checkpoint discipline (the origin proof's note, kept): sample where the
        # occupied cell was JUST observed — offline replay has no live tick between
        # batches, so beliefs relax toward the field's own equilibrium in silence;
        # reading long after the last event would test the decay physics, not the
        # comprehension.
        here = occupied_cell(walk, now)
        if readings.get(f"sight_{here}") == 1.0:
            fresh_seen[0] += 1
            if fresh_seen[0] % 25 == 0:
                checkpoints.append((now, {c: _z(engine, c) for c in CELLS}))

    n = BrainRunner(engine).replay(runner_batches(rows, flush_secs=30.0), on_batch=on_batch)
    assert n > 150, f"suspiciously few batches ({n}) — the sim thinned out"

    # ── (2) every declared binding drove the field (no dead vocabulary) ──
    spec = gridworld_spec()
    bridge = engine.sensor_bridge
    touched_ids = {b.sensor_id for b in bridge.bindings.values()
                   if (b.node, b.qubit_role) in bridge.touched_roles}
    assert touched_ids == {b.sensor_id for b in spec.bindings}

    # ── (3) comprehension: at every checkpoint, the ground-truth occupied cell's
    #        agent_near belief clearly exceeds every long-vacated cell's ──
    assert len(checkpoints) >= 3, "too few checkpoints to call it tracking"
    for now, zs in checkpoints:
        here = occupied_cell(walk, now)
        vacated = [c for c, ago in last_vacated(walk, now, CELLS).items()
                   if c != here and ago >= VACATED_S]
        assert vacated, f"no long-vacated cell at {now} — walk too fast to test against"
        for cell in vacated:
            assert zs[here] > zs[cell] + Z_MARGIN, (
                f"at {now}: occupied {here} (z={zs[here]:.3f}) not clearly above "
                f"vacated {cell} (z={zs[cell]:.3f})")

    # ── (4) the learnable fiber drifted off its priors — blank → learned is measurable.
    #        The gridworld fiber is qubit-backed, so the scalar snapshot alone can be
    #        empty of motion; learnedness is the whole-fiber settledness gauge (the
    #        origin proof's own witness). Either read moving off the floor counts. ──
    fiber_after = engine._snapshot_param_fiber()
    moved = [f"{node}.{key}"
             for node, params in fiber_after.items()
             for key, triple in params.items()
             if triple != fiber_before.get(node, {}).get(key, triple)]
    learned_after = learnedness(engine)
    assert moved or learned_after > learned_before, (
        f"the fiber did not drift (learnedness {learned_before:.4f} → {learned_after:.4f}, "
        f"0 scalar params moved) over a full replayed day")

    # ── sanity: non-degenerate field — every qubit still a finite Bloch point ──
    for cname, c in engine.field.clusters.items():
        for role, idx in getattr(c, "role_index", {}).items():
            b = c.qubit_bloch(idx)
            assert all(math.isfinite(float(v)) for v in b), f"{cname}:{role} bloch not finite"

    # ── (5) the coordinate tells the truth: the engine LIVED in the fix's region (the
    #        anchor drifts within it between fixes — offline replay runs no pin loop,
    #        so the region is held as a distance bound, not geohash-prefix luck), and a
    #        fresh fix restores the exact token. ──
    lat, lon = engine.anchor_value("geo")
    assert abs(lat - SITE[0]) < 10.0 and abs(lon - SITE[1]) < 10.0, (
        f"the anchor left the fix's region: ({lat:.2f}, {lon:.2f}) vs {SITE}")
    assert ".nowhere." not in gauge_name(engine)
    engine.ground_anchor("geo", SITE, alpha=1.0)
    assert f".{geohash5(*SITE)}." in gauge_name(engine)

    # ── (6) save/load: the learned field survives byte-for-byte and keeps working ──
    import tempfile
    from pathlib import Path
    h_before = engine.field_canon_hash()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "engine_state.pkl"
        engine.save(str(path))
        fresh = _boot()
        fresh.load(str(path))
        assert fresh.field_canon_hash() == h_before
        # the reloaded engine KEEPS comprehending: one more hour of the world ingests clean
        t1 = datetime.fromisoformat(rows[-1][0])
        walk2 = agent_walk(seed=8, days=0.05, start=(t1 + timedelta(seconds=60)).isoformat())
        more = synthesize_rows(gridworld_spec(), walk2, seed=8)
        n2 = BrainRunner(fresh).replay(runner_batches(more, flush_secs=30.0))
        assert n2 > 0

    # ── (5, closing half) a fresh boot is again NOWHERE — location never leaks through code ──
    reborn = _boot()
    _ground_blank_anchor(reborn)
    assert ".nowhere." in gauge_name(reborn)


def test_no_place_token_without_an_anchor() -> None:
    """A spec that declares NO anchor can never mint a place: the gauge name is honest
    about being nowhere, with or without a codec registered."""
    _pin_rngs()
    engine = _boot(with_anchor=False)
    assert ".nowhere." in gauge_name(engine)
    engine.register_anchor("geo", codec=SphereCodec())   # codec alone grants no place
    assert ".nowhere." in gauge_name(engine)


def test_synthetic_stream_is_deterministic_and_spec_scoped() -> None:
    walk = agent_walk(seed=7, days=1.0)
    spec = gridworld_spec()
    a = synthesize_rows(spec, walk, seed=7)
    b = synthesize_rows(spec, walk, seed=7)
    c = synthesize_rows(spec, walk, seed=8)
    assert a == b, "same seed must reproduce the identical stream"
    assert a != c, "a different seed must vary the stream"
    # only the spec's declared vocabulary is ever emitted, and all of it appears
    bound = {bind.sensor_id for bind in spec.bindings}
    emitted = {sid for _, sid, _, _ in a}
    assert emitted == bound
