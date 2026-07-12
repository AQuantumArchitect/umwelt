"""Wall-clock pacing + the blank mixed-belief floor — the sparse-cadence levers.

Pins the two fixes the first foreign-cadence world (daily market bars) forced:
  1. spec.tick_s honors the silence between sparse batches as bounded free-evolution
     catch-up — OFF by default and byte-identical to the origin behavior when off;
  2. a blank boot starts ANALOG dissipative beliefs maximally mixed (the ground pole
     is a false certainty for a continuous quantity), while event/unitary roles keep
     their semantic ground start.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from umwelt.boot import build_engine
from umwelt.spec import roles
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec

T0 = datetime(2026, 1, 5, 21, 0, tzinfo=timezone.utc)


def _register_vocab():
    # idempotent: a continuous analog reading + an event role, engine-neutral names
    if roles.role_input_mode("t_level") != "dissipative" or not roles.is_analog_role("t_level"):
        roles.register_role_mode("t_level", "dissipative", analog=True)
    if roles.role_input_mode("t_ping") != "unitary":
        roles.register_role_mode("t_ping", "unitary")


def _spec(tick_s: float | None) -> DomainSpec:
    _register_vocab()
    return DomainSpec(
        name="pacing-world",
        nodes=(NodeSpec("top", parent=None, kind="root", roles=("t_level",)),
               NodeSpec("cell", parent="top", roles=("t_level", "t_ping"))),
        bindings=(BindingSpec("lvl", zone="cell", role="t_level",
                              normalizer={"type": "range", "lo": 0.0, "hi": 10.0}),
                  BindingSpec("png", zone="cell", role="t_ping", normalizer="binary")),
        tick_s=tick_s,
    )


def _z(engine, role="t_level"):
    return float(engine.field.clusters["cell"].role_bloch(role)[2])


def _drive(engine, days: int, gap_days: float = 1.0):
    for i in range(days):
        engine.ingest(sensor_readings={"lvl": 9.0},
                      now=T0 + timedelta(days=gap_days * i))
    return _z(engine)


def test_blank_boot_mixes_analog_dissipative_beliefs():
    engine = build_engine(spec=_spec(None), population=False)
    cluster = engine.field.clusters["cell"]
    level = cluster.role_bloch("t_level")
    ping = cluster.role_bloch("t_ping")
    assert abs(float(level[2])) < 1e-9, "analog dissipative belief must boot UNKNOWN"
    assert float(ping[2]) > 0.9, "event role keeps its semantic ground start"


def test_off_means_off_no_catchup_ever_runs():
    """tick_s=None (default): the pacing path provably never fires, whatever the wall
    gaps look like. (Wall time already reaches the LEARNING layers legitimately —
    calibration is time-aware — so the pin is the catch-up counter, not a hash.)"""
    engine = build_engine(spec=_spec(None), population=False)
    for i in range(6):
        engine.ingest(sensor_readings={"lvl": 9.0}, now=T0 + timedelta(days=i))
    assert engine._wall_catchup_steps == 0
    assert engine._last_bridged_inputs is None


def test_pacing_defeats_sparse_cadence_starvation():
    """The starvation pin: daily batches on a paced world (tick_s=1h, zero-order
    hold) drive the belief far further than the same batches tick-driven — and land
    near the dense-republish reference the market harness had to emulate by hand."""
    paced = build_engine(spec=_spec(3600.0), population=False)
    unpaced = build_engine(spec=_spec(None), population=False)
    z_paced = _drive(paced, days=6)
    z_unpaced = _drive(unpaced, days=6)
    assert z_paced < 0.0, "high reading must drive toward the excited pole"
    assert abs(z_paced) > 2.0 * abs(z_unpaced), (
        f"pacing did not defeat starvation: paced z={z_paced:+.3f} "
        f"unpaced z={z_unpaced:+.3f}")

    # the dense reference: the same reading delivered every simulated hour by hand
    dense = build_engine(spec=_spec(None), population=False)
    for i in range(6 * 24):
        dense.ingest(sensor_readings={"lvl": 9.0}, now=T0 + timedelta(hours=i))
    z_dense = _z(dense)
    assert abs(z_paced - z_dense) < 0.15, (
        f"zero-order hold should approximate the dense stream: "
        f"paced {z_paced:+.3f} vs dense {z_dense:+.3f}")


def test_hold_spans_one_gap_only():
    """A channel absent from the previous batch contributes nothing during the gap:
    after a batch with NO readings, the next gap free-evolves and the belief eases
    back toward uncertainty instead of riding a stale reading forever."""
    engine = build_engine(spec=_spec(3600.0), population=False)
    _drive(engine, days=6)
    settled = _z(engine)
    engine.ingest(sensor_readings={}, now=T0 + timedelta(days=7))   # silence batch
    engine.ingest(sensor_readings={}, now=T0 + timedelta(days=9))   # gap held EMPTY
    relaxed = _z(engine)
    assert abs(relaxed) < abs(settled), (
        f"a stale reading must not hold forever: |{settled:+.3f}| -> |{relaxed:+.3f}|")


def test_catchup_is_bounded():
    """A year-long gap costs at most _wall_catchup_max substeps (bounded compute)."""
    engine = build_engine(spec=_spec(3600.0), population=False)
    engine._wall_catchup_max = 8
    engine.ingest(sensor_readings={"lvl": 9.0}, now=T0)
    import time as _t
    t0 = _t.perf_counter()
    engine.ingest(sensor_readings={"lvl": 9.0}, now=T0 + timedelta(days=365))
    assert (_t.perf_counter() - t0) < 5.0
    assert engine._step > 0