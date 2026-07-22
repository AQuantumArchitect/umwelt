"""The self-forecast seam: a spec that declares `forecast_leaves` gets a live self-forecast
organ for free, and it honestly discriminates the facets it can foresee from the ones it can't.

Pins the shipped wiring (build_engine auto-attaches, ingest auto-steps) so the dormant-organ
regression (`DEFAULT_FORECAST_LEAVES` empty, nothing attaches) can't silently return.
"""
from umwelt.boot import build_engine
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec
from umwelt.projection.cognifold_trace import cognifold_trace
from examples.shaman_self.world import FACETS, SHAMAN_SELF_SPEC, self_signal_stream


def test_forecast_leaves_autoattach_from_spec():
    """A spec that declares forecast_leaves → build_engine attaches the organ over exactly those."""
    eng = build_engine(spec=SHAMAN_SELF_SPEC, population=False)
    assert eng.forecast_surface is not None
    assert set(eng.forecast_surface.leaves) == set(SHAMAN_SELF_SPEC.forecast_leaves)


def test_no_forecast_leaves_no_organ():
    """A spec WITHOUT forecast_leaves → no organ, byte-identical to a non-foreseeing world."""
    spec = DomainSpec(
        name="plain",
        nodes=(NodeSpec("root", parent=None, kind="root", roles=("x",)),),
        bindings=(BindingSpec("s_x", zone="root", role="x",
                              normalizer="binary", force_observe=True),),
    )
    eng = build_engine(spec=spec, population=False)
    assert eng.forecast_surface is None


def test_self_foresees_itself_and_honestly_discriminates():
    """Ingest alone (no manual surface stepping) drives the organ; the field foresees its
    steady facets far better than its near-white one, and the cognifold trace carries the ladder."""
    eng = build_engine(spec=SHAMAN_SELF_SPEC, population=False, calibration=True, fractal=True)
    for now, readings in self_signal_stream(seed=7, ticks=900, dt_min=2.0):
        eng.ingest(sensor_readings=readings, now=now)   # ingest AUTO-steps the organ

    preds = eng.forecast_surface.predictions()
    assert preds, "ingest did not step the organ — no live forecast rungs"

    trace = cognifold_trace(eng, world="shaman_self")
    regs_fc = [r for r in trace["registers"] if r.get("forecast")]
    assert len(regs_fc) == len(FACETS), f"{len(regs_fc)} of {len(FACETS)} facets carry a ladder"

    best = {r["role"]: max(e["skill"] for e in r["forecast"])
            for r in trace["registers"] if r.get("forecast")}
    # Honest discrimination: the steady facets are foreseen; the near-white one is not.
    # Margins are generous (measured: conviction≈0.94, comprehension≈0.91, momentum≈0.63).
    assert best["conviction"] > best["momentum"] + 0.15, best
    assert best["comprehension"] > best["momentum"] + 0.10, best
