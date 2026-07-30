"""The unified belief GAUGE — value + confidence + reliability + forecast_skill.

Generalizes the benchmarking gym's four independently-derived learnings back into the
engine's canonical readout: host.api.beliefs() (and the hearth /beliefs endpoint) now
carry, per belief, the calibrated value, the Bloch-radius confidence, the learned
reliability (ObservationTrust), and — when a forecast surface is attached — the
forecast skill. All additive: value/confidence unchanged; new fields default None.
"""
from datetime import datetime, timedelta

from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec, OutputSpec
from umwelt.host import GameHost


def _spec():
    return DomainSpec(
        name="d",
        nodes=(NodeSpec("root", parent=None, kind="root", roles=("level",)),
               NodeSpec("a", parent="root", roles=("level",))),
        bindings=(BindingSpec("a_in", zone="a", role="level", normalizer="binary",
                              force_observe=True, collapse_alpha=0.5),),
        outputs=(OutputSpec("a_out", node="a", role="level"),))


def _host():
    h = GameHost()
    h.register_world(_spec(), population=False)
    return h


def _feed(h, values):
    t = datetime(2026, 7, 20, 9, 0)
    for i, v in enumerate(values):
        h.engine.ingest(sensor_readings={"a_in": v}, now=t + timedelta(minutes=5 * i))


def test_role_gauge_matches_beliefs():
    h = _host()
    _feed(h, [1] * 5)
    value, conf = h.engine.field.clusters["a"].role_gauge("level")
    b = h.beliefs()["a.level"]
    assert abs(b.value - value) < 1e-9 and abs(b.confidence - conf) < 1e-9


def test_reliability_is_populated_and_tracks_consistency():
    consistent = _host(); _feed(consistent, [1] * 8)     # obs sits on belief -> reliable
    noisy = _host(); _feed(noisy, [1, 0] * 4)             # obs swings -> noisy
    r_consistent = consistent.beliefs()["a.level"].reliability
    r_noisy = noisy.beliefs()["a.level"].reliability
    assert r_consistent is not None and r_noisy is not None
    assert r_consistent > r_noisy, (r_consistent, r_noisy)


def test_obs_trust_always_on_even_without_learn_collapse():
    # LEARN_COLLAPSE is off by default; reliability must STILL be tracked/readable.
    import umwelt.learning.observation_trust as ot_mod
    assert ot_mod.LEARN_COLLAPSE is False
    h = _host(); _feed(h, [1] * 4)
    assert getattr(h.engine, "_obs_trust", None) is not None
    assert h.beliefs()["a.level"].reliability is not None


def test_forecast_skill_none_without_surface():
    h = _host(); _feed(h, [1] * 4)
    # no forecast surface attached -> forecast_skill is cleanly absent, not an error
    assert h.beliefs()["a.level"].forecast_skill is None


def test_value_confidence_still_calibrated_backcompat():
    h = _host(); _feed(h, [1] * 6)
    b = h.beliefs()["a.level"]
    assert 0.0 <= b.value <= 1.0 and 0.0 <= b.confidence <= 1.0
    assert b.value > 0.5  # pushed toward the observed pole
