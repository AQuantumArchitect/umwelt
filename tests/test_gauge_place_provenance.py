"""Place is provenance, not radius.

gauge_name's place segment may only be minted by an anchor that was GROUNDED by
evidence (engine.ground_anchor) or restored from an artifact carrying a fix. An
un-grounded anchor qubit that merely DRIFTS off maximally-mixed under field dynamics
must keep reading `nowhere` — drift is not a coordinate anyone gave the engine.
Found by a 13-day real-house replay: the geo qubit crossed the r=0.5 radius gate and
the gauge minted a geohash for a house whose location was genuinely unknown.
"""
from __future__ import annotations

from umwelt.boot import build_engine
from umwelt.projection.gauge_name import gauge_name
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec
from umwelt.substrate.bloch import bloch_to_location, location_to_bloch


def _spec() -> DomainSpec:
    return DomainSpec(
        name="provenance-world",
        nodes=(
            NodeSpec("world", parent=None, kind="root", roles=("presence_signal",)),
            NodeSpec("region_a", parent="world", roles=("presence_signal",),
                     role_modes={"presence_signal": "unitary"}),
            NodeSpec("_geo", parent="world", kind="clock", roles=("geo",),
                     role_modes={"geo": "unitary"}),
        ),
        bindings=(BindingSpec("sig_a", zone="region_a", role="presence_signal",
                              normalizer="binary", force_observe=True),),
        anchors={"geo": {"note": "declared, never assumed"}},
    )


class _SphereCodec:
    def encode(self, value):
        return location_to_bloch(*value)

    def decode(self, bloch):
        return bloch_to_location(*bloch)


def _boot():
    engine = build_engine(spec=_spec(), population=False)
    engine.register_anchor("geo", codec=_SphereCodec())
    engine.delocate_anchor("geo")
    return engine


def _force_drift(engine) -> None:
    """Park the geo qubit at a pure pole WITHOUT grounding — the strongest version of
    what long field evolution does to an un-grounded anchor."""
    cluster, idx = engine._anchor_qubit("geo")
    assert cluster is not None
    cluster.observe_qubit(idx, (0.0, 0.0, 1.0), alpha=1.0)


def test_ungrounded_drift_never_mints_a_place() -> None:
    engine = _boot()
    assert ".nowhere." in gauge_name(engine)
    _force_drift(engine)
    assert ".nowhere." in gauge_name(engine), \
        "an un-grounded anchor drifted off maximally-mixed and the gauge named a place"


def test_grounding_still_mints_and_delocate_revokes() -> None:
    engine = _boot()
    engine.ground_anchor("geo", (25.0, 55.0), alpha=1.0)
    named = gauge_name(engine)
    assert ".nowhere." not in named, f"grounded anchor still unnamed: {named}"
    engine.delocate_anchor("geo")
    assert ".nowhere." in gauge_name(engine), "delocate did not revoke the place token"
