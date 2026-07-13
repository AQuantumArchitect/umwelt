"""A binding that targets a zone or role that doesn't exist must never build silently.

Found via the first foreign consumer (umwelt-market): a typo'd zone/role in a
BindingSpec used to either vanish with no signal at all (bad zone) or log a warning
while still registering (bad role) — both landmines for a new domain author. Now
SensorBridge.register() raises; the spec-application membrane (apply_spec_bindings /
_apply_spec_bindings) still catches it and skips that one binding, so one typo can
never break the rest of a spec's bindings — but it is always logged, and never silently
half-registered.
"""
from __future__ import annotations

import logging

import pytest

from umwelt.boot import build_engine
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec
from umwelt.membranes.ingress import SensorBridge
from umwelt.spec.build import build_graph_from_spec


def _tiny_spec(bindings: tuple) -> DomainSpec:
    nodes = (
        NodeSpec("root", parent=None, kind="root", roles=()),
        NodeSpec("cell", parent="root", roles=("level",),
                 role_modes={"level": "dissipative"}),
    )
    return DomainSpec(name="tiny", nodes=nodes, bindings=bindings)


def test_register_raises_on_nonexistent_zone():
    graph = build_graph_from_spec(_tiny_spec(()))
    bridge = SensorBridge(graph)
    with pytest.raises(ValueError, match="does not exist in the graph"):
        bridge.register("s1", zone="nonexistent", qubit_role="level")
    assert "s1" not in bridge.bindings


def test_register_raises_on_undeclared_role():
    graph = build_graph_from_spec(_tiny_spec(()))
    bridge = SensorBridge(graph)
    with pytest.raises(ValueError, match="only declares roles"):
        bridge.register("s1", zone="cell", qubit_role="not_a_role")
    assert "s1" not in bridge.bindings


def test_build_engine_skips_a_bad_binding_but_logs_it_and_keeps_the_rest(caplog):
    spec = _tiny_spec((
        BindingSpec("bad_zone", zone="nonexistent", role="level", normalizer="binary"),
        BindingSpec("bad_role", zone="cell", role="not_a_role", normalizer="binary"),
        BindingSpec("good", zone="cell", role="level", normalizer="binary"),
    ))
    with caplog.at_level(logging.WARNING):
        engine = build_engine(spec=spec, population=False)
    assert "good" in engine.sensor_bridge.bindings
    assert "bad_zone" not in engine.sensor_bridge.bindings
    assert "bad_role" not in engine.sensor_bridge.bindings
    messages = " ".join(r.message for r in caplog.records)
    assert "bad_zone" in messages and "bad_role" in messages
