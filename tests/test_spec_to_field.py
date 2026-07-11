"""P1 milestone: a DomainSpec becomes a stepping field with a diff-stable gauge.

A tiny gridworld spec (the proof-gate domain's seed) drives the whole substrate path:
schema → graph → engine priors → clusters → evolution → canonical gauge snapshot.
"""
from __future__ import annotations

import json

from umwelt.spec.schema import BindingSpec, BridgeSpec, DomainSpec, DriverSpec, NodeSpec
from umwelt.spec.build import build_graph_from_spec
from umwelt.spec import roles as role_registry
from umwelt.substrate.field import QuantumField
from umwelt.substrate.param_bundles import configure_param_bundles
from umwelt.projection.gauge import field_gauge, in_rest_window, driver_phase


def tiny_grid_spec() -> DomainSpec:
    cells = ["cell_0_0", "cell_0_1", "cell_1_0", "cell_1_1"]
    nodes = [NodeSpec("grid", parent=None, kind="root", roles=("agent_near",))]
    for c in cells:
        nodes.append(NodeSpec(
            c, parent="grid", roles=("agent_near", "resource"),
            role_modes={"agent_near": "unitary", "resource": "dissipative"},
            params={"gamma": (0.04, 0.01, 0.001, 0.2)},
        ))
    bridges = (
        BridgeSpec("cell_0_0", "cell_0_1", shared_roles=("agent_near",), kind="open"),
        BridgeSpec("cell_0_0", "cell_1_0", shared_roles=("agent_near",), kind="open"),
        BridgeSpec("cell_1_1", "cell_0_1", shared_roles=("agent_near",), kind="gated"),
        BridgeSpec("cell_1_1", "cell_1_0", shared_roles=("agent_near",), kind="door"),  # alias
    )
    bindings = tuple(
        BindingSpec(f"sight_{c}", zone=c, role="agent_near", normalizer="binary",
                    force_observe=True)
        for c in cells
    ) + tuple(
        BindingSpec(f"resource_{c}", zone=c, role="resource",
                    normalizer={"type": "range", "lo": 0.0, "hi": 10.0})
        for c in cells
    )
    return DomainSpec(
        name="tiny-grid",
        nodes=tuple(nodes),
        bridges=bridges,
        bindings=bindings,
        drivers=(DriverSpec("day", node="_clock", role="phase", period_s=1200.0),),
    )


def test_spec_builds_graph_and_registers_roles():
    spec = tiny_grid_spec()
    graph = build_graph_from_spec(spec)
    names = [n.name for n in graph.all_nodes()]
    assert names[0] == "grid" and len(names) == 5
    # blocker-4 data path: the spec's role modes landed in the registry
    assert role_registry.role_input_mode("agent_near") == "unitary"
    assert role_registry.role_input_mode("resource") == "dissipative"
    # the driver's anchor role is out-of-band analog
    assert role_registry.is_driver_role("phase") and role_registry.is_analog_role("phase")
    # bridge kind aliases canonicalized
    kinds = {(b.source, b.target): b.kind for b in graph.bridges}
    assert kinds[("cell_1_1", "cell_1_0")] == "gated"


def test_field_steps_and_gauge_is_diff_stable():
    spec = tiny_grid_spec()
    graph = build_graph_from_spec(spec)
    configure_param_bundles(graph, spec)
    # engine DNA landed on the root; spec params landed per node
    assert graph.root.param_bundle.get("gamma") is not None
    assert graph.find("cell_0_0").param_bundle.get("gamma") == 0.04

    field = QuantumField(graph)
    assert set(field.clusters) == {"grid", "cell_0_0", "cell_0_1", "cell_1_0", "cell_1_1"}

    for _ in range(25):
        field.step()

    snap1 = field_gauge(field)
    snap2 = field_gauge(field)
    assert snap1 == snap2                       # canonical read: same state → same snapshot
    blob = json.dumps(snap1, sort_keys=True)    # diff-stable = JSON-serializable + ordered
    assert "cell_0_0" in snap1 and "bloch" in snap1["cell_0_0"]
    for name, g in snap1.items():
        for role, xyz in g["bloch"].items():
            assert all(abs(v) <= 1.0 + 1e-6 for v in xyz), (name, role, xyz)
    assert json.loads(blob) == snap1


def test_rest_window_and_driver_phase_defaults():
    spec = tiny_grid_spec()
    graph = build_graph_from_spec(spec)
    field = QuantumField(graph)
    # no _clock node in this spec → no phase, and None is never "in the window"
    assert driver_phase(field) is None
    assert in_rest_window(None) is False
    assert in_rest_window(0.5) is True and in_rest_window(0.9) is False
