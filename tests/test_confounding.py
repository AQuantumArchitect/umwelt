"""Graph-derived confounding loops — an actuator confounds the learned role it projects onto.
Uniform: read from projections, no per-actuator code. See meerkat/brain/confounding.py.
"""
from __future__ import annotations

from umwelt.learning.confounding import (
    actuator_confounding, confounding_loops, confounded_now,
)
from umwelt.membranes.tendril import Tendril
from umwelt.substrate.graph import WorldGraph, WorldNode


def _graph():
    studio = WorldNode(name="studio", roles=["presence", "environment", "energy"])
    kitchen = WorldNode(name="kitchen", roles=["presence", "environment", "energy"])
    studio.add_child(kitchen)
    # a light: its state projects to the kitchen's LEARNED environment role → confounds it
    kitchen.add_child(WorldNode(name="kasa_kitchen_light", roles=["light_state"],
                                kind="actuator", projection={"light_state": "environment"}))
    # a plug: projects to energy (learned) → confounds energy
    kitchen.add_child(WorldNode(name="kasa_plug", roles=["power_draw"],
                                kind="actuator", projection={"power_draw": "energy"}))
    # a motion SENSOR (not an actuator) — must not appear
    kitchen.add_child(WorldNode(name="motion", roles=["occupancy"], kind="sensor",
                                projection={"occupancy": "presence"}))
    return WorldGraph(root=studio, bridges=[])


def test_actuator_confounds_its_projected_learned_role():
    surface = confounding_loops(_graph())
    assert surface["kitchen"] == {"environment", "energy"}   # light→env, plug→energy
    assert "studio" not in surface                            # studio has no actuators


def test_sensor_is_not_a_confounder():
    # only kind=="actuator" nodes are read; the motion sensor's projection is ignored
    surface = confounding_loops(_graph())
    assert "presence" not in surface.get("kitchen", set())


def test_projection_to_device_leaf_role_is_not_confounding():
    # an actuator projecting onto a DEVICE-leaf role (not a learned zone role) → no loop
    root = WorldNode(name="studio", roles=["presence"])
    dev = WorldNode(name="lamp", roles=["light_state"], kind="actuator",
                    projection={"light_state": "brightness"})   # brightness lives on no learned node
    root.add_child(dev)
    assert confounding_loops(WorldGraph(root=root, bridges=[])) == {}


def test_confounded_now_intersects_surface_with_recent_actuation():
    g = _graph()
    # kitchen env was just actuated (light fired); energy was not
    now = confounded_now(g, {"kitchen": {"environment"}})
    assert now == {"kitchen": {"environment"}}
    # a recently-actuated role that ISN'T on the surface → not confounded
    assert confounded_now(g, {"kitchen": {"presence"}}) == {}


def test_actuator_confounding_maps_each_actuator_to_its_cluster_and_roles():
    # the per-actuator map is what the learning router resolves a recent dispatch through
    m = actuator_confounding(_graph())
    assert m["kasa_kitchen_light"] == ("kitchen", {"environment"})
    assert m["kasa_plug"] == ("kitchen", {"energy"})
    assert "motion" not in m                                   # sensors don't confound
    # confounding_loops is the union of this map by cluster
    assert confounding_loops(_graph()) == {"kitchen": {"environment", "energy"}}


def test_tendril_dispatch_echo_decays_over_window_and_is_none_when_idle():
    class _T(Tendril):
        graph_node = "kasa_kitchen_light"
        echo_window = 100.0
        def __init__(self):
            self.ts = None
        def last_dispatch_ts(self):
            return self.ts

    t = _T()
    assert t.dispatch_echo(now_ts=1000.0) is None             # never dispatched → no echo
    t.ts = 1000.0
    assert t.dispatch_echo(now_ts=1000.0) == 1.0              # fresh dispatch → full echo
    assert t.dispatch_echo(now_ts=1050.0) == 0.5             # half-window → half echo
    assert t.dispatch_echo(now_ts=1100.0) is None            # past the window → no echo
    assert t.dispatch_echo(now_ts=1300.0) is None            # long stale → no echo
