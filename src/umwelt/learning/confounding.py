"""Confounding loops — which LEARNED roles does actuation make self-causing?

Derived ENTIRELY from the world graph, no per-actuator code. An actuator's PROJECTION is its
bottom-up coupling into the belief — the learned role its device state feeds (the graph already
encodes `kasa_switch_kitchen.light_state → kitchen_environment`). So an actuator that projects
onto a learned role closes a self-causing loop: the brain commands the device, the device's
state — and its physical effect, sensed back through the matching signal — moves that
learned role, and the world model "learns" a change it itself drove (the downstream-from-us trap).

This is UNIFORM: every actuator, read straight from its projection. No special case, no
actuator-kind map, no hand-declaration. The AC was never the exception — EVERY actuator confounds
the role it projects onto (kitchen light → kitchen_environment, dining light → dining_activity, a
plug → energy, the AC → environment). The hand-coded ACActuator.recent_intervention saw one loop
and missed the rest; this replaces it.

confounding_loops(graph) → {cluster_name: set[role]} — the static confounding SURFACE (which
learned roles CAN be self-caused). The dynamic gate — which of these were actually ACTUATED
recently, so are confounded *right now* — layers on top from the actuators' dispatch records
(actuated_roles below). See learning_router.py.
"""
from __future__ import annotations

_DEVICE_KINDS = frozenset({"actuator", "sensor", "component", "appliance"})


def actuator_confounding(graph) -> dict[str, tuple[str, set[str]]]:
    """{actuator_node_name: (cluster_name, {confounded roles})} — the per-actuator confounding,
    so a recent dispatch BY A NAMED ACTUATOR maps to the cluster + learned roles it confounds.
    Read from each actuator node's projection — uniform, graph-derived. A device's projection is
    bottom-up (its role → its PARENT cluster's role), so each target is resolved against the
    actuator's nearest LEARNED ancestor (the cluster it projects INTO), not globally — robust to
    roles like 'environment' shared across regions, and merge-correct (under node-merge the ancestor
    is the merged root and the targets are its region-qualified roles). The tendril declares only
    its graph_node identity; THIS derives the confounding — no per-actuator cluster hand-coding."""
    out: dict[str, tuple[str, set[str]]] = {}

    def walk(node, learned_ancestor):
        is_learned = (bool(getattr(node, "roles", None))
                      and getattr(node, "kind", None) not in _DEVICE_KINDS)
        cluster = node if is_learned else learned_ancestor
        if getattr(node, "kind", None) == "actuator" and learned_ancestor is not None:
            anc_roles = getattr(learned_ancestor, "roles", None) or []
            roles = {t for t in (getattr(node, "projection", None) or {}).values()
                     if t in anc_roles}      # projection targets that land on a learned role
            if roles:
                out[node.name] = (learned_ancestor.name, roles)
        for child in getattr(node, "children", {}).values():
            walk(child, cluster)

    walk(graph.root, None)
    return out


def confounding_loops(graph) -> dict[str, set[str]]:
    """The static confounding surface aggregated by cluster: {cluster_name: {learned roles that
    actuation can make self-causing}}. The union over actuator_confounding() — every actuator,
    uniform, graph-derived, no special case (the AC is just one entry among them)."""
    out: dict[str, set[str]] = {}
    for cluster, roles in actuator_confounding(graph).values():
        out.setdefault(cluster, set()).update(roles)
    return out


def confounded_now(graph, actuated_roles: dict[str, set[str]]) -> dict[str, set[str]]:
    """Intersect the static surface with what was ACTUATED recently. `actuated_roles` =
    {cluster: {roles whose actuator fired within the recent window}} (assembled from the
    dispatch records / projections of recently-commanded devices). Returns the roles whose
    world-model learning is self-caused RIGHT NOW — the set the learning router should discount."""
    surface = confounding_loops(graph)
    out: dict[str, set[str]] = {}
    for cluster, roles in actuated_roles.items():
        hit = roles & surface.get(cluster, set())
        if hit:
            out[cluster] = hit
    return out
