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


# ── Actor-keyed confounding (FL-core Phase 3) ─────────────────────────────────
# Extends the graph-derived surface with WHO acted. The static surface stays the
# graph law; actor tags make multi-mind hygiene possible without replacing it.

_ACTOR_INTENT_LOG_ATTR = "_actor_intent_log"


def record_actor_intent(engine, actor_id: str, intent_name: str, t=None) -> None:
    """Append (actor_id, intent_name, t) onto the engine for multi-actor hygiene."""
    log = getattr(engine, _ACTOR_INTENT_LOG_ATTR, None)
    if log is None:
        log = []
        setattr(engine, _ACTOR_INTENT_LOG_ATTR, log)
    log.append((str(actor_id), str(intent_name), t))
    # Bound memory
    if len(log) > 512:
        del log[:-256]


def actor_intent_log(engine) -> list:
    return list(getattr(engine, _ACTOR_INTENT_LOG_ATTR, []) or [])


def actor_confounded_now(
    graph,
    actuated_roles: dict[str, set[str]],
    *,
    actor_id: str | None = None,
    actor_actuated: dict[str, dict[str, set[str]]] | None = None,
) -> dict[str, set[str]]:
    """Graph surface ∩ recent actuation, optionally restricted to one actor.

    `actor_actuated` maps actor_id → {cluster: {roles}} for multi-mind. When
    actor_id is set and actor_actuated is provided, only that actor's recent
    actuations count. When actor_id is None, falls back to confounded_now
    (graph-only, single-mind compatible).
    """
    if actor_id is not None and actor_actuated is not None:
        roles = actor_actuated.get(actor_id, {})
        return confounded_now(graph, roles)
    return confounded_now(graph, actuated_roles)
