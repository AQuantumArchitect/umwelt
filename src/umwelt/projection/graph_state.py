"""graph_state — the canonical projection of the world graph: the website's SINGLE SOURCE OF TRUTH.

The mind is a graph (root → regions → entities → devices), each node carrying a param fiber, a belief
cluster (Bloch qubits + purity), wired by learned couplings, under a context gauge with an agency
qubit. Every surface an app serves should be a *projection* of that one graph — "projection" is the
brain's own vocabulary (`WorldNode.projection` maps roles parent→child; observe = partial collapse).

This serializer is that one projection. It COMPOSES the existing cheap reads — it does not reinvent:

  • topology   — `graph.all_nodes()` walk + `graph.bridges` (the node tree + lateral couplings).
  • per-node   — `transparency.model_snapshot` rows REGROUPED onto their owning node: `bloch_cluster`
    (belief), `param_fiber` (knobs), plus `committed` (the discrete truth from the world model).
  • globals    — `agency.snapshot()`, `competence.competence_snapshot()`, `reservoir.context_gauge()`,
    and the transparency `summary`. Because the params/clusters/summary ARE `model_snapshot`'s output,
    `/api/transparency` is a strict SUBSET of this → provable parity (see tests/brain/test_graph_state).

SELF-DESCRIBING: every organ carries a `type` tag naming its projection-type, so the frontend maps
`type → renderer` without knowing the page (one registry, every node that emits a type renders it, at
every depth — the fractal payoff). Cheap O(1) reads only: NEVER `reservoir.context()` / density-matrix
rebuild / fractal-bridge recompute — safe to poll + cache on the RDK. See docs/WEBSITE.md, transparency.py.
"""
from __future__ import annotations

from typing import Any

from umwelt.projection import transparency as _transparency

# The organ projection-types this snapshot emits. The frontend panel registry must cover this set;
# the test asserts no organ escapes it (so a new organ can't silently render as nothing).
KNOWN_ORGAN_TYPES = frozenset({
    "bloch_cluster", "param_fiber", "committed",          # per-node
    "agency", "competence", "run_gauge", "body", "ingest", "summary",   # global
})


def _graph(reservoir: Any):
    field = getattr(reservoir, "field", None)
    return getattr(field, "graph", None) or getattr(reservoir, "graph", None)


def _edges(graph: Any) -> list[dict]:
    """Lateral couplings (bridges) as source→target edges — the graph's non-tree links."""
    out: list[dict] = []
    for b in (getattr(graph, "bridges", None) or []):
        try:
            out.append({
                "source": b.source,
                "target": b.target,
                "kind": getattr(b, "kind", "door"),
                "shared_roles": list(getattr(b, "shared_roles", []) or []),
                "weight": round(float(getattr(b, "coupling_weight", 0.0)), 4),
                "is_tendril": bool(getattr(b, "is_tendril", False)),
            })
        except Exception:
            continue
    return out


def _committed_nodes(reservoir: Any) -> dict:
    """The discrete COMMITTED truth per node (what collapsed to, not the live belief) — cheap read off
    the classical world model (no density-matrix rebuild)."""
    try:
        world = getattr(reservoir, "world", None)
        if world is not None:
            return (world.context() or {}).get("nodes", {}) or {}
    except Exception:
        pass
    return {}


def graph_state(reservoir: Any) -> dict:
    """The whole legible graph as ONE self-describing snapshot: topology (nodes + edges) with each
    node's organs, plus the global organs. The source every surface projects from."""
    graph = _graph(reservoir)

    # Reuse the whole-graph projection — params/clusters/summary are model_snapshot's, regrouped by node.
    # Guarded: a broken/foreign reservoir must degrade to topology-only, never 500 the whole projection.
    try:
        snap = _transparency.model_snapshot(reservoir)
    except Exception:
        snap = {"params": [], "clusters": [], "summary": {}}
    params_by_node: dict[str, list] = {}
    for p in snap.get("params", []):
        params_by_node.setdefault(p.get("node"), []).append(p)
    clusters_by_name = {c.get("name"): c for c in snap.get("clusters", [])}
    committed = _committed_nodes(reservoir)

    nodes: list[dict] = []
    if graph is not None:
        try:
            walk = list(graph.all_nodes())
        except Exception:
            walk = []
        for n in walk:
          try:                                          # one bad node can't blank the whole topology
            organs: list[dict] = []
            cl = clusters_by_name.get(n.name)
            if cl is not None:
                organs.append({"type": "bloch_cluster", **cl})
            pf = params_by_node.get(n.name)
            if pf:
                organs.append({"type": "param_fiber", "params": pf})
            cm = committed.get(n.name)
            if cm:
                organs.append({"type": "committed", **cm})
            nodes.append({
                "path": n.path,
                "name": n.name,
                "kind": getattr(n, "kind", "region"),
                "depth": n.depth,
                "is_leaf": n.is_leaf,
                "parent": (n.parent.name if n.parent is not None else None),
                "children": list(n.children.keys()),
                "roles": list(getattr(n, "roles", []) or []),
                "grid_pos": list(getattr(n, "grid_pos", (0, 0))),
                "folded": bool(getattr(n, "folded", False)),
                "organs": organs,
            })
          except Exception:
            continue

    globals_: list[dict] = []
    try:
        ag = getattr(reservoir, "agency", None)
        if ag is not None:
            globals_.append({"type": "agency", **ag.snapshot()})
    except Exception:
        pass
    try:
        from umwelt.learning import competence as _competence
        globals_.append({"type": "competence", **_competence.competence_snapshot(reservoir)})
    except Exception:
        pass
    try:
        if hasattr(reservoir, "context_gauge"):
            globals_.append({"type": "run_gauge", **(reservoir.context_gauge() or {})})
    except Exception:
        pass
    # (The origin appended a compute-body self-sense organ here; an app with a hardware
    # body registers its own "body" organ through its projection layer.)
    try:
        # the ingest edge: what arrived that bound to NOTHING (foreign / mislabelled / unwired) — the
        # gap between the data's vocabulary and the engine's bindings. Surfaced so a foreign-world ingest
        # is legible instead of a silent drop. See SensorBridge.unmatched_snapshot.
        sb = getattr(reservoir, "sensor_bridge", None)
        if sb is not None and hasattr(sb, "unmatched_snapshot"):
            globals_.append({"type": "ingest", "unmatched": sb.unmatched_snapshot()})
    except Exception:
        pass
    globals_.append({"type": "summary", **snap.get("summary", {})})

    return {
        "version": 1,
        "topology": {"nodes": nodes, "edges": _edges(graph) if graph is not None else []},
        "globals": globals_,
    }
