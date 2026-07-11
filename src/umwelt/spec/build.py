"""spec/build — rebuild a world graph from a DomainSpec, and extract one back.

The de-risking heart of the world=data seam, inherited from the origin deployment:
rather than hand-transcribe topology, EXTRACT the spec by introspecting a live graph,
then prove completeness by RECONSTRUCTION — `build_graph_from_spec(extract_spec(g))`
must be identical to `g`. Parity by construction.

Topology only here (nodes + bridges + sectors). Param bundles attach AFTER the graph is
built, bridge coupling params are learned-from-0 (None at build time), and the post-build
transforms stay in code, flagged on the spec. Role registration is the build-time half of
blocker-4: every role mode a NodeSpec declares is pushed into the spec.roles registry so
the substrate never asks an I/O edge how to classify a role.
"""
from __future__ import annotations

from umwelt.spec.schema import (DomainSpec, NodeSpec, BridgeSpec,
                                canonical_bridge_kind, canonical_node_kind)
from umwelt.spec import roles as role_registry
from umwelt.substrate.graph import WorldGraph, WorldNode, Bridge


def _t(x):
    """Normalize a list/tuple/None to a tuple (specs are frozen → hashable, deterministic)."""
    return tuple(x) if x is not None else None


def register_spec_roles(spec: DomainSpec) -> None:
    """Push every role mode the spec declares into the role registry — the data path that
    replaced the substrate's import of the sensor edge. NodeSpec.role_modes entries win;
    driver anchor roles register as out-of-band analog automatically."""
    for ns in spec.nodes:
        for role, mode in (ns.role_modes or {}).items():
            role_registry.register_role_mode(role, mode)
    for ds in spec.drivers:
        role_registry.register_driver_role(ds.role)


def extract_spec(graph: WorldGraph, name: str = "extracted") -> DomainSpec:
    """Walk a live graph → a DomainSpec. DFS order (root first, parents before children)
    so a reconstruction in the same order can always resolve each node's parent."""
    nodes: list[NodeSpec] = []
    for n in graph.all_nodes():                         # DFS, parent-before-child
        nodes.append(NodeSpec(
            name=n.name,
            parent=(n.parent.name if n.parent is not None else None),
            roles=tuple(n.roles or ()),
            kind=getattr(n, "kind", "region"),
            projection=dict(n.projection) if n.projection else {},
            features=tuple(n.features or ()),
            grid_pos=tuple(getattr(n, "grid_pos", (0, 0))),
            cluster_backend=getattr(n, "cluster_backend", None),
            role_modes=(dict(n.role_modes) if getattr(n, "role_modes", None) else None),
            connectivity=_t(getattr(n, "connectivity", None)),
            folded=bool(getattr(n, "folded", False)),
        ))
    bridges = tuple(
        BridgeSpec(
            source=b.source, target=b.target,
            shared_roles=tuple(b.shared_roles or ()),
            kind=getattr(b, "kind", "gated"),
            role_map=dict(b.role_map) if b.role_map else {},
        )
        for b in (graph.bridges or [])
    )
    return DomainSpec(name=name, nodes=tuple(nodes), bridges=bridges)


def build_graph_from_spec(spec: DomainSpec) -> WorldGraph:
    """Reconstruct the topology (nodes + bridges + sectors) from a spec. Nodes are created
    in spec order (root first) and re-parented by name. Bridges rebuild with
    coupling_param=None (learned from 0). Also registers the spec's role vocabulary."""
    register_spec_roles(spec)
    by_name: dict[str, WorldNode] = {}
    root: WorldNode | None = None
    for ns in spec.nodes:
        node = WorldNode(
            name=ns.name,
            roles=list(ns.roles),
            kind=canonical_node_kind(ns.kind),
            projection=(dict(ns.projection) if ns.projection else None),
            features=list(ns.features),
            grid_pos=tuple(ns.grid_pos),
            cluster_backend=ns.cluster_backend,
            role_modes=(dict(ns.role_modes) if ns.role_modes else None),
            connectivity=(list(ns.connectivity) if ns.connectivity is not None else None),
            folded=ns.folded,
        )
        by_name[ns.name] = node
        if ns.parent is None:
            if root is not None:
                raise ValueError(f"spec has two roots: {root.name!r} and {ns.name!r}")
            root = node
        else:
            parent = by_name.get(ns.parent)
            if parent is None:
                raise ValueError(f"node {ns.name!r} references parent {ns.parent!r} not yet built "
                                 "(spec nodes must be in parent-before-child order)")
            parent.add_child(node)
    if root is None:
        raise ValueError("spec has no root node (a NodeSpec with parent=None)")

    bridges = [
        Bridge(
            source=bs.source, target=bs.target,
            shared_roles=list(bs.shared_roles),
            kind=canonical_bridge_kind(bs.kind),
            role_map=dict(bs.role_map) if bs.role_map else {},
        )
        for bs in spec.bridges
    ]
    # The overlapping fractal-sector cover rides as-is (only .name/.members/.roles are read).
    return WorldGraph(root=root, bridges=bridges, sectors=list(spec.sectors))
