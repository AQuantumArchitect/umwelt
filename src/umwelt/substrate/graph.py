"""
Recursive World Graph — the ontology of the world being modeled.

The world is a tree of nodes. Each node can contain children,
forming a fractal structure at arbitrary depth:

    world (root)
    +-- region_a
    +-- region_b
    +-- region_c
    |   +-- component_1
    |   |   +-- subcomponent_1a
    |   |   +-- subcomponent_1b
    |   +-- component_2
    |   +-- component_3
    +-- region_d
    |   +-- gated_link
    +-- region_e

Each node carries:
  - roles: qubit axes defining this node's fiber (can differ per node)
  - children: sub-nodes forming deeper structure
  - projection: how this node's roles map onto parent's roles
  - bridges: lateral connections between sibling nodes

The graph IS the ontology. The quantum probability field and
classical world model are two views projected from this structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from umwelt.substrate.params import ParameterBundle, ScalarParam


@dataclass
class WorldNode:
    """A node in the recursive world graph."""

    name: str
    roles: list[str] = field(default_factory=list)
    children: dict[str, "WorldNode"] = field(default_factory=dict)
    features: list[str] = field(default_factory=list)
    # Maps this node's role -> parent's role for upward projection
    projection: dict[str, str] | None = None
    kind: str = "region"
    grid_pos: tuple[int, int] = (0, 0)
    param_bundle: "ParameterBundle | None" = field(default=None, repr=False)
    # Per-node cluster backend override ("cumulant" pins this node to the
    # CumulantCluster regardless of the global flag — e.g. a merged big node whose
    # full ρ would be the OOM monster). None → the field's global backend choice.
    cluster_backend: "str | None" = field(default=None, repr=False)
    # Explicit per-role input-mode override {role: "unitary"|"dissipative"}. Needed
    # when roles are renamed (e.g. merged "bedroom_presence") so role_input_mode's
    # name-based classification doesn't misfire. None → classify by role name.
    role_modes: "dict[str, str] | None" = field(default=None, repr=False)
    # ZZ connectivity for this node's cluster + H-tower basis. None → dense all-
    # pairs; an edge set of (role_a, role_b) pairs → only those couple (a big
    # merged cluster's physical adjacency, not n² all-pairs). Keeps the cluster
    # and basis consistent. See resolve_zz_pairs.
    connectivity: "list | None" = field(default=None, repr=False)
    # FOLDED (MANIFOLD device edge): this node's STATE roles live in the root manifold; it gets no
    # separate field cluster. The node stays for IDENTITY (bindings, dispatch, de-confounding) + param
    # reads; state routes via merged_zone_role(name, role). Set by the root's merge_to_manifold.
    folded: bool = field(default=False, repr=False)
    _parent: "WorldNode | None" = field(default=None, repr=False)

    def add_child(self, child: "WorldNode") -> "WorldNode":
        """Add a child node. Sets parent reference."""
        child._parent = self
        self.children[child.name] = child
        return child

    @property
    def parent(self) -> "WorldNode | None":
        return self._parent

    @property
    def path(self) -> str:
        """Dot-separated path from root, e.g. 'world.region.device'."""
        parts = []
        node: WorldNode | None = self
        while node is not None:
            parts.append(node.name)
            node = node._parent
        return ".".join(reversed(parts))

    @property
    def depth(self) -> int:
        d = 0
        node = self._parent
        while node is not None:
            d += 1
            node = node._parent
        return d

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def walk(self) -> Iterator["WorldNode"]:
        """Depth-first traversal of this node and all descendants."""
        yield self
        for child in self.children.values():
            yield from child.walk()

    def leaves(self) -> list["WorldNode"]:
        return [n for n in self.walk() if n.is_leaf]

    def find(self, path: str) -> "WorldNode | None":
        """Find a descendant by dot-separated path relative to this node."""
        parts = path.split(".", 1)
        child = self.children.get(parts[0])
        if child is None:
            return None
        if len(parts) == 1:
            return child
        return child.find(parts[1])

    def siblings(self) -> list["WorldNode"]:
        if self._parent is None:
            return []
        return [c for c in self._parent.children.values() if c is not self]


@dataclass
class Bridge:
    """A lateral connection between two nodes."""

    source: str  # node name
    target: str  # node name
    shared_roles: list[str]
    kind: str = "door"  # open, door, wall, tendril
    coupling_param: "ScalarParam | None" = field(default=None, repr=False)
    role_map: dict[str, str] = field(default_factory=dict)

    _KIND_DEFAULTS: dict[str, float] = field(
        default_factory=lambda: {"open": 1.0, "door": 0.7, "wall": 0.3},
        init=False,
        repr=False,
    )

    @property
    def is_tendril(self) -> bool:
        """Directed cross-role coupling (region → actuator). No symmetric reconcile."""
        return bool(self.role_map)

    @property
    def coupling_base(self) -> float:
        if self.coupling_param is not None:
            return self.coupling_param.value
        return self._KIND_DEFAULTS.get(self.kind, 0.5)

    @property
    def coupling_weight(self) -> float:
        return self.coupling_base * len(self.shared_roles)


@dataclass
class WorldGraph:
    """Complete world graph: root node + lateral bridges."""

    root: WorldNode
    bridges: list[Bridge] = field(default_factory=list)
    # Overlapping fractal pre-cluster cover (b9.30). Each entry has .name/.members/.roles; realized as
    # intra-sector couplings by merge_rooms under UMWELT_SECTORS. Empty = no sectors (today's default).
    sectors: list = field(default_factory=list)

    def find(self, path: str) -> WorldNode | None:
        """Find a node by name. Searches the full tree."""
        if path == self.root.name:
            return self.root
        # Try direct child lookup first (common case)
        found = self.root.find(path)
        if found is not None:
            return found
        # Search all nodes by name
        for node in self.root.walk():
            if node.name == path:
                return node
        return None

    def all_nodes(self) -> list[WorldNode]:
        return list(self.root.walk())

    def nodes_with_roles(self) -> list[WorldNode]:
        return [n for n in self.root.walk() if n.roles]

    def bridges_for(self, node_name: str) -> list[Bridge]:
        return [
            b
            for b in self.bridges
            if b.source == node_name or b.target == node_name
        ]

    @property
    def max_depth(self) -> int:
        return max(n.depth for n in self.root.walk())
