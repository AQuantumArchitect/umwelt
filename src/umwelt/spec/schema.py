"""DomainSpec — a world as DATA, not code.

The single declarative source of truth per domain: a frozen-dataclass manifest that fully
describes a world's topology (nodes), lateral structure (bridges), measurement vocabulary
(bindings + normalizers), outputs (tendrils), and time (periodic drivers). The engine runs
ANY spec; no domain is a code path. Proven by the blank-slate gate: an unconfigured engine
boots from an arbitrary spec and comprehends a synthetic stream (proofs/blank_slate.py).

Extracted from the meerkat deployment's HouseSpec seam, generalized: kind vocabularies are
neutral with domain aliases, anchors replace the hard-coded location, and the output/driver
schema is new (the deployment wired those imperatively; here they are data).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from umwelt.spec.normalizers import resolve_normalizer

# Canonical node/bridge kind vocabularies, with domain-dialect aliases so a spec can speak
# its own language ("zone" for a region, "door" for a gated link) without schema change.
NODE_KINDS = ("root", "region", "environment", "clock", "anchor", "signal",
              "actuator", "entity", "component", "synthetic")
NODE_KIND_ALIASES = {"zone": "region", "sensor": "signal", "earth": "anchor",
                     "appliance": "component", "person": "entity"}
BRIDGE_KINDS = ("open", "gated", "wall", "tendril")
BRIDGE_KIND_ALIASES = {"door": "gated"}


def canonical_node_kind(kind: str) -> str:
    return NODE_KIND_ALIASES.get(kind, kind)


def canonical_bridge_kind(kind: str) -> str:
    return BRIDGE_KIND_ALIASES.get(kind, kind)


@dataclass(frozen=True)
class NodeSpec:
    """One node in the world graph, parent-referenced — the canonical unit. A region, a
    device, a synthetic node (_clock / an anchor) and the root are all NodeSpecs; `kind` +
    depth distinguish them. Captures every field a WorldNode needs to be reconstructed
    identically. `parent=None` ⇒ the root."""
    name: str
    parent: str | None = None
    roles: tuple = ()                       # qubit roles
    kind: str = "region"
    projection: dict = field(default_factory=dict)   # this node's role → parent's role
    features: tuple = ()                    # semantic tags
    grid_pos: tuple = (0, 0)                # (x,y) for layout
    cluster_backend: str | None = None      # "cumulant" pins the backend
    role_modes: dict | None = None          # {role: "unitary"|"dissipative"} override
    connectivity: tuple | None = None       # sparse ZZ edge set (merged nodes), else None=dense
    folded: bool = False                    # manifold: state lives in parent
    reduce: str | None = None               # derived belief over children's shared role:
                                            # "max" | "mean" | "or" (see spec/derived.py)
    params: dict = field(default_factory=dict)   # learnable priors for this node:
                                                 # {name: (default, sigma, lo, hi)}


@dataclass(frozen=True)
class BridgeSpec:
    """A lateral edge between two nodes. role_map non-empty ⇒ tendril (directed, region→actuator)."""
    source: str
    target: str
    shared_roles: tuple = ()
    kind: str = "gated"                     # open | gated | wall | tendril (+ domain aliases)
    role_map: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SectorSpec:
    """A fractal pre-cluster: a named GROUP of regions that belong together spatially or
    functionally, so their shared roles couple THROUGH the sector — not just via the sparse
    pairwise bridges. The cover is OVERLAPPING, not a partition: a region may appear in
    several sectors, so `members` across sectors need not be disjoint. In the cumulant
    substrate this is natural — a region's role keeps ONE marginal (1-RDM) and simply
    participates in more pair-cumulants, one per sector it joins.

    `roles` = which shared roles the sector couples. Realized as extra intra-sector
    cross-region ZZ edges on a merged cluster (opt-in via `enable_sectors`)."""
    name: str
    members: tuple = ()
    roles: tuple = ()


@dataclass(frozen=True)
class BindingSpec:
    """A signal → (node, role) binding with a DECLARATIVE normalizer.

    THE MEASUREMENT MODEL: a signal is a weak measurement with strength k and detector
    efficiency η — `collapse_alpha` was always the folded product of the two. Declaring
    them separately lets the Belavkin path use k·η as the measurement strength while η
    stays the signal's OWN reliability coordinate (the trust web learns it online). Both
    optional: unset → the binding uses collapse_alpha, or the caller's global default."""
    sensor_id: str
    zone: str                               # target node name (field name kept for
                                            # signature-compatibility with the origin seam)
    role: str
    normalizer: Any = "binary"              # str | {type,**params} | callable → resolve_normalizer
    weight: float = 1.0
    event_type: str = ""                    # "" | "contact" | "affective" | ...
    force_observe: bool = False
    collapse_alpha: float | None = None
    strength: float | None = None           # measurement strength k
    efficiency: float | None = None         # detector efficiency η ∈ [0,1]
    description: str = ""

    def build_normalizer(self) -> Callable[[float], float]:
        return resolve_normalizer(self.normalizer)

    def measurement_alpha(self) -> float | None:
        """The effective collapse strength this binding declares: k·η when the measurement
        model is set, else collapse_alpha, else None (caller applies its global default)."""
        if self.strength is not None:
            eta = 1.0 if self.efficiency is None else float(self.efficiency)
            return float(self.strength) * eta
        return self.collapse_alpha


@dataclass(frozen=True)
class OutputSpec:
    """One output tendril as data: how a decision leaves the field.

    The engine reads the named node/role continuously (the field IS the decision surface),
    decodes it into a device-unit command, gates it, and emits an Action. Unit mapping and
    clamps live HERE at the edge — the field stays unit-free. `shadow=True` is the default
    and the law: a new output decides-but-does-not-dispatch until the app flips it."""
    name: str                               # stable tendril id → Action reason "<name>_auto"
    node: str                               # graph node whose belief drives it
    role: str = "level"                     # the role read off that node
    kind: str = "binary"                    # binary | scalar | categorical | vector
    decode: Any = "sticky"                  # decoder: name | {type,**params} (egress registry)
    codomain: tuple = (0.0, 1.0)            # device-unit clamp at the edge
    readback_sensor: str | None = None      # signal id watched for user corrections
    reward_channel: str = "override"        # which reward channel corrections release
    gates: dict = field(default_factory=dict)     # enable_param, rate_limit_s, deadband, ...
    coupling: dict = field(default_factory=dict)  # committed-belief geometry: rest, priors
    dispatch: dict = field(default_factory=dict)  # opaque routing payload for the app dispatcher
    shadow: bool = True


@dataclass(frozen=True)
class DriverSpec:
    """A periodic driver as data: the domain's clock. The engine anchors the named node/role
    qubit to the driver's phase each tick (out of band, analog). `type` resolves against the
    driver registry in clocks/drivers.py — "harmonic" is the engine default; domains register
    their own (an ephemeris, an exchange session calendar, a game tick)."""
    name: str
    node: str = "_clock"
    role: str = "phase"
    type: str = "harmonic"
    period_s: float = 86400.0
    params: dict = field(default_factory=dict)
    rest_window: tuple | None = None        # per-cycle quiet band (phase fractions)


@dataclass(frozen=True)
class DomainSpec:
    """The whole world as one declarative manifest. `build_graph_from_spec()` consumes the
    topology; the post-build graph TRANSFORMS (subdomains/merge/manifold) stay in code,
    FLAGGED here (the spec never expresses an algorithm). Channel maps ride along so a
    domain's wire vocabulary is declared in one place; they are opaque to the engine —
    the app's ingest adapter interprets them."""
    name: str
    nodes: tuple = ()                       # tuple[NodeSpec] — the topology (root + all nodes)
    bridges: tuple = ()                     # tuple[BridgeSpec]
    bindings: tuple = ()                    # tuple[BindingSpec]
    sectors: tuple = ()                     # tuple[SectorSpec]
    outputs: tuple = ()                     # tuple[OutputSpec] — decisions as data
    drivers: tuple = ()                     # tuple[DriverSpec] — time as data
    channel_maps: dict = field(default_factory=dict)   # named opaque transport maps
    anchors: dict = field(default_factory=dict)        # e.g. {"geo": {"lat":..., "lon":...}};
                                                       # NO default — unanchored means unanchored
    # The world's natural tick in wall seconds. None (default) = tick-driven: one dt
    # per ingest regardless of wall gaps — the origin's dense-polled behavior. Set it
    # and the engine honors the silence between sparse batches as bounded
    # free-evolution catch-up (see BeliefEngine.ingest wall-clock pacing).
    tick_s: float | None = None
    # IGNORED signals — ids that arrive on the wire but are DELIBERATELY not bound, each
    # with a reason. Declaring them turns the ingest gap from a scary "N unbound" into the
    # truth: "0 actionable, N explained." Pattern = exact id or `prefix_*` wildcard.
    ignored: tuple = ()                     # tuple[(pattern, reason)]
    # declarative learning-vocabulary extensions (consumed by learning/reward registry)
    param_channels: tuple = ()              # tuple[(exact_or_prefix, channel_name)]
    param_key_normalizer: str | None = None # dotted "module:fn" hook, default identity
    # post-build flags (trigger in-code transforms; the spec never expresses the algorithm)
    enable_subdomains: bool = False
    merge_groups: bool = False
    manifold: bool = False
    enable_sectors: bool = False
    extensions: dict = field(default_factory=dict)     # opaque app-side flags


def load_spec(ref: str) -> DomainSpec:
    """Resolve a ``module:ATTR`` reference to a DomainSpec — the ONE string form every spec
    seam accepts (the ``UMWELT_SPEC`` boot env, demo ``--spec`` flags). Raises loudly on a
    bad ref — a mistyped world must never silently boot a different one."""
    module_name, _, attr = ref.partition(":")
    if not module_name or not attr:
        raise ValueError(f"spec ref must be 'module:ATTR', got {ref!r}")
    import importlib
    spec = getattr(importlib.import_module(module_name), attr)
    if not isinstance(spec, DomainSpec):
        raise TypeError(f"{ref} resolved to {type(spec).__name__}, not a DomainSpec")
    return spec


def match_ignored(sensor_id: str, ignored: tuple) -> str | None:
    """If sensor_id is declared-ignored, return its reason, else None. Pattern = exact id
    or `prefix_*`."""
    for pat, reason in (ignored or ()):
        if pat == sensor_id or (pat.endswith("*") and sensor_id.startswith(pat[:-1])):
            return reason
    return None
