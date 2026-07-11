"""graph_state — the canonical projection of the world graph (an app's single source of truth).

Ported from the origin deployment's projection gate (meerkat tests/brain/test_graph_state.py)
onto the gridworld engine: a self-describing topology (nodes+edges) whose per-node organs
REGROUP transparency.model_snapshot field-for-field (provable parity), every organ carries a
known `type`, and the snapshot stays CHEAP (never calls engine.context()). The origin's
compute-body organ tests stayed with the origin (umwelt cut body_state — an app registers its
own body organ); its house-vocabulary foreign-graph cases are re-voiced on a station world.
See umwelt/projection/graph_state.py.
"""
from __future__ import annotations

from tests.test_spec_to_field import tiny_grid_spec
from umwelt.projection import graph_state as G
from umwelt.projection import transparency as T


def _engine():
    from umwelt.boot import build_engine
    return build_engine(spec=tiny_grid_spec())


def _flat_params(gs) -> list:
    return [p for n in gs["topology"]["nodes"] for o in n["organs"]
            if o["type"] == "param_fiber" for p in o["params"]]


def _flat_clusters(gs) -> list:
    return [{k: v for k, v in o.items() if k != "type"}
            for n in gs["topology"]["nodes"] for o in n["organs"] if o["type"] == "bloch_cluster"]


def _global(gs, t):
    return next((g for g in gs["globals"] if g["type"] == t), None)


def test_envelope_and_topology_shape():
    gs = G.graph_state(_engine())
    assert gs["version"] == 1
    assert set(("topology", "globals")).issubset(gs)
    nodes = gs["topology"]["nodes"]
    assert len(nodes) >= 5
    for n in nodes:
        assert set(("path", "name", "kind", "depth", "is_leaf", "parent", "children",
                    "roles", "organs")).issubset(n)
    # edges carry the bridge shape
    assert gs["topology"]["edges"]
    for e in gs["topology"]["edges"]:
        assert set(("source", "target", "kind", "weight")).issubset(e)


def test_organs_are_self_describing():
    """Every organ (per-node + global) carries a `type` in the known set — so none silently
    renders as nothing in an app's panel registry."""
    gs = G.graph_state(_engine())
    seen = set()
    for n in gs["topology"]["nodes"]:
        for o in n["organs"]:
            assert "type" in o
            seen.add(o["type"])
    for g in gs["globals"]:
        assert "type" in g
        seen.add(g["type"])
    assert seen <= G.KNOWN_ORGAN_TYPES, f"organ types escaped the registry: {seen - G.KNOWN_ORGAN_TYPES}"
    # the global organs an app's chrome depends on are present
    assert {"agency", "competence", "run_gauge", "summary"} <= {g["type"] for g in gs["globals"]}


def test_parity_with_model_snapshot():
    """The whole point of single-source-of-truth: graph_state's params/clusters/summary ARE
    model_snapshot's, regrouped onto the topology — so a transparency surface is a strict subset."""
    engine = _engine()
    gs = G.graph_state(engine)
    snap = T.model_snapshot(engine)
    # params: same multiset (regrouped by node, nothing dropped or duplicated)
    gp = _flat_params(gs)
    assert len(gp) == len(snap["params"])
    assert sorted(p["name"] for p in gp) == sorted(p["name"] for p in snap["params"])
    # clusters: same set, field-for-field
    gc = {c["name"]: c for c in _flat_clusters(gs)}
    sc = {c["name"]: c for c in snap["clusters"]}
    assert set(gc) == set(sc)
    for name in gc:
        assert gc[name] == sc[name]
    # summary: identical (incl. competence/learnedness/skill)
    summ = _global(gs, "summary")
    assert {k: summ[k] for k in snap["summary"]} == dict(snap["summary"])


def test_snapshot_is_cheap_never_calls_context(monkeypatch):
    """The live-board discipline: graph_state must build from O(1) reads only — NEVER the heavy
    engine.context() (density-matrix rebuild + fractal/bridge recompute). Poison context() and
    the snapshot must still succeed."""
    engine = _engine()

    def _boom(*a, **k):
        raise AssertionError("graph_state must not call engine.context()")

    monkeypatch.setattr(engine, "context", _boom, raising=False)
    gs = G.graph_state(engine)            # must not raise
    assert gs["topology"]["nodes"]


def test_frontend_view_contract():
    """Lock the organs a projection frontend depends on, so a backend refactor that drops one
    breaks here instead of silently white-screening the app's graph page."""
    gs = G.graph_state(_engine())
    gtypes = {g["type"] for g in gs["globals"]}
    assert {"agency", "competence", "run_gauge", "summary"} <= gtypes
    # the topology drill-in (glance / child cards) needs belief clusters on real nodes
    assert any(o["type"] == "bloch_cluster" for n in gs["topology"]["nodes"] for o in n["organs"])
    # the transparency flat view needs params + clusters to flatten
    assert _flat_params(gs) and _flat_clusters(gs)
    # competence carries the breakdown a console panel shows
    cmp = _global(gs, "competence")
    assert {"competence", "learnedness", "skill"} <= set(cmp)
    # the ingest organ makes a foreign-world ingest legible (unmatched signals surface)
    ing = _global(gs, "ingest")
    assert ing is not None and "unmatched" in ing


# ── edge cases: foreign-shaped + broken inputs (graph_state must be topology-AGNOSTIC + never 500) ──
class _FakeNode:
    def __init__(self, name, kind="region", roles=None, parent=None):
        self.name = name; self.kind = kind; self.roles = roles or []
        self.children = {}; self.grid_pos = (0, 0); self.folded = False
        self._parent = parent
        if parent is not None:
            parent.children[name] = self
    @property
    def parent(self): return self._parent
    @property
    def path(self):
        p, n = [], self
        while n: p.append(n.name); n = n._parent
        return ".".join(reversed(p))
    @property
    def depth(self):
        d, n = 0, self._parent
        while n: d += 1; n = n._parent
        return d
    @property
    def is_leaf(self): return not self.children
    def walk(self):
        yield self
        for c in self.children.values():
            yield from c.walk()


class _FakeGraph:
    def __init__(self, root): self.root = root; self.bridges = []
    def all_nodes(self): return list(self.root.walk())


class _FakeEngine:
    """Minimal engine with a FOREIGN-shaped graph and no field/agency/etc. — mimics what a
    different domain (or a half-built one) looks like to graph_state."""
    def __init__(self, graph): self.graph = graph; self.field = None


def _foreign_graph():
    # a world with names + roles the gridworld build never has
    root = _FakeNode("station", kind="root", roles=["crewed"])
    _FakeNode("airlock", roles=["crewed", "pressure"], parent=root)
    bay = _FakeNode("cargo_bay", roles=["stocked"], parent=root)
    _FakeNode("charger", kind="device", roles=["power"], parent=bay)
    return _FakeGraph(root)


def test_foreign_shaped_graph_projects_structurally():
    """graph_state is TOPOLOGY-AGNOSTIC: a foreign world (alien names/roles, no field) still
    projects a clean topology — the structural half of 'ingest a different world' works
    without code changes."""
    gs = G.graph_state(_FakeEngine(_foreign_graph()))
    names = {n["name"] for n in gs["topology"]["nodes"]}
    assert names == {"station", "airlock", "cargo_bay", "charger"}
    station = next(n for n in gs["topology"]["nodes"] if n["name"] == "station")
    assert station["children"] == ["airlock", "cargo_bay"]
    ch = next(n for n in gs["topology"]["nodes"] if n["name"] == "charger")
    assert ch["path"] == "station.cargo_bay.charger" and ch["depth"] == 2 and ch["is_leaf"]
    # no field → no belief/param organs, but the envelope + summary global still hold
    assert next((g for g in gs["globals"] if g["type"] == "summary"), None) is not None


def test_model_snapshot_failure_degrades_to_topology(monkeypatch):
    """If the whole-graph projection throws (a broken/foreign field), graph_state must DEGRADE
    to topology-only, never 500. Poison model_snapshot and assert nodes still come back."""
    monkeypatch.setattr(G._transparency, "model_snapshot",
                        lambda r: (_ for _ in ()).throw(RuntimeError("boom")))
    gs = G.graph_state(_FakeEngine(_foreign_graph()))
    assert len(gs["topology"]["nodes"]) == 4        # topology survived the projection failure


def test_empty_and_none_engine_never_crash():
    assert G.graph_state(None)["topology"]["nodes"] == []
    assert G.graph_state(_FakeEngine(_FakeGraph(_FakeNode("solo"))))["topology"]["nodes"][0]["name"] == "solo"


def test_node_organs_attach_to_owning_node():
    gs = G.graph_state(_engine())
    by_name = {n["name"]: n for n in gs["topology"]["nodes"]}
    # a belief cluster organ, if present on a node, is named for that node
    for n in gs["topology"]["nodes"]:
        for o in n["organs"]:
            if o["type"] == "bloch_cluster":
                assert o["name"] == n["name"]
    # params attached to a node all declare that node
    for n in gs["topology"]["nodes"]:
        for o in n["organs"]:
            if o["type"] == "param_fiber":
                assert all(p["node"] == n["name"] for p in o["params"])
    assert "grid" in by_name        # the root is present
