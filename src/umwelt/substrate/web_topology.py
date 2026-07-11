"""Phase 6 (fractal-web) — learnable web topology.

Today the bridge web is hand-declared (region adjacency). This makes the adjacency
SPARSE + LEARNABLE: grow a weak bridge between two un-bridged clusters that
co-MOVE, and decay+prune a learned bridge whose coupling falls toward zero. The
"web-ish" part of the north star — emergent connectivity, not declared. See
project_fractal_web.

Three invariants keep it honest and safe:

  • HALO STAYS CLOSED (N=0). A grown edge carries the SAME ScalarParam coupling
    (`coupling_{src}_{tgt}`, lo=0..hi=1.5) as every declared bridge — it is the
    same gauge coordinate. Phase 6 grows MORE of an in-gauge structure; it does
    not introduce a new off-gauge weight. Learning the coupling up/down is a
    logged geometric transport like any other fiber param.

  • STRUCTURE IS SACRED. Only kind="learned" edges are ever grown, decayed, or
    removed. The declared topology (door/open/wall/tendril) and structural
    parent-child coupling are never touched — they are the floor.

  • SPARSE BY CONSTRUCTION. A hard cap bounds the number of learned edges, so the
    web never collapses to dense O(n²). Growth picks the single strongest
    co-moving candidate per evolve tick; pruning removes the weakest decayed edge.

Co-movement, not co-state: two empty regions both sitting at z=-1 have high cosine
similarity but are not coupled. We score the windowed PEARSON CORRELATION of the
shared role's z over a rolling buffer — do they rise and fall TOGETHER — which is
the signal that an edge should exist.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field as dc_field

import numpy as np

from umwelt.substrate.params import ScalarParam
from umwelt.substrate.graph import Bridge

LEARNED_KIND = "learned"


def make_learned_bridge(source: str, target: str, roles, coupling: float = 0.1) -> Bridge:
    """A learned web edge: a normal Bridge whose coupling is the SAME ScalarParam
    type as any declared bridge (`coupling_{src}_{tgt}`, lo=0..hi=1.5) — an in-gauge
    coordinate, so growing one keeps the halo closed. Used both to grow at runtime
    and to re-create a persisted learned edge on load."""
    return Bridge(
        source=source, target=target, shared_roles=list(roles), kind=LEARNED_KIND,
        coupling_param=ScalarParam(
            name=f"coupling_{source}_{target}", value=coupling,
            sigma=0.1, lo=0.0, hi=1.5,
        ),
    )


@dataclass
class WebTopologyConfig:
    window: int = 16            # rolling samples used for the correlation
    grow_threshold: float = 0.6  # |corr| above which an unbridged pair earns an edge
    keep_threshold: float = 0.3  # realized corr below which a learned edge decays
    decay: float = 0.7          # coupling *= decay when a learned edge underperforms
    floor: float = 0.02         # learned coupling below this → edge removed
    grow_value: float = 0.1     # initial coupling of a freshly-grown weak edge
    max_learned: int = 8        # hard sparsity cap on learned edges
    min_samples: int = 8        # need this much history before growing/pruning
    min_activity: float = 0.15  # FALLBACK PRIOR only. The live floor is the learned, qubit-
                                # backed gauge coordinate `web_min_activity` on the root bundle
                                # (resolved each evolve tick); this value is used solely when
                                # there is no fiber (standalone use). Both clusters must MOVE
                                # above the floor to correlate — else a near-flat pair making
                                # one coincident step hits a spurious corr=1.0 (real-data lesson).


@dataclass
class WebTopology:
    """Rolling co-movement tracker + grow/prune policy over learned bridges."""

    config: WebTopologyConfig = dc_field(default_factory=WebTopologyConfig)
    # (cluster_name, role) → deque of recent z samples
    _buf: dict = dc_field(default_factory=dict, repr=False)

    def __post_init__(self):
        # The LIVE activity floor. Resolved each evolve tick from the gauge coordinate
        # (root bundle `web_min_activity`, qubit-backed); the config value is only the
        # fallback/prior for standalone use with no fiber.
        self._floor = self.config.min_activity
        # The rest of the grow/prune policy is gauge-backed too (#361); these hold the live
        # values resolved each tick (observe / maybe_evolve), seeded to the config prior so a
        # standalone WebTopology with no fiber behaves identically.
        self._window = self.config.window
        self._min_samples = self.config.min_samples
        # The floor is learned through the ONE learner object (supervised mode: collapse toward the
        # observed noise-floor quantile). See universal_learner.py / project_universal_learning_law.
        from umwelt.learning.universal_learner import UniversalLearner
        self._learner = UniversalLearner()

    # ── sampling ─────────────────────────────────────────────────
    def observe(self, field) -> None:
        """Append the current shared-role z of every cluster to its ring buffer.
        Cheap: one float per (cluster, role) per call."""
        w = int(self._resolve(getattr(field, "graph", None), "web_window", self.config.window))
        self._window = w
        for name, cluster in field.clusters.items():
            if name.startswith("_"):
                continue
            for role in cluster.qubit_roles:
                key = (name, role)
                dq = self._buf.get(key)
                if dq is None:
                    dq = self._buf[key] = deque(maxlen=w)
                dq.append(float(cluster.role_bloch(role)[2]))  # z component

    def _corr(self, a: str, b: str, role: str) -> float:
        da, db = self._buf.get((a, role)), self._buf.get((b, role))
        if da is None or db is None:
            return 0.0
        n = min(len(da), len(db))
        if n < self._min_samples:
            return 0.0
        xa = np.array(list(da)[-n:])
        xb = np.array(list(db)[-n:])
        # Activity floor: both clusters must genuinely MOVE. Without this, a near-flat
        # pair that makes one coincident step yields a spurious corr=1.0 — the real-data
        # eval grew exactly such phantom edges on short, quiet episodes. The floor is the
        # LIVE gauge coordinate (self._floor, learned + qubit-backed), not a constant.
        if xa.std() < self._floor or xb.std() < self._floor:
            return 0.0
        return float(np.corrcoef(xa, xb)[0, 1])

    # ── policy ───────────────────────────────────────────────────
    def _existing_pairs(self, graph) -> set:
        pairs = set()
        for b in graph.bridges:
            pairs.add(frozenset((b.source, b.target)))
        return pairs

    def _learned_edges(self, graph) -> list:
        return [b for b in graph.bridges if b.kind == LEARNED_KIND]

    def _resolve(self, graph, key: str, fallback: float) -> float:
        """Live-read a gauge coordinate from the root bundle (qubit-backed by _bind_param_fiber),
        or fall back to the config prior when there's no fiber (standalone use). The single read
        path for every WebTopologyConfig constant (#361 — totality of constants)."""
        root = getattr(graph, "root", None)
        if root is not None and getattr(root, "param_bundle", None) is not None:
            if root.param_bundle.get_param(key) is not None:
                return float(root.param_bundle.get(key))
        return fallback

    def _resolve_floor(self, graph) -> float:
        """The live activity floor = the gauge coordinate `web_min_activity`."""
        return self._resolve(graph, "web_min_activity", self.config.min_activity)

    def _learn_floor(self, field, graph) -> None:
        """Learn the floor toward the observed quiescent-activity level — a partial
        collapse of the `web_min_activity` qubit, exactly like every other fiber param.
        The noise floor is the LOWER QUARTILE of per-cluster movement (a robust statistic,
        not a tuned knob); the observation's uncertainty is the activity spread itself.
        No-op when the param isn't on the fiber (standalone graphs) or there isn't enough
        of an activity distribution to estimate from."""
        root = getattr(graph, "root", None)
        if root is None or getattr(root, "param_bundle", None) is None:
            return
        if root.param_bundle.get_param("web_min_activity") is None:
            return
        root_name = getattr(root, "name", None)
        acts = []
        for name, cluster in field.clusters.items():
            if name.startswith("_") or name == root_name:
                continue
            stds = [float(np.std(list(self._buf[(name, r)])))
                    for r in cluster.qubit_roles
                    if self._buf.get((name, r)) is not None
                    and len(self._buf[(name, r)]) >= self._min_samples]
            if stds:
                acts.append(max(stds))
        if len(acts) < 4:
            return
        acts = np.array(acts)
        target = float(np.quantile(acts, 0.25))          # noise floor = lower quartile (the measured target)
        obs_sigma = max(1e-3, float(np.std(acts)))        # data-derived observation σ
        # Supervised collapse toward the measured noise floor, via the ONE learner object.
        self._learner.observe(root.param_bundle.get_param("web_min_activity"), target, obs_sigma)

    def maybe_evolve(self, field, graph) -> dict:
        """One grow/prune pass. Returns {'grown': [...], 'pruned': [...]} for logging.
        Idempotent on a stable field (no spurious grow/prune)."""
        grown: list = []
        pruned: list = []
        cfg = self.config
        # Read the LIVE gauge coordinates (root bundle) before any correlation, then learn the floor.
        # Every grow/prune constant is a live-read coordinate now (#361), seeded to the config prior
        # for parity; _corr/_learn_floor read self._floor + self._min_samples set here.
        self._floor = self._resolve_floor(graph)
        self._min_samples = int(self._resolve(graph, "web_min_samples", cfg.min_samples))
        keep_threshold = self._resolve(graph, "web_keep_threshold", cfg.keep_threshold)
        decay = self._resolve(graph, "web_decay", cfg.decay)
        floor = self._resolve(graph, "web_floor", cfg.floor)
        grow_threshold = self._resolve(graph, "web_grow_threshold", cfg.grow_threshold)
        grow_value = self._resolve(graph, "web_grow_value", cfg.grow_value)
        max_learned = int(self._resolve(graph, "web_max_learned", cfg.max_learned))

        # ── PRUNE / DECAY first (frees cap budget for a better edge) ──
        for b in self._learned_edges(graph):
            realized = max((abs(self._corr(b.source, b.target, r))
                            for r in b.shared_roles), default=0.0)
            if realized < keep_threshold and b.coupling_param is not None:
                b.coupling_param.value = max(0.0, b.coupling_param.value * decay)
            if b.coupling_param is not None and b.coupling_param.value < floor:
                graph.bridges.remove(b)
                pruned.append((b.source, b.target))

        # ── GROW the single strongest co-moving unbridged pair (if under cap) ──
        if len(self._learned_edges(graph)) < max_learned:
            existing = self._existing_pairs(graph)
            # The root aggregator is not a web peer — it already gathers its children
            # via projection; a learned bridge to it would double-count, not discover.
            root_name = graph.root.name if graph.root is not None else None
            names = [n for n in field.clusters
                     if not n.startswith("_") and n != root_name]
            best = None  # (abs_corr, a, b, [roles])
            for i, a in enumerate(names):
                ca = field.clusters[a]
                for b in names[i + 1:]:
                    if frozenset((a, b)) in existing:
                        continue
                    cb = field.clusters[b]
                    shared = [r for r in ca.qubit_roles if r in cb.role_index]
                    if not shared:
                        continue
                    role_corrs = [(abs(self._corr(a, b, r)), r) for r in shared]
                    mx, role = max(role_corrs, key=lambda t: t[0])
                    if mx >= grow_threshold and (best is None or mx > best[0]):
                        best = (mx, a, b, [role])
            if best is not None:
                _, a, b, roles = best
                graph.bridges.append(make_learned_bridge(a, b, roles, grow_value))
                grown.append((a, b))

        # The floor is a gauge coordinate — learn it toward the observed noise floor.
        self._learn_floor(field, graph)
        return {"grown": grown, "pruned": pruned}
