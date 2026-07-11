"""Forecast comprehension circuit — per-φ-cluster foresight geometry.

The forebrain enriches its present model with a projection of how that model will
be one φ-rung ahead, and READS OFF three cheap cumulant-space quantities. For a
cluster c, three reference states at the cluster's natural φ-horizon Δτ_c:

  present     z_c(t)       — the live Bloch-z (held in e1)
  endogenous  ẑ_c^self     — forecast_z: the pure-H rollout (γ=0, no inputs) =
                             "where I drift under my OWN learned dynamics"
  enriched    ẑ_c^enr      — the forecast brains' published stream = "what actually
                             tends to happen" (learned inputs / exogenous patterns)

From those (all O(n) vector ops on e1 — "easy compute" on the cumulant engine):

  disparity      D_c = ‖ẑ^enr − z(t)‖                 is my slice of reality moving?
  downstream     endo_c = cos(Δself, Δenr), gated by   did I cause it? (R_c = is c
                 tendril-reachability R_c               graph-downstream of an actuator)
                 downstream_c = R_c · max(0, endo_c)
  meta_surprise  S_c = ‖ẑ^enr − ẑ^self‖                the forecast projects I'll be
                                                        surprised → exogenous shock

The gem: S_c high ∧ downstream_c ≈ 0 → an exogenous event is coming that I cannot
derive from my own state ("not downstream from us"). Clusters with no actuator in
their subtree (an external periodic driver) are R_c = 0 by construction — their
predicted change is exogenous, correctly.

OBSERVE-ONLY: this computes + exposes the geometry; it never drives behaviour. The
contract is to validate the foresight reads true on live data before anything acts
on it. See the forecast-comprehension vision + [[project_confidence_gauge_braid]].
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from umwelt.clocks.phi_clock import fib_strides_at

_EPS = 1e-9
_DEVICE_KINDS = frozenset({"actuator", "sensor", "component", "appliance"})


def tendril_reachable_clusters(graph, cluster_names) -> set[str]:
    """The R_c gate: cluster names DOWNSTREAM of one of our actuators (we can
    influence them). A cluster is reachable if its subtree contains an actuator
    leaf, OR it is bridge-connected to such a cluster (our actuation propagates
    along the tendril bridges). Clusters with no path from an actuator —
    an external periodic driver — are NOT downstream: their predicted change is
    exogenous by construction. Coarse (cluster-level) but it captures the decisive
    split (the external driver is upstream of us; a driven output is downstream)."""
    names = set(cluster_names)
    has_actuator: set[str] = set()
    for name in names:
        node = graph.find(name) if hasattr(graph, "find") else None
        if node is None:
            continue
        for sub in node.walk():
            if sub is node:
                continue
            if getattr(sub, "kind", None) == "actuator":
                has_actuator.add(name)
                break
    reachable = set(has_actuator)
    # One hop along bridges: a cluster bridged to an actuated cluster is downstream.
    for b in getattr(graph, "bridges", []) or []:
        s, t = getattr(b, "source", None), getattr(b, "target", None)
        if s in has_actuator and t in names:
            reachable.add(t)
        if t in has_actuator and s in names:
            reachable.add(s)
    return reachable


def _phi_horizon(dt_factor: float = 1.0) -> int:
    """The single natural φ-rung look-ahead, in field steps: the first Fibonacci
    stride of the φ-clock (8 at live dt_factor=1.0; slides up under replay)."""
    return int(fib_strides_at(max(1.0, float(dt_factor)), 1)[0])


class ForecastComprehension:
    """Read-only per-cluster forecast-comprehension overlay. Hand it the live field
    and the reachable-cluster set; each tick, pass the enriched forecast (the
    forecast brains' stream) and get back the per-cluster foresight geometry."""

    def __init__(self, field, reachable: set[str] | None = None,
                 dt_factor: float = 1.0):
        self.field = field
        self.reachable = set(reachable or ())
        self.horizon = _phi_horizon(dt_factor)
        # Latest enriched z per (cluster, role), fed by the forecast-stream tap. The
        # tap writes; comprehend() reads. Latest-value wins (forecasts refresh ~30s).
        # Each entry is (z, monotonic_ts): a leaf older than stale_after_s DROPS OUT of
        # comprehend() — the resident_a constant-1.984 lesson (2026-07-07): the forecast
        # tiers went quiet and the comparison kept scoring live reality against a fossil,
        # printing the same meta-surprise for DAYS. No data must read as absent, not as
        # a frozen number.
        self._enriched: dict[str, dict[str, tuple[float, float]]] = {}
        self.stale_after_s = 600.0

    def feed(self, cluster: str, role: str, z: float, confidence: float = 1.0) -> None:
        """Tap entry point: record one leaf's enriched forecast (the forecast brains'
        published z, after the merged_zone_role remap). confidence≤0 is ignored."""
        if confidence is not None and confidence <= 0.0:
            return
        import time as _time
        self._enriched.setdefault(cluster, {})[role] = (float(z), _time.monotonic())

    def comprehend(self, enriched: dict[str, dict[str, float]] | None = None) -> dict[str, dict]:
        """enriched = {cluster_name: {role: enriched_z}} (partial coverage is fine —
        only forecast leaves participate). None → use the fed buffer (the live path).
        Returns {cluster_name: {disparity, downstream, meta_surprise, endo_align,
        reachable, horizon, n_leaves}} for every cumulant cluster the stream covers."""
        if enriched is None:
            # live path: unpack the fed buffer, dropping leaves past the staleness window
            import time as _time
            now = _time.monotonic()
            enriched = {
                c: {r: z for r, (z, ts) in leaves.items()
                    if (now - ts) < self.stale_after_s}
                for c, leaves in self._enriched.items()
            }
        out: dict[str, dict] = {}
        for name, leaf_z in enriched.items():
            cluster = self.field.clusters.get(name)
            if cluster is None or not leaf_z:
                continue
            from umwelt.substrate.backend import cluster_kind
            if cluster_kind(cluster) != "cumulant":
                continue  # forecast_z is the cumulant pure-H rollout; only big/merged
                          # clusters have it. Others skipped.
            ridx = cluster.role_index
            roles = [r for r in leaf_z if r in ridx]
            if not roles:
                continue
            idx = np.array([ridx[r] for r in roles])
            z_now = np.asarray(cluster.e1[idx, 2], dtype=float)
            z_self = np.asarray(cluster.forecast_z(self.horizon), dtype=float)[idx]
            z_enr = np.array([float(leaf_z[r]) for r in roles])

            d_self = z_self - z_now          # what my own dynamics produce
            d_enr = z_enr - z_now            # what the world is forecast to do
            disparity = float(np.linalg.norm(d_enr))
            meta_surprise = float(np.linalg.norm(z_enr - z_self))
            ns, ne = np.linalg.norm(d_self), np.linalg.norm(d_enr)
            endo_align = float(np.dot(d_self, d_enr) / (ns * ne)) if ns > _EPS and ne > _EPS else 0.0
            R = 1.0 if name in self.reachable else 0.0
            out[name] = {
                "disparity": disparity,
                "meta_surprise": meta_surprise,
                "endo_align": endo_align,
                "downstream": R * max(0.0, endo_align),
                "reachable": bool(R),
                "exogenous": bool(meta_surprise > _EPS and R * max(0.0, endo_align) < 0.5),
                "horizon": self.horizon,
                "n_leaves": len(roles),
            }
        return out
