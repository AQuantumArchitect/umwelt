"""GAUGE — read the field's gauge state once, canonically.

The engine's state is a set of qubit Bloch vectors (the gauge field) plus each cluster's
comprehension DEPTH — how many meta-scales it climbs (the φ-clock / depth-gating). Both are
gauge-quantities in the geometric sense: the Bloch position and the Berry phase a cluster
accumulates around a cycle don't depend on how FAST you traversed it. That speed-invariance
is exactly what lets us run different parts of the engine at different comprehension rates and
still compare what each LEARNED — and it's why a git diff of these snapshots is meaningful.

  cluster_gauge(cluster, node)  → one cluster's roles' bloch + purity + comprehension scale
  field_gauge(field)            → all clusters, sorted (diff-stable)
  comprehension_scale(node[, lvl]) get/set the per-subtree speed (depth-gating knob)
  driver_phase(field, node, role)  → the phase [0,1) read off a periodic driver's clock qubit
  in_rest_window(phase, window)    → the per-cycle Poincaré section (phase in the rest window)
"""
from __future__ import annotations

import numpy as np

from umwelt.substrate.fractal_stack import cluster_max_scale_level
from umwelt.substrate.bloch import bloch_to_phase

# Default rest window as a PHASE FRACTION of the primary driver's cycle — a mid-cycle
# quiet band for stroboscopic (once-per-cycle, same-phase) sampling. Domain drivers
# override it (clocks.drivers.PeriodicDriver.rest_window).
DEFAULT_REST_WINDOW = (0.45, 0.65)


def _node_map(field) -> dict:
    g = getattr(field, "graph", None) or getattr(field, "_graph", None)
    if g is None:
        return {}
    for attr in ("nodes_with_roles", "nodes"):
        fn = getattr(g, attr, None)
        try:
            seq = fn() if callable(fn) else fn
            return {n.name: n for n in seq}
        except Exception:
            continue
    return {}


def cluster_gauge(cluster, node=None) -> dict:
    """One cluster's gauge state: per-role Bloch (rounded for diff-stability) + purity, plus
    its comprehension SCALE — how many meta-scales it climbs (its φ-clock depth). Two clusters
    with different scales run at different comprehension speeds; that difference shows here.

    A dense QubitCluster has ONE joint purity (the whole density matrix). A ProductQubitCluster
    (the param fiber `_fiber_*`, the trust/EMA banks) is N INDEPENDENT qubits — it has no single
    scalar purity, so we emit a per-role purity (each qubit's Bloch radius |r|). Without this branch
    `field_gauge` raised AttributeError on every product cluster, so the gauge could not read the
    learned coordinates it exists to certify."""
    import math
    bloch = {r: [round(float(v), 6) for v in cluster.role_bloch(r)] for r in cluster.role_index}
    g = {
        "roles": sorted(cluster.role_index),
        "bloch": {r: bloch[r] for r in sorted(bloch)},
    }
    scalar_purity = getattr(cluster, "purity", None)
    if scalar_purity is not None:
        g["purity"] = round(float(scalar_purity), 6)
    else:
        # per-qubit purity for a product cluster: |r| of each role's Bloch vector
        g["purity"] = {r: round(math.sqrt(sum(v * v for v in bloch[r])), 6)
                       for r in sorted(bloch)}
    # Input-side confidence: the edge-supplied VALIDITY of the last observation that
    # moved each role — the symmetric counterpart to purity (the output-side belief
    # confidence). Both are gauge quantities; together they make the health contract
    # visible in the clocktape ledger + the post-comprehended stream. Only present for
    # roles that have been observed-with-confidence (sensor edge supplies it).
    obs_conf = getattr(cluster, "_obs_confidence", None)
    if obs_conf:
        ridx = cluster.role_index
        g["obs_confidence"] = {
            r: round(float(obs_conf[ridx[r]]), 6)
            for r in sorted(ridx) if ridx[r] in obs_conf
        }
    if node is not None:
        try:
            g["scale_level"] = int(cluster_max_scale_level(node))
        except Exception:
            pass
    return g


def field_gauge(field) -> dict:
    """Every cluster's gauge state, keyed + sorted → a diff-stable snapshot of the whole field."""
    nm = _node_map(field)
    return {name: cluster_gauge(field.clusters[name], nm.get(name))
            for name in sorted(field.clusters)}


def driver_phase(field, node: str = "_clock", role: str | None = None):
    """The field's current phase [0,1) read off a periodic driver's clock qubit, else None.

    `node`/`role` name the driver's anchor qubit (clocks.drivers.PeriodicDriver declares
    them); the phase is the equatorial angle of that qubit's Bloch vector — the belief's
    OWN clock reading, not wall time. `role=None` → the clock cluster's first role."""
    clk = field.clusters.get(node)
    if clk is None:
        return None
    roles = getattr(clk, "role_index", {})
    if role is None:
        role = next(iter(sorted(roles)), None)
    if role is None or role not in roles:
        return None
    b = clk.role_bloch(role)
    return round(bloch_to_phase(float(b[0]), float(b[1])), 6)


def in_rest_window(phase, window: tuple[float, float] = DEFAULT_REST_WINDOW) -> bool:
    """Is the driver phase in the per-cycle rest window? The Poincaré section for
    stroboscopic (once-per-cycle, same-phase) sampling — the natural commit boundary
    for the clock tape."""
    if phase is None:
        return False
    lo, hi = window
    return lo <= (phase % 1.0) < hi


def comprehension_scale(node, level: int | None = None) -> int:
    """Get (or SET, if `level` given) a subtree's comprehension depth — the per-cluster
    speed knob. Higher level = climbs more meta-scales = comprehends more slowly/deeply.
    This is the existing depth-gating override (node.max_scale_level), named for what it is."""
    if level is not None:
        node.max_scale_level = int(level)
    return int(cluster_max_scale_level(node))
