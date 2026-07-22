"""cognifold_trace — the belief-field as a renderable Bloch atlas (reasoning-transparency export).

Where `graph_state` projects the whole legible graph for a web console, this projection emits the
*bulk full-gauge* readout a 3D field renderer wants: one flat list of belief REGISTERS, each carrying
its Bloch spherical coordinates plus the unified gauge (value/confidence/reliability/forecast_skill),
wired by the couplings that already exist. A separate renderer (SpaceWheat's 3D cognifold field, which
reads per-register Bloch spherical coords) draws it directly — turning the belief-field into an
instrument you can watch reason.

It COMPOSES the existing cheap reads — it invents no new physics:

  • geometry — a pure Bloch-Cartesian→spherical transform of each leaf's own Bloch vector
    (`cluster.role_bloch(role)`), through the repo's `substrate.bloch` atlas so it stays consistent
    with every other chart. `r_bloch` IS the confidence coordinate; `purity` is the PER-REGISTER
    single-qubit form (1+|r|²)/2 (NOT the cluster's joint Tr(ρ²)).
  • gauge    — `reliability` + `forecast_skill` ride in from `host.api.beliefs()` (the canonical unified
    gauge); both may be `null` (a never-observed leaf / no forecaster attached).
  • forecast — per leaf, the fractal horizon ladder from an attached `ForecastSurface.predictions()`,
    else empty (the default: `DEFAULT_FORECAST_LEAVES` is empty, so a domain-free world has no leaves).
  • edges    — the couplings that already exist, re-voiced onto register indices: cross-cluster
    `graph.bridges` (kind "bridge"), intra-cluster learned `zz`/`xy` from `transparency.model_snapshot`
    (kinds "zz"/"xy" — the persistent Hamiltonian SKELETON), and the beliefs' actual shared
    information `mi` (kind "mi", bits) from the fast connected-correlation sensor (the PULSE — empty at
    the product resting state, alight where correlation genuinely lives).
  • manifold — the per-cluster higher-order structure of the comprehension manifold: how BOUND each
    cluster's beliefs are (`total_correlation`), the binding's CHARACTER (`o_information` → `grain`:
    chorus=redundancy / conspiracy=synergy, dense-exact-only), the VI `metric` between beliefs, and the
    MI-graph `constellations`. See substrate/higher_order.py, substrate/information_geometry.py.

Additive + cheap: per-cluster reads on small clusters, never `engine.context()`. Degrades to an
empty-but-valid envelope on a broken/foreign engine rather than raising. See graph_state.py, host/api.py.
"""
from __future__ import annotations

import math
from typing import Any

from umwelt.projection import transparency as _transparency
from umwelt.projection.graph_state import _graph as _resolve_graph
from umwelt.substrate.backend import is_param_fiber
from umwelt.substrate.bloch import bloch_radius, qubit_purity

FORMAT = "cognifold_trace_v1"
_EPS = 1e-12


# ── the pure geometry (a single register's chart) ─────────────────────────────

def register_geometry(x: float, y: float, z: float) -> dict:
    """Pure Bloch-Cartesian (x,y,z) → the register's spherical + gauge geometry.

    Inverse-consistent with the `substrate.bloch` atlas: `r_bloch = |r|` is the confidence
    coordinate, `purity = (1+|r|²)/2` is the single-qubit form (per-register, recomputed from THIS
    leaf's radius — never the cluster's joint Tr(ρ²)). The round-trip law holds to float precision:
    given {theta, phi, r_bloch, r_xy}, (x,y,z) reconstructs as
        x = r_xy·cos(phi),  y = r_xy·sin(phi),  z = r_bloch·cos(theta)
    (pinned in tests/test_cognifold_trace.py)."""
    x, y, z = float(x), float(y), float(z)
    r_bloch = bloch_radius(x, y, z)          # |r| — the confidence coordinate
    denom = r_bloch if r_bloch > _EPS else _EPS
    theta = math.acos(max(-1.0, min(1.0, z / denom)))
    phi = math.atan2(y, x)
    r_xy = math.hypot(x, y)
    p0 = (z + 1.0) / 2.0
    purity = float(qubit_purity((x, y, z)))  # (1+|r|²)/2 — this leaf's own radius
    return {
        "p0": p0, "p1": 1.0 - p0,
        "r_xy": r_xy, "r_bloch": r_bloch,
        "phi": phi, "theta": theta, "purity": purity,
    }


# ── input normalization: reuse host.api.beliefs() whether host or raw engine ──

def _engine_and_beliefs(host_or_engine: Any):
    """Return (engine, beliefs_map). `beliefs_map` is `{"node.role": Belief}` from the canonical
    `host.api.beliefs()` readout — called directly on a GameHost, or via a GameHost bound to a raw
    engine (the worker path). Both routes share ONE gauge implementation; never raises."""
    obj = host_or_engine
    # A raw BeliefEngine carries `.field.clusters`; a host/worker carries `.engine`.
    if hasattr(obj, "field") and hasattr(getattr(obj, "field", None), "clusters"):
        eng = obj
    elif hasattr(obj, "engine"):
        try:
            eng = obj.engine
        except Exception:
            eng = obj
    else:
        eng = obj

    beliefs_map: dict = {}
    try:
        if hasattr(obj, "beliefs") and callable(obj.beliefs):
            beliefs_map = obj.beliefs()
        else:
            from umwelt.host.api import GameHost
            h = GameHost()
            h._engine = eng
            beliefs_map = h.beliefs()
    except Exception:
        beliefs_map = {}
    return eng, beliefs_map


def _world_name(eng: Any, world: str | None) -> str:
    if world:
        return str(world)
    try:
        root = getattr(getattr(eng, "graph", None), "root", None)
        if root is not None and getattr(root, "name", None):
            return str(root.name)
    except Exception:
        pass
    return "umwelt"


def _belief_clusters(eng: Any):
    """The belief clusters, in field order — the SAME set transparency/graph_state renders as
    `bloch_cluster` organs (skip the param fibers and the `_`-prefixed internals). Yields
    (name, cluster, roles) with roles in `qubit_roles` order (so transparency's zz/xy `a`/`b`
    role names index the registers built here)."""
    field = getattr(eng, "field", None)
    clusters = getattr(field, "clusters", None) or {}
    for name, c in clusters.items():
        if str(name).startswith("_"):
            continue
        try:
            if is_param_fiber(c):
                continue
        except Exception:
            continue
        roles = list(getattr(c, "qubit_roles", ()) or ())
        if roles:
            yield name, c, roles


def _forecasts_by_leaf(eng: Any) -> dict:
    """`{(node, role): [{"horizon_min", "z_pred", "skill"}, ...]}` from an attached ForecastSurface.
    Empty when no forecaster is attached, or its leaf allowlist is empty (the domain-free default)."""
    out: dict = {}
    fs = getattr(eng, "forecast_surface", None)
    if fs is None:
        return out
    try:
        preds = fs.predictions()  # {(node, role, horizon): {z_pred, skill, horizon_min, ...}}
    except Exception:
        return out
    for (node, role, _h), d in preds.items():
        try:
            hm = float(d.get("horizon_min"))
        except (TypeError, ValueError):
            continue
        entry = {
            "horizon_min": int(hm) if hm.is_integer() else round(hm, 3),
            "z_pred": d.get("z_pred"),
            "skill": d.get("skill"),
            "skill_vs_persistence": d.get("skill_vs_persistence"),
        }
        out.setdefault((node, role), []).append(entry)
    for leaf in out:
        out[leaf].sort(key=lambda e: (e["horizon_min"] is None, e["horizon_min"]))
    return out


# ── edges: re-voice the couplings that already exist onto register indices ────

def _bridge_edges(eng: Any, reg_id: dict) -> list[dict]:
    """Cross-cluster bridge couplings (kind "bridge"). Same source as graph_state's `_edges`
    (`graph.bridges`), expanded from node↔node to register↔register per shared role (or, for a
    directed tendril, per `role_map` pair). weight = the bridge's coupling_weight."""
    out: list[dict] = []
    graph = _resolve_graph(eng)
    for b in (getattr(graph, "bridges", None) or []):
        try:
            weight = round(float(getattr(b, "coupling_weight", 0.0)), 6)
            role_map = dict(getattr(b, "role_map", {}) or {})
            if role_map:
                pairs = list(role_map.items())          # directed tendril: src_role → tgt_role
            else:
                pairs = [(r, r) for r in (getattr(b, "shared_roles", None) or [])]
            for src_role, tgt_role in pairs:
                i = reg_id.get((b.source, src_role))
                j = reg_id.get((b.target, tgt_role))
                if i is None or j is None or i == j:
                    continue
                out.append({"i": i, "j": j, "weight": weight, "kind": "bridge"})
        except Exception:
            continue
    return out


def _intra_edges(eng: Any, reg_id: dict) -> list[dict]:
    """Intra-cluster learned couplings from transparency.model_snapshot: `zz` Ising (kind "zz",
    weight = |J|) and `xy` exchange (kind "xy", weight = |k| = √(kxx²+kyy²))."""
    out: list[dict] = []
    try:
        snap = _transparency.model_snapshot(eng)
    except Exception:
        return out
    for cl in snap.get("clusters", []) or []:
        name = cl.get("name")
        for e in cl.get("zz", []) or []:
            i = reg_id.get((name, e.get("a")))
            j = reg_id.get((name, e.get("b")))
            if i is None or j is None or i == j:
                continue
            out.append({"i": i, "j": j,
                        "weight": round(abs(float(e.get("j", 0.0))), 6), "kind": "zz"})
        for e in cl.get("xy", []) or []:
            i = reg_id.get((name, e.get("a")))
            j = reg_id.get((name, e.get("b")))
            if i is None or j is None or i == j:
                continue
            mag = math.hypot(float(e.get("kxx", 0.0)), float(e.get("kyy", 0.0)))
            out.append({"i": i, "j": j, "weight": round(mag, 6), "kind": "xy"})
    return out


def _mi_edges(eng: Any, reg_id: dict, *, floor: float = 1e-4) -> list:
    """Real quantum mutual-information edges: I(A:B) in bits between co-located beliefs, via the
    fast connected-correlation sensor (substrate.mutual_information — product pairs cost only the
    gate, so this is ~free). DISTINCT from the zz/xy Hamiltonian-COUPLING edges: those are learned
    potential; this is the beliefs' ACTUAL shared information right now. Empty when the field sits
    at product (the common resting state, since observations decorrelate) and lights up where
    correlation genuinely lives. Cumulant clusters only (dense joint rho carries a spurious residue)."""
    out: list = []
    field = getattr(eng, "field", None)
    if field is None:
        return out
    try:
        from umwelt.substrate.mutual_information import field_mi_scan
        for node, pairs in field_mi_scan(field, floor=floor, cumulant_only=True).items():
            for p in pairs:
                i = reg_id.get((node, p["role_i"]))
                j = reg_id.get((node, p["role_j"]))
                if i is None or j is None or i == j:
                    continue
                out.append({"i": i, "j": j, "weight": round(float(p["mi_bits"]), 6), "kind": "mi"})
    except Exception:
        pass
    return out


def _actions_section(eng: Any, reg_id: dict) -> list:
    """The decision layer: each output tendril's current committed recommendation, wired to the
    belief register that DRIVES it (OutputSpec names a node+role → a register). The `shadow` /
    `gated` fields are the "why did it act — or NOT" honesty: a shadow output decides visibly but
    dispatches nothing (earned-autonomy un-flipped), so every decision is a ghost first. PURE
    reads only — never steps a tendril (that mutates the commit pump); guarded per-tendril,
    degrades to []."""
    out: list = []
    tendrils = getattr(eng, "tendrils", None) or []
    for t in tendrils:
        try:
            node = getattr(t, "node", None)
            role = getattr(t, "role", None)
            commit = getattr(t, "commit", None)
            level = float(getattr(commit, "level", 0.0)) if commit is not None else 0.0
            conf = float(getattr(commit, "confidence", 0.0)) if commit is not None else 0.0
            dec = getattr(getattr(t, "_decoder", None), "__name__", "") or ""
            kind = str(getattr(getattr(t, "spec", None), "kind", "") or "")
            if "linear" in dec:
                lo, hi = getattr(t, "codomain", (0.0, 1.0))
                value = float(lo) + level * (float(hi) - float(lo))
            else:  # sticky / binary
                value = 1.0 if level >= 0.5 else 0.0
            shadow = bool(getattr(t, "shadow", True))
            try:
                enabled = bool(t.enabled()) if hasattr(t, "enabled") else True
            except Exception:
                enabled = True
            held = []
            if shadow:
                held.append("shadow")
            if not enabled:
                held.append("disabled")
            sp = getattr(t, "spec", None)
            actuator = ""
            if sp is not None:
                actuator = (getattr(sp, "dispatch", {}) or {}).get("actuator_id", getattr(t, "name", ""))
            out.append({
                "id": getattr(t, "name", ""),
                "node": node, "role": role,
                "value": round(value, 4),
                "committed_level": round(level, 4),
                "confidence": round(conf, 4),
                "decode": dec.strip("_").replace("_decoder", ""),
                "kind": kind,
                "shadow": shadow,
                "gated": bool(held),
                "gates_held": held,
                "actuator_id": actuator,
                "driving_register": reg_id.get((node, role)),
            })
        except Exception:
            continue
    return out


# ── the manifold section: the field's higher-order shape and grain ────────────

def _manifold_clusters(eng: Any) -> list[dict]:
    """Per-cluster higher-order structure of the comprehension manifold, over the SAME belief
    clusters the registers/edges are built from: `total_correlation` (how bound), `o_information` →
    `grain` (the binding's character — chorus/conspiracy, dense-exact-only, else 'flat'), the VI
    `metric` distance matrix between beliefs, and the MI-graph `constellations`. Only clusters with
    ≥2 beliefs appear (a lone belief has nothing to bind). Guarded per-cluster, degrades to [] —
    never raises. Reuses substrate.higher_order + substrate.information_geometry (which ride the same
    cumulants the MI sensor already touches)."""
    from umwelt.substrate.higher_order import cluster_higher_order
    from umwelt.substrate.information_geometry import constellations, vi_distance_matrix

    out: list[dict] = []
    for name, c, roles in _belief_clusters(eng):
        try:
            ho = cluster_higher_order(c)
            if ho is None:                       # <2 roles → no higher-order structure
                continue
            cons = constellations(c)
            m_roles, D = vi_distance_matrix(c)
            oi = ho["o_information"]
            out.append({
                "name": name,
                "roles": list(roles),
                "n": ho["n"],
                "total_correlation": round(float(ho["total_correlation"]), 6),
                "o_information": None if oi is None else round(float(oi), 6),
                "grain": ho["grain"],
                "tier": ho["tier"],
                "max_corr_norm": round(float(ho["max_corr_norm"]), 6),
                "constellations": cons,
                "metric": {"roles": list(m_roles),
                           "distance": [[round(float(v), 6) for v in row] for row in D]},
            })
        except Exception:
            continue
    return out


# ── the export ────────────────────────────────────────────────────────────────

def cognifold_trace(host_or_engine: Any, *, world: str | None = None) -> dict:
    """The belief-field as a `cognifold_trace_v1` document: a flat register list (Bloch spherical
    geometry + unified gauge + per-leaf forecast) plus the coupling edges, indexed so `edges[k].i/.j`
    address `registers`. Reuses `host.api.beliefs()` (gauge) and `transparency`/`graph.bridges`
    (topology). Cheap, additive, guarded — degrades to a valid empty envelope, never raises."""
    eng, beliefs_map = _engine_and_beliefs(host_or_engine)
    forecasts = _forecasts_by_leaf(eng)

    manifold_clusters = _manifold_clusters(eng)
    # (node, role) → its cluster-local constellation id, so each register can name the community it
    # belongs to (None where the cluster has no higher-order structure — a lone belief).
    constellation_of: dict[tuple, int] = {}
    for cl in manifold_clusters:
        for comp_id, comp in enumerate(cl["constellations"]):
            for role in comp:
                constellation_of[(cl["name"], role)] = comp_id

    registers: list[dict] = []
    reg_id: dict[tuple, int] = {}
    for name, c, roles in _belief_clusters(eng):
        for role in roles:
            try:
                bloch = c.role_bloch(role)
                x, y, z = float(bloch[0]), float(bloch[1]), float(bloch[2])
            except Exception:
                continue
            geom = register_geometry(x, y, z)
            rid = len(registers)
            reg_id[(name, role)] = rid
            belief = beliefs_map.get(f"{name}.{role}")
            registers.append({
                "id": rid, "node": name, "role": role,
                **geom,
                "value": geom["p0"], "confidence": geom["r_bloch"],
                "reliability": getattr(belief, "reliability", None),
                "forecast_skill": getattr(belief, "forecast_skill", None),
                "forecast": forecasts.get((name, role), []),
                "constellation": constellation_of.get((name, role)),
            })

    edges = _bridge_edges(eng, reg_id) + _intra_edges(eng, reg_id) + _mi_edges(eng, reg_id)

    return {
        "world": _world_name(eng, world),
        "format": FORMAT,
        "n_registers": len(registers),
        "registers": registers,
        "edges": edges,
        "actions": _actions_section(eng, reg_id),
        "manifold": {"clusters": manifold_clusters},
    }
