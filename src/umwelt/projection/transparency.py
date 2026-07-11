"""Model transparency — a LIGHTWEIGHT, legible snapshot of what the brain has become (b9.10).

The dissolution (b9.9) made every internal knob a named, bounded, gauge-tracked fiber param, and the
field is a set of cumulant clusters with learned couplings. That makes the model finally *showable*.
This serializer walks the live structure with cheap O(1) reads only — NO density-matrix rebuild, NO
`reservoir.context()` (that path is heavy + cached for the console). Three views:

  • params   — every learnable parameter (value within its [lo,hi], how settled, how many times learned)
  • clusters — the field's belief clusters (roles + Bloch z + purity): the world's organization
  • web      — the learned couplings within each cluster (zz Ising + xy exchange): what's linked to what

Read-only; safe to poll. Feeds /api/transparency → ui/transparency.html (the /transparency page).
"""
from __future__ import annotations

from typing import Any


def _param_row(node_name: str, p: Any) -> dict:
    snap = p.snapshot()                       # {name, value, sigma, prior_mean, drift, updates, frozen}
    lo, hi = getattr(p, "lo", None), getattr(p, "hi", None)
    prior_sigma = getattr(p, "prior_sigma", None)
    # "settled" = how far the posterior width has shrunk from its prior (1 = fully settled, 0 = unlearned).
    settled = None
    if prior_sigma:
        try:
            settled = max(0.0, min(1.0, 1.0 - (p.sigma / prior_sigma)))
        except Exception:
            settled = None
    # position of the value within its hard range, for a bar render
    frac = None
    if lo is not None and hi is not None and hi > lo:
        frac = max(0.0, min(1.0, (p.value - lo) / (hi - lo)))
    snap.update(node=node_name, lo=lo, hi=hi, settled=settled, frac=frac)
    return snap


def _role_groups(role_names: list, zvals: dict, sectors) -> dict | None:
    """The COMPREHENSIBLE view of a merged cluster: group its '{region}_{role}' names by
    owning region, and overlay the (overlapping) sector cover with derived aggregate reads. Pure
    projection — the one-cluster manifold is untouched; regions and sectors are VIEWS over the
    flat role list, so the flat `roles` stays the single source of the bloch data (groups
    reference by name). Returns None when no role carries a folded prefix (per-region spokes,
    bare fibers), keeping those rows byte-identical."""
    # Folded-topology grouping needs the fold transform's role map (arrives with the
    # generic fold transform — see substrate/param_bundles._attach note). Until then no
    # role carries a folded prefix, so there is nothing to group.
    return None
    split_merged_role = merged_role = None  # unreachable; kept for the fold-transform seam
    regions: dict[str, list] = {}
    ungrouped: list = []
    for rn in role_names:
        sp = split_merged_role(rn)
        (regions.setdefault(sp[0], []) if sp else ungrouped).append(rn)
    if not regions:
        return None
    out: dict = {"regions": [{"name": z, "roles": rs} for z, rs in regions.items()]}
    if ungrouped:
        out["ungrouped"] = ungrouped
    secs = []
    for sec in (sectors or ()):
        for role in getattr(sec, "roles", ()) or ():
            members = [merged_role(m, role) for m in sec.members]
            members = [m for m in members if m in zvals]
            if len(members) < 2:
                continue
            zs = [zvals[m] for m in members]
            # z_max ≈ "any member up" (the OR-ish asserted read); z_mean = the sector's tone.
            entry = {"name": sec.name, "role": role, "members": members,
                     "z_mean": round(sum(zs) / len(zs), 3), "z_max": round(max(zs), 3)}
            # a sector HUB qubit (UMWELT_SECTOR_QUBITS) carries the sector's own LEARNED belief
            hub = zvals.get(merged_role(sec.name, role))
            if hub is not None:
                entry["z_self"] = round(hub, 3)
            secs.append(entry)
    if secs:
        out["sectors"] = secs
    return out


def _cluster_row(name: str, c: Any, sectors=()) -> dict:
    roles = list(getattr(c, "qubit_roles", ()) or ())
    rdata = []
    for r in roles:
        try:
            b = c.role_bloch(r)
            rdata.append({"role": r, "x": round(float(b[0]), 3),
                          "y": round(float(b[1]), 3), "z": round(float(b[2]), 3)})
        except Exception:
            continue
    purity = getattr(c, "purity", None)
    if callable(purity):
        try:
            purity = purity()
        except Exception:
            purity = None
    from umwelt.substrate.backend import cluster_kind
    kind = cluster_kind(c)
    is_cum = kind == "cumulant"
    info = {"name": name, "n_qubits": len(roles), "roles": rdata,
            "purity": (round(float(purity), 3) if purity is not None else None),
            "kind": kind}
    groups = _role_groups(roles, {d["role"]: d["z"] for d in rdata}, sectors)
    if groups:
        info["groups"] = groups
    # the coupling web (cumulant clusters carry learned h/zz/xy)
    if is_cum and roles:
        zz = []
        for (i, j), J in (getattr(c, "_zz", {}) or {}).items():
            if abs(float(J)) < 1e-6:
                continue
            zz.append({"a": roles[i], "b": roles[j], "j": round(float(J), 4)})
        xy = []
        for (i, j), kk in (getattr(c, "_xy", {}) or {}).items():
            kxx, kyy = float(kk[0]), float(kk[1])
            if abs(kxx) < 1e-6 and abs(kyy) < 1e-6:
                continue
            xy.append({"a": roles[i], "b": roles[j], "kxx": round(kxx, 4), "kyy": round(kyy, 4)})
        info["zz"] = sorted(zz, key=lambda d: -abs(d["j"]))
        info["xy"] = sorted(xy, key=lambda d: -(abs(d["kxx"]) + abs(d["kyy"])))
    return info


def model_snapshot(reservoir: Any) -> dict:
    """The whole legible model — params + clusters + the coupling web. Cheap reads only."""
    field = getattr(reservoir, "field", None)
    graph = getattr(field, "graph", None) or getattr(reservoir, "graph", None)

    # ── params: walk every node's bundle, de-duping shared archetype bundles (count each once) ──
    params: list[dict] = []
    seen: set[int] = set()
    if graph is not None:
        try:
            nodes = graph.all_nodes()
        except Exception:
            nodes = []
        for node in nodes:
            b = getattr(node, "param_bundle", None)
            if b is None or id(b) in seen:
                continue
            seen.add(id(b))
            for _, p in getattr(b, "params", {}).items():
                try:
                    params.append(_param_row(getattr(node, "name", "?"), p))
                except Exception:
                    continue

    # ── clusters + web: skip the param fiber (_*) and product (non-belief) clusters ──
    clusters: list[dict] = []
    web_edges = 0
    from umwelt.substrate.backend import is_param_fiber
    sectors = list(getattr(graph, "sectors", ()) or ()) if graph is not None else []
    for name, c in (getattr(field, "clusters", {}) or {}).items():
        if name.startswith("_") or is_param_fiber(c):
            continue
        try:
            row = _cluster_row(name, c, sectors=sectors)
        except Exception:
            continue
        web_edges += len(row.get("zz", [])) + len(row.get("xy", []))
        clusters.append(row)

    # "learned" = the value has MOVED from its prior. drift (in prior-sigma units) survives a reload;
    # update_count does NOT (it resets when the forebrain restarts off the pickle), so counting updates
    # would read 0 right after every deploy even though the learned values persist. Drift is honest.
    def _moved(p) -> bool:
        return abs(p.get("drift") or 0.0) > 0.01 or (p.get("updates") or 0) > 0
    for p in params:
        p["learned"] = _moved(p)
    learned = sum(1 for p in params if p["learned"])
    # competence = learnedness × prediction-skill (b9.15): the signal that EARNS the agency fold.
    # Surfaced here so the operator can WATCH the engine earn its takeover (see competence.py). Cheap
    # reads, membrane-guarded — a competence read must never break the transparency envelope.
    try:
        from umwelt.learning import competence as _competence
        comp = _competence.competence_snapshot(reservoir)
    except Exception:
        comp = {"competence": None, "learnedness": None, "skill": None}
    # storage health (the 211GB WAL incident #403, made a full ledger in b9.38): store sizes
    # plus the archive+prune ledger (prune_state.json, written by ops/maintenance/
    # prune_stores.py) so retention drift is visible on /transparency instead of discovered
    # at a wedged boot. Cheap stats + one small JSON read; never a DB query.
    wal_mb = None
    stores = None
    try:
        import json as _json
        import os as _os
        _db = _os.environ.get("UMWELT_DB_PATH", "")
        if _db:
            _dir = _os.path.dirname(_db) or "."

            def _mb(p):
                return round(_os.path.getsize(p) / 1e6, 1) if _os.path.exists(p) else None

            def _pair(p):
                return {"file_mb": _mb(p), "wal_mb": _mb(p + "-wal") or 0.0}

            wal_mb = _mb(_db + "-wal")
            stores = {
                "events": _pair(_db),
                "surprise": _pair(_os.environ.get(
                    "SURPRISE_DB_PATH", _os.path.join(_dir, "surprise.db"))),
                "streams": _pair(_os.environ.get(
                    "UMWELT_STREAMS_DB", _os.path.join(_dir, "streams.db"))),
            }
            # b9.53: the full STORE REGISTRY census rides the block — every declared store
            # with its bound + size (an undeclared store cannot appear; a declared one always
            # does). Same law as coverage_audit, applied to data.
            # (The origin appended its full store-registry census here; an app with
            # declared stores rides its own registry through this block.)
            _ps = _os.path.join(_dir, "prune_state.json")
            if _os.path.exists(_ps):
                with open(_ps) as _f:
                    _led = _json.load(_f)
                stores["last_prune"] = _led.get("last_prune")
                stores["rows_pruned"] = _led.get("rows_pruned")
                stores["rows_archived"] = _led.get("rows_archived")
    except Exception:
        stores = None
    return {
        # most-moved first (drift), so what the brain has actually shaped floats to the top
        "params": sorted(params, key=lambda p: -(abs(p.get("drift") or 0.0))),
        "clusters": sorted(clusters, key=lambda c: -c["n_qubits"]),
        "summary": {
            "param_count": len(params), "params_learned": learned,
            "cluster_count": len(clusters), "web_edges": web_edges,
            "events_wal_mb": wal_mb,
            "stores": stores,
            **comp,
        },
    }
