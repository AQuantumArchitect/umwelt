"""field_unify — ONE canonical field state that unpacks into the CLUSTERED form OR the MANIFOLD "1-matrix" form.

Luke's keystone: the live forebrain wants the field CLUSTERED (many small cumulant clusters — fast, real-time,
and decoupled, which is why the viz reads "hub-and-spokes"); the forecast/learning brain wants the WHOLE field
as ONE connected topology (the manifold / "1 matrix" — where every qubit can couple to every other, the
"webway"). The two are byte-identical AT THE DECOUPLED STATE, because a cumulant cluster's EOM for qubit i
depends only on i's own (e1, e2) + the couplings that TOUCH i — never on cross-cluster correlations when the
cross-cluster couplings are zero. So:

    manifold  =  one CumulantCluster over all N qubits, carrying the full e1[N,3] / e2[N,N,3,3] / h / zz / xy
    clusters  =  the manifold SLICED into per-cluster sub-blocks; cross-cluster e2 + couplings DROPPED

`pack(clusters)` builds the canonical state (cross-cluster e2 = the product e1_i⊗e1_j; cross couplings = whatever
the input carries — none for separate clusters, the full web for a single learned manifold). `to_clusters`
slices it back (the forebrain: drops the web → fast). `to_manifold` materializes the one connected matrix (the
forecast brain: keeps the web → a collapse propagates everywhere a coupling reaches). Same pickle, two unpackings.
"""
from __future__ import annotations

import hashlib
import pickle

import numpy as np

from umwelt.substrate.cumulant_cluster import CumulantCluster

CANON_VERSION = 1


def belief_clusters(field) -> dict:
    """The belief-bearing clusters of a field — excludes the '_'-prefixed param fibers + product clusters."""
    from umwelt.substrate.backend import is_param_fiber
    return {n: c for n, c in field.clusters.items()
            if not n.startswith("_") and not is_param_fiber(c)
            and getattr(c, "qubit_roles", None)}


def pack_field(field) -> dict:
    """Canonical state of a whole field's belief clusters (the on-demand bridge for save/load)."""
    return pack(belief_clusters(field))


def canon_hash(canon: dict) -> str:
    """Content hash of the canonical field state — the field's GAUGE COORDINATE. Two forms (clustered /
    manifold) carrying the same hash are PROVABLY the same brain; an unchanged hash across a save = an empty
    diff = provable non-training (the clock-tape gauge ↔ git contract). 16 hex chars."""
    h = hashlib.sha256()
    # the LEARNED content (beliefs + couplings) — NOT the partition (topology), so re-saving the same brain
    # in the same form gives the same hash regardless of incidental cluster grouping = clean non-training signal.
    for arr in (canon["e1"], canon["e2"], canon["h"]):
        h.update(np.ascontiguousarray(np.asarray(arr, float)).round(10).tobytes())
    for tag, d in (("zz", canon["zz"]), ("xy", canon["xy"])):
        for k in sorted(d):
            h.update(repr((tag, k, np.round(np.asarray(d[k], float), 10).tolist())).encode())
    return h.hexdigest()[:16]


def _modes_of(c) -> dict:
    diss = getattr(c, "_diss", set()) or set()
    return {r: ("dissipative" if i in diss else "unitary") for i, r in enumerate(c.qubit_roles)}


def _diss_rates_of(c) -> dict:
    """role → per-qubit dissipative rate (only for dissipative qubits with a non-default rate)."""
    per = getattr(c, "_gamma_diss_per_qubit", {}) or {}
    return {c.qubit_roles[i]: float(r) for i, r in per.items() if 0 <= i < len(c.qubit_roles)}


def _state_of(c):
    """Canonical cumulant state (e1[n,3], e2[n,n,3,3], h[n,3], zz, xy) for a cluster of EITHER kind.
    A CUMULANT cluster is read verbatim (exact). A DENSE cluster's ρ + learned H are decomposed to the
    1-/2-RDM + Pauli coefficients — high-fidelity (the 2nd-order cumulant closure, ≈belief 0.97–1.0), the
    same truncation the merged manifold cluster already runs; >2-body correlations are dropped (that's how a
    small dense cluster JOINS the cumulant manifold)."""
    roles = list(c.qubit_roles)
    n = len(roles)
    gamma = float(getattr(c, "gamma", getattr(getattr(c, "evolver", None), "gamma", 0.05)) or 0.05)
    dt = float(getattr(c, "dt", getattr(getattr(c, "evolver", None), "dt", 0.01)) or 0.01)
    if hasattr(c, "_xy"):                                  # CUMULANT — verbatim, exact
        return (np.asarray(c.e1, float), np.asarray(c.e2, float), np.asarray(c._h, float),
                dict(c._zz), dict(c._xy), _modes_of(c), _diss_rates_of(c), gamma, dt)
    # DENSE — decompose ρ (1-/2-RDM) and H (Pauli coefficients)
    from umwelt.substrate.density_matrix import pauli_x, pauli_y, pauli_z
    P = (pauli_x, pauli_y, pauli_z)
    dim = 2 ** n
    rho = np.asarray(c.rho)
    sig = [[P[a](i, n) for a in range(3)] for i in range(n)]
    e1 = np.zeros((n, 3)); e2 = np.zeros((n, n, 3, 3))
    for i in range(n):
        for a in range(3):
            e1[i, a] = float(np.real(np.trace(rho @ sig[i][a])))
    for i in range(n):
        for j in range(n):
            if i == j:
                e2[i, j] = np.outer(e1[i], e1[i])         # diagonal = product (cumulant convention)
            else:
                for a in range(3):
                    for b in range(3):
                        e2[i, j, a, b] = float(np.real(np.trace(rho @ sig[i][a] @ sig[j][b])))
    H = np.asarray(getattr(c.evolver, "H_base"))
    h = np.zeros((n, 3)); zz = {}; xy = {}
    for i in range(n):
        for a in range(3):
            h[i, a] = float(np.real(np.trace(H @ sig[i][a]))) / dim
    for i in range(n):
        for j in range(i + 1, n):
            J = float(np.real(np.trace(H @ sig[i][2] @ sig[j][2]))) / dim
            kxx = float(np.real(np.trace(H @ sig[i][0] @ sig[j][0]))) / dim
            kyy = float(np.real(np.trace(H @ sig[i][1] @ sig[j][1]))) / dim
            if abs(J) > 1e-12:
                zz[(i, j)] = J
            if abs(kxx) > 1e-12 or abs(kyy) > 1e-12:
                xy[(i, j)] = (kxx, kyy)
    modes = _modes_of(c) if hasattr(c, "_diss") else {r: "unitary" for r in roles}
    return (e1, e2, h, zz, xy, modes, _diss_rates_of(c), gamma, dt)


def pack(clusters: dict) -> dict:
    """Assemble the canonical global state from a dict {name: CumulantCluster}. The global qubit order is the
    concatenation of the clusters in iteration order. Within-cluster e2 blocks are copied verbatim; cross-cluster
    e2 is the product (uncorrelated) e1_i⊗e1_j; couplings are mapped to global indices (intra only, for separate
    clusters; the whole web if `clusters` is a single manifold). Faithful: also stores per-qubit mode + diss rate,
    and per-cluster gamma/dt so the unpackers rebuild identical dynamics."""
    names = list(clusters)
    partition = []                     # [(name, [roles], gamma, dt)]
    modes_g, diss_g, zz, xy = {}, {}, {}, {}
    e1s, hs, e2blocks, offsets, off = [], [], [], {}, 0
    for name in names:
        e1c, e2c, hc, zzc, xyc, modes, rates, gamma, dt = _state_of(clusters[name])
        roles = list(clusters[name].qubit_roles)
        partition.append((name, roles, gamma, dt))
        offsets[name] = o = off
        off += len(roles)
        e1s.append(e1c); hs.append(hc); e2blocks.append((o, e2c))
        for r in roles:
            g = f"{name}\x1f{r}"
            modes_g[g] = modes.get(r, "unitary")
            if r in rates:
                diss_g[g] = rates[r]
        for (li, lj), J in zzc.items():
            if J:
                zz[(o + li, o + lj)] = float(J)
        for (li, lj), k in xyc.items():
            kxx, kyy = (k if not np.isscalar(k) else (k, k))
            if kxx or kyy:
                xy[(o + li, o + lj)] = (float(kxx), float(kyy))
    N = off
    e1 = np.concatenate(e1s, axis=0) if e1s else np.zeros((0, 3))
    h = np.concatenate(hs, axis=0) if hs else np.zeros((0, 3))
    e2 = np.zeros((N, N, 3, 3))
    for o, e2c in e2blocks:            # diagonal blocks = each cluster's own e2 (verbatim)
        nb = e2c.shape[0]
        e2[o:o + nb, o:o + nb] = e2c
    for a in names:                    # cross blocks = product e1_i ⊗ e1_j (uncorrelated)
        oa, na = offsets[a], len(clusters[a].qubit_roles)
        for b in names:
            if a == b:
                continue
            ob, nb = offsets[b], len(clusters[b].qubit_roles)
            e2[oa:oa + na, ob:ob + nb] = np.einsum("ia,jb->ijab", e1[oa:oa + na], e1[ob:ob + nb])
    return {"version": CANON_VERSION, "partition": partition, "modes": modes_g, "diss_rates": diss_g,
            "e1": e1, "e2": e2, "h": h, "zz": zz, "xy": xy}


def _global_roles(partition):
    """Canonical (cluster, role, global_name) triples in qubit order. global_name is unique across clusters."""
    out = []
    for (name, roles, _g, _dt) in partition:
        for r in roles:
            out.append((name, r, f"{name}\x1f{r}"))
    return out


def _build(node, roles, modes, diss_rates, gamma, dt):
    kwargs = dict(gamma=gamma, dt=dt, role_modes=modes)
    rates = {r: diss_rates[r] for r in roles if r in diss_rates}
    if rates:                                  # else the constructor's scalar default (5.0) applies
        kwargs["gamma_diss"] = rates
    return CumulantCluster(node, roles, **kwargs)


def to_clusters(canon: dict) -> dict:
    """Slice the canonical state into the CLUSTERED form (the forebrain). Each cluster gets its own e1/e2/h
    sub-block + only the INTRA-cluster couplings — cross-cluster couplings are dropped (decoupled, fast)."""
    part, e1, e2, h = canon["partition"], canon["e1"], canon["e2"], canon["h"]
    zz, xy, modes, rates = canon["zz"], canon["xy"], canon["modes"], canon["diss_rates"]
    spans, off = {}, 0
    for (name, roles, _g, _dt) in part:
        spans[name] = (off, len(roles)); off += len(roles)
    out = {}
    for (name, roles, gamma, dt) in part:
        o, n = spans[name]
        cm = {r: modes[f"{name}\x1f{r}"] for r in roles}
        cr = {r: rates[f"{name}\x1f{r}"] for r in roles if f"{name}\x1f{r}" in rates}
        c = _build(name, roles, cm, cr, gamma, dt)
        c.e1[:] = e1[o:o + n]
        c.e2[:] = e2[o:o + n, o:o + n]
        zloc = {(gi - o, gj - o): J for (gi, gj), J in zz.items() if o <= gi < o + n and o <= gj < o + n}
        xloc = {(gi - o, gj - o): k for (gi, gj), k in xy.items() if o <= gi < o + n and o <= gj < o + n}
        c.set_couplings(h_fields=h[o:o + n], zz=zloc, xy=xloc)
        out[name] = c
    return out


def to_manifold(canon: dict, *, node: str = "manifold") -> CumulantCluster:
    """Materialize the canonical state as ONE connected CumulantCluster over all qubits (the forecast brain).
    Keeps EVERY coupling, intra- and cross-cluster — the full web. A collapse propagates wherever a coupling
    reaches. gamma/dt taken from the partition (must be uniform across clusters for a single-scalar manifold)."""
    triples = _global_roles(canon["partition"])
    roles = [g for (_n, _r, g) in triples]
    modes = {g: canon["modes"][g] for g in roles}
    rates = {g: canon["diss_rates"][g] for g in roles if g in canon["diss_rates"]}
    gammas = {g for (_n, _r, _g, _dt) in [] for g in []}     # placeholder; gamma below
    gset = {g for (_n, _rs, g, _dt) in canon["partition"]}
    dset = {d for (_n, _rs, _g, d) in canon["partition"]}
    gamma = next(iter(gset)) if len(gset) == 1 else max(gset)   # uniform expected; fall back to max
    dt = next(iter(dset)) if len(dset) == 1 else min(dset)
    c = _build(node, roles, modes, rates, gamma, dt)
    c.e1[:] = canon["e1"]
    c.e2[:] = canon["e2"]
    c.set_couplings(h_fields=canon["h"], zz=dict(canon["zz"]), xy=dict(canon["xy"]))
    return c


def _fill_cumulant(c, gidx, e1, e2, h, zz, xy):
    """Populate an existing cumulant cluster from the canonical global state at qubit positions `gidx`."""
    c.e1[:] = e1[gidx]
    c.e2[:] = e2[np.ix_(gidx, gidx)]
    pos = {g: k for k, g in enumerate(gidx)}
    zloc = {(pos[gi], pos[gj]): J for (gi, gj), J in zz.items() if gi in pos and gj in pos}
    xloc = {(pos[gi], pos[gj]): k for (gi, gj), k in xy.items() if gi in pos and gj in pos}
    c.set_couplings(h_fields=h[gidx], zz=zloc, xy=xloc)


def restore_into(canon: dict, field) -> int:
    """Populate an EXISTING built field's CUMULANT belief clusters from the canonical state, matched by role.
    Handles both the clustered build (a cluster named N → globals 'N\\x1f{role}') and a field_unify manifold
    (whose roles ARE the global names). Dense clusters are skipped (they load ρ natively — the cumulant
    canonical can't reconstruct >2-body ρ). Returns the number of clusters populated."""
    triples = _global_roles(canon["partition"])
    gi_of = {g: i for i, (_c, _r, g) in enumerate(triples)}
    e1, e2, h, zz, xy = canon["e1"], canon["e2"], canon["h"], canon["zz"], canon["xy"]
    done = 0
    for name, c in field.clusters.items():
        if name.startswith("_") or not hasattr(c, "_xy"):
            continue
        roles = list(c.qubit_roles)
        gA = [gi_of.get(f"{name}\x1f{r}") for r in roles]
        gB = [gi_of.get(r) for r in roles]
        gidx = gA if all(g is not None for g in gA) else (gB if all(g is not None for g in gB) else None)
        if gidx is not None:
            _fill_cumulant(c, gidx, e1, e2, h, zz, xy)
            done += 1
    return done


def manifold_from_pickle(path, *, builder=None, node: str = "manifold"):
    """Load a brain pickle (saved natively in CLUSTER form) and materialize it as the 1-matrix MANIFOLD — the
    forecast/dream brain's whole connected field. Derives the fold on demand (build clustered → load → pack →
    to_manifold), so the forebrain's hot save path stays the cheap native clustered write."""
    if builder is None:
        from umwelt.boot import build_reservoir as builder
    r = builder()
    r.load(str(path))
    canon = pack_field(r.field)
    return to_manifold(canon, node=node)


def save(canon: dict, path) -> None:
    with open(path, "wb") as fh:
        pickle.dump(canon, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load(path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)
