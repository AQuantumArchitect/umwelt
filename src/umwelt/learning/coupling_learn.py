"""coupling_learn — learn a 2-qubit coupling J from REAL data by the predictive contrast, and validate it
by held-out forecast-surprise reduction. The production core shared by the dream topology organ and the
experiment CLIs (experiments/cooccurrence.py, learn_coupling_predictive.py, dream_topology_sweep.py).

The lever (see project docs / the lab thread): a rich, branchable forecast needs COUPLED beliefs. The field
learns a conditional coupling J between two qubits so that a collapse of the leader propagates to the
follower. The objective is the PREDICTIVE CONTRAST, not a static correlation (which has no gradient under
free-run):

    D(J) = follower_z | leader=+1   −   follower_z | leader=−1     (responds to J; alive gradient)

J is fit (Newton-damped) so D(J) matches the contrast the DATA shows. Two backends, one loop:
  • DENSE cluster      → perturb evolver.H_base with an exchange matrix, free-run via FieldRolloutForecaster
  • CUMULANT cluster   → perturb the sparse _xy exchange slot, free-run via forecast_z (n>16, no 2ⁿ H)

The data target D* is a lagged co-occurrence from events.db (leader_t → follower_{t+lag}), forecast/synthetic
rows excluded. An edge is only KEPT if it reduces forecast surprise on HELD-OUT real data (k-fold CV) — never
a synthetic shadow, which could be under-specified in ways that matter.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime

import numpy as np

_SX = np.array([[0, 1], [1, 0]], complex)
_SY = np.array([[0, -1j], [1j, 0]], complex)


# ─────────────────────────── events.db → predictive contrast D* ───────────────────────────

def _stream(con: sqlite3.Connection, event_type: str, device_like: str | None,
            exclude_forecast: bool = True, since: str | None = None, limit: int | None = None):
    clauses = ["event_type = ?"]
    params: list[object] = [event_type]
    if device_like:
        clauses.append("source_device LIKE ?")
        params.append(device_like)
    if exclude_forecast:                     # the brain's OWN forecasts — learning from them is circular
        clauses.append("source_device NOT LIKE 'forecast_%'")
        clauses.append("synthetic = 0")
    if since:
        clauses.append("timestamp >= ?")
        params.append(since)
    where = " AND ".join(clauses)
    if limit:    # cap to the most-recent N rows — the 3.6 GB live db has millions/stream; parsing every row
        # in Python pinned the A55 for >15 min/pass. A few thousand recent samples estimate the contrast fine.
        sql = (f"SELECT timestamp, value FROM (SELECT timestamp, value FROM events WHERE {where} "
               f"ORDER BY timestamp DESC LIMIT {int(limit)}) ORDER BY timestamp")
    else:
        sql = f"SELECT timestamp, value FROM events WHERE {where} ORDER BY timestamp"
    ts, vals = [], []
    for tstr, vstr in con.execute(sql, params):
        try:
            v = float(json.loads(vstr) if vstr and vstr[0] not in "+-.0123456789" else vstr)
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
        try:
            t = datetime.fromisoformat(tstr).timestamp()
        except ValueError:
            continue
        ts.append(t); vals.append(v)
    return np.asarray(ts, float), np.asarray(vals, float)


def _zmap(values: np.ndarray) -> np.ndarray:
    """Robustly map a raw stream to belief-z ∈ [−1,1] by its own p10..p90 (unit-free, outlier-safe)."""
    if values.size == 0:
        return values
    lo, hi = np.percentile(values, [10, 90])
    if hi <= lo:
        hi = lo + 1e-9
    return np.clip(2.0 * (values - lo) / (hi - lo) - 1.0, -1.0, 1.0)


def pull_stream(db_path: str, event_type: str, device: str | None = None, since: str | None = None,
                limit: int | None = None):
    """Open the db read-only and return (epoch_seconds, raw_values) for one real stream, or None.
    `limit` caps to the most-recent N rows (bounds the parse cost on the big live db)."""
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return None
    try:
        return _stream(con, event_type, device, since=since, limit=limit)
    finally:
        con.close()


def contrast_from_arrays(ta, va, tb, vb, lag_s: float = 30.0, tol_s: float = 60.0):
    """D* = E[b_z(t+lag) | a high] − E[..| a low] from two pre-pulled streams (or None if too sparse).
    Follower → robust belief-z; LEADER → high/low split on its RAW midpoint (a near-constant/binary leader
    like an occupancy stream that is mostly-1.0 collapses under percentile-z; the midpoint keeps present/absent)."""
    if ta.size < 20 or tb.size < 20:
        return None
    zb = _zmap(vb)
    a_mid = 0.5 * (float(np.min(va)) + float(np.max(va)))
    order = np.argsort(tb)
    tb_s, zb_s = tb[order], zb[order]
    want = ta + lag_s
    idx = np.clip(np.searchsorted(tb_s, want), 1, len(tb_s) - 1)
    left, right = idx - 1, idx
    pick = np.where(np.abs(tb_s[left] - want) <= np.abs(tb_s[right] - want), left, right)
    ok = np.abs(tb_s[pick] - want) <= tol_s
    if ok.sum() < 20:
        return None
    a_hi = (va > a_mid) & ok
    a_lo = (va <= a_mid) & ok
    if a_hi.sum() < 5 or a_lo.sum() < 5:
        return None
    b_hi = float(np.mean(zb_s[pick[a_hi]]))
    b_lo = float(np.mean(zb_s[pick[a_lo]]))
    return dict(contrast=b_hi - b_lo, b_given_hi=b_hi, b_given_lo=b_lo,
                n_hi=int(a_hi.sum()), n_lo=int(a_lo.sum()), n_match=int(ok.sum()))


def measured_contrast(db_path: str, *, a_type: str, b_type: str, a_device=None,
                      b_device=None, lag_s=30.0, tol_s=60.0, since=None):
    """Convenience: pull both streams + compute the contrast. Stream types are DOMAIN
    vocabulary, so both are required — the engine ships no default stream names."""
    a = pull_stream(db_path, a_type, a_device, since)
    b = pull_stream(db_path, b_type, b_device, since)
    if a is None or b is None:
        return None
    return contrast_from_arrays(a[0], a[1], b[0], b[1], lag_s=lag_s, tol_s=tol_s)


# ─────────────────────────── the backend-agnostic coupling learner ───────────────────────────

def learn_coupling(field, node: str, a_role: str, b_role: str, D_target: float, *,
                   horizon: int = 8, lr: float = 0.8, steps: int = 20, eps: float = 0.05,
                   operator: str = "auto", clamp_reach: bool = True) -> dict:
    """Learn coupling J so the predictive contrast D(J)=b|a=+1 − b|a=−1 matches D_target. Newton-damped
    (stable because dD/dJ≠0 under collapse). Side-effect-free: the coupling is restored on exit.
    Backend-agnostic: a CUMULANT cluster (has _xy) perturbs the sparse exchange slot + free-runs via
    forecast_z; a DENSE cluster perturbs H_base + rolls via FieldRolloutForecaster.
    Returns dict(J, D, D_target, D_raw, op_mode, reach, converged, runaway)."""
    c = field.clusters[node]
    pa, qa = c.role_index[a_role], c.role_index[b_role]
    op_mode = ("antiexchange" if D_target < 0 else "exchange") if operator == "auto" else operator

    if hasattr(c, "_xy"):                          # CUMULANT backend
        e10, e20 = c.e1.copy(), c.e2.copy()
        key = (pa, qa) if (pa, qa) in c._xy else (qa, pa)
        xy0 = c._xy.get(key, (0.0, 0.0))

        def set_coupling(J):
            c.set_couplings(xy={(pa, qa): (J, J) if op_mode == "exchange" else (J, -J)})

        def restore():
            c.e1[:], c.e2[:] = e10, e20
            c.set_couplings(xy={key: xy0})

        def branch_read(cz):
            c.e1[:], c.e2[:] = e10, e20
            c.observe_qubit(pa, (0.0, 0.0, float(cz)), alpha=1.0)
            return float(c.forecast_z(horizon)[qa])
    else:                                          # DENSE backend
        from umwelt.substrate.density_matrix import _single_qubit_op
        from umwelt.foresight.forecast_rollout import FieldRolloutForecaster
        n = c.n_qubits
        H0 = c.evolver.H_base.copy()
        XX = _single_qubit_op(_SX, pa, n) @ _single_qubit_op(_SX, qa, n)
        YY = _single_qubit_op(_SY, pa, n) @ _single_qubit_op(_SY, qa, n)
        exch = (XX + YY if op_mode == "exchange" else XX - YY).astype(H0.dtype)

        def set_coupling(J):
            c.evolver.H_base = (H0 + J * exch).astype(H0.dtype)

        def restore():
            c.evolver.H_base = H0

        def branch_read(cz):
            fc = FieldRolloutForecaster(field, dt_seconds=1.0)
            snap = fc._snapshot()
            try:
                c.observe_qubit(pa, (0.0, 0.0, float(cz)), alpha=1.0)
                return fc.forecast_freerun(horizon, [(node, b_role)])[(node, b_role)]
            finally:
                fc._restore(snap)

    def predictive_contrast(J):
        set_coupling(J)
        try:
            return branch_read(+1.0) - branch_read(-1.0)
        finally:
            restore()

    tgt, reach = D_target, None
    if clamp_reach:
        reach = abs(predictive_contrast(4.0 if D_target >= 0 else -4.0))
        tgt = float(np.clip(D_target, -0.9 * reach, 0.9 * reach))
    J, D, runaway = 0.0, predictive_contrast(0.0), False
    for _ in range(steps):
        err = tgt - D
        if abs(err) < 0.02:
            break
        dD = (predictive_contrast(J + eps) - predictive_contrast(J - eps)) / (2 * eps)
        J += lr * err / (abs(dD) + 0.05)
        if abs(J) > 20.0:
            runaway = True
            break
        D = predictive_contrast(J)
    restore()
    return dict(J=float(J), D=float(D), D_target=float(tgt), D_raw=float(D_target), op_mode=op_mode,
                reach=(None if reach is None else float(reach)),
                converged=(abs(tgt - D) < 0.05 and not runaway), runaway=runaway)


def apply_coupling(field, node: str, a_role: str, b_role: str, J: float, family: str = "exchange") -> bool:
    """Persistently install a learned coupling on a CUMULANT cluster's _xy slot (survives the pickle).
    Returns True if applied. Dense clusters aren't persisted here (their H is rebuilt from params)."""
    c = field.clusters[node]
    if not hasattr(c, "_xy"):
        return False
    pa, qa = c.role_index[a_role], c.role_index[b_role]
    c.set_couplings(xy={(pa, qa): (J, J) if family == "exchange" else (J, -J)})
    return True


# ─────────────────────────── held-out, surprise-only validation (k-fold CV) ───────────────────────────

def kfold_validate_edge(field, node, lead, foll, ta, va, tb, vb, *, folds=4, lag_s=60.0, tol_s=120.0,
                        shrink=0.5, horizon=8, margin=0.02) -> dict:
    """k-fold time-CV: for each fold, learn J on the OTHER folds' contrast (shrunk), then measure forecast-
    surprise reduction on the held-out fold. surprise_off=|D*_test| (no coupling predicts no response),
    surprise_on=|D*_test − D(J)|. An edge is ROBUST iff a majority of folds validate AND the mean reduction
    clears the margin AND no fold is strongly hurt (min > −margin). No synthetic shadows; only real held-out
    data judges. Returns dict(robust, mean_reduction, std, J, family, n_folds, reductions)."""
    allt = np.concatenate([ta, tb]) if ta.size and tb.size else np.array([])
    if allt.size < 40:
        return dict(robust=False, reason="too sparse", mean_reduction=0.0, J=0.0, family=None, n_folds=0)
    bounds = np.linspace(float(allt.min()), float(allt.max()), folds + 1)
    reductions, Js, families = [], [], []
    for f in range(folds):
        lo, hi = bounds[f], bounds[f + 1]
        tr_a = (ta < lo) | (ta >= hi); tr_b = (tb < lo) | (tb >= hi)      # train = all but this fold
        te_a = (ta >= lo) & (ta < hi); te_b = (tb >= lo) & (tb < hi)      # test  = this fold
        mtr = contrast_from_arrays(ta[tr_a], va[tr_a], tb[tr_b], vb[tr_b], lag_s=lag_s, tol_s=tol_s)
        mte = contrast_from_arrays(ta[te_a], va[te_a], tb[te_b], vb[te_b], lag_s=lag_s, tol_s=tol_s)
        if mtr is None or mte is None:
            continue
        res = learn_coupling(field, node, lead, foll, shrink * mtr["contrast"], horizon=horizon)
        if res["runaway"] or abs(res["J"]) > 10.0:
            continue
        off, on = abs(mte["contrast"]), abs(mte["contrast"] - res["D"])
        reductions.append(off - on); Js.append(res["J"]); families.append(res["op_mode"])
    n = len(reductions)
    if n == 0:
        return dict(robust=False, reason="no valid folds", mean_reduction=0.0, J=0.0, family=None, n_folds=0)
    mean_red = float(np.mean(reductions))
    frac_pos = sum(1 for r in reductions if r > 0) / n
    # ROBUST = validated on a majority of folds AND a positive mean reduction AND most held-out folds help.
    # (One noisy regime where the edge hurts shouldn't veto a coupling that reduces surprise most of the time —
    # but a coupling that only helps on average while hurting half the folds is just overfit, and fails frac_pos.)
    robust = (n >= math.ceil(folds / 2) and mean_red > margin and frac_pos >= 0.6)
    return dict(robust=robust, mean_reduction=mean_red, std=float(np.std(reductions)),
                frac_pos=round(frac_pos, 2), worst=round(min(reductions), 3),
                J=float(np.mean(Js)), family=max(set(families), key=families.count),
                n_folds=n, reductions=[round(r, 3) for r in reductions])
