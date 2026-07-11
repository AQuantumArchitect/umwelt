"""Tests for the dream topology-mutation organ: the backend-agnostic coupling learner, the held-out
k-fold surprise validator, and grow_couplings (scan→validate→consolidate). Selection is PURELY held-out
forecast-surprise reduction on REAL data — no synthetic shadows — so these build synthetic event streams
with a KNOWN follow relationship (robust) vs none (rejected) and assert the organ grows only the real edge.
"""
import json
import os
import sqlite3

import numpy as np

from umwelt.learning import coupling_learn as cl
from umwelt.foresight import dream_topology
from umwelt.substrate.cumulant_cluster import CumulantCluster

UNITARY = {"a": "unitary", "b": "unitary"}


class _FakeField:
    def __init__(self, clusters):
        self.clusters = clusters


def _fresh_field():
    cum = CumulantCluster("z", ["a", "b"], role_modes=UNITARY)
    return _FakeField({"z": cum}), cum


def _follow_streams(n=400, dt=20.0, block=600.0, lag=60.0, follow=True, seed=0):
    """Leader a toggles in blocks; follower b is high-after-a (follow) or random (not)."""
    rng = np.random.default_rng(seed)
    ta = np.arange(n) * dt
    a_block = ((ta // block).astype(int) % 2)
    va = a_block.astype(float)                                   # presence 0/1
    tb = ta + lag
    if follow:
        vb = np.where(a_block > 0.5, 0.85, 0.15) + 0.03 * rng.standard_normal(n)
    else:
        vb = rng.random(n)                                      # no relationship
    return ta, va, tb, vb


def test_learn_coupling_cumulant_converges():
    """The backend-agnostic learner fits J on a cumulant cluster so D(J) matches a positive target."""
    field, _ = _fresh_field()
    res = cl.learn_coupling(field, "z", "a", "b", 0.2, horizon=8)
    assert not res["runaway"]
    assert res["converged"], res
    assert res["op_mode"] == "exchange" and res["J"] > 0


def test_kfold_validates_real_follow_and_rejects_noise():
    """An edge with a consistent held-out follow relationship is ROBUST; pure noise is not."""
    field, _ = _fresh_field()
    ta, va, tb, vb = _follow_streams(follow=True)
    good = cl.kfold_validate_edge(field, "z", "a", "b", ta, va, tb, vb, folds=4, lag_s=60, tol_s=120)
    assert good["robust"], good
    assert good["mean_reduction"] > 0 and good["frac_pos"] >= 0.6

    field2, _ = _fresh_field()
    ta, va, tb, vb = _follow_streams(follow=False, seed=1)
    noise = cl.kfold_validate_edge(field2, "z", "a", "b", ta, va, tb, vb, folds=4, lag_s=60, tol_s=120)
    assert not noise["robust"], noise


def _synthetic_db(path, follow=True):
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE events (timestamp TEXT, source_device TEXT, event_type TEXT, "
                "value TEXT, metadata TEXT, synthetic INT DEFAULT 0)")
    ta, va, tb, vb = _follow_streams(follow=follow)
    from datetime import datetime, timedelta, timezone
    t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    rows = []
    for t, v in zip(ta, va):
        rows.append(((t0 + timedelta(seconds=float(t))).isoformat(), "devA", "evA", json.dumps(float(v)), "{}", 0))
    for t, v in zip(tb, vb):
        rows.append(((t0 + timedelta(seconds=float(t))).isoformat(), "devB", "evB", json.dumps(float(v)), "{}", 0))
    con.executemany("INSERT INTO events VALUES (?,?,?,?,?,?)", rows)
    con.commit(); con.close()


_RESOLVE = lambda role: {"a": ("evA", None), "b": ("evB", None)}.get(role)


def test_grow_couplings_consolidates_only_real_edge(tmp_path):
    """grow_couplings with consolidate=True writes the surviving J to _xy (persists); a no-relationship
    stream consolidates nothing. consolidate=False is side-effect-free."""
    db = str(tmp_path / "ev.db")
    _synthetic_db(db, follow=True)

    # propose-only: _xy must stay zero
    field, cum = _fresh_field()
    rep = dream_topology.grow_couplings(field, db, resolve=_RESOLVE, consolidate=False,
                                        folds=4, lag_s=60, tol_s=120, edge_floor=0.05)
    assert all(k == (0.0, 0.0) for k in cum._xy.values()), "propose-only must not mutate _xy"
    assert rep["z"]["proposals"], "the real follow edge should be proposed"

    # consolidate: the robust edge is written to _xy and survives a snapshot round-trip
    field2, cum2 = _fresh_field()
    rep2 = dream_topology.grow_couplings(field2, db, resolve=_RESOLVE, consolidate=True,
                                         folds=4, lag_s=60, tol_s=120, edge_floor=0.05)
    assert rep2["z"]["applied"], rep2
    assert cum2._xy[(0, 1)] != (0.0, 0.0), "consolidated edge must write _xy"
    assert cum2.snapshot()["xy"], "the grown coupling must persist through the pickle"


def test_grow_couplings_noise_grows_nothing(tmp_path):
    db = str(tmp_path / "ev_noise.db")
    _synthetic_db(db, follow=False)
    field, cum = _fresh_field()
    rep = dream_topology.grow_couplings(field, db, resolve=_RESOLVE, consolidate=True,
                                        folds=4, lag_s=60, tol_s=120, edge_floor=0.05)
    applied = rep.get("z", {}).get("applied", [])
    assert not applied, f"no real signal → consolidate nothing, got {applied}"
    assert all(k == (0.0, 0.0) for k in cum._xy.values())


def test_dream_loop_status_exposes_topology():
    from umwelt.foresight import dream_loop
    st = dream_loop.status()
    assert "topology" in st and "topology_lookback_days" in st


def test_clone_validate_then_apply_to_live(tmp_path):
    """The live-loop safety pattern: validate+consolidate on a CLONE, then apply the grown couplings to a
    separate LIVE field via apply_coupling — the live field ends with the same learned _xy (never touched
    during the heavy validation)."""
    db = str(tmp_path / "ev.db")
    _synthetic_db(db, follow=True)
    clone_field, clone_cum = _fresh_field()
    live_field, live_cum = _fresh_field()

    rep = dream_topology.grow_couplings(clone_field, db, resolve=_RESOLVE, consolidate=True,
                                        folds=4, lag_s=60, tol_s=120, edge_floor=0.05)
    grown = rep["z"]["applied"]
    assert grown, "the real follow edge should grow on the clone"
    assert all(k == (0.0, 0.0) for k in live_cum._xy.values()), "live field untouched during validation"

    from umwelt.learning.coupling_learn import apply_coupling
    for e in grown:
        assert apply_coupling(live_field, e["node"], e["leader"], e["follower"], e["J"], e["family"])
    assert live_cum._xy[(0, 1)] == clone_cum._xy[(0, 1)] != (0.0, 0.0)
