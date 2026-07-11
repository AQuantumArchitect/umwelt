"""Model-transparency serializer — the legible model snapshot (params + clusters + coupling web).

Ported from the origin deployment's serializer gate (meerkat tests/brain/test_transparency.py)
onto the gridworld engine. The origin's folded-topology grouping cases (its region-merge +
sector overlay) stayed with the origin: the fold transform is not in this library, so
_role_groups is pinned to its no-fold contract instead. The dissolved-knob names asserted here
are the engine's own fiber vocabulary (driver_alpha / agency_tau_days), not any domain's.
"""
from __future__ import annotations

import json

from tests.test_spec_to_field import tiny_grid_spec
from umwelt.projection import transparency


def _engine():
    from umwelt.boot import build_engine
    return build_engine(spec=tiny_grid_spec())


def test_snapshot_envelope_and_param_shape():
    snap = transparency.model_snapshot(_engine())
    assert set(("params", "clusters", "summary")).issubset(snap)
    s = snap["summary"]
    # a built world has many fiber params and a belief cluster per region
    assert s["param_count"] > 50
    assert s["cluster_count"] >= 5
    assert s["param_count"] == len(snap["params"])
    p = snap["params"][0]
    for k in ("name", "node", "value", "lo", "hi", "settled", "frac", "updates",
              "frozen", "learned", "drift"):
        assert k in p, f"param row missing {k}"


def test_clusters_carry_roles_and_kind():
    snap = transparency.model_snapshot(_engine())
    # the dissolved engine knobs are present (every knob is named + shown)
    names = {p["name"] for p in snap["params"]}
    assert {"driver_alpha", "agency_tau_days"}.issubset(names)
    c = snap["clusters"][0]
    assert c["n_qubits"] >= 1 and c["roles"]
    assert c["kind"] in ("cumulant", "dense", "product")
    assert all(set(("role", "x", "y", "z")).issubset(r) for r in c["roles"])


def test_param_fiber_clusters_excluded():
    snap = transparency.model_snapshot(_engine())
    for c in snap["clusters"]:
        assert not c["name"].startswith("_")        # the _params fiber is not a belief cluster


def test_clusters_carry_bloch_and_purity():
    snap = transparency.model_snapshot(_engine())
    for c in snap["clusters"]:
        assert c["purity"] is None or 0.0 <= c["purity"] <= 1.0001
        for r in c["roles"]:
            for axis in ("x", "y", "z"):
                assert -1.0001 <= r[axis] <= 1.0001            # Bloch components in range


def test_snapshot_web_shape_when_couplings_present():
    # plant a cumulant cluster with a learned alignment edge into the field, then assert
    # the web surfaces with the right shape.
    from umwelt.substrate.cumulant_cluster import CumulantCluster
    engine = _engine()
    cum = CumulantCluster("annex", ["agent_near", "resource"],
                          role_modes={"agent_near": "unitary", "resource": "unitary"})
    (i, j) = next(iter(cum._zz.keys()))
    cum._zz[(i, j)] = 0.5                                   # plant a learned alignment edge
    engine.field.clusters["annex"] = cum
    snap = transparency.model_snapshot(engine)
    row = next(c for c in snap["clusters"] if c["name"] == "annex")
    assert row.get("zz"), "zz web edge did not surface"
    e = row["zz"][0]
    assert set(("a", "b", "j")).issubset(e) and isinstance(e["j"], float)
    assert snap["summary"]["web_edges"] >= 1


def test_role_groups_pin_the_no_fold_contract():
    """No fold transform in this library → no role carries a folded prefix → grouping is
    None and rows stay byte-identical (the origin's merged-topology grouping stays there)."""
    g = transparency._role_groups(["agent_near", "resource"],
                                  {"agent_near": 0.1, "resource": 0.0}, [])
    assert g is None


def test_serializer_never_throws_on_empty_engine():
    # the membrane: a bare object yields the empty envelope, not an exception
    snap = transparency.model_snapshot(object())
    assert snap["params"] == [] and snap["clusters"] == []


def test_snapshot_is_json_serializable():
    # the projection is a wire format — it must serialize as-is
    snap = transparency.model_snapshot(_engine())
    json.dumps(snap)


def test_summary_stores_block_reads_prune_ledger(tmp_path, monkeypatch):
    """Datastream health: summary.stores carries store sizes + the prune ledger
    (prune_state.json), all cheap file reads."""
    db = tmp_path / "umwelt_events.db"
    db.write_bytes(b"x" * 2_000_000)
    (tmp_path / "umwelt_events.db-wal").write_bytes(b"x" * 500_000)
    (tmp_path / "prune_state.json").write_text(json.dumps(
        {"last_prune": "2026-07-05T09:00:00+00:00", "rows_pruned": 10, "rows_archived": 15}))
    monkeypatch.setenv("UMWELT_DB_PATH", str(db))
    st = transparency.model_snapshot(_engine())["summary"]["stores"]
    assert st["events"]["file_mb"] == 2.0
    assert st["events"]["wal_mb"] == 0.5
    assert st["surprise"]["file_mb"] is None      # absent store reads as None, never throws
    assert st["last_prune"] == "2026-07-05T09:00:00+00:00"
    assert st["rows_archived"] == 15


def test_summary_stores_none_without_db_path(monkeypatch):
    monkeypatch.delenv("UMWELT_DB_PATH", raising=False)
    s = transparency.model_snapshot(_engine())["summary"]
    assert s["stores"] is None and s["events_wal_mb"] is None
