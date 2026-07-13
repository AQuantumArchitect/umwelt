"""Earned-autonomy policy + ledger pins: watch is the default, promotion is explicit,
topology_change can never auto-apply, and the competence summary counts honestly."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from umweltforge.policy import (CHANGE_TYPES, NEVER_AUTO, WardenPolicy,
                                append_ledger, competence_summary, read_ledger)


def test_fresh_policy_is_all_watch():
    pol = WardenPolicy.fresh("w")
    assert set(pol.dials) == set(CHANGE_TYPES)
    assert all(mode == "watch" for mode in pol.dials.values())


def test_missing_file_loads_all_watch(tmp_path):
    pol = WardenPolicy.load(tmp_path / "nope.json", "w")
    assert all(mode == "watch" for mode in pol.dials.values())


def test_promote_demote_roundtrip(tmp_path):
    p = tmp_path / "policy.json"
    pol = WardenPolicy.fresh("w")
    pol.promote("param_tune")
    pol.save(p)
    loaded = WardenPolicy.load(p, "w")
    assert loaded.mode("param_tune") == "run"
    loaded.demote("param_tune")
    loaded.save(p)
    assert WardenPolicy.load(p, "w").mode("param_tune") == "watch"


def test_never_auto_is_unpromotable_and_clamped_on_load(tmp_path):
    pol = WardenPolicy.fresh("w")
    for ct in NEVER_AUTO:
        with pytest.raises(ValueError):
            pol.promote(ct)
    # A hand-edited (or tampered) file saying "run" is clamped back to watch.
    p = tmp_path / "policy.json"
    p.write_text(json.dumps({"world": "w", "dials": {
        "topology_change": "run", "param_tune": "run", "made_up": "run"}}))
    loaded = WardenPolicy.load(p, "w")
    assert loaded.mode("topology_change") == "watch"
    assert loaded.mode("param_tune") == "run"
    assert "made_up" not in loaded.dials


def test_unknown_change_type_raises():
    pol = WardenPolicy.fresh("w")
    with pytest.raises(ValueError):
        pol.promote("made_up")
    with pytest.raises(ValueError):
        pol.demote("made_up")


def test_ledger_appends_and_survives_torn_lines(tmp_path):
    p = tmp_path / "ledger.jsonl"
    append_ledger(p, {"world": "w", "action": "proposed", "proposal_id": "p1",
                      "change_type": "param_tune"})
    append_ledger(p, {"world": "w", "action": "auto_applied", "proposal_id": "p1",
                      "change_type": "param_tune"})
    with p.open("a") as f:
        f.write("{torn json\n")
    entries = read_ledger(p)
    assert len(entries) == 2 and all("ts" in e for e in entries)


def test_competence_summary_counts_per_type(tmp_path):
    p = tmp_path / "ledger.jsonl"
    append_ledger(p, {"action": "proposed", "proposal_id": "p1",
                      "change_type": "param_tune"})
    append_ledger(p, {"action": "proposed", "proposal_id": "p2",
                      "change_type": "binding_add"})
    append_ledger(p, {"action": "validation_failed", "proposal_id": "p2",
                      "change_type": "binding_add"})
    append_ledger(p, {"action": "auto_applied", "proposal_id": "p1",
                      "change_type": "param_tune"})
    append_ledger(p, {"action": "accepted", "proposal_id": "p2"})   # ct via p2
    s = competence_summary(read_ledger(p))
    assert s["param_tune"] == {"proposed": 1, "auto_applied": 1,
                               "validation_failed": 0, "accepted": 0, "rejected": 0}
    assert s["binding_add"]["proposed"] == 1
    assert s["binding_add"]["validation_failed"] == 1
    assert s["binding_add"]["accepted"] == 1


def test_workspace_create_open_and_layout(tmp_path):
    from umweltforge.workspace import ForgeWorkspace, module_name
    ws = ForgeWorkspace.create("my-world", "a small test domain", root=tmp_path)
    assert ws.rant_path.read_text() == "a small test domain"
    assert ws.guide_path.exists() and "dissipative" in ws.guide_path.read_text()
    assert ws.policy_path.exists()
    assert ws.spec_ref() == "world_my_world:SPEC"
    assert module_name("my-world") == "world_my_world"
    with pytest.raises(FileExistsError):
        ForgeWorkspace.create("my-world", "again", root=tmp_path)
    opened = ForgeWorkspace.open("my-world", root=tmp_path)
    assert opened.root == ws.root
    with pytest.raises(FileNotFoundError):
        ForgeWorkspace.open("ghost", root=tmp_path)


def test_package_imports_without_sdk():
    # The lazy-import discipline: everything except ClaudeForgeAgent construction
    # works with claude_agent_sdk absent from the environment.
    import umweltforge  # noqa: F401
    import umweltforge.agent
    import umweltforge.prompts  # noqa: F401
    try:
        import claude_agent_sdk  # noqa: F401
        has_sdk = True
    except ImportError:
        has_sdk = False
    if not has_sdk:
        with pytest.raises(RuntimeError, match="umwelt-engine\\[forge\\]"):
            umweltforge.agent.ClaudeForgeAgent()
