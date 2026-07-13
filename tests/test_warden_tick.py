"""The warden's pins — earned autonomy enforced mechanically, all offline.

watch = ledgered proposal, module untouched. run = staging gate, then apply +
restart. topology_change never auto-applies, even if the policy file is hand-edited
to say run. A session that rewrites policy.json is detected, reverted, and treated
as all-watch. Malformed findings poison nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from umweltforge.agent import AgentResult
from umweltforge.policy import WardenPolicy, read_ledger
from umweltforge.warden import apply_accepted, warden_tick
from umweltforge.workspace import ForgeWorkspace

BASE_MODULE = """\
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec

SPEC = DomainSpec(
    name="warden-tiny",
    nodes=(
        NodeSpec("top", parent=None, kind="root", roles=()),
        NodeSpec("area_a", parent="top", roles=("level",),
                 params={"gamma_diss": (0.05, 0.01, 0.0, 1.0)}),
    ),
    bindings=(BindingSpec("sig_a", zone="area_a", role="level",
                          normalizer="binary"),),
)
"""

TUNED_MODULE = BASE_MODULE.replace("(0.05, 0.01, 0.0, 1.0)", "(0.10, 0.01, 0.0, 1.0)")
BROKEN_MODULE = BASE_MODULE.replace('zone="area_a"', 'zone="ghost"')


class WardenFake:
    """Writes a scripted findings.json into the tick dir (or misbehaves on cue)."""

    def __init__(self, findings=None, misbehave=None):
        self.findings = findings
        self.misbehave = misbehave          # None | "no_file" | "bad_json" | "tamper"
        self.tick_dirs: list[Path] = []

    def run(self, workspace, prompt, *, system_prompt, allowed_tools):
        tick_dir = Path(workspace)
        self.tick_dirs.append(tick_dir)
        assert allowed_tools == ("Read", "Write")   # no shell for the warden
        if self.misbehave == "no_file":
            return AgentResult(ok=True)
        if self.misbehave == "bad_json":
            (tick_dir / "findings.json").write_text("{not json")
            return AgentResult(ok=True)
        if self.misbehave == "tamper":
            # tick_dir = <ws>/warden/ticks/<ts> → policy.json at <ws>/warden/
            policy_path = tick_dir.parents[1] / "policy.json"
            pol = json.loads(policy_path.read_text())
            pol["dials"] = {ct: "run" for ct in pol["dials"]}
            policy_path.write_text(json.dumps(pol))
        (tick_dir / "findings.json").write_text(json.dumps(self.findings))
        return AgentResult(ok=True)


class StubClient:
    def __init__(self):
        self.stops: list[str] = []
        self.starts: list[str] = []

    def health(self):
        return {"world": "w", "step": 42, "seed_profile": "blank"}

    def state(self):
        return {"globals": [{"type": "ingest", "unmatched": {
            "actionable": 1, "sensors": [{"sensor_id": "sig_new", "count": 9}]}}]}

    def recommendations(self):
        return []

    def stop_world(self, name):
        self.stops.append(name)

    def start_world(self, name):
        self.starts.append(name)


def _proposal(pid="p1", change_type="param_tune", module=TUNED_MODULE):
    return {"id": pid, "change_type": change_type, "rationale": "test",
            "expected_effect": "test", "new_module": module}


@pytest.fixture()
def ws(tmp_path):
    w = ForgeWorkspace.create("tiny", "a rant", root=tmp_path)
    w.module_path.write_text(BASE_MODULE)
    return w


def _actions(ws):
    return [e["action"] for e in read_ledger(ws.ledger_path)]


def test_all_watch_proposes_and_touches_nothing(ws, tmp_path):
    client = StubClient()
    fake = WardenFake({"findings": [{"severity": "info", "summary": "s",
                                     "evidence": "e"}],
                       "proposals": [_proposal()]})
    result = warden_tick("tiny", agent=fake, client=client, root=tmp_path)
    assert result.watched == ["p1"] and not result.applied and not result.rejected
    assert ws.module_path.read_text() == BASE_MODULE
    assert client.stops == [] and client.starts == []
    assert "proposed" in _actions(ws) and "auto_applied" not in _actions(ws)
    # the tick dir preserved the evidence trail
    tick_dir = fake.tick_dirs[0]
    assert (tick_dir / "context.json").exists()
    assert (tick_dir / "p1.diff").exists()
    ctx = json.loads((tick_dir / "context.json").read_text())
    assert ctx["unmatched"]["actionable"] == 1      # lifted out of state


def test_promoted_change_type_auto_applies_with_restart(ws, tmp_path):
    policy = WardenPolicy.load(ws.policy_path, "tiny")
    policy.promote("param_tune")
    policy.save(ws.policy_path)
    client = StubClient()
    result = warden_tick("tiny", agent=WardenFake(
        {"findings": [], "proposals": [_proposal()]}), client=client,
        root=tmp_path)
    assert result.applied == ["p1"]
    assert ws.module_path.read_text() == TUNED_MODULE
    assert client.stops == ["tiny"] and client.starts == ["tiny"]
    assert "auto_applied" in _actions(ws)
    backups = list(ws.attempts_dir.glob("pre_apply_*_p1.py"))
    assert len(backups) == 1 and backups[0].read_text() == BASE_MODULE


def test_invalid_proposal_fails_staging_and_touches_nothing(ws, tmp_path):
    policy = WardenPolicy.load(ws.policy_path, "tiny")
    policy.promote("param_tune")
    policy.save(ws.policy_path)
    client = StubClient()
    result = warden_tick("tiny", agent=WardenFake(
        {"findings": [], "proposals": [_proposal(module=BROKEN_MODULE)]}),
        client=client, root=tmp_path)
    assert result.rejected == ["p1"] and not result.applied
    assert ws.module_path.read_text() == BASE_MODULE
    assert client.stops == [] and client.starts == []
    failed = [e for e in read_ledger(ws.ledger_path)
              if e["action"] == "validation_failed"]
    assert failed and "ghost" in failed[0]["failures"]


def test_topology_change_never_auto_applies(ws, tmp_path):
    # Even a hand-edited policy.json saying run is clamped at load.
    pol = json.loads(ws.policy_path.read_text())
    pol["dials"]["topology_change"] = "run"
    ws.policy_path.write_text(json.dumps(pol))
    client = StubClient()
    result = warden_tick("tiny", agent=WardenFake(
        {"findings": [], "proposals": [_proposal(change_type="topology_change")]}),
        client=client, root=tmp_path)
    assert result.watched == ["p1"] and not result.applied
    assert ws.module_path.read_text() == BASE_MODULE


def test_no_apply_flag_watches_everything(ws, tmp_path):
    policy = WardenPolicy.load(ws.policy_path, "tiny")
    policy.promote("param_tune")
    policy.save(ws.policy_path)
    result = warden_tick("tiny", agent=WardenFake(
        {"findings": [], "proposals": [_proposal()]}), client=StubClient(),
        root=tmp_path, apply=False)
    assert result.watched == ["p1"] and not result.applied
    assert ws.module_path.read_text() == BASE_MODULE


def test_malformed_findings_ledger_and_stop(ws, tmp_path):
    for misbehave in ("no_file", "bad_json"):
        result = warden_tick("tiny", agent=WardenFake(misbehave=misbehave),
                             client=StubClient(), root=tmp_path)
        assert result.error and not result.proposals
    assert _actions(ws).count("tick_malformed") == 2
    assert ws.module_path.read_text() == BASE_MODULE


def test_policy_tampering_is_reverted_and_forces_watch(ws, tmp_path):
    before = ws.policy_path.read_text()
    client = StubClient()
    result = warden_tick("tiny", agent=WardenFake(
        {"findings": [], "proposals": [_proposal()]}, misbehave="tamper"),
        client=client, root=tmp_path)
    # the mid-session promotion to "run" must NOT have been honored...
    assert result.watched == ["p1"] and not result.applied
    assert client.stops == [] and client.starts == []
    # ...and the file is back to its pre-tick bytes, with the tamper ledgered.
    assert ws.policy_path.read_text() == before
    assert "policy_tampered" in _actions(ws)


def test_accept_verdict_applies_a_watched_proposal(ws, tmp_path):
    client = StubClient()
    warden_tick("tiny", agent=WardenFake(
        {"findings": [], "proposals": [_proposal()]}), client=client,
        root=tmp_path)
    assert ws.module_path.read_text() == BASE_MODULE     # watched, not applied
    applied, detail = apply_accepted("tiny", "p1", client=client, root=tmp_path)
    assert applied, detail
    assert ws.module_path.read_text() == TUNED_MODULE
    assert client.stops == ["tiny"] and client.starts == ["tiny"]
    accepted = [e for e in read_ledger(ws.ledger_path) if e["action"] == "accepted"]
    assert accepted and accepted[0]["by"] == "cli"
    # an unknown id is a clean miss
    applied, detail = apply_accepted("tiny", "nope", client=client, root=tmp_path)
    assert not applied and "no proposal" in detail
