"""The compile pipeline's pins — all offline, no SDK, no API key, no network.

THE pin of the wave: a lying agent provably cannot register a broken world. The
pipeline re-runs the deterministic gate in a fresh subprocess after every authoring
session; the agent's own claim (AgentResult.ok) is transport-level noise.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from umweltforge.agent import AgentResult, resolve_agent
from umweltforge.pipeline import compile_world, run_validation
from umweltforge.workspace import ForgeWorkspace, module_name

GOOD_MODULE = """\
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec

SPEC = DomainSpec(
    name="scripted-tiny",
    nodes=(
        NodeSpec("top", parent=None, kind="root", roles=()),
        NodeSpec("area_a", parent="top", roles=("level",)),
    ),
    bindings=(BindingSpec("sig_a", zone="area_a", role="level",
                          normalizer="binary"),),
)
"""

BAD_MODULE = GOOD_MODULE.replace('zone="area_a"', 'zone="ghost_node"')


class ScriptedAgent:
    """Writes a queued module text per attempt; records every prompt it saw."""

    def __init__(self, scripts, ok=True):
        self.scripts = list(scripts)
        self.ok = ok
        self.prompts: list[str] = []

    def run(self, workspace, prompt, *, system_prompt, allowed_tools):
        self.prompts.append(prompt)
        text = self.scripts.pop(0) if self.scripts else None
        if text is not None:
            fname = module_name(Path(workspace).name) + ".py"
            (Path(workspace) / fname).write_text(text)
        return AgentResult(ok=self.ok)


class StubClient:
    def __init__(self):
        self.created: list[dict] = []
        self.stops: list[str] = []
        self.starts: list[str] = []

    def create_world(self, name, spec, **knobs):
        self.created.append({"name": name, "spec": spec, **knobs})
        return {"name": name, "port": 1}

    def stop_world(self, name):
        self.stops.append(name)
        return {"name": name, "running": False}

    def start_world(self, name):
        self.starts.append(name)
        return {"name": name, "running": True}


def make_good_agent():
    return ScriptedAgent([GOOD_MODULE])


def test_good_agent_registers_with_spec_path(tmp_path):
    client = StubClient()
    result = compile_world("tiny", "a tiny test domain",
                           agent=ScriptedAgent([GOOD_MODULE]), client=client,
                           root=tmp_path)
    assert result.ok and result.registered and result.attempts == 1
    assert result.report["ok"] is True
    [created] = client.created
    assert created["name"] == "tiny"
    assert created["spec"] == "world_tiny:SPEC"
    assert created["spec_path"] == str(tmp_path / "tiny")
    ws = ForgeWorkspace.open("tiny", root=tmp_path)
    assert ws.module_path.exists()
    assert (ws.attempts_dir / "attempt_01" / "report.json").exists()
    ledger = [json.loads(l) for l in ws.ledger_path.read_text().splitlines()]
    assert any(e["action"] == "compiled" and e["attempts"] == 1 for e in ledger)


def test_lying_agent_cannot_register_a_broken_world(tmp_path):
    """Agent writes garbage and claims success; the independent gate says no,
    every attempt, and nothing is ever registered."""
    client = StubClient()
    result = compile_world("liar", "a domain",
                           agent=ScriptedAgent(["this is not python ][",
                                               "also not python ]["], ok=True),
                           client=client, root=tmp_path, max_attempts=2)
    assert not result.ok and not result.registered
    assert result.attempts == 2
    assert client.created == []
    assert "workspace kept" in result.error
    assert result.report["ok"] is False


def test_fixit_agent_sees_previous_failure_report(tmp_path):
    agent = ScriptedAgent([BAD_MODULE, GOOD_MODULE])
    result = compile_world("fixit", "a domain", agent=agent,
                           client=StubClient(), root=tmp_path)
    assert result.ok and result.attempts == 2
    assert "previous attempt FAILED" not in agent.prompts[0]
    assert "ghost_node" in agent.prompts[1]         # the exact failure, fed back
    assert "bindings_strict" in agent.prompts[1]


def test_no_register_mode_needs_no_client(tmp_path):
    result = compile_world("solo", "a domain", agent=ScriptedAgent([GOOD_MODULE]),
                           client=None, root=tmp_path, register=False)
    assert result.ok and not result.registered
    with pytest.raises(ValueError):
        compile_world("solo2", "a domain", agent=ScriptedAgent([GOOD_MODULE]),
                      client=None, root=tmp_path, register=True)


def test_run_validation_survives_a_crashing_module(tmp_path):
    d = tmp_path / "ws"
    d.mkdir()
    (d / "world_boom.py").write_text("raise RuntimeError('import bomb')\n")
    ok, report = run_validation(d, "world_boom:SPEC")
    assert not ok
    assert any("import bomb" in c["detail"] for c in report["checks"])


def test_resolve_agent_env_seam(monkeypatch):
    monkeypatch.setenv("UMWELT_FORGE_AGENT", "test_forge_pipeline:make_good_agent")
    agent = resolve_agent()
    assert isinstance(agent, ScriptedAgent)
    # explicit ref beats the env
    agent2 = resolve_agent("test_forge_pipeline:make_good_agent")
    assert isinstance(agent2, ScriptedAgent)
    with pytest.raises(ValueError):
        resolve_agent("not-a-ref")


def test_cli_parser_builds_without_sdk():
    from umweltforge.cli import build_parser
    ap = build_parser()
    args = ap.parse_args(["new", "w", "--rant", "hi", "--no-register"])
    assert args.name == "w" and args.no_register
    args = ap.parse_args(["warden", "promote", "w", "param_tune"])
    assert args.dial_action == "promote"
    args = ap.parse_args(["warden", "tick", "w", "--no-apply"])
    assert args.no_apply


def test_missing_sdk_error_is_actionable(monkeypatch):
    monkeypatch.delenv("UMWELT_FORGE_AGENT", raising=False)
    try:
        import claude_agent_sdk  # noqa: F401
        pytest.skip("SDK installed in this environment")
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        resolve_agent()
