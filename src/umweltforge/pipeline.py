"""The compile pipeline: rant → authored module → INDEPENDENT gate → running world.

The discipline that makes an embedded LLM safe to build on: the agent's success
claim is never trusted. After every authoring session the pipeline re-runs the
deterministic gate (umwelt.spec.validate) in a FRESH subprocess — fresh because
role/normalizer registries are process-global and module caching would make an
in-process re-validation of an edited module unsound. Only a green gate registers
the world with the daemon, and the world's manifest carries spec_path so every
future respawn imports the module from the workspace.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from umweltforge.agent import COMPILE_TOOLS, ForgeAgent
from umweltforge.policy import append_ledger
from umweltforge.prompts import authoring_system_prompt, authoring_task_prompt
from umweltforge.workspace import ForgeWorkspace


@dataclass
class CompileResult:
    ok: bool
    world: str
    spec_ref: str = ""
    spec_path: str = ""
    attempts: int = 0
    report: "dict | None" = None       # last parsed ValidationReport.to_dict()
    registered: bool = False
    error: str = ""
    agent_errors: list = field(default_factory=list)


def run_validation(module_dir: Path, spec_ref: str, *,
                   timeout_s: float = 300.0) -> "tuple[bool, dict]":
    """Run the deterministic gate against a module dir, in a fresh subprocess.

    Returns (ok, report_dict). A gate that crashes or times out is a failure with a
    synthesized report — never an exception (the pipeline's loop needs a verdict).
    """
    module_dir = Path(module_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(module_dir), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    try:
        out = subprocess.run(
            [sys.executable, "-m", "umwelt.spec.validate", spec_ref, "--json"],
            env=env, cwd=str(module_dir), capture_output=True, text=True,
            timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, {"ok": False, "spec": spec_ref,
                       "checks": [{"name": "gate_subprocess", "ok": False,
                                   "skipped": False,
                                   "detail": f"gate timed out after {timeout_s}s"}]}
    try:
        report = json.loads(out.stdout)
    except json.JSONDecodeError:
        report = {"ok": False, "spec": spec_ref,
                  "checks": [{"name": "gate_subprocess", "ok": False,
                              "skipped": False,
                              "detail": f"gate emitted no JSON (rc={out.returncode}): "
                                        f"{(out.stderr or out.stdout)[-2000:]}"}]}
    return bool(out.returncode == 0 and report.get("ok")), report


def compile_world(name: str, rant: str, *, agent: ForgeAgent, client=None,
                  root: "Path | str | None" = None, max_attempts: int = 3,
                  register: bool = True,
                  world_knobs: "dict | None" = None) -> CompileResult:
    """The full pipeline. `client` is an umweltd.client.UmweltClient (or None with
    register=False for author-and-validate-only)."""
    if register and client is None:
        raise ValueError("register=True needs a client (or pass register=False)")

    ws = ForgeWorkspace.create(name, rant, root=root)
    spec_ref = ws.spec_ref()
    result = CompileResult(ok=False, world=name, spec_ref=spec_ref,
                           spec_path=str(ws.root))

    last_report_json: "str | None" = None
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        result.attempts = attempt
        prompt = authoring_task_prompt(rant, ws.module_file, spec_ref,
                                       last_report_json=last_report_json)
        agent_result = agent.run(
            ws.root, prompt,
            system_prompt=authoring_system_prompt(name, ws.module_file, spec_ref),
            allowed_tools=COMPILE_TOOLS)
        if not agent_result.ok:
            result.agent_errors.append(agent_result.error)

        # The independent verdict — regardless of what the agent claims.
        ok, report = run_validation(ws.root, spec_ref)
        result.report = report
        ws.snapshot_attempt(attempt, report)
        if ok:
            result.ok = True
            break
        last_report_json = json.dumps(report, indent=1)

    if not result.ok:
        result.error = (f"gate still red after {result.attempts} attempt(s); "
                        f"workspace kept for a human: {ws.root}")
        return result

    if register:
        client.create_world(name, spec=spec_ref, spec_path=str(ws.root),
                            **(world_knobs or {}))
        result.registered = True
    append_ledger(ws.ledger_path, {"world": name, "action": "compiled",
                                   "attempts": result.attempts,
                                   "registered": result.registered})
    return result
