"""The warden tick — one-shot inspection of a running world, under earned autonomy.

Cron-able: gather the world's live surfaces into a context bundle, run a read-only
agent session whose entire output contract is one findings.json, ledger every
proposal, and — ONLY for change-types a human has dialed to "run" — validate the
proposed module in staging (the same fresh-subprocess gate as compile; the agent's
confidence counts for nothing) and apply it with a stop→start restart.

Tamper discipline: policy.json is hashed before the session and re-checked after.
The warden agent has no shell and no reason to touch it; if it changed anyway, the
pre-tick copy is restored, the tick is ledgered "policy_tampered", and everything
is treated as watch.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from umweltforge.agent import WARDEN_TOOLS, ForgeAgent
from umweltforge.pipeline import run_validation
from umweltforge.policy import (CHANGE_TYPES, NEVER_AUTO, WardenPolicy,
                                append_ledger, read_ledger)
from umweltforge.prompts import warden_system_prompt, warden_task_prompt
from umweltforge.workspace import ForgeWorkspace

CONTEXT_CAP_BYTES = 200_000


@dataclass
class TickResult:
    world: str
    tick_id: str
    findings: list = field(default_factory=list)
    proposals: list = field(default_factory=list)
    applied: list = field(default_factory=list)     # auto-applied proposal ids
    rejected: list = field(default_factory=list)    # failed staging validation
    watched: list = field(default_factory=list)     # ledgered, awaiting a human
    error: str = ""


def gather_context(client, ws: ForgeWorkspace, policy: WardenPolicy,
                   *, max_bytes: int = CONTEXT_CAP_BYTES) -> dict:
    """The world's live surfaces, priority-ordered under a byte cap: health and the
    shadow decisions always fit; the full state projection is dropped first if the
    world is huge (its unmatched-signal gap is lifted out separately so binding_add
    evidence survives the cut)."""
    ctx: dict = {"world": ws.name, "dials": dict(policy.dials),
                 "ledger_tail": read_ledger(ws.ledger_path)[-20:]}
    try:
        ctx["health"] = client.health()
        ctx["recommendations"] = client.recommendations()
        state = client.state()
        for g in (state.get("globals") or []):
            if isinstance(g, dict) and "unmatched" in g:
                ctx["unmatched"] = g["unmatched"]
        if len(json.dumps(state)) <= max_bytes - len(json.dumps(ctx)):
            ctx["state"] = state
        else:
            ctx["state_dropped"] = ("full state projection exceeded the context "
                                    "cap; health/recommendations/unmatched kept")
    except Exception as exc:
        ctx["gather_error"] = f"{type(exc).__name__}: {exc}"
    return ctx


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parse_findings(path: Path) -> "tuple[dict | None, str]":
    if not path.exists():
        return None, "agent wrote no findings.json"
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"findings.json is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return None, "findings.json must be a JSON object"
    findings = data.get("findings", [])
    proposals = data.get("proposals", [])
    if not isinstance(findings, list) or not isinstance(proposals, list):
        return None, "findings/proposals must be lists"
    seen_ids = set()
    for p in proposals:
        if not isinstance(p, dict):
            return None, "each proposal must be an object"
        pid, ct, mod = p.get("id"), p.get("change_type"), p.get("new_module")
        if not pid or pid in seen_ids:
            return None, f"proposal id missing or duplicated: {pid!r}"
        seen_ids.add(pid)
        if ct not in CHANGE_TYPES:
            return None, f"proposal {pid!r}: unknown change_type {ct!r}"
        if not isinstance(mod, str) or not mod.strip():
            return None, f"proposal {pid!r}: new_module must be the full module text"
    return {"findings": findings, "proposals": proposals}, ""


def _stage_and_apply(ws: ForgeWorkspace, client, proposal: dict,
                     tick_id: str) -> "tuple[bool, str]":
    """Validate the proposed module in staging; on green, back up + replace the live
    module and restart the world (stop snapshots via SIGTERM; start re-imports
    through spec_path and replays the log tail). Returns (applied, detail)."""
    staging = ws.staging_dir / tick_id
    staging.mkdir(parents=True, exist_ok=True)
    (staging / ws.module_file).write_text(proposal["new_module"])
    ok, report = run_validation(staging, ws.spec_ref())
    if not ok:
        failures = [c["detail"] for c in report.get("checks", [])
                    if not c.get("ok") and not c.get("skipped")]
        return False, "; ".join(f for f in failures if f) or "gate failed"
    backup = ws.attempts_dir / f"pre_apply_{tick_id}_{proposal['id']}.py"
    if ws.module_path.exists():
        shutil.copy2(ws.module_path, backup)
    tmp = ws.module_path.with_suffix(".py.tmp")
    tmp.write_text(proposal["new_module"])
    tmp.replace(ws.module_path)                     # atomic on POSIX
    client.stop_world(ws.name)
    client.start_world(ws.name)
    return True, ""


def warden_tick(name: str, *, agent: ForgeAgent, client,
                root: "Path | str | None" = None, apply: bool = True) -> TickResult:
    ws = ForgeWorkspace.open(name, root=root)
    policy = WardenPolicy.load(ws.policy_path, name)
    policy_hash_before = _sha256(ws.policy_path) if ws.policy_path.exists() else ""
    policy_backup = (ws.policy_path.read_bytes()
                     if ws.policy_path.exists() else None)

    tick_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tick_dir = ws.ticks_dir / tick_id
    n = 0
    while tick_dir.exists():                        # same-second reruns
        n += 1
        tick_dir = ws.ticks_dir / f"{tick_id}_{n}"
    tick_dir.mkdir(parents=True)
    result = TickResult(world=name, tick_id=tick_dir.name)

    ctx = gather_context(client, ws, policy)
    (tick_dir / "context.json").write_text(json.dumps(ctx, indent=1))
    if ws.module_path.exists():
        shutil.copy2(ws.module_path, tick_dir / ws.module_file)

    agent.run(tick_dir, warden_task_prompt(name, ws.module_file),
              system_prompt=warden_system_prompt(CHANGE_TYPES, policy.dials),
              allowed_tools=WARDEN_TOOLS)

    parsed, err = _parse_findings(tick_dir / "findings.json")
    if parsed is None:
        append_ledger(ws.ledger_path, {"world": name, "action": "tick_malformed",
                                       "tick": result.tick_id, "detail": err})
        result.error = err
        return result

    # Tamper check: the session must not have touched the autonomy dials.
    policy_hash_after = _sha256(ws.policy_path) if ws.policy_path.exists() else ""
    tampered = policy_hash_after != policy_hash_before
    if tampered:
        if policy_backup is not None:
            ws.policy_path.write_bytes(policy_backup)
        append_ledger(ws.ledger_path, {"world": name, "action": "policy_tampered",
                                       "tick": result.tick_id})

    result.findings = parsed["findings"]
    result.proposals = parsed["proposals"]
    current = ws.module_path.read_text() if ws.module_path.exists() else ""

    for p in result.proposals:
        diff = "".join(difflib.unified_diff(
            current.splitlines(keepends=True),
            p["new_module"].splitlines(keepends=True),
            fromfile=ws.module_file, tofile=f"{ws.module_file} ({p['id']})"))
        (tick_dir / f"{p['id']}.diff").write_text(diff)
        append_ledger(ws.ledger_path, {
            "world": name, "action": "proposed", "tick": result.tick_id,
            "proposal_id": p["id"], "change_type": p["change_type"],
            "rationale": p.get("rationale", ""),
            "module_sha256": hashlib.sha256(p["new_module"].encode()).hexdigest()})

        mode = "watch" if tampered else policy.mode(p["change_type"])
        if not apply or mode != "run" or p["change_type"] in NEVER_AUTO:
            result.watched.append(p["id"])
            continue
        applied, detail = _stage_and_apply(ws, client, p, result.tick_id)
        if applied:
            result.applied.append(p["id"])
            current = p["new_module"]               # later diffs read against it
            append_ledger(ws.ledger_path, {
                "world": name, "action": "auto_applied", "tick": result.tick_id,
                "proposal_id": p["id"], "change_type": p["change_type"]})
        else:
            result.rejected.append(p["id"])
            append_ledger(ws.ledger_path, {
                "world": name, "action": "validation_failed",
                "tick": result.tick_id, "proposal_id": p["id"],
                "change_type": p["change_type"], "failures": detail})

    return result


def find_proposal(ws: ForgeWorkspace, proposal_id: str) -> "tuple[dict, str] | None":
    """Locate a ledgered proposal's full text: scan ticks newest-first for a
    findings.json containing the id. Returns (proposal, tick_id) or None."""
    for tick_dir in sorted(ws.ticks_dir.iterdir(), reverse=True):
        f = tick_dir / "findings.json"
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for p in data.get("proposals", []):
            if isinstance(p, dict) and p.get("id") == proposal_id:
                return p, tick_dir.name
    return None


def apply_accepted(name: str, proposal_id: str, *, client,
                   root: "Path | str | None" = None) -> "tuple[bool, str]":
    """The human-verdict path: accept a watched proposal — the SAME
    staging-validate-apply gate as auto-apply, with by:'cli' on the ledger."""
    ws = ForgeWorkspace.open(name, root=root)
    hit = find_proposal(ws, proposal_id)
    if hit is None:
        return False, f"no proposal {proposal_id!r} found in any tick"
    proposal, tick_id = hit
    applied, detail = _stage_and_apply(ws, client, proposal, f"accept_{tick_id}")
    append_ledger(ws.ledger_path, {
        "world": name, "action": "accepted" if applied else "validation_failed",
        "proposal_id": proposal_id, "change_type": proposal.get("change_type"),
        "by": "cli", **({"failures": detail} if not applied else {})})
    return applied, detail
