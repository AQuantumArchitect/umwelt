"""The ForgeAgent seam — how the forge runs an LLM session, and how tests avoid one.

Production: ClaudeForgeAgent wraps the Claude Agent SDK (optional extra, imported
lazily) — a one-shot agentic session with a scoped cwd and a restricted tool set.
Tests and offline runs: any callable-class implementing the same `run` shape,
resolved via --agent / UMWELT_FORGE_AGENT ("module:factory").

AgentResult.ok is TRANSPORT-level only (the session ran to completion). It is never
evidence the authored module is valid — the pipeline re-runs the deterministic gate
in a fresh subprocess regardless.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

# Compile sessions may run the gate themselves while iterating; warden sessions
# only read context and write findings.json.
COMPILE_TOOLS = ("Read", "Write", "Edit", "Bash")
WARDEN_TOOLS = ("Read", "Write")


@dataclass
class AgentResult:
    ok: bool                    # transport-level completion ONLY — never trusted
    error: str = ""
    transcript_path: "Path | None" = None


@runtime_checkable
class ForgeAgent(Protocol):
    def run(self, workspace: Path, prompt: str, *, system_prompt: str,
            allowed_tools: tuple) -> AgentResult: ...


class ClaudeForgeAgent:
    """The production agent: one Claude Agent SDK session, cwd-scoped, tool-restricted.

    Auth is the ANTHROPIC_API_KEY env var (the SDK reads it directly). Deliberately
    thin — everything behind the seam so SDK API drift stays a one-file fix and no
    repo test ever imports it.
    """

    def __init__(self, *, model: "str | None" = None):
        try:
            import claude_agent_sdk  # noqa: F401 — presence check only
        except ImportError as exc:
            raise RuntimeError(
                "the forge's embedded agent needs the Claude Agent SDK:\n"
                '    pip install "umwelt-engine[forge]"\n'
                "and an Anthropic API key in the environment:\n"
                "    export ANTHROPIC_API_KEY=sk-ant-...\n"
                "(offline/scripted agents: set UMWELT_FORGE_AGENT=module:factory)"
            ) from exc
        self.model = model

    def run(self, workspace: Path, prompt: str, *, system_prompt: str,
            allowed_tools: tuple) -> AgentResult:
        import asyncio
        import json

        from claude_agent_sdk import ClaudeAgentOptions, query

        workspace = Path(workspace)
        transcript = workspace / "transcript.jsonl"
        options = ClaudeAgentOptions(
            allowed_tools=list(allowed_tools),
            cwd=str(workspace),
            permission_mode="acceptEdits",
            system_prompt=system_prompt,
            **({"model": self.model} if self.model else {}),
        )

        async def _drain() -> None:
            with transcript.open("a") as f:
                async for message in query(prompt=prompt, options=options):
                    try:
                        f.write(json.dumps(_jsonable(message)) + "\n")
                    except Exception:
                        f.write(json.dumps({"repr": repr(message)}) + "\n")

        try:
            asyncio.run(_drain())
            return AgentResult(ok=True, transcript_path=transcript)
        except Exception as exc:
            return AgentResult(ok=False, error=f"{type(exc).__name__}: {exc}",
                               transcript_path=transcript if transcript.exists()
                               else None)


def _jsonable(message) -> dict:
    d = getattr(message, "__dict__", None)
    if isinstance(d, dict):
        return {k: repr(v) for k, v in d.items()}
    return {"repr": repr(message)}


def resolve_agent(spec: "str | None" = None) -> ForgeAgent:
    """Resolve the agent implementation: explicit 'module:factory' ref (--agent flag)
    > UMWELT_FORGE_AGENT env > the production ClaudeForgeAgent."""
    ref = spec or os.environ.get("UMWELT_FORGE_AGENT")
    if ref:
        module_name, _, attr = ref.partition(":")
        if not module_name or not attr:
            raise ValueError(f"agent ref must be 'module:factory', got {ref!r}")
        factory = getattr(importlib.import_module(module_name), attr)
        agent = factory()
        if not isinstance(agent, ForgeAgent):
            raise TypeError(f"{ref} built {type(agent).__name__}, which has no "
                            f"ForgeAgent-shaped run()")
        return agent
    return ClaudeForgeAgent()
