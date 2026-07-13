"""Earned autonomy — the warden's per-world, per-change-type authority dial + ledger.

The engine's own law (shadow by default: decide visibly, dispatch nothing until
promoted) applied to the warden itself. Every change-type starts at "watch"
(propose-only); a HUMAN promotes a change-type to "run" via the CLI once the ledger
shows competence. The warden cannot promote itself: promotion is only reachable from
the CLI, the tick hash-checks policy.json around the agent session, and
topology_change is structurally un-promotable this wave.

The ledger is append-only JSONL — the competence record a promotion decision reads:

    {"ts", "world", "action": "compiled",          "attempts": N}
    {"ts", "world", "action": "proposed",          "tick", "proposal_id",
     "change_type", "rationale", "module_sha256"}
    {"ts", "world", "action": "auto_applied",      "tick", "proposal_id", "change_type"}
    {"ts", "world", "action": "validation_failed", "tick", "proposal_id",
     "change_type", "failures"}
    {"ts", "world", "action": "promoted"|"demoted","change_type", "by": "cli"}
    {"ts", "world", "action": "accepted"|"rejected","proposal_id", "by": "cli"}
    {"ts", "world", "action": "tick_malformed"|"policy_tampered", "tick"}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

CHANGE_TYPES = ("normalizer_tune", "param_tune", "binding_add", "binding_remove",
                "topology_change")

# Structurally un-promotable this wave: a topology change replays the whole event
# history onto a different graph — always a human call.
NEVER_AUTO = frozenset({"topology_change"})

WATCH, RUN = "watch", "run"


@dataclass
class WardenPolicy:
    world: str
    dials: dict            # change_type -> "watch" | "run"

    @classmethod
    def fresh(cls, world: str) -> "WardenPolicy":
        return cls(world=world, dials={ct: WATCH for ct in CHANGE_TYPES})

    @classmethod
    def load(cls, path: Path, world: str) -> "WardenPolicy":
        """Missing file → all watch. Unknown change-types are dropped; NEVER_AUTO
        entries are clamped back to watch even if the file says run (tamper-proofing:
        the policy file lives in a directory the warden agent can read)."""
        pol = cls.fresh(world)
        if not Path(path).exists():
            return pol
        try:
            data = json.loads(Path(path).read_text())
        except (json.JSONDecodeError, OSError):
            return pol
        for ct, mode in (data.get("dials") or {}).items():
            if ct in CHANGE_TYPES and mode in (WATCH, RUN):
                pol.dials[ct] = WATCH if ct in NEVER_AUTO else mode
        return pol

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(
            {"world": self.world, "version": 1, "dials": self.dials,
             "updated": _now_iso()}, indent=1, sort_keys=True))

    def mode(self, change_type: str) -> str:
        return self.dials.get(change_type, WATCH)

    def promote(self, change_type: str) -> None:
        if change_type not in CHANGE_TYPES:
            raise ValueError(f"unknown change type {change_type!r}; "
                             f"known: {CHANGE_TYPES}")
        if change_type in NEVER_AUTO:
            raise ValueError(f"{change_type!r} can never auto-apply — it stays "
                             f"propose-only by design")
        self.dials[change_type] = RUN

    def demote(self, change_type: str) -> None:
        if change_type not in CHANGE_TYPES:
            raise ValueError(f"unknown change type {change_type!r}; "
                             f"known: {CHANGE_TYPES}")
        self.dials[change_type] = WATCH


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_ledger(path: Path, entry: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"ts": _now_iso(), **entry}, sort_keys=True) + "\n")


def read_ledger(path: Path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue                 # a torn line never poisons the record
    return out


def competence_summary(entries: list) -> dict:
    """Per change-type counts a human promotion decision reads."""
    out = {ct: {"proposed": 0, "auto_applied": 0, "validation_failed": 0,
                "accepted": 0, "rejected": 0} for ct in CHANGE_TYPES}
    by_id: dict[str, str] = {}          # proposal_id -> change_type (for verdicts)
    for e in entries:
        ct = e.get("change_type")
        action = e.get("action")
        pid = e.get("proposal_id")
        if ct in out and pid:
            by_id[pid] = ct
        if action in ("proposed", "auto_applied", "validation_failed") and ct in out:
            out[ct][action] += 1
        elif action in ("accepted", "rejected"):
            ct = ct if ct in out else by_id.get(pid or "")
            if ct in out:
                out[ct][action] += 1
    return out
