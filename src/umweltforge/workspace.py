"""The forge workspace — one directory per authored world, the compiler's whole state.

Layout under <forge_root>/<world-name>/:

    rant.txt                the raw domain description, verbatim
    GUIDE.md                the distilled authoring instructions (prompts.AUTHORING_GUIDE)
    world_<name>.py         the generated DomainSpec module (attr SPEC) — this dir is
                            the world's manifest spec_path, so the daemon imports from
                            here at every boot
    attempts/               per-attempt snapshots: the module + its validation report
    warden/
        policy.json         the earned-autonomy dials (policy.WardenPolicy)
        ledger.jsonl        the append-only competence ledger
        ticks/<ts>/         per-tick context bundle, module copy, findings.json
        staging/            candidate modules under validation before an apply

The module file is world_<name>.py, not a generic world.py, so the spec ref in
world.json is self-identifying and two forge worlds can never shadow each other on a
shared sys.path.
"""
from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path("~/.umwelt/forge")


def forge_root(override: "Path | str | None" = None) -> Path:
    if override is not None:
        return Path(override).expanduser()
    env = os.environ.get("UMWELT_FORGE_ROOT")
    return Path(env).expanduser() if env else DEFAULT_ROOT.expanduser()


def module_name(world: str) -> str:
    return "world_" + re.sub(r"[^a-z0-9_]", "_", world.lower())


@dataclass
class ForgeWorkspace:
    root: Path          # <forge_root>/<name>
    name: str

    # ── paths ────────────────────────────────────────────────────────────────────
    @property
    def rant_path(self) -> Path: return self.root / "rant.txt"
    @property
    def guide_path(self) -> Path: return self.root / "GUIDE.md"
    @property
    def module_file(self) -> str: return module_name(self.name) + ".py"
    @property
    def module_path(self) -> Path: return self.root / self.module_file
    @property
    def attempts_dir(self) -> Path: return self.root / "attempts"
    @property
    def warden_dir(self) -> Path: return self.root / "warden"
    @property
    def policy_path(self) -> Path: return self.warden_dir / "policy.json"
    @property
    def ledger_path(self) -> Path: return self.warden_dir / "ledger.jsonl"
    @property
    def ticks_dir(self) -> Path: return self.warden_dir / "ticks"
    @property
    def staging_dir(self) -> Path: return self.warden_dir / "staging"

    def spec_ref(self) -> str:
        return f"{module_name(self.name)}:SPEC"

    # ── lifecycle ────────────────────────────────────────────────────────────────
    @classmethod
    def create(cls, name: str, rant: str,
               root: "Path | str | None" = None) -> "ForgeWorkspace":
        """Create a fresh workspace. Refuses to overwrite an existing one — a
        workspace holds a world's authoring history and warden ledger."""
        if not re.match(r"^[a-z0-9][a-z0-9_-]{0,63}$", name):
            raise ValueError(f"world name {name!r} must match [a-z0-9][a-z0-9_-]*")
        ws = cls(root=forge_root(root) / name, name=name)
        if ws.root.exists():
            raise FileExistsError(f"forge workspace already exists: {ws.root}")
        for d in (ws.root, ws.attempts_dir, ws.warden_dir, ws.ticks_dir,
                  ws.staging_dir):
            d.mkdir(parents=True, exist_ok=True)
        ws.rant_path.write_text(rant)
        from umweltforge.prompts import AUTHORING_GUIDE
        ws.guide_path.write_text(AUTHORING_GUIDE)
        from umweltforge.policy import WardenPolicy
        WardenPolicy.fresh(name).save(ws.policy_path)
        return ws

    @classmethod
    def open(cls, name: str, root: "Path | str | None" = None) -> "ForgeWorkspace":
        ws = cls(root=forge_root(root) / name, name=name)
        if not ws.root.is_dir():
            raise FileNotFoundError(
                f"no forge workspace for {name!r} at {ws.root} — was this world "
                f"created with `umwelt-forge new`?")
        return ws

    def snapshot_attempt(self, n: int, report: "dict | None") -> Path:
        """Preserve attempt n's module + validation report for the human audit trail."""
        import json
        dest = self.attempts_dir / f"attempt_{n:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        if self.module_path.exists():
            shutil.copy2(self.module_path, dest / self.module_file)
        (dest / "report.json").write_text(json.dumps(report, indent=1)
                                          if report is not None else "null")
        return dest
