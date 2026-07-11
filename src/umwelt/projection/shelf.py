"""Shelf reader — the READ side of the brain-lineage DAG, as PROD code.

The brain selects which artifact to boot from (the origin deployment's boot selector) and the console lists the
catalogue (api._shelf_brains), so a lean read view over the shelf's append-only `index.jsonl` belongs in the
production package — NOT in the experiments/ lineage harness. The WRITE side (ops/archive.sh) and the rig that
mints artifacts stay in the lineage repo; this is just the reader they share.

Artifacts are content-addressed (sha256 + 8-char alias) with a parametric gauge-name `label` and lineage
`parents`. The archive lives OUTSIDE the code repo — `$UMWELT_ARCHIVE_ROOT` else `~/ws/smrthaus/archive`
(staging) → the deployment's external data drive. Nothing here writes; an unreachable archive yields an empty
view (callers degrade to the live mind / floor).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# default MUST match ops/archive.sh (`$HOME/ws/smrthaus/archive`) so reader and writer agree.
ARCHIVE_ROOT = Path(os.environ.get("UMWELT_ARCHIVE_ROOT", str(Path.home() / "ws" / "smrthaus" / "archive")))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


class Shelf:
    """Read view over the artifact lineage ledger (`<ARCHIVE_ROOT>/index.jsonl`)."""

    def __init__(self, archive_root: Path = ARCHIVE_ROOT):
        self.artifacts = _read_jsonl(Path(archive_root) / "index.jsonl")
        # content-addressed: full sha + 8-char alias → row; + the parametric gauge-name `label`
        # (newest wins, so re-minting the same name resolves to the latest artifact).
        self._by_id: dict[str, dict] = {}
        self._by_label: dict[str, dict] = {}
        for r in self.artifacts:
            sha = r.get("sha256", "")
            if sha:
                self._by_id[sha] = r
                self._by_id[sha[:8]] = r
            lbl = r.get("label")
            if lbl:
                self._by_label[lbl] = r

    def resolve(self, ident: str) -> dict | None:
        """Resolve a brain reference: exact/8-char sha, exact label, then unique label prefix."""
        if ident in self._by_id:
            return self._by_id[ident]
        if ident in self._by_label:
            return self._by_label[ident]
        hit = self._by_id.get(ident[:8])
        if hit is not None:
            return hit
        pref = [r for lbl, r in self._by_label.items() if lbl.startswith(ident)]
        return pref[0] if len(pref) == 1 else None

    def lineage(self, ident: str) -> list[dict]:
        """This artifact then its parents (depth-first up the ancestry)."""
        seen, chain, stack = set(), [], [ident]
        while stack:
            row = self.resolve(stack.pop(0))
            if row is None:
                continue
            key = row.get("sha256", "")[:8]
            if key in seen:
                continue
            seen.add(key)
            chain.append(row)
            for p in row.get("parents", []) or []:
                stack.append(p)
        return chain

    def best(self, kind: str | None = None, by: str = "score") -> dict | None:
        cands = [r for r in self.artifacts
                 if (kind is None or r.get("kind") == kind) and r.get(by) is not None]
        return max(cands, key=lambda r: r[by], default=None)

    def brains(self) -> list[dict]:
        """Brain artifacts newest-first — the catalogue the console brain-picker + boot select read."""
        return [r for r in reversed(self.artifacts) if r.get("kind") == "brain"]

    def field_nodes(self) -> list[dict]:
        """Field-state gauge nodes newest-first (kind=field) — the brain's content-addressed learning
        history (field_gauge): each is a canonical state that unpacks into the clustered OR manifold form."""
        return [r for r in reversed(self.artifacts) if r.get("kind") == "field"]
