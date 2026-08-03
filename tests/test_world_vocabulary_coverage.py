"""World vocabulary coverage — every role/node a yurt world declares must have a real glyph.

`umwelt.projection.emoji`'s own contract is domain-registration: "the engine ships only the
neutral fallback... registered by the domain" (emoji.py's header). Nothing enforced that contract —
a world could declare a role, never register it, and the render would just silently look a little
duller forever. Found the hard way, twice: hive-ops had 7/30 roles falling through to the generic
🔵/⚪/🔴 default (the fleet's original 5 roles + the wake organ's 2, never wired into
`register_role_emoji`), and yurt-mood had ALL 8 roles generic despite its own docstring already
having designed the glyph vocabulary in prose. Both went unnoticed until a human eyeballed a
cognifold screenshot. This test turns that into a hard, automatic failure.

Each world is checked in its OWN subprocess — `ROLE_EMOJI`/`NODE_ICONS` are module-level globals in
emoji.py, so importing two worlds' vocabularies in one process would cross-contaminate coverage
(a role `hive-ops` registers would falsely "cover" the same role name in a different, unrelated
world). Importing `world` (not calling `vocabulary.register()` directly) exercises the exact path
production boot uses (`umweltd.worker.WorldHost.__init__` → `_call_ref(manifest["vocabulary"])` or,
for the worlds that self-register at import time like hive-ops/yurt-mood, the `world.py` import
itself) — same discovery path `hearth_client.py`'s `YURT_ENV_DEFAULT` and `scripts/cognifold.sh`
already use to reach the yurt tree from a sibling repo.

If the yurt worlds tree isn't mounted here (this repo runs standalone in plenty of environments),
the whole module skips rather than failing — this is a coverage check ON that tree, not a claim
umwelt requires it to exist.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

YURT_WORLDS_DIR_DEFAULT = "/mnt/c/Users/Luke Spooner/ws-win/yurt/worlds"
UMWELT_SRC = str(Path(__file__).resolve().parents[1] / "src")

# 2026-08-03: the verdance allowlist is CLOSED. The axis pole meanings turned out to be
# documented in the BAR workspace itself (quantum_bar_ai/world.py docstring + campaign_manifold.py
# CAMPAIGN_AXES comments); verdance_cognifold_bridge.py transcribed them into glyph pairs and both
# worlds now carry a real vocabulary.py sourced from the same table. Empty set kept (not deleted)
# so the next genuinely-untranscribable world has the mechanism ready.
NO_VOCAB_ALLOWLIST: set[str] = set()

# Boot shims that re-export another world's SPEC verbatim rather than declaring their own — testing
# them would just duplicate that world's result, not check anything new.
_SKIP_PREFIXES = ("_",)


def _yurt_worlds_dir() -> Path | None:
    raw = os.environ.get("YURT_WORLDS_DIR", YURT_WORLDS_DIR_DEFAULT)
    p = Path(raw)
    return p if p.is_dir() else None


def _discover_worlds(worlds_dir: Path) -> list[str]:
    out = []
    for child in sorted(worlds_dir.iterdir()):
        if not child.is_dir() or child.name.startswith(_SKIP_PREFIXES):
            continue
        if (child / "world.py").exists():
            out.append(child.name)
    return out


_AUDIT_SCRIPT = r"""
import json, sys
sys.path.insert(0, {umwelt_src!r})
sys.path.insert(0, {world_dir!r})
try:
    world = __import__("world")
except Exception as exc:
    print(json.dumps({{"import_error": repr(exc)}}))
    raise SystemExit(0)

from umwelt.projection.emoji import ROLE_EMOJI, NODE_ICONS

spec = world.SPEC
roles, nodes = set(), set()
for node in spec.nodes:
    nodes.add(node.name)
    roles.update(node.roles or ())

print(json.dumps({{
    "name": spec.name,
    "unregistered_roles": sorted(r for r in roles if r not in ROLE_EMOJI),
    "unregistered_nodes": sorted(n for n in nodes if n not in NODE_ICONS),
    "total_roles": len(roles),
    "total_nodes": len(nodes),
}}))
"""


def _audit_world(world_dir: Path) -> dict:
    script = _AUDIT_SCRIPT.format(umwelt_src=UMWELT_SRC, world_dir=str(world_dir))
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, (
        f"world {world_dir.name!r} audit subprocess crashed:\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    assert lines, f"world {world_dir.name!r} audit produced no output; stderr={proc.stderr}"
    return json.loads(lines[-1])


def _world_names() -> list[str]:
    worlds_dir = _yurt_worlds_dir()
    if worlds_dir is None:
        return []
    return _discover_worlds(worlds_dir)


@pytest.fixture(scope="module")
def worlds_dir() -> Path:
    d = _yurt_worlds_dir()
    if d is None:
        pytest.skip(
            "yurt worlds tree not mounted here "
            f"(set YURT_WORLDS_DIR, or expected {YURT_WORLDS_DIR_DEFAULT!r}) — "
            "this is a coverage check ON that tree, not a hard umwelt dependency"
        )
    return d


@pytest.mark.parametrize("world_name", _world_names())
def test_world_role_and_node_emoji_coverage(worlds_dir, world_name):
    """Every role/node a world's own SPEC declares must be registered — no silent generic default."""
    result = _audit_world(worlds_dir / world_name)

    if world_name in NO_VOCAB_ALLOWLIST:
        assert "import_error" not in result, (
            f"{world_name} is on the no-vocabulary allowlist but now fails to import "
            f"({result.get('import_error')}) — investigate before trusting the allowlist"
        )
        return

    assert "import_error" not in result, f"{world_name} failed to import: {result['import_error']}"
    assert not result["unregistered_roles"], (
        f"{world_name}: roles falling through to the generic emoji default: "
        f"{result['unregistered_roles']} — register_role_emoji() for each in its vocabulary.py"
    )
    assert not result["unregistered_nodes"], (
        f"{world_name}: nodes with no icon: {result['unregistered_nodes']} — "
        f"register_node_icon() for each in its vocabulary.py"
    )


def test_no_vocab_allowlist_is_still_accurate(worlds_dir):
    """The allowlist names worlds with genuinely NO vocabulary.py — catch drift either way:
    a world regaining a vocabulary.py (allowlist entry stale, should be removed) or losing one
    it used to have (a silent regression the allowlist would otherwise hide)."""
    discovered = set(_discover_worlds(worlds_dir))
    has_vocab = {n for n in discovered if (worlds_dir / n / "vocabulary.py").exists()}
    stale_allowlist = NO_VOCAB_ALLOWLIST & has_vocab
    assert not stale_allowlist, (
        f"{stale_allowlist} now have a vocabulary.py — remove from NO_VOCAB_ALLOWLIST "
        "and let the real coverage test hold them to the standard"
    )
