"""The no-domain-vocabulary gate.

The engine's core promise is that src/umwelt/ knows NOTHING about any particular world:
no houses, no rooms, no residents, no astronomy, no vendor names. Domain vocabulary lives
in specs, registered normalizers/roles/drivers, and examples/ — never in the engine.

This test is the structural enforcement: it fails the moment a banned word appears in
engine source. Provenance citations are allowed per-file via ALLOW (a docstring crediting
the meerkat origin is honest, a `room` variable is a leak).
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "umwelt"

BANNED = re.compile(
    r"\b(meerkat|austin|solar|celestial|resident|room|rooms|house|houses|zone|zones|"
    r"presence|kasa|zigbee|mqtt|weather|sunrise|sunset|moon|dimmer|thermostat)\b",
    re.IGNORECASE,
)

# file (relative to src/umwelt) -> words permitted there, for provenance/docstring honesty
ALLOW: dict[str, set[str]] = {
    "_util.py": {"meerkat"},
}


def test_engine_source_is_domain_free():
    leaks: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = str(path.relative_to(SRC))
        allowed = {w.lower() for w in ALLOW.get(rel, set())}
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for m in BANNED.finditer(line):
                word = m.group(1).lower()
                if word in allowed:
                    continue
                leaks.append(f"{rel}:{lineno}: '{m.group(1)}' in: {line.strip()[:90]}")
    assert not leaks, (
        "domain vocabulary leaked into the engine "
        f"({len(leaks)} hits):\n" + "\n".join(leaks[:40])
    )
