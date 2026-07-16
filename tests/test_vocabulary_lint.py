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
    # schema.py: "zone" is CODE — the NODE_KIND_ALIASES entry {"zone": "region"} and the
    # BindingSpec.zone field name (kept for signature-compatibility with the origin seam),
    # plus the comment/docstring lines documenting exactly those. "meerkat" is the
    # provenance citation in the module docstring.
    "spec/schema.py": {"zone", "meerkat"},
    # training_backbone.py: "zone" is CODE — _apply_spec_bindings reads BindingSpec.zone
    # and passes SensorBridge.register(zone=…), both field/kwarg names kept for
    # signature-compatibility with the origin seam (same rationale as spec/schema.py).
    "learning/training_backbone.py": {"zone"},
    # ingress.py: "zone" is CODE — SensorBridge.register(zone=...) / node_params(zone)
    # keep the origin seam's kwarg name so BindingSpec.zone and ported callers register
    # unchanged (apply_spec_bindings passes zone=b.zone).
    "membranes/ingress.py": {"zone"},
    # validate.py: "zone" is CODE — the strict gate reads BindingSpec.zone and calls
    # SensorBridge.register(zone=…) with no membrane guard (same origin-seam rationale
    # as ingress.py).
    "spec/validate.py": {"zone"},
    # kits: BindingSpec.zone kwarg (origin seam field name) in declarative specs
    "kits/fog/cassette.py": {"zone"},
    "kits/attention/cassette.py": {"zone"},
    "kits/market/cassette.py": {"zone"},
    "kits/dream/cassette.py": {"zone"},
}


def test_engine_source_is_domain_free():
    leaks: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        allowed = {w.lower() for w in ALLOW.get(rel, set())}
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in BANNED.finditer(line):
                word = m.group(1).lower()
                if word in allowed:
                    continue
                leaks.append(f"{rel}:{lineno}: '{m.group(1)}' in: {line.strip()[:90]}")
    assert not leaks, (
        "domain vocabulary leaked into the engine "
        f"({len(leaks)} hits):\n" + "\n".join(leaks[:40])
    )
