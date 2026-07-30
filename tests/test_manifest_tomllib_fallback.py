"""py3.10 CI: MANIFEST.toml must load without bare stdlib-only tomllib."""
from __future__ import annotations

from pathlib import Path

import pytest

from umweltd.packet import read_manifest_toml


def test_read_manifest_toml_parses_minimal(tmp_path: Path) -> None:
    (tmp_path / "MANIFEST.toml").write_text(
        'pythonpath = ["../checkout"]\n[pins]\numwelt_sha = "abc"\n',
        encoding="utf-8",
    )
    man = read_manifest_toml(tmp_path)
    assert man.get("pythonpath") == ["../checkout"]
    assert man.get("pins", {}).get("umwelt_sha") == "abc"


def test_read_manifest_toml_missing_is_empty(tmp_path: Path) -> None:
    assert read_manifest_toml(tmp_path) == {}
