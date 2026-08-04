"""Heredity: a forged world can descend from an existing one.

Before this, every forge attempt authored from nothing. Variation with no
inheritance is not evolution — it is repeated invention, and nothing accumulates
across attempts. `coral/gym_lineage_select.py` has implemented ε-greedy selection
over lineages the whole time and had never been given world variants to choose
between; this is what produces them.

The gate is deliberately unchanged: a variant earns registration exactly as an
original does, from an independent subprocess that does not trust the agent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from umweltforge import pipeline as pl            # noqa: E402
from umweltforge.prompts import authoring_task_prompt  # noqa: E402


def test_no_parent_is_unchanged(tmp_path):
    assert pl._read_parent(None) == (None, None)
    p = authoring_task_prompt("a rant", "world_x.py", "world_x:SPEC")
    assert "VARIANT" not in p and "parent spec" not in p


def test_parent_module_file_is_read(tmp_path):
    f = tmp_path / "world_soil.py"
    f.write_text("SPEC = 'parent body'", encoding="utf-8")
    src, name = pl._read_parent(f)
    assert src == "SPEC = 'parent body'" and name == "world_soil"


def test_parent_directory_prefers_world_py(tmp_path):
    d = tmp_path / "wargen_self"
    d.mkdir()
    (d / "vocabulary.py").write_text("# not the spec", encoding="utf-8")
    (d / "world.py").write_text("SPEC = 'the real one'", encoding="utf-8")
    src, name = pl._read_parent(d)
    assert src == "SPEC = 'the real one'" and name == "world"


def test_parent_directory_skips_vocabulary_when_no_world_py(tmp_path):
    """A forged workspace holds exactly one module and it is not vocabulary.py."""
    d = tmp_path / "forged"
    d.mkdir()
    (d / "vocabulary.py").write_text("# nope", encoding="utf-8")
    (d / "world_forge_probe.py").write_text("SPEC = 'forged body'", encoding="utf-8")
    src, name = pl._read_parent(d)
    assert name == "world_forge_probe"


def test_missing_parent_degrades_to_no_lineage(tmp_path):
    """An unreadable parent must not crash an attempt — it just has no ancestor."""
    assert pl._read_parent(tmp_path / "nope.py") == (None, None)


def test_prompt_carries_the_parent_and_asks_for_a_deliberate_diff():
    p = authoring_task_prompt("a rant", "world_x.py", "world_x:SPEC",
                              parent_source="SPEC = 'ancestor'",
                              parent_name="world_soil")
    assert "VARIANT" in p
    assert "SPEC = 'ancestor'" in p
    assert "world_soil" in p
    # the point of a variant is a measurable difference, not a rewrite
    assert "changed" in p


def test_parent_and_failure_report_coexist():
    """A retry of a variant must keep BOTH its ancestor and the gate's complaint."""
    p = authoring_task_prompt("a rant", "world_x.py", "world_x:SPEC",
                              last_report_json='{"ok": false}',
                              parent_source="SPEC = 'ancestor'",
                              parent_name="world_soil")
    assert "ancestor" in p and "previous validation report" in p
