"""The spec gate's own gate — validate_spec surfaces every authoring error loudly.

The load-bearing pin: `apply_spec_bindings` is membrane-guarded (a bad binding is
skipped with a warning so a running world never loses its good bindings), which is
exactly wrong for AUTHORING — so `bindings_strict` must surface what boot swallows.
Contrast tests/test_binding_validation.py, which pins the skip-and-log behavior.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

from examples.gridworld.world import gridworld_spec
from umwelt.spec.schema import BindingSpec, DomainSpec, DriverSpec, NodeSpec, OutputSpec
from umwelt.spec.validate import validate_spec


def _tiny_spec(**overrides) -> DomainSpec:
    """A minimal valid world: a root, one region, one binary signal."""
    base = dict(
        name="tiny",
        nodes=(
            NodeSpec("top", parent=None, kind="root", roles=()),
            NodeSpec("area_a", parent="top", roles=("level",)),
        ),
        bindings=(BindingSpec("sig_a", zone="area_a", role="level",
                              normalizer="binary"),),
    )
    base.update(overrides)
    return DomainSpec(**base)


def test_gridworld_passes_every_check():
    report = validate_spec(gridworld_spec())
    assert report.ok, report.summary()
    assert [c.name for c in report.checks] == [
        "resolve", "schema_sanity", "topology_build", "bindings_strict",
        "boot_blank", "synthetic_exercise", "save_load_roundtrip"]
    assert not any(c.skipped for c in report.checks)


def test_tiny_spec_passes():
    report = validate_spec(_tiny_spec())
    assert report.ok, report.summary()


def test_two_roots_fail_schema_sanity():
    spec = _tiny_spec(nodes=(
        NodeSpec("top", parent=None, kind="root", roles=()),
        NodeSpec("second_top", parent=None, kind="root", roles=()),
        NodeSpec("area_a", parent="top", roles=("level",)),
    ))
    report = validate_spec(spec)
    assert not report.ok
    failed = {c.name for c in report.failures()}
    assert "schema_sanity" in failed
    assert "multiple roots" in next(c for c in report.checks
                                    if c.name == "schema_sanity").detail


def test_orphan_parent_fails_topology_build():
    spec = _tiny_spec(nodes=(
        NodeSpec("top", parent=None, kind="root", roles=()),
        NodeSpec("area_a", parent="nowhere_node", roles=("level",)),
    ))
    report = validate_spec(spec)
    assert not report.ok
    topo = next(c for c in report.checks if c.name == "topology_build")
    assert not topo.ok and "nowhere_node" in topo.detail
    # downstream checks skip rather than cascade
    assert all(c.skipped for c in report.checks
               if c.name in ("bindings_strict", "boot_blank",
                             "synthetic_exercise", "save_load_roundtrip"))


def test_binding_to_unknown_node_fails_bindings_strict():
    """THE pin: the exact error apply_spec_bindings would swallow is surfaced,
    naming the binding."""
    spec = _tiny_spec(bindings=(
        BindingSpec("sig_a", zone="area_a", role="level", normalizer="binary"),
        BindingSpec("sig_ghost", zone="no_such_node", role="level",
                    normalizer="binary"),
    ))
    report = validate_spec(spec)
    assert not report.ok
    strict = next(c for c in report.checks if c.name == "bindings_strict")
    assert not strict.ok
    assert "sig_ghost" in strict.detail and "no_such_node" in strict.detail
    # the world WOULD boot (that's the membrane guard) — the gate must still fail
    exercised = next(c for c in report.checks if c.name == "synthetic_exercise")
    assert exercised.skipped


def test_binding_to_undeclared_role_fails_bindings_strict():
    spec = _tiny_spec(bindings=(
        BindingSpec("sig_a", zone="area_a", role="no_such_role",
                    normalizer="binary"),))
    report = validate_spec(spec)
    strict = next(c for c in report.checks if c.name == "bindings_strict")
    assert not strict.ok and "no_such_role" in strict.detail


def test_unknown_normalizer_type_fails_bindings_strict():
    spec = _tiny_spec(bindings=(
        BindingSpec("sig_a", zone="area_a", role="level",
                    normalizer={"type": "made_up_norm"}),))
    report = validate_spec(spec)
    strict = next(c for c in report.checks if c.name == "bindings_strict")
    assert not strict.ok and "made_up_norm" in strict.detail


def test_dead_normalizer_fails_synthetic_exercise():
    # range with lo == hi maps every reading to 0.0 — vocabulary that can never
    # move the field, caught before the coverage law would pass vacuously.
    spec = _tiny_spec(bindings=(
        BindingSpec("sig_a", zone="area_a", role="level",
                    normalizer={"type": "range", "lo": 5.0, "hi": 5.0}),))
    report = validate_spec(spec)
    assert not report.ok
    ex = next(c for c in report.checks if c.name == "synthetic_exercise")
    assert not ex.ok and "sig_a" in ex.detail and "dead" in ex.detail


def test_driver_role_bindings_are_exempt_from_coverage():
    spec = _tiny_spec(
        drivers=(DriverSpec("day", period_s=86400.0),),
        bindings=(
            BindingSpec("sig_a", zone="area_a", role="level", normalizer="binary"),
            BindingSpec("clock_sig", zone="_clock", role="phase",
                        normalizer={"type": "cyclic", "period": 86400.0}),
        ))
    report = validate_spec(spec)
    assert report.ok, report.summary()
    ex = next(c for c in report.checks if c.name == "synthetic_exercise")
    assert "clock_sig" in ex.detail        # exemption is reported, not silent


def test_shadow_law_enforced_and_waivable():
    spec = _tiny_spec(outputs=(
        OutputSpec("push_a", node="area_a", role="level", shadow=False),))
    strict = validate_spec(spec)
    sanity = next(c for c in strict.checks if c.name == "schema_sanity")
    assert not strict.ok and "shadow" in sanity.detail
    waived = validate_spec(spec, require_shadow=False)
    assert waived.ok, waived.summary()


def test_bad_ref_fails_resolve():
    report = validate_spec("no.such.module:SPEC")
    assert not report.ok
    assert report.checks[0].name == "resolve" and not report.checks[0].ok


def test_cli_green_and_json_shape(tmp_path):
    (tmp_path / "world_spec.py").write_text(
        "from examples.gridworld.world import gridworld_spec\n"
        "SPEC = gridworld_spec()\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])
    out = subprocess.run(
        [sys.executable, "-m", "umwelt.spec.validate", "world_spec:SPEC", "--json"],
        env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    payload = json.loads(out.stdout)
    assert payload["ok"] is True and payload["spec"] == "gridworld-3x3"
    assert {c["name"] for c in payload["checks"]} >= {
        "resolve", "bindings_strict", "synthetic_exercise"}


def test_cli_red_exit_code(tmp_path):
    (tmp_path / "bad_spec.py").write_text(
        "from umwelt.spec.schema import DomainSpec, NodeSpec, BindingSpec\n"
        "SPEC = DomainSpec(name='bad', nodes=("
        "NodeSpec('top', parent=None, kind='root', roles=()),"
        "NodeSpec('area_a', parent='top', roles=('level',)),),"
        "bindings=(BindingSpec('sig_x', zone='ghost', role='level',"
        " normalizer='binary'),))\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])
    out = subprocess.run(
        [sys.executable, "-m", "umwelt.spec.validate", "bad_spec:SPEC", "--json"],
        env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 1
    payload = json.loads(out.stdout)
    assert payload["ok"] is False
    strict = next(c for c in payload["checks"] if c["name"] == "bindings_strict")
    assert "sig_x" in strict["detail"]
