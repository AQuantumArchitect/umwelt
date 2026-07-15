"""Phase 2 host API contract tests."""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from examples.fledgeling_fog.world import FOG_SPEC, place_names
from umwelt.host import GameHost, Intent
from umwelt.host.api import Belief


def _host() -> GameHost:
    h = GameHost()
    h.register_world(FOG_SPEC, population=False)
    return h


def test_eta_zero_is_noop():
    h = _host()
    places = place_names()
    before = {
        p: tuple(float(v) for v in h.engine.field.clusters[p].role_bloch("agent_near"))
        for p in places
    }
    r = h.observe("scout", f"scout_{places[0]}", 1.0, confidence=0.0)
    assert r.get("accepted") is False
    assert r.get("reason") == "eta_zero"
    after = {
        p: tuple(float(v) for v in h.engine.field.clusters[p].role_bloch("agent_near"))
        for p in places
    }
    for p in places:
        for a, b in zip(before[p], after[p]):
            assert abs(a - b) < 1e-9, f"{p} changed under η=0"


def test_positive_eta_moves_belief():
    h = _host()
    p0 = place_names()[0]
    p1 = place_names()[1]
    # Drive p0 empty then occupied so motion is measurable from either pole
    for _ in range(6):
        h.observe("scout", f"scout_{p0}", 0.0, confidence=1.0)
        h.step()
    low = h.belief_value(p0, "agent_near").value
    for _ in range(8):
        h.observe("scout", f"scout_{p0}", 1.0, confidence=1.0)
        h.step()
    high = h.belief_value(p0, "agent_near").value
    assert high > low + 0.05, f"low={low:.3f} high={high:.3f}"
    # untouched place stays distinct from heavily observed p0
    other = h.belief_value(p1, "agent_near").value
    assert abs(high - other) > 0.02


def test_shadow_intend_no_world_side_effect():
    h = _host()
    before = len(h.world_side_effects)
    d = h.intend("agent_a", Intent(actor_id="agent_a", name="claim_safe", shadow=True))
    assert d.mode == "shadow"
    assert d.dispatched is False
    assert len(h.world_side_effects) == before
    assert len(h.live_dispatches) == 0


def test_beliefs_return_calibrated_not_bloch_required():
    h = _host()
    out = h.beliefs("scout")
    assert out
    sample = next(iter(out.values()))
    assert isinstance(sample, Belief)
    assert 0.0 <= sample.value <= 1.0
    assert 0.0 <= sample.confidence <= 1.0 + 1e-9
    # no 'bloch' key in public Belief
    assert not hasattr(sample, "bloch")


def test_kill_reload_state_survives():
    h = _host()
    p0 = place_names()[0]
    for _ in range(8):
        h.observe("scout", f"scout_{p0}", 1.0, confidence=1.0)
        h.step()
    digest = h.field_canon_hash()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "host.pkl"
        h.save(path)
        h2 = _host()
        h2.load(path)
        assert h2.field_canon_hash() == digest


def test_step_turn_advances():
    h = _host()
    t0 = h.now
    h.step_turn(3)
    assert h.turn == 3
    assert h.now > t0


def test_phase1_happy_path_has_no_raw_ingest():
    """Structural: demo happy path uses host observe, not raw engine ingest calls."""
    demo = Path(__file__).resolve().parents[1] / "examples" / "fledgeling_fog" / "demo.py"
    text = demo.read_text(encoding="utf-8")
    # Strip comments/docstrings for call-site check
    code_lines = []
    for line in text.splitlines():
        stripped = line.split("#", 1)[0]
        code_lines.append(stripped)
    code = "\n".join(code_lines)
    # Drop module docstring
    if code.lstrip().startswith('"""'):
        end = code.lstrip().find('"""', 3)
        if end != -1:
            code = code.lstrip()[end + 3 :]
    assert "engine.ingest" not in code
    assert ".ingest(" not in code or "observe_many" in code
    assert "GameHost" in text
    assert "observe_many" in text
