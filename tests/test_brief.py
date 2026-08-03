"""brief — the universal human world read composed purely from wire dicts.

Pins the law the module exists for: NO vocabulary/registry import is needed —
everything renders from what the trace itself carries (north/south emoji,
node_icon, reliability, surprise), and the prose is formatted from exactly the
dict `--json` would print."""
from __future__ import annotations

from umwelt.projection.brief import compose_brief, render_brief

HEALTH = {"world": "toy", "step": 42, "last_event_ts": "2026-01-01T00:00:00+00:00"}

TRACE = {
    "world": "toy",
    "registers": [
        {"node": "pantry", "role": "grain", "node_icon": "🏺",
         "north_emoji": "🌾", "south_emoji": "🕳️",
         "value": 0.95, "confidence": 0.9, "purity": 0.95, "r_xy": 0.0,
         "reliability": 0.1, "surprise": 0.2},
        {"node": "pantry", "role": "bread", "node_icon": "🏺",
         "north_emoji": "🍞", "south_emoji": "🕳️",
         "value": 0.02, "confidence": 0.8, "purity": 0.9, "r_xy": 0.6,
         "reliability": None, "surprise": None},
        {"node": "yard", "role": "agent_near", "node_icon": "🌳",
         "north_emoji": "🧍", "south_emoji": "🌫️",
         "value": 0.5, "confidence": 0.1, "purity": 0.5, "r_xy": 0.0,
         "reliability": 0.8, "surprise": None},
    ],
    "edges": [
        {"i": 0, "j": 1, "weight": 0.2, "kind": "zz"},
        {"i": 0, "j": 2, "weight": 0.11, "kind": "mi"},
    ],
}

RECS = [{"actuator_id": "harvest", "command": {"on": True}, "node": "pantry",
         "role": "grain", "value": 1, "confidence": 0.8, "reason": "auto"}]


def test_compose_groups_by_node_with_traveling_icons():
    b = compose_brief(HEALTH, TRACE, RECS)
    assert b["world"] == "toy"
    assert b["registers"] == 3
    assert set(b["nodes"]) == {"pantry", "yard"}
    assert b["nodes"]["pantry"]["icon"] == "🏺"       # from the trace, not NODE_ICONS
    assert b["nodes"]["yard"]["icon"] == "🌳"


def test_state_glyphs_come_from_trace_poles():
    b = compose_brief(HEALTH, TRACE, None)
    roles = {r["role"]: r["glyph"] for r in b["nodes"]["pantry"]["roles"]}
    assert roles["grain"] == "🌾"                     # z>0.3 → north pole glyph
    assert roles["bread"].startswith("🕳️")            # z<-0.3 → south pole glyph
    assert "💠" in roles["bread"]                     # r_xy>0.5 → coherence modifier
    yard = b["nodes"]["yard"]["roles"][0]
    assert yard["glyph"] == "⚪"                      # mid-axis → neutral


def test_movers_low_trust_mi_and_shadow():
    b = compose_brief(HEALTH, TRACE, RECS)
    assert b["movers"][0]["belief"] == "pantry.grain"          # highest surprise first
    assert b["low_observation_trust"] == ["pantry.grain"]      # reliability 0.1 < 0.3
    assert b["live_mi_edges"] == 1                             # mi kind counted apart
    assert b["shadow_recommendations"] == RECS


def test_render_is_formatted_from_the_same_dict():
    b = compose_brief(HEALTH, TRACE, RECS)
    text = render_brief(b)
    assert "world toy" in text and "step 42" in text
    assert "🏺" in text and "🌳" in text               # traveling node icons render
    assert "🌾" in text                                # pole glyph renders
    assert "movers: pantry.grain" in text
    assert "low observation-trust: pantry.grain" in text
    assert "harvest → {'on': True}" in text and "driven by pantry.grain" in text


def test_quiet_world_briefs_honestly_empty():
    b = compose_brief({"world": "quiet"}, {"world": "quiet", "registers": [], "edges": []})
    assert b["registers"] == 0 and b["movers"] == [] and b["live_mi_edges"] == 0
    text = render_brief(b)
    assert "never/unknown" in text                     # absent last_event_ts stays honest
