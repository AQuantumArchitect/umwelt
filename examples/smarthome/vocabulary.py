"""The smart-home domain vocabulary — every registration the origin deployment makes.

The engine ships ZERO domain words (lint-enforced); a domain speaks by registering its
vocabulary at import. This module is the worked example of that pattern, restoring the
home idioms the extraction moved out of the engine: role input modes, home normalizers,
the render glyphs, and the home's reward manifest.

Call `register_smarthome_vocabulary()` once before building a home spec.
"""
from __future__ import annotations

from umwelt.spec import roles
from umwelt.spec.normalizers import register_normalizer, regime_norm


def _temp_f(center: float, width: float, from_c: bool = False):
    """Regime classifier in °F with optional °C→°F conversion (the home convention)."""
    base = regime_norm(center=center, width=width)
    if from_c:
        return lambda v: base(v * 9.0 / 5.0 + 32.0)
    return base


def _contact_presence(val: float) -> float:
    """Contact sensor for the occupancy observe path: closed (1.0) → -1.0 (drives the
    target toward occupied under the flipped device-echo convention), open (0.0) → 0.0
    (equatorial = uncertain)."""
    return -1.0 if val > 0.5 else 0.0


_REGISTERED = False


def register_smarthome_vocabulary() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    # ── role input modes (were the origin's _UNITARY_ROLES / _ANALOG_ROLES tables) ──
    for role in ("presence", "activity", "open", "locked", "door_state", "occupancy",
                 "active", "power_state", "heating", "motion", "state"):
        roles.register_role_mode(role, "unitary")
    for role in ("temperature", "humidity", "pressure", "conditions"):
        roles.register_role_mode(role, "dissipative", analog=True)
    for role in ("light_state", "power_draw"):
        roles.register_observe_role(role)

    # ── home normalizers ──
    register_normalizer("temp_f", _temp_f)
    register_normalizer("contact_presence", lambda: _contact_presence)

    # ── render glyphs (the origin's emoji maps, now app data) ──
    from umwelt.projection.emoji import register_node_icon, register_role_emoji
    register_role_emoji("presence", {"pos": "🏠", "zero": "🚪", "neg": "🚶", "coherent": "⚡"})
    register_role_emoji("activity", {"pos": "🏃", "zero": "🧘", "neg": "💤", "coherent": "⚡"})
    register_role_emoji("temperature", {"pos": "🌡️", "zero": "🌤️", "neg": "❄️", "coherent": "🌀"})
    register_role_emoji("light_state", {"pos": "💡", "zero": "↔️", "neg": "🌑", "coherent": "⚡"})
    for node, icon in (("bedroom", "🛏️"), ("bathroom", "🚿"), ("kitchen", "🍳"),
                       ("living", "🛋️"), ("hallway", "🚶"), ("exterior", "🌍")):
        register_node_icon(node, icon)
    from umwelt.substrate.ground import register_collapse_emoji
    register_collapse_emoji("presence", "🏠", "🚶")
    register_collapse_emoji("activity", "🏃", "💤")

    # ── the home's reward manifest additions ──
    from umwelt.learning.reward.registry import (declare_outcome_channel,
                                                 register_param_channel)
    declare_outcome_channel("sleep")
    register_param_channel(exact={"presence_decay"}, channel="override")

    # ── the home's known place + solar clock ──
    from umwelt.projection.gauge_name import register_known_place
    register_known_place("austin", 30.267, -97.743)
    from examples.smarthome.solar import register_solar_driver
    register_solar_driver()
