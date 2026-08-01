"""Actuation autonomy — the cutover control plane (b9.3).

Every actuator is built but SHADOW-gated by a `*_enabled` param (default 0 → computes but never dispatches).
"Waking the body up" = flipping those to AUTO, one actuator at a time, only after a parity gate proves the
shadow path decides identically to the legacy loop ("auto-on after parity", the campaign decision). This
module is the single place that knows the mapping and owns the flip — so the flip is gauge-tracked, per-
actuator, reversible, and safety-guarded, rather than scattered literal `bundle.update(...)` calls.

It deliberately touches NONE of the actuator decision logic (that is being dissolved into qubits in Lane A,
b9.5/b9.6). It only operates the enable flags + reports posture, so it survives that rework unchanged. The
operator-facing autonomy switch (b9.8, watch/suggest/run) wires to set_posture(); the release config sets
the flags at deploy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

AUTO = "auto"          # the flag is on → the actuator drives the device
SHADOW = "shadow"      # the flag is off → it decides but never dispatches
ON, OFF = 1.0, 0.0
_FLIP_ALPHA = 1.0      # a posture flip is a hard set, not a gentle learn


@dataclass(frozen=True)
class ActuatorAutonomy:
    key: str                 # short stable name
    enable_param: str        # the param-bundle flag that gates dispatch
    label: str               # operator-facing label
    auto_capable: bool       # may this go AUTO? (locks/plugs: no — recommend-only, enforced here AND in the tendril)
    parity_proven: bool      # is there a green shadow-vs-legacy parity gate today?
    parity_ref: str          # where that evidence lives
    extra_params: tuple = field(default_factory=tuple)   # flags flipped together (e.g. lights' commit path)


# The engine ships an EMPTY registry — an actuator surface is DOMAIN DATA, declared by the
# deployment (register_actuator_autonomy), never engine code. The origin deployment's catalog
# (its device banks, in its campaign's cutover order) moves to its example/app.
REGISTRY: list[ActuatorAutonomy] = []
_BY_KEY: dict[str, ActuatorAutonomy] = {}


def register_actuator_autonomy(entry: ActuatorAutonomy) -> None:
    """Register one actuator's autonomy row. Duplicate keys raise — two surfaces fighting
    over one enable flag is a config bug, not a merge."""
    if entry.key in _BY_KEY:
        raise ValueError(f"actuator autonomy {entry.key!r} already registered")
    REGISTRY.append(entry)
    _BY_KEY[entry.key] = entry


def _get(bundle, name: str, default: float = 0.0) -> float:
    try:
        v = bundle.get(name, default)
        return float(v if v is not None else default)
    except Exception:
        return default


def posture(bundle, key: str) -> str:
    a = _BY_KEY[key]
    return AUTO if _get(bundle, a.enable_param) >= 0.5 else SHADOW


def set_posture(bundle, key: str, auto: bool, *, force: bool = False) -> dict:
    """Flip an actuator shadow⇄auto via its enable flag(s), gauge-tracked (the bundle.update turns the
    param's gauge clock). Refuses to go AUTO for a non-auto-capable actuator (locks/plugs), or for one
    without a parity gate unless `force=True` (auto-on AFTER parity). Returns the resulting record."""
    a = _BY_KEY[key]
    if auto and not a.auto_capable:
        return {"key": key, "changed": False, "posture": posture(bundle, key),
                "refused": "not auto-capable (recommend-only)"}
    if auto and not a.parity_proven and not force:
        return {"key": key, "changed": False, "posture": posture(bundle, key),
                "refused": "no parity gate yet (pass force=True to override)"}
    target = ON if auto else OFF
    before = posture(bundle, key)
    for p in (a.enable_param, *a.extra_params):
        bundle.update(p, target, _FLIP_ALPHA)
    after = posture(bundle, key)
    return {"key": key, "changed": before != after, "posture": after, "was": before,
            "params": [a.enable_param, *a.extra_params]}


def report(bundle, reservoir=None) -> dict:
    """The whole autonomy surface — for the console, the gauge, and the deploy decision.
    Given the live reservoir, each actuator also carries its EARNED competence breakdown
    (competence.actuator_competence, b9.44) so the Watch⇄Run switch shows what its family has
    learned — the earning is visible; the flip stays operator-owned."""
    comp: dict = {}
    if reservoir is not None:
        try:
            from umwelt.learning.competence import actuator_competence
            comp = actuator_competence(reservoir)
        except Exception:
            comp = {}
    items = []
    for a in REGISTRY:
        items.append({"key": a.key, "label": a.label, "posture": posture(bundle, a.key),
                      "auto_capable": a.auto_capable, "parity_proven": a.parity_proven,
                      "enable_param": a.enable_param, "parity_ref": a.parity_ref,
                      "competence": comp.get(a.key)})
    return {"actuators": items,
            "auto_count": sum(1 for i in items if i["posture"] == AUTO),
            "shadow_count": sum(1 for i in items if i["posture"] == SHADOW)}
