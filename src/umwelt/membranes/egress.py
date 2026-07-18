"""Egress — how decisions leave the field, as DATA (blocker 7 of the extraction).

Three pieces:
  * `Action` — the one command wire format every tendril emits.
  * `SpecTendril` + `build_tendrils(engine, spec)` — an OutputSpec becomes a live tendril:
    the engine reads the named node/role continuously, pumps a slow COMMITTED belief
    (rise=coupling / fall=decay — the two-timescale geometry that decouples a flappy
    perception from a sticky actuation), decodes it through a registered decoder, gates
    it, and emits an Action. Unit mapping + the device clamp live HERE at the edge; the
    field stays unit-free.
  * `OutputSurface` — routes emitted Actions: SHADOW / RECOMMEND decisions are recorded
    for the app to read; AUTO decisions go to the injected `dispatch(Action)` callable.
    The dispatcher is ALWAYS app code — the engine never knows a transport.

The law: a new output is SHADOW by default (`OutputSpec.shadow=True`) — it decides every
tick, its decisions are visible, and it dispatches nothing until the app flips it. Earned
autonomy is the app's call, not the engine's default.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from umwelt._util import clamp
from umwelt.membranes.tendril import CommittedBelief, Tendril

logger = logging.getLogger(__name__)


@dataclass
class Action:
    """A command to send to an output device/channel."""
    actuator_id: str
    command: dict            # device-specific command payload (opaque to the engine)
    node: str                # source node in the world model
    role: str                # source role
    value: int               # committed value (+1 or -1)
    confidence: float        # confidence at collapse time
    reason: str              # why (periodic, query, confidence, "<tendril>_auto", ...)

    def __repr__(self):
        return (
            f"Action({self.actuator_id}: {self.command} "
            f"<- {self.node}.{self.role}={'+'if self.value>0 else '-'} "
            f"conf={self.confidence:.2f} [{self.reason}])"
        )


# ── the decoder registry: OutputSpec.decode → (tendril, level, prev) -> command dict ──
# A decoder turns the committed belief into a device-unit command. "sticky" is the binary
# base case (purity-derived hysteresis); "linear" maps level onto the codomain. Domains
# register richer decoders (a color ramp, a categorical picker) via register_decoder.

def _sticky_decoder(tendril: "SpecTendril", level: float, prev_command: dict | None) -> dict:
    prev_on = bool((prev_command or {}).get("on", False))
    on = tendril.commit.commit(prev_on, tendril.hysteresis_scale)
    return {"on": on}


def _linear_decoder(tendril: "SpecTendril", level: float, prev_command: dict | None) -> dict:
    lo, hi = tendril.codomain
    return {"value": clamp(lo + level * (hi - lo), lo, hi)}


DECODERS: dict[str, Callable] = {
    "sticky": _sticky_decoder,
    "linear": _linear_decoder,
}


def register_decoder(name: str, fn: Callable) -> None:
    """Register a domain decoder: fn(tendril, level, prev_command) -> command dict."""
    if name in DECODERS:
        raise ValueError(f"decoder {name!r} already registered")
    DECODERS[name] = fn


def resolve_decoder(spec) -> tuple[Callable, dict]:
    """OutputSpec.decode (name | {type,**params}) → (decoder fn, params)."""
    d = spec.decode
    if isinstance(d, str):
        d = {"type": d}
    if callable(d):
        return d, {}
    name = d.get("type", "sticky")
    fn = DECODERS.get(name)
    if fn is None:
        raise ValueError(f"unknown decoder {name!r}; known: {sorted(DECODERS)}")
    return fn, {k: v for k, v in d.items() if k != "type"}


class SpecTendril(Tendril):
    """One OutputSpec, alive. The uniform mechanics every declared output shares:

        evidence = the source qubit's level             (field read, unit-free)
        commit.pump(evidence, coupling, decay)          (two-timescale committed belief)
        command  = decoder(commit)                      (device units, clamped — the edge)
        gates    → Action(reason="<name>_auto"|"_rec")  (or None while a gate holds)

    Learnable geometry: `coupling`/`decay` seed from OutputSpec.coupling and are nudged by
    operator overrides (apply_override) via the proportional-nudge law — a corrected
    output learns to rise faster or linger differently, per output, from its own
    readback channel."""

    def __init__(self, engine, spec):
        self.engine = engine
        self.spec = spec
        self.name = spec.name
        self.node = spec.node
        self.role = spec.role
        self.shadow = bool(spec.shadow)
        self.graph_node = spec.node
        self.route_mode = "auto"
        self.codomain = tuple(spec.codomain)
        self.readback_sensor = spec.readback_sensor
        self.reward_channel = spec.reward_channel
        c = dict(spec.coupling or {})
        self.commit = CommittedBelief(spec.name, rest_z=float(c.get("rest_z", -1.0)))
        self.coupling = float(c.get("coupling", 0.12))     # rise rate (evidence pump)
        self.decay = float(c.get("decay", 0.03))           # linger rate (relax to rest)
        self.hysteresis_scale = float(c.get("hysteresis_scale", 0.5))
        self.override_alpha = float(c.get("override_alpha", 0.6))
        self.override_lr = float(c.get("override_lr", 0.2))
        g = dict(spec.gates or {})
        self.enable_param = g.get("enable_param")          # root fiber gate, else always-on
        self.rate_limit_s = float(g.get("rate_limit_s", 30.0))
        self.deadband = float(g.get("deadband", 0.05))
        self.echo_window = float(g.get("echo_window", 300.0))
        self._decoder, self._decoder_params = resolve_decoder(spec)
        self._last_command: dict | None = None
        self._last_level: float | None = None
        self._last_dispatch_ts: float | None = None
        self._last_override_ts: float | None = None

    # ── reads ─────────────────────────────────────────────────────────────────────
    def _evidence(self) -> float:
        cluster = self.engine.field.clusters.get(self.node)
        if cluster is None or self.role not in getattr(cluster, "role_index", {}):
            return 0.0
        z = float(cluster.role_bloch(self.role)[2])
        return (z + 1.0) / 2.0

    def _root_param(self, key: str, default: float) -> float:
        rb = getattr(getattr(self.engine, "graph", None), "root", None)
        if rb is None or rb.param_bundle is None:
            return default
        return float(rb.param_bundle.get(key, default))

    def enabled(self) -> bool:
        if self.enable_param is None:
            return True
        return self._root_param(self.enable_param, 0.0) >= 0.5

    def observed(self) -> float | None:
        if not self.readback_sensor:
            return None
        lr = self.engine.sensor_bridge.latest_raw(self.readback_sensor)
        return None if lr is None else float(lr[0])

    def last_dispatch_ts(self) -> float | None:
        return self._last_dispatch_ts

    # ── the tick ─────────────────────────────────────────────────────────────────
    def step(self, now_ts: float | None = None) -> Action | None:
        now_ts = time.time() if now_ts is None else float(now_ts)
        self.commit.pump(self._evidence(), self.coupling, self.decay)
        command = self._decoder(self, self.commit.level, self._last_command,
                                **self._decoder_params)
        level = self.commit.level
        # gates, in suppression order (each can only SKIP a dispatch, never add one)
        if not self.enabled():
            self._last_command = command
            self._last_level = level
            return None
        if (self._last_dispatch_ts is not None
                and now_ts - self._last_dispatch_ts < self.rate_limit_s):
            return None
        if command == self._last_command and self._last_command is not None:
            return None                                     # nothing new to say
        desired = command.get("value", 1.0 if command.get("on") else 0.0)
        if not self.out_of_preference(float(desired), self.observed(), self.deadband):
            return None                                     # reality already holds it
        action = Action(
            actuator_id=self.spec.dispatch.get("actuator_id", self.name),
            command={**command, **{k: v for k, v in self.spec.dispatch.items()
                                   if k != "actuator_id"}},
            node=self.node, role=self.role,
            value=1 if level >= 0.5 else -1,
            confidence=self.commit.confidence,
            reason=self.route_reason(),
        )
        self._last_command = command
        self._last_level = level
        self._last_dispatch_ts = now_ts
        return action

    # ── learning from the operator ───────────────────────────────────────────────
    def apply_override(self) -> None:
        """When the readback disagrees with our last decision shortly after we made it,
        the operator corrected us: pull the committed belief toward the observed reality
        (a hard-ish collapse) and nudge the rise/fall geometry by the proportional law so
        the same correction is needed less next time. Releases on the declared reward
        channel's vocabulary (the app reads reward_channel; the engine just learns)."""
        obs = self.observed()
        if obs is None or self._last_level is None:
            return
        span = max(self.codomain[1] - self.codomain[0], 1e-9)
        want = clamp((obs - self.codomain[0]) / span, 0.0, 1.0)
        err = want - self.commit.level
        # deadband is declared in DEVICE units (the edge owns units); compare in
        # level space by normalizing it through the codomain span.
        if abs(err) < self.deadband / span:
            return
        # collapse the belief toward what the operator revealed
        self.commit.cluster.observe_qubit(0, (0.0, 0.0, 2.0 * want - 1.0),
                                          self.override_alpha)
        # nudge the geometry: they wanted MORE → rise faster; LESS → linger shorter
        from umwelt.learning.meta_idioms import proportional_nudge
        ratio = 1.0 + self.override_lr * err
        if err > 0:
            self.coupling = clamp(proportional_nudge(self.coupling, ratio), 0.005, 0.9)
        else:
            self.decay = clamp(proportional_nudge(self.decay, 1.0 - self.override_lr * err),
                               0.002, 0.5)
        self._last_override_ts = time.time()

    def snapshot(self) -> dict:
        return {
            "name": self.name, "node": self.node, "role": self.role,
            "shadow": self.shadow, "enabled": self.enabled(),
            "commit": self.commit.snapshot(),
            "coupling": round(self.coupling, 5), "decay": round(self.decay, 5),
            "last_command": self._last_command,
            "last_dispatch_ts": self._last_dispatch_ts,
        }

    # ── persistence (the engine snapshot's tendril block) ────────────────────────
    def state_dict(self) -> dict:
        """Full-precision continuation state: the commit qubit + dispatch memory +
        learned rise/fall geometry — everything a resumed engine needs for its next
        tendril tick to be bit-identical to a never-stopped one. Measured to matter:
        without this block an incremental boot (snapshot + log tail) forks from a
        from-log replay on its first tail batch (the 2026-07-18 lease-drill chain
        fork's third cause, after RNG stream position and param display-rounding)."""
        return {
            "commit_mats": self.commit.cluster.state_matrices(),
            "last_command": self._last_command,
            "last_level": self._last_level,
            "last_dispatch_ts": self._last_dispatch_ts,
            "last_override_ts": self._last_override_ts,
            "coupling": float(self.coupling),
            "decay": float(self.decay),
        }

    def load_state_dict(self, data: dict) -> None:
        mats = data.get("commit_mats")
        if mats:
            self.commit.cluster.load_matrices(mats)
        self._last_command = data.get("last_command")
        self._last_level = data.get("last_level")
        self._last_dispatch_ts = data.get("last_dispatch_ts")
        self._last_override_ts = data.get("last_override_ts")
        if data.get("coupling") is not None:
            self.coupling = float(data["coupling"])
        if data.get("decay") is not None:
            self.decay = float(data["decay"])


def build_tendrils(engine, spec) -> list[SpecTendril]:
    """Every OutputSpec in the spec becomes a live tendril on the engine's uniform
    surface. Bad output specs fail LOUDLY at boot (a mis-declared output must never
    silently not exist)."""
    return [SpecTendril(engine, o) for o in (spec.outputs or ())]


class OutputSurface:
    """Routes emitted Actions. AUTO + non-shadow → the injected dispatcher; SHADOW and
    RECOMMEND decisions are recorded on `recommendations` for the app to read (the
    ghost layer). `history` keeps the recent dispatch record either way."""

    def __init__(self, dispatch: Callable[[Action], None] | None = None,
                 history: int = 256):
        self.dispatch = dispatch
        self.recommendations: deque[Action] = deque(maxlen=history)
        self.history: deque[tuple[float, Action, bool]] = deque(maxlen=history)

    def route(self, actions: list[Action], tendrils: list | None = None) -> int:
        """Route a tick's actions. Returns the number actually dispatched."""
        shadow_names = {t.name for t in (tendrils or []) if getattr(t, "shadow", False)}
        dispatched = 0
        now = time.time()
        for a in actions:
            is_auto = a.reason.endswith("_auto")
            name = a.reason.rsplit("_", 1)[0]
            live = is_auto and name not in shadow_names and self.dispatch is not None
            if live:
                try:
                    self.dispatch(a)
                    dispatched += 1
                except Exception as e:
                    logger.warning("dispatch of %s failed: %s", a.actuator_id, e)
                    live = False
            if not live:
                self.recommendations.append(a)
            self.history.append((now, a, live))
        return dispatched

    def snapshot(self) -> dict:
        return {
            "pending_recommendations": len(self.recommendations),
            "recent": [
                {"ts": ts, "actuator": a.actuator_id, "reason": a.reason,
                 "dispatched": live}
                for ts, a, live in list(self.history)[-10:]
            ],
        }
