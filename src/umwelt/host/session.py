"""WorldSession — shared ground, N private minds (FL-core Phase 3).

One classical ground/scene plus a map of private GameHosts (each a blank engine
with optional per-observer channel masks). Intents are tagged by actor_id so
self-action hygiene can key on who acted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from umwelt.host.api import Belief, Decision, GameHost, Intent, Observation

_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


@dataclass
class GroundState:
    """Shared classical scene (positions, flags) — not a belief field."""
    entities: dict[str, dict] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)

    def set_entity(self, entity_id: str, **attrs) -> None:
        self.entities.setdefault(entity_id, {}).update(attrs)

    def get(self, entity_id: str, key: str, default=None):
        return self.entities.get(entity_id, {}).get(key, default)


class WorldSession:
    """N private umwelten over one ground truth scene."""

    def __init__(self) -> None:
        self.ground = GroundState()
        self.minds: dict[str, GameHost] = {}
        self._spec = None
        self._t = _EPOCH
        self._tick_s = 1.0
        self._cost_notes: list[str] = []

    def register_world(self, spec, *, tick_s: float | None = None, start=None) -> "WorldSession":
        self._spec = spec
        if tick_s is not None:
            self._tick_s = float(tick_s)
        else:
            drivers = getattr(spec, "drivers", ()) or ()
            self._tick_s = float(drivers[0].period_s) if drivers else 1.0
        if start is not None:
            self._t = start
        return self

    def add_mind(
        self,
        observer_id: str,
        *,
        channel_mask: set[str] | None = None,
        population: bool = False,
    ) -> GameHost:
        if self._spec is None:
            raise RuntimeError("register_world first")
        host = GameHost()
        host.register_world(
            self._spec,
            tick_s=self._tick_s,
            start=self._t,
            population=population,
            observer_id=observer_id,
            channel_mask=channel_mask,
        )
        self.minds[observer_id] = host
        return host

    def mind(self, observer_id: str) -> GameHost:
        return self.minds[observer_id]

    def observe(self, obs: Observation) -> dict:
        host = self.minds.get(obs.observer_id)
        if host is None:
            raise KeyError(f"unknown observer {obs.observer_id!r}")
        t = obs.t if obs.t is not None else self._t
        return host.observe(
            obs.observer_id, obs.channel, obs.value, obs.confidence, t=t
        )

    def observe_raw(
        self,
        observer_id: str,
        channel: str,
        value: float,
        confidence: float = 1.0,
        t: datetime | None = None,
    ) -> dict:
        return self.observe(
            Observation(observer_id, channel, value, confidence, t or self._t)
        )

    def intend(self, actor_id: str, intent: Intent | str, **payload) -> Decision:
        """Intent applies to the actor's own mind only (private umwelt)."""
        host = self.minds.get(actor_id)
        if host is None:
            raise KeyError(f"unknown actor/mind {actor_id!r}")
        return host.intend(actor_id, intent, **payload)

    def beliefs(self, observer_id: str, query: str | None = None) -> dict[str, Belief]:
        return self.minds[observer_id].beliefs(observer_id, query=query)

    def step(self, t: datetime | None = None) -> dict[str, dict]:
        if t is not None:
            self._t = t
        else:
            from datetime import timedelta

            self._t = self._t + timedelta(seconds=self._tick_s)
        return {oid: h.step(t=self._t) for oid, h in self.minds.items()}

    def step_turn(self, n: int = 1) -> list[dict[str, dict]]:
        return [self.step() for _ in range(max(0, int(n)))]

    def apply_ground_event(
        self,
        *,
        channel: str,
        value: float,
        observers: list[str] | None = None,
        confidence: float = 1.0,
    ) -> None:
        """Push the same ground event into selected minds (who can sense it)."""
        targets = observers if observers is not None else list(self.minds)
        for oid in targets:
            self.observe_raw(oid, channel, value, confidence=confidence, t=self._t)

    def measure_cost(self, n_agents: int, *, ticks: int = 5) -> dict:
        """Cheap multi-engine cost probe: boot n_agents and step empty ticks."""
        import time

        if self._spec is None:
            raise RuntimeError("register_world first")
        t0 = time.perf_counter()
        tmp = WorldSession().register_world(self._spec, tick_s=self._tick_s)
        for i in range(n_agents):
            tmp.add_mind(f"agent_{i}", population=False)
        boot_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        for _ in range(ticks):
            tmp.step()
        step_s = time.perf_counter() - t1
        note = (
            f"N={n_agents} multi-engine: boot={boot_s:.4f}s, "
            f"{ticks} steps={step_s:.4f}s ({step_s / max(ticks, 1):.5f}s/step)"
        )
        self._cost_notes.append(note)
        return {
            "n_agents": n_agents,
            "boot_s": boot_s,
            "step_total_s": step_s,
            "step_mean_s": step_s / max(ticks, 1),
            "note": note,
        }

    @property
    def cost_notes(self) -> list[str]:
        return list(self._cost_notes)
