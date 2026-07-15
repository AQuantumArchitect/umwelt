"""Agency loop — sub-routines, attention budget, earned automation (Phase 4).

Sub-routines are named policies that emit intents on a schedule under an attention
budget. Promotion shadow→live is explicit and gated by measured success count.
FF / time-contraction pauses when surprise or rest gate fires.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from umwelt.host.api import Decision, GameHost, Intent


@dataclass
class AttentionBudget:
    """Finite attention units spent by sub-routines each turn."""
    capacity: float = 10.0
    free: float = 10.0

    def reset(self) -> None:
        self.free = self.capacity

    def spend(self, cost: float) -> bool:
        if cost > self.free:
            return False
        self.free -= cost
        return True

    @property
    def low(self) -> bool:
        return self.free < 0.25 * self.capacity


@dataclass
class SubRoutine:
    """Named policy: emit an intent when scheduled and budget allows."""
    name: str
    intent_name: str
    period_turns: int = 1
    attention_cost: float = 1.0
    actor_id: str = "player"
    payload: dict = field(default_factory=dict)
    enabled: bool = True
    successes: int = 0
    shadow: bool = True
    auto_live: bool = False  # only after explicit promotion
    _last_fire_turn: int = -10**9

    def due(self, turn: int) -> bool:
        if not self.enabled:
            return False
        return (turn - self._last_fire_turn) >= self.period_turns


@dataclass
class PromotionGate:
    """Earned automation: shadow → live only after N measured successes."""
    min_successes: int = 3
    promoted: set[str] = field(default_factory=set)

    def record_success(self, routine: SubRoutine) -> None:
        routine.successes += 1

    def can_auto_intend(self, routine: SubRoutine) -> bool:
        """Shadow auto-intend is earned only after min_successes measured successes."""
        return routine.successes >= self.min_successes or routine.auto_live

    def can_promote(self, routine: SubRoutine) -> bool:
        return routine.successes >= self.min_successes

    def promote(self, routine: SubRoutine) -> bool:
        if not self.can_promote(routine):
            return False
        routine.shadow = False
        routine.auto_live = True
        self.promoted.add(routine.name)
        return True


@dataclass
class TimeContraction:
    """FF while attention is free; pause on surprise or rest."""
    ff_enabled: bool = True
    paused: bool = False
    reason: str = ""

    def update(
        self,
        *,
        attention: AttentionBudget,
        surprise: float = 0.0,
        rest: bool = False,
        surprise_threshold: float = 0.35,
    ) -> None:
        if rest or surprise >= surprise_threshold:
            self.paused = True
            self.reason = "rest" if rest else "surprise"
            self.ff_enabled = False
            return
        if attention.low:
            # low free attention → host may FF (when not surprised)
            self.ff_enabled = True
            self.paused = False
            self.reason = "ff"
            return
        self.ff_enabled = False
        self.paused = False
        self.reason = "normal"


class AgencyLoop:
    """Owns sub-routines + budget + promotion over a GameHost."""

    def __init__(
        self,
        host: GameHost,
        *,
        attention: AttentionBudget | None = None,
        promotion: PromotionGate | None = None,
    ) -> None:
        self.host = host
        self.attention = attention or AttentionBudget()
        self.promotion = promotion or PromotionGate()
        self.routines: dict[str, SubRoutine] = {}
        self.clock = TimeContraction()
        self.decisions: list[Decision] = []
        self._self_confound_guards: list[str] = []

    def add_routine(self, routine: SubRoutine) -> None:
        self.routines[routine.name] = routine

    def teach_success(self, routine_name: str) -> None:
        r = self.routines[routine_name]
        self.promotion.record_success(r)

    def promote(self, routine_name: str) -> bool:
        return self.promotion.promote(self.routines[routine_name])

    def tick(
        self,
        *,
        surprise: float = 0.0,
        rest: bool = False,
        success_check: Callable[[SubRoutine, Decision], bool] | None = None,
    ) -> list[Decision]:
        """One agency tick: update FF gate, fire due routines under budget."""
        self.attention.reset()
        self.clock.update(
            attention=self.attention, surprise=surprise, rest=rest
        )
        if self.clock.paused:
            return []

        fired: list[Decision] = []
        turn = self.host.turn
        for r in self.routines.values():
            if not r.due(turn):
                continue
            if not self.attention.spend(r.attention_cost):
                continue
            # Auto-intend only after N measured successes (PromotionGate.min_successes).
            # Live dispatch still requires explicit promote() → auto_live.
            if not self.promotion.can_auto_intend(r):
                continue
            shadow = r.shadow and not r.auto_live
            intent = Intent(
                actor_id=r.actor_id,
                name=r.intent_name,
                payload=dict(r.payload),
                shadow=shadow,
            )
            decision = self.host.intend(r.actor_id, intent)
            r._last_fire_turn = turn
            fired.append(decision)
            self.decisions.append(decision)
            # Self-confound guard: shadow/live intents must tag actor.
            self._assert_no_self_confound(r, decision)
            if success_check is not None and success_check(r, decision):
                self.teach_success(r.name)
        return fired

    def _assert_no_self_confound(self, routine: SubRoutine, decision: Decision) -> None:
        """Record that automation intents stay actor-tagged (no anonymous world write)."""
        if not decision.actor_id:
            raise AssertionError(
                f"routine {routine.name} produced untagged intent — self-confound risk"
            )
        self._self_confound_guards.append(
            f"{routine.name}:{decision.actor_id}:{decision.mode}"
        )

    def patrol_demo_ready(self, routine_name: str, n_successes: int) -> bool:
        """Teach N successes then allow shadow auto-intend."""
        for _ in range(n_successes):
            self.teach_success(routine_name)
        r = self.routines[routine_name]
        return r.successes >= n_successes
