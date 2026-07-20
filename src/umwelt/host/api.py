"""GameHost — plain observe / intend / beliefs / step over a BeliefEngine.

Maps:
  observe → binding channel + η (confidence)
  intend  → shadow|live Decision via OutputSurface / tendrils
  beliefs → calibrated scalar + confidence (not Bloch z by default)
  step / step_turn → game cadence ingest ticks
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from umwelt.boot import build_engine
from umwelt.substrate.bloch import bloch_radius

_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Observation:
    observer_id: str
    channel: str
    value: float
    confidence: float = 1.0  # η ∈ [0,1]; 0 is a no-op
    t: datetime | None = None


@dataclass(frozen=True)
class Intent:
    actor_id: str
    name: str
    payload: dict = field(default_factory=dict)
    shadow: bool = True


@dataclass(frozen=True)
class Decision:
    intent_name: str
    actor_id: str
    mode: str  # "shadow" | "live"
    command: dict
    confidence: float
    dispatched: bool
    node: str = ""
    role: str = ""


@dataclass(frozen=True)
class Belief:
    node: str
    role: str
    value: float  # calibrated scalar, typically [0,1] from z
    confidence: float  # |r| ∈ [0,1] — Bloch radius: how settled the belief is
    # The unified gauge, filled from latent engine machinery when available
    # (both default None so every existing value/confidence consumer is untouched):
    reliability: float | None = None   # learned observation trust (ObservationTrust
    #   alpha ∈ [~0.1, ~0.97]): how consistent this leaf's readings are. None if the
    #   leaf has never been observed.
    forecast_skill: float | None = None  # shortest-horizon forecaster skill (1-error_ema)
    #   from an attached forecast surface. None if no forecaster is attached.


def _forecast_skill(eng, node: str, role: str):
    """Shortest-horizon forecaster skill for one leaf, if a forecast surface is
    attached to the engine (foresight/ForecastSurface). Defensive: any surface
    without the expected shape, or none at all, yields None — so 'learn by
    forecasting' percolates into the belief gauge when present and is simply absent
    otherwise. No behaviour change for engines without a forecaster."""
    fs = getattr(eng, "forecast_surface", None)
    if fs is None:
        return None
    try:
        preds = fs.predictions()  # {(node, role, horizon): {"skill", ...}}
        cands = [(h, d.get("skill")) for (n, r, h), d in preds.items()
                 if n == node and r == role and d.get("skill") is not None]
        if not cands:
            return None
        return round(float(min(cands, key=lambda hs: hs[0])[1]), 4)
    except Exception:
        return None


class GameHost:
    """Thin host face: one world, one mind (use WorldSession for multi-mind)."""

    def __init__(self) -> None:
        self._engine = None
        self._spec = None
        self._t: datetime = _EPOCH
        self._tick_s: float = 1.0
        self._turn: int = 0
        self._world_side_effects: list[dict] = []
        self._live_dispatch_log: list[dict] = []
        self._actor_intents: list[tuple[str, str, datetime]] = []
        self._channel_mask: set[str] | None = None  # None = all channels allowed
        self._observer_id: str = "default"

    # ── registration ──────────────────────────────────────────────────────────

    def register_world(
        self,
        spec,
        *,
        tick_s: float | None = None,
        start: datetime | None = None,
        population: bool = False,
        observer_id: str = "default",
        channel_mask: set[str] | None = None,
        dispatch: Callable | None = None,
    ):
        """Boot a blank engine from a DomainSpec (or module:ATTR string)."""
        self._spec = spec
        self._observer_id = observer_id
        self._channel_mask = set(channel_mask) if channel_mask is not None else None

        def _dispatch(action) -> None:
            rec = {
                "actuator_id": action.actuator_id,
                "command": dict(action.command),
                "node": action.node,
                "role": action.role,
                "reason": action.reason,
            }
            self._live_dispatch_log.append(rec)
            self._world_side_effects.append(rec)
            if dispatch is not None:
                dispatch(action)

        self._engine = build_engine(
            spec=spec, population=population, dispatch=_dispatch
        )
        if tick_s is not None:
            self._tick_s = float(tick_s)
        else:
            drivers = getattr(self._spec, "drivers", ()) or ()
            self._tick_s = float(drivers[0].period_s) if drivers else 1.0
        self._t = start if start is not None else _EPOCH
        self._turn = 0
        return self

    @property
    def engine(self):
        if self._engine is None:
            raise RuntimeError("register_world first")
        return self._engine

    @property
    def spec(self):
        return self._spec

    @property
    def now(self) -> datetime:
        return self._t

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def world_side_effects(self) -> list[dict]:
        return list(self._world_side_effects)

    @property
    def live_dispatches(self) -> list[dict]:
        return list(self._live_dispatch_log)

    # ── observe / intend / beliefs / step ─────────────────────────────────────

    def observe(
        self,
        observer_id: str,
        channel: str,
        value: float,
        confidence: float = 1.0,
        t: datetime | None = None,
    ) -> dict:
        """Ingest one observation. confidence (η) ≤ 0 is a no-op on the field."""
        eng = self.engine
        if t is not None:
            self._t = t
        eta = float(confidence)
        # Per-observer mask: unknown channels are dropped (no field drive).
        if self._channel_mask is not None and channel not in self._channel_mask:
            return {"accepted": False, "reason": "channel_masked", "step": eng._step}
        if eta <= 0.0:
            # Explicit no-op path: do not call ingest with conf=0 via process that
            # might still advance clocks — still allow a pure time tick if needed,
            # but with empty readings so field gets no measurement.
            result = eng.ingest(sensor_readings={}, now=self._t, confidence={})
            result = dict(result)
            result["accepted"] = False
            result["reason"] = "eta_zero"
            return result
        result = eng.ingest(
            sensor_readings={channel: float(value)},
            now=self._t,
            confidence={channel: eta},
        )
        result = dict(result)
        result["accepted"] = True
        return result

    def observe_many(
        self,
        observer_id: str,
        readings: dict[str, float],
        confidence: dict[str, float] | None = None,
        t: datetime | None = None,
    ) -> dict:
        """Batch observe (host happy-path for demos/proofs)."""
        eng = self.engine
        if t is not None:
            self._t = t
        conf = dict(confidence or {})
        filtered: dict[str, float] = {}
        conf_out: dict[str, float] = {}
        for ch, val in readings.items():
            if self._channel_mask is not None and ch not in self._channel_mask:
                continue
            eta = float(conf.get(ch, 1.0))
            if eta <= 0.0:
                continue
            filtered[ch] = float(val)
            conf_out[ch] = eta
        if not filtered:
            result = eng.ingest(sensor_readings={}, now=self._t, confidence={})
            result = dict(result)
            result["accepted"] = False
            result["reason"] = "all_dropped"
            return result
        result = eng.ingest(
            sensor_readings=filtered, now=self._t, confidence=conf_out
        )
        result = dict(result)
        result["accepted"] = True
        return result

    def intend(self, actor_id: str, intent: Intent | str, **payload) -> Decision:
        """Emit an intent → Decision (shadow by default).

        Shadow: recorded, no world side effect via live dispatch.
        Live: only when the matching tendril is non-shadow AND host dispatch fires.
        """
        eng = self.engine
        if isinstance(intent, str):
            intent = Intent(actor_id=actor_id, name=intent, payload=dict(payload))
        else:
            actor_id = intent.actor_id or actor_id

        self._actor_intents.append((actor_id, intent.name, self._t))

        # Tag confounding surface with actor (Phase 3 hygiene).
        try:
            from umwelt.learning.confounding import record_actor_intent

            record_actor_intent(eng, actor_id, intent.name, self._t)
        except Exception:
            pass

        tendrils = list(getattr(eng, "tendrils", None) or [])
        match = next((t for t in tendrils if t.name == intent.name), None)
        if match is None:
            # Ad-hoc intent: shadow recommendation only — no tendril, no world effect.
            return Decision(
                intent_name=intent.name,
                actor_id=actor_id,
                mode="shadow",
                command=dict(intent.payload),
                confidence=0.0,
                dispatched=False,
            )

        # Force a tendril step read.
        action = match.step(
            now_ts=self._t.timestamp()
            if hasattr(self._t, "timestamp")
            else None
        )
        want_live = (not intent.shadow) and (not match.shadow)
        surface = getattr(eng, "output_surface", None)
        dispatched = False
        command = dict(intent.payload)
        conf = 0.0
        node = match.node
        role = match.role
        if action is not None:
            command = dict(action.command)
            conf = float(action.confidence)
            node = action.node
            role = action.role
            if want_live and surface is not None and surface.dispatch is not None:
                # Temporarily treat as live for this route
                was_shadow = match.shadow
                match.shadow = False
                try:
                    n = surface.route([action], tendrils=[match])
                    dispatched = n > 0
                finally:
                    match.shadow = was_shadow
            elif surface is not None:
                surface.recommendations.append(action)
                surface.history.append((self._t.timestamp(), action, False))

        mode = "live" if dispatched else "shadow"
        if want_live and not dispatched and action is not None:
            # Explicit live request but tendril still shadow-gated → stay shadow.
            mode = "shadow"
        return Decision(
            intent_name=intent.name,
            actor_id=actor_id,
            mode=mode,
            command=command,
            confidence=conf,
            dispatched=dispatched,
            node=node,
            role=role,
        )

    def beliefs(
        self,
        observer_id: str | None = None,
        query: str | None = None,
    ) -> dict[str, Belief]:
        """Return calibrated {node.role → Belief(value, confidence)}.

        value = (z+1)/2 ∈ [0,1]; confidence = Bloch radius |r| ∈ [0,1].
        Default face never returns raw Bloch z.
        """
        del observer_id  # single-mind host; multi-mind uses WorldSession
        eng = self.engine
        # latent gauge coordinates, read once per call (both degrade to {} if the
        # engine hasn't accrued them yet — additive, never raises):
        trust_snap = {}
        _ot = getattr(eng, "_obs_trust", None)
        if _ot is not None:
            try:
                trust_snap = _ot.snapshot()  # {"node.role": {"innov_ema","alpha"}}
            except Exception:
                trust_snap = {}
        out: dict[str, Belief] = {}
        for cname, cluster in eng.field.clusters.items():
            roles = getattr(cluster, "role_index", {}) or {}
            for role in roles:
                key = f"{cname}.{role}"
                if query is not None and query not in key and query != cname and query != role:
                    continue
                bloch = cluster.role_bloch(role)
                x, y, z = float(bloch[0]), float(bloch[1]), float(bloch[2])
                value = (z + 1.0) / 2.0
                conf = bloch_radius(x, y, z)
                rel = trust_snap.get(key, {}).get("alpha")
                out[key] = Belief(
                    node=cname, role=role, value=value, confidence=conf,
                    reliability=rel,
                    forecast_skill=_forecast_skill(eng, cname, role),
                )
        return out

    def belief_value(self, node: str, role: str) -> Belief:
        key = f"{node}.{role}"
        b = self.beliefs(query=node).get(key)
        if b is None:
            return Belief(node=node, role=role, value=0.5, confidence=0.0)
        return b

    def step(self, t: datetime | None = None, *, dt_s: float | None = None) -> dict:
        """Advance game time one tick (empty ingest advances field clocks)."""
        eng = self.engine
        if t is not None:
            self._t = t
        else:
            self._t = self._t + timedelta(seconds=dt_s if dt_s is not None else self._tick_s)
        result = eng.ingest(sensor_readings={}, now=self._t, confidence={})
        return dict(result)

    def step_turn(self, n: int = 1) -> list[dict]:
        """Advance n game turns."""
        results = []
        for _ in range(max(0, int(n))):
            self._turn += 1
            results.append(self.step())
        return results

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        self.engine.save(str(path))

    def load(self, path: str | Path) -> None:
        self.engine.load(str(path))

    def field_canon_hash(self) -> str:
        return self.engine.field_canon_hash()

    def snapshot(self) -> dict[str, Any]:
        return {
            "turn": self._turn,
            "t": self._t.isoformat(),
            "tick_s": self._tick_s,
            "observer_id": self._observer_id,
            "hash": self.field_canon_hash(),
            "side_effects": len(self._world_side_effects),
        }
