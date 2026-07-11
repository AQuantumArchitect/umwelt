"""
Berry Tape — the field's phenomenological record.

The Berry phase is the genuine geometric phase of the quantum state. Each
qubit's Bloch vector traces a path on the sphere under the Lindblad dynamics,
and the exact two-level Berry connection accumulates along it:

    dγ = -½ (1 - cosθ) dφ · |r|

(see BlochGeometricPhase in substrate/params.py). The global phase is the sum
of the per-qubit γ. This is real physics, not a parameter-drift metric: it
returns to a prior value ONLY when the state genuinely retraces a loop, one
equatorial loop gives -π (so a spinor needs 4π to return), and a path enclosing
zero net solid angle returns γ → 0. The phase is a "process clock" — how far
the field has actually moved through state space, independent of wall-clock time.

The Berry Tape records significant events stamped not by when they happened
but by where in this geometric journey they occurred. Two events 6 hours apart
in clock time might be Berry-adjacent if the field was quiet in between.

Three layers:

    1. TICKER — global phase (sum of per-qubit geometric phases) and velocity
    2. STAMPS — significant events stamped with their Berry phase position
    3. RETURNS — detection of recurring topological patterns (same event
       at Berry-adjacent phases = a real geometric return, "I've seen this")

The tape is circular (bounded memory). Old stamps roll off. Returns are
detected by comparing new stamps against the full buffer.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


# ================================================================
# Layer 1: Global Phase Ticker
# ================================================================

class BerryTicker:
    """Collects per-qubit geometric phases into one global scalar.

    The global phase is the sum of the qubits' Bloch-trajectory geometric
    phases (BlochGeometricPhase), across the whole field.

    Velocity (dφ/step) measures how fast the state is currently moving.
    High velocity = the field is wriggling fast = high-information moment.
    Low velocity = a settled world, settled state.
    """

    def __init__(self, velocity_alpha: float = 0.05):
        self.phase: float = 0.0
        self.velocity: float = 0.0         # EMA of phase rate of change
        self._prev_phase: float = 0.0
        self._velocity_alpha = velocity_alpha

    def tick(self, all_phases: list[float]):
        """Called once per engine step with Berry phases from all bundles.

        Args:
            all_phases: list of per-qubit geometric phases (γ) from the field.
        """
        self.phase = sum(all_phases)
        delta = self.phase - self._prev_phase
        self._prev_phase = self.phase

        a = self._velocity_alpha
        self.velocity = a * delta + (1.0 - a) * self.velocity

    @property
    def speed(self) -> float:
        """Absolute velocity — directionless measure of learning rate."""
        return abs(self.velocity)

    def snapshot(self) -> dict:
        return {
            "phase": round(float(self.phase), 6),
            "velocity": round(float(self.velocity), 8),
            "speed": round(float(self.speed), 8),
        }


# ================================================================
# Layer 2: Event Stamps
# ================================================================

@dataclass(frozen=True)
class BerryStamp:
    """One event stamped on the Berry tape.

    Immutable record of something that happened at a particular point
    in the field's learning journey.
    """
    phase: float          # global Berry phase when this event occurred
    event_type: str       # category: "contact", "motion", "collapse", "spike", ...
    source: str           # signal id or node name
    detail: str           # human-readable: "front_door opened", "signal +3.2"
    bloch_z_snap: dict[str, float] = field(default_factory=dict)  # compact field state


class BerryStamper:
    """Records significant events on the Berry tape.

    The tape is a circular buffer (deque). Events are stamped with the
    current global Berry phase, not clock time. The tape preserves
    temporal ordering (events always append) but the spacing between
    stamps reflects learning activity, not elapsed seconds.
    """

    def __init__(self, max_stamps: int = 2000):
        self.tape: deque[BerryStamp] = deque(maxlen=max_stamps)
        self._significance_threshold: float = 0.1  # minimum Bloch-z delta to stamp

    def stamp(
        self,
        phase: float,
        event_type: str,
        source: str,
        detail: str,
        bloch_z_snap: dict[str, float] | None = None,
    ) -> BerryStamp:
        """Record an event on the tape."""
        s = BerryStamp(
            phase=phase,
            event_type=event_type,
            source=source,
            detail=detail,
            bloch_z_snap=bloch_z_snap or {},
        )
        self.tape.append(s)
        return s

    def stamp_sensor_event(
        self,
        phase: float,
        sensor_id: str,
        raw_value: float,
        event_type: str = "sensor",
        bloch_z_snap: dict[str, float] | None = None,
    ) -> BerryStamp | None:
        """Stamp a signal reading if it's significant enough.

        Contact-style signals (0/1) always stamp on transitions.
        Continuous signals stamp if the value is extreme or changed sharply.
        """
        detail = f"{sensor_id}={raw_value}"
        return self.stamp(phase, event_type, sensor_id, detail, bloch_z_snap)

    def stamp_collapse(
        self,
        phase: float,
        node: str,
        role: str,
        old_state: str,
        new_state: str,
        bloch_z_snap: dict[str, float] | None = None,
    ) -> BerryStamp:
        """Stamp a collapse transition — the sky committing to the ground."""
        detail = f"{node}.{role}: {old_state} -> {new_state}"
        return self.stamp(phase, "collapse", node, detail, bloch_z_snap)

    def recent(self, n: int = 20) -> list[BerryStamp]:
        """Last n stamps on the tape."""
        return list(self.tape)[-n:]

    def stamps_in_range(self, lo: float, hi: float) -> list[BerryStamp]:
        """All stamps with phase in [lo, hi]."""
        return [s for s in self.tape if lo <= s.phase <= hi]

    def stamps_by_type(self, event_type: str) -> list[BerryStamp]:
        """All stamps of a given type."""
        return [s for s in self.tape if s.event_type == event_type]

    def snapshot(self, n: int = 50) -> list[dict]:
        """Recent stamps as serializable dicts."""
        return [
            {
                "phase": round(s.phase, 4),
                "type": s.event_type,
                "source": s.source,
                "detail": s.detail,
            }
            for s in list(self.tape)[-n:]
        ]


# ================================================================
# Layer 3: Geometric Return Detection
# ================================================================

@dataclass
class GeometricReturn:
    """A detected recurrence: same event type at Berry-adjacent phases.

    This is the topological fingerprint of a daily rhythm, a habit,
    a recurring pattern — detected without any notion of clock time.
    """
    event_type: str
    source: str
    current_phase: float
    previous_phase: float
    phase_delta: float     # how far apart in Berry-space
    occurrences: int       # how many times this return has been seen
    confidence: float      # how tight the clustering is (0..1)


class ReturnDetector:
    """Detects recurring topological patterns on the Berry tape.

    For each event type+source pair, maintains a history of Berry phases
    where that event occurred. When a new event arrives, checks if any
    previous occurrence is "Berry-close" — meaning the system was in a
    similar state of learning when this happened before.

    The definition of "close" adapts: initially broad (detect anything),
    tightening as more data accumulates and the return period stabilizes.

    This is how the field detects a routine without knowing what a day
    is — it sees that door-open-style events cluster at certain Berry
    phases, and those phases recur with a characteristic spacing.
    """

    def __init__(self, max_history: int = 500):
        # {(event_type, source): [phase1, phase2, ...]}
        self._phase_history: dict[tuple[str, str], deque[float]] = {}
        self._max_history = max_history

        # Detected returns — the recognized rhythms
        self.returns: dict[tuple[str, str], GeometricReturn] = {}

        # Adaptive tolerance: starts broad, tightens as we see more
        self._base_tolerance: float = 5.0  # Berry phase units

    def observe(self, stamp: BerryStamp) -> GeometricReturn | None:
        """Process a new stamp. Returns a GeometricReturn if one is detected."""
        key = (stamp.event_type, stamp.source)
        if key not in self._phase_history:
            self._phase_history[key] = deque(maxlen=self._max_history)

        history = self._phase_history[key]

        # Check for returns: is the current phase close to any previous occurrence?
        best_match: tuple[float, float] | None = None  # (prev_phase, delta)
        for prev_phase in history:
            delta = abs(stamp.phase - prev_phase)
            if delta < 1e-6:
                continue  # skip self (shouldn't happen but guard)
            if best_match is None or delta < best_match[1]:
                best_match = (prev_phase, delta)

        history.append(stamp.phase)

        if best_match is None:
            return None

        prev_phase, delta = best_match

        # Adaptive tolerance based on history depth
        n_events = len(history)
        tolerance = self._base_tolerance / math.sqrt(max(1, n_events - 1))

        if delta > tolerance:
            return None

        # Detected a return! Update or create the return record.
        existing = self.returns.get(key)
        if existing is not None:
            existing.current_phase = stamp.phase
            existing.previous_phase = prev_phase
            existing.phase_delta = delta
            existing.occurrences += 1
            # Confidence tightens as returns cluster
            existing.confidence = min(1.0, existing.occurrences / max(1, n_events * 0.5))
        else:
            existing = GeometricReturn(
                event_type=stamp.event_type,
                source=stamp.source,
                current_phase=stamp.phase,
                previous_phase=prev_phase,
                phase_delta=delta,
                occurrences=1,
                confidence=1.0 / max(1, n_events),
            )
            self.returns[key] = existing

        logger.debug(
            "Geometric return: %s/%s at phase=%.2f (prev=%.2f, delta=%.2f, "
            "occurrences=%d, confidence=%.3f)",
            stamp.event_type, stamp.source,
            stamp.phase, prev_phase, delta,
            existing.occurrences, existing.confidence,
        )
        return existing

    def active_returns(self, min_confidence: float = 0.1) -> list[GeometricReturn]:
        """All detected returns above confidence threshold."""
        return [
            r for r in self.returns.values()
            if r.confidence >= min_confidence
        ]

    def snapshot(self) -> dict:
        returns = self.active_returns()
        return {
            "n_tracked_patterns": len(self._phase_history),
            "n_returns_detected": len(returns),
            "returns": [
                {
                    "event": f"{r.event_type}/{r.source}",
                    "phase_delta": round(r.phase_delta, 4),
                    "occurrences": r.occurrences,
                    "confidence": round(r.confidence, 3),
                }
                for r in sorted(returns, key=lambda r: -r.confidence)[:20]
            ],
        }


# ================================================================
# Unified Berry Tape
# ================================================================

class BerryTape:
    """The complete Berry tape: ticker + stamps + return detection.

    The field's phenomenological record. Everything that matters
    happened somewhere on this tape, ordered by the geometry of
    learning rather than the passage of clock time.

    Usage:
        tape = BerryTape()

        # Each engine step:
        tape.tick(all_berry_phases)

        # When something significant happens:
        tape.stamp_sensor("front_door_contact", 1.0, "contact")
        tape.stamp_collapse("region_a", "occupancy", "empty", "occupied")

        # Query:
        tape.ticker.velocity    # how fast is the field learning right now?
        tape.recent_stamps(10)  # last 10 events on the tape
        tape.returns()          # recognized recurring patterns
    """

    def __init__(self, max_stamps: int = 2000, max_history: int = 500):
        self.ticker = BerryTicker()
        self.stamper = BerryStamper(max_stamps=max_stamps)
        self.detector = ReturnDetector(max_history=max_history)

        # Significance filtering for polling signals.
        # Event-driven signals always stamp; polling signals only stamp
        # when value changes by more than this threshold.
        self._last_stamped: dict[str, float] = {}
        self._env_threshold: float = 0.5  # raw value delta (e.g., 0.5 degrees)

    def tick(self, all_berry_phases: list[float]):
        """Update the global phase ticker. Call once per engine step."""
        self.ticker.tick(all_berry_phases)

    def stamp_sensor(
        self,
        sensor_id: str,
        raw_value: float,
        event_type: str = "sensor",
        event_driven: bool = False,
        bloch_z_snap: dict[str, float] | None = None,
    ) -> GeometricReturn | None:
        """Stamp a signal event and check for geometric returns.

        Event-driven signals stamp on actual value transitions (0↔1).
        Heartbeat republishes (same value as last stamp) are dropped so
        the tape reflects real state changes, not polling cadence.

        Polling signals stamp only when the value changes by more than
        the env threshold.
        """
        prev = self._last_stamped.get(sensor_id)
        if event_driven:
            if prev is not None and raw_value == prev:
                return None  # heartbeat republish, not a transition
        else:
            if prev is not None and abs(raw_value - prev) < self._env_threshold:
                return None  # value hasn't changed enough to be interesting
        self._last_stamped[sensor_id] = raw_value

        stamp = self.stamper.stamp_sensor_event(
            phase=self.ticker.phase,
            sensor_id=sensor_id,
            raw_value=raw_value,
            event_type=event_type,
            bloch_z_snap=bloch_z_snap,
        )
        if stamp is not None:
            return self.detector.observe(stamp)
        return None

    def stamp_collapse(
        self,
        node: str,
        role: str,
        old_state: str,
        new_state: str,
        bloch_z_snap: dict[str, float] | None = None,
    ) -> GeometricReturn | None:
        """Stamp a collapse transition and check for geometric returns."""
        stamp = self.stamper.stamp_collapse(
            phase=self.ticker.phase,
            node=node,
            role=role,
            old_state=old_state,
            new_state=new_state,
            bloch_z_snap=bloch_z_snap,
        )
        return self.detector.observe(stamp)

    def stamp_custom(
        self,
        event_type: str,
        source: str,
        detail: str,
        bloch_z_snap: dict[str, float] | None = None,
    ) -> GeometricReturn | None:
        """Stamp an arbitrary event on the tape."""
        stamp = self.stamper.stamp(
            phase=self.ticker.phase,
            event_type=event_type,
            source=source,
            detail=detail,
            bloch_z_snap=bloch_z_snap,
        )
        return self.detector.observe(stamp)

    def recent_stamps(self, n: int = 20) -> list[BerryStamp]:
        """Last n stamps on the tape."""
        return self.stamper.recent(n)

    def returns(self, min_confidence: float = 0.1) -> list[GeometricReturn]:
        """All recognized recurring patterns."""
        return self.detector.active_returns(min_confidence)

    def snapshot(self) -> dict:
        """Full tape state for API."""
        return {
            "ticker": self.ticker.snapshot(),
            "recent_stamps": self.stamper.snapshot(n=30),
            "geometric_returns": self.detector.snapshot(),
            "tape_length": len(self.stamper.tape),
        }
