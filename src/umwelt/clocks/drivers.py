"""Periodic drivers — the domain's clocks, as pluggable ports (blocker 1 of the extraction).

A PeriodicDriver is anything with a deterministic phase the field should comprehend: a
day, an exchange session, a game tick, an ephemeris. Each tick the engine partially
collapses the driver's anchor qubit toward `target_bloch(now)` with strength
`driver_alpha × (learned per-driver trust)` — the phase is fixed physics; its
COMPREHENSION is learned (the field learns to anticipate the cycle, and the anchor
calibrates down as anticipation skill rises).

The engine ships one driver type: `harmonic` — a pure sinusoid clock of a given period,
no astronomy. Domains register their own via `register_driver` (the origin deployment's
sky-ephemeris driver lives in its example, not here). `DriverSpec.type` resolves
against this registry at boot.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Callable, Protocol, runtime_checkable

from umwelt.substrate.bloch import phase_to_bloch


@runtime_checkable
class PeriodicDriver(Protocol):
    """The port every clock implements. `node`/`role` name the anchor qubit; `phase(now)`
    ∈ [0,1); `target_bloch(now)` is the Bloch point the anchor collapses toward;
    `rest_window` is the per-cycle quiet band (phase fractions) for stroboscopic
    sampling, or None for the engine default."""
    name: str
    node: str
    role: str
    period_s: float
    rest_window: tuple[float, float] | None

    def phase(self, now: datetime) -> float: ...
    def target_bloch(self, now: datetime) -> tuple[float, float, float]: ...


class HarmonicDriver:
    """A pure sinusoid clock: phase = elapsed/period mod 1, anchored at a fixed epoch so
    the phase is deterministic across restarts. The engine default (period = 86400 s)."""

    def __init__(self, name: str = "harmonic_day", *, node: str = "_clock",
                 role: str = "phase", period_s: float = 86400.0,
                 phase_at_epoch: float = 0.0,
                 rest_window: tuple[float, float] | None = None):
        self.name = name
        self.node = node
        self.role = role
        self.period_s = float(period_s)
        self.phase_at_epoch = float(phase_at_epoch)
        self.rest_window = rest_window
        self._epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)

    def phase(self, now: datetime) -> float:
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        elapsed = (now - self._epoch).total_seconds()
        return (self.phase_at_epoch + elapsed / self.period_s) % 1.0

    def target_bloch(self, now: datetime) -> tuple[float, float, float]:
        return phase_to_bloch(self.phase(now))


# ── the driver registry: DriverSpec.type → factory ──────────────────────────────────
DRIVER_FACTORIES: dict[str, Callable] = {
    "harmonic": lambda spec: HarmonicDriver(
        spec.name, node=spec.node, role=spec.role, period_s=spec.period_s,
        rest_window=spec.rest_window, **(spec.params or {})),
}


def register_driver(type_name: str, factory: Callable) -> None:
    """Register a domain driver type: factory(DriverSpec) -> PeriodicDriver."""
    if type_name in DRIVER_FACTORIES:
        raise ValueError(f"driver type {type_name!r} already registered")
    DRIVER_FACTORIES[type_name] = factory


def build_driver(spec) -> PeriodicDriver:
    """Resolve a DriverSpec to a live driver. Unknown types raise — a spec that names a
    clock we can't build should fail loudly."""
    factory = DRIVER_FACTORIES.get(spec.type)
    if factory is None:
        raise ValueError(f"unknown driver type {spec.type!r}; known: {sorted(DRIVER_FACTORIES)}")
    return factory(spec)


def rest_window_of(driver, default: tuple[float, float] = (0.45, 0.65)) -> tuple[float, float]:
    """A driver's rest window, falling back to the engine default phase band."""
    w = getattr(driver, "rest_window", None)
    return tuple(w) if w else default


def anticipation_phase_error(predicted: float, actual: float) -> float:
    """Circular phase error ∈ [0, 0.5] — the anticipation-skill metric for a driver."""
    d = abs((predicted - actual) % 1.0)
    return min(d, 1.0 - d)


def phase_rate(period_s: float, dt_s: float) -> float:
    """Phase advanced in dt_s — the free-evolution rate the H-tower should learn to carry."""
    return (dt_s / period_s) % 1.0


def phase_angle(phase: float) -> float:
    """Phase fraction → radians on the Bloch equator."""
    return 2.0 * math.pi * (phase % 1.0)
