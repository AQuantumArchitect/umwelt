"""SolarDriver — the smart-home domain's clock: local apparent solar time.

The origin deployment's day is not wall-clock UTC; it is the sun's apparent motion at the
house's longitude (UTC + longitude + equation-of-time corrections), continuous across
midnight and replay windows. This is the canonical example of a domain-registered
PeriodicDriver: the engine ships only the harmonic clock; astronomy lives here.

Register it once at app import:

    from examples.smarthome.solar import register_solar_driver
    register_solar_driver()
    # then in the spec: DriverSpec("sun", type="solar", params={"lon": -97.743})
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from umwelt.substrate.bloch import phase_to_bloch

DAY_MINUTES = 24.0 * 60.0


def _equation_of_time_min(dt: datetime) -> float:
    """Equation-of-time correction in minutes."""
    doy = dt.timetuple().tm_yday
    B = math.radians(360 / 365 * (doy - 81))
    return 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)


def solar_phase_minutes(dt: datetime, lon: float) -> float:
    """Monotonic apparent-solar minutes: UTC minutes + longitude + equation of time.
    Stays continuous across midnight while phase-locking rhythms to the solar day."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.timestamp() / 60.0 + lon * 4.0 + _equation_of_time_min(dt_utc)


class SolarDriver:
    """Local apparent solar time on a Bloch equator — phase 0.5 ≈ solar noon."""

    def __init__(self, name: str = "sun", *, node: str = "_clock",
                 role: str = "local_solar_time", lon: float = 0.0,
                 rest_window: tuple[float, float] | None = (0.45, 0.65)):
        self.name = name
        self.node = node
        self.role = role
        self.lon = float(lon)
        self.period_s = 86400.0
        self.rest_window = rest_window

    def phase(self, now: datetime) -> float:
        return (solar_phase_minutes(now, self.lon) / DAY_MINUTES) % 1.0

    def target_bloch(self, now: datetime) -> tuple[float, float, float]:
        return phase_to_bloch(self.phase(now))


def register_solar_driver() -> None:
    """Make DriverSpec(type="solar") resolvable. params: lon (degrees east)."""
    from umwelt.clocks.drivers import DRIVER_FACTORIES
    if "solar" in DRIVER_FACTORIES:
        return
    DRIVER_FACTORIES["solar"] = lambda spec: SolarDriver(
        spec.name, node=spec.node, role=spec.role,
        rest_window=spec.rest_window or (0.45, 0.65), **(spec.params or {}))
