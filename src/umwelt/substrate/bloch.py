"""bloch — the ATLAS of the belief manifold (dissolution M1, b9.32).

The system's one state geometry is the Bloch ball; every domain that touches it does so
through a chart — a named (embed, read) pair between domain coordinates and the ball.
Before this module each domain hand-rolled its own converter pair in its own file
(qubit_param, solar_clock, earth_gear, dimmer_encoding), and purity was written four
ways. The atlas gathers them: ONE implementation per map, each domain module re-exports
its chart's functions (call sites unchanged — this is pure relocation), and the
round-trip law `read(embed(v)) == v` is pinned by one parametrized test
(tests/brain/test_bloch_atlas.py).

Deliberately NOT merged: the charts themselves. `location_to_bloch` and `phase_to_bloch`
are genuinely different maps — the atlas names that fact instead of hiding it. Same
computation once; different computations visibly different. (`earth_gear.fixes_to_bloch`
— a spherical-centroid AGGREGATION over embeds, not an invertible chart — stays with its
domain, as does diagnostics' stringify-everything serializer.)

Purity, one concept in its call shapes:
  purity_from_rho(ρ)   — exact Tr(ρ²) on a joint density matrix (the full-ρ backend);
  qubit_purity(bloch)  — the single-qubit form (1+|r|²)/2, vectorized over (…, 3) rows
                         (the cumulant e1 stack; the classical z-only limit);
  bloch_radius(x,y,z)  — |r| itself, the confidence coordinate (qubit_param.purity_r).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from umwelt._util import clamp01


# ── the charts ────────────────────────────────────────────────────────────────

def value_to_bloch_z(value: float, lo: float, hi: float) -> float:
    """Linear rescale value∈[lo,hi] → Bloch z∈[-1,+1] (clipped)."""
    if hi <= lo:
        return 0.0
    z = 2.0 * (value - lo) / (hi - lo) - 1.0
    return max(-1.0, min(1.0, z))


def bloch_z_to_value(z: float, lo: float, hi: float) -> float:
    """Inverse of value_to_bloch_z."""
    return lo + (z + 1.0) * (hi - lo) / 2.0


def phase_to_bloch(phase: float) -> tuple[float, float, float]:
    """Map day phase ∈ [0,1) to an equatorial Bloch vector."""
    theta = 2.0 * math.pi * (float(phase) % 1.0)
    return (math.cos(theta), math.sin(theta), 0.0)


def bloch_to_phase(x: float, y: float) -> float:
    """Map an equatorial Bloch vector back to day phase in [0, 1)."""
    return (math.atan2(float(y), float(x)) / (2.0 * math.pi)) % 1.0


def location_to_bloch(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    """A point on Earth's surface → its unit Bloch vector. latitude → z (pole vs equator),
    longitude → azimuth on the x-y plane. |r| = 1 — a pure, fully-known location."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    return (math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat))


def bloch_to_location(x: float, y: float, z: float) -> tuple[float, float]:
    """Inverse: Bloch vector → (lat, lon) in degrees. Robust to a sub-unit radius (a
    not-yet-pure / learning gear) by normalizing onto the sphere first."""
    r = math.sqrt(x * x + y * y + z * z) or 1.0
    lat = math.degrees(math.asin(max(-1.0, min(1.0, z / r))))
    lon = math.degrees(math.atan2(y, x))
    return (lat, lon)


def preference_to_bloch(brightness: float, color_temp: float) -> tuple[float, float, float]:
    """(brightness, color_temp) ∈ [0,1]² → Bloch (x, y, z) target."""
    b = clamp01(float(brightness))
    t = clamp01(float(color_temp))
    z = 1.0 - 2.0 * b               # z = -1 full bright, +1 off
    x = 2.0 * t - 1.0               # x = -1 warm, +1 cool
    return (x, 0.0, z)


def bloch_to_preference(x: float, y: float, z: float) -> tuple[float, float]:
    """Bloch (x, y, z) → (brightness, color_temp) ∈ [0,1]². Ignores y."""
    brightness = (1.0 - float(z)) / 2.0
    color_temp = (float(x) + 1.0) / 2.0
    return (clamp01(brightness), clamp01(color_temp))


@dataclass(frozen=True)
class Chart:
    """A named (embed, read) pair onto the Bloch ball. `embed` takes domain coordinates
    to the ball; `read` inverts it (up to the chart's stated information loss — e.g. the
    preference chart ignores y). The atlas below is the complete census of where the
    manifold touches the world; a sensor binding is a chart plus a measurement model."""
    name: str
    embed: Callable
    read: Callable


SCALAR_Z = Chart("scalar_z", value_to_bloch_z, bloch_z_to_value)
DAY_PHASE = Chart("day_phase", phase_to_bloch, bloch_to_phase)
EARTH_SPHERE = Chart("earth_sphere", location_to_bloch, bloch_to_location)
LIGHT_PREFERENCE = Chart("light_preference", preference_to_bloch, bloch_to_preference)

ATLAS: tuple[Chart, ...] = (SCALAR_Z, DAY_PHASE, EARTH_SPHERE, LIGHT_PREFERENCE)


# ── purity, one concept ───────────────────────────────────────────────────────

def bloch_radius(x: float, y: float, z: float) -> float:
    """|r| ∈ [0,1] — the confidence coordinate: 1=pure (certain), 0=maximally mixed."""
    return math.sqrt(x * x + y * y + z * z)


def qubit_purity(bloch) -> np.ndarray | float:
    """Single-qubit purity Tr(ρ²) = (1+|r|²)/2 from Bloch coordinates. Vectorized:
    accepts one (3,) vector or a stack (…, 3) of them (e.g. a cumulant e1 matrix) and
    returns matching shape. The classical z-only limit is the same formula at x=y=0."""
    arr = np.asarray(bloch, dtype=float)
    out = 0.5 * (1.0 + np.sum(np.square(arr), axis=-1))
    return float(out) if out.ndim == 0 else out


def purity_from_rho(rho: np.ndarray) -> float:
    """Exact Tr(ρ²) on a joint density matrix — 1.0 pure, 1/dim maximally mixed."""
    return float(np.real(np.trace(rho @ rho)))
