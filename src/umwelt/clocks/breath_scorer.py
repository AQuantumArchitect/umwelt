"""Breath scorer: does a set of driven cycles *breathe* quasiperiodically?

Given a timeseries of per-key ``(fill, phase)`` observations — e.g. an ecology's
per-caste (population fill, metabolic phase), or any collection of driven nodes —
this lifts each key onto the Bloch sphere (``theta = pi * fill``, ``phi = 2*pi*phase``),
accumulates its Berry geometric phase with :class:`BlochGeometricPhase`, and drives a
:class:`BerryTape`. A genuine limit cycle traces a closed Bloch loop, so its accumulated
phase *returns* to a prior value — the tape's :class:`GeometricReturn` is that retrace,
detected with no notion of clock time.

The score rewards three things and penalises their opposites:

* **RETURNS** — the process clock comes back to a prior phase (quasiperiodic breathing),
  not drifting forever (runaway) and not sitting still (flat).
* **Multiplicity** — more than one distinct cycle in the phase-velocity spectrum
  ("many stable cycles of different varieties").
* **Golden-ratio structure** — the dominant periods sit at ``phi**k`` relationships
  (a maximally non-resonant / KAM-torus target: cycles that never phase-lock into a
  brittle resonance), tested against :data:`umwelt.clocks.phi_clock.PHI`.

This is deliberately generic (no ecology imports) so it can score any driven-node
cluster. numpy is used only for the spectral (FFT) step, matching the rest of the repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sin, log
from typing import Iterable, Mapping, Sequence

import numpy as np

from umwelt.clocks.berry_tape import BerryTape, GeometricReturn
from umwelt.clocks.phi_clock import PHI
from umwelt.substrate.params import BlochGeometricPhase

_LOG_PHI = log(PHI)


@dataclass
class BreathReport:
    """The verdict on one run of driven cycles."""

    steps: int
    return_count: int = 0
    returns: list[dict] = field(default_factory=list)  # summarised GeometricReturns
    dominant_periods: list[float] = field(default_factory=list)  # in steps, strongest first
    peak_powers: list[float] = field(default_factory=list)  # spectral power of each dominant period
    golden_ratio_score: float = 0.0  # 0..1 — how phi-related the dominant periods are
    multiplicity: float = 0.0  # 0..1 — 1 peak -> 0, 2 -> 0.5, 3+ -> 1
    flatness: float = 0.0  # 0..1 — fraction of steps with ~no phase motion
    breathing_score: float = 0.0  # 0..1 composite

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "return_count": self.return_count,
            "dominant_periods": [round(p, 3) for p in self.dominant_periods],
            "golden_ratio_score": round(self.golden_ratio_score, 4),
            "multiplicity": round(self.multiplicity, 4),
            "flatness": round(self.flatness, 4),
            "breathing_score": round(self.breathing_score, 4),
        }


def _bloch(fill: float, phase: float, purity: float) -> list[float]:
    """Map (fill, phase) in [0,1] to a Bloch vector; purity<1 damps the step (mixed state)."""
    theta = pi * min(1.0, max(0.0, fill))
    phi = 2.0 * pi * phase
    r = min(1.0, max(0.0, purity))
    st = sin(theta)
    return [r * st * cos(phi), r * st * sin(phi), r * cos(theta)]


def _golden_ratio_score(periods: Sequence[float], powers: Sequence[float], *, tol: float = 0.12) -> float:
    """Power-weighted fraction of dominant-period *pairs* whose ratio is a power of phi.

    A pair ``(p_hi, p_lo)`` scores when ``log(p_hi/p_lo)`` is within ``tol`` (in units of
    ``log phi``) of an integer ``k >= 1`` — i.e. the periods are at ``phi, phi**2, ...``.
    Needs at least two peaks (a lone cycle has no ratio to be golden with).
    """
    n = len(periods)
    if n < 2:
        return 0.0
    num = 0.0
    den = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            hi, lo = max(periods[i], periods[j]), min(periods[i], periods[j])
            if lo <= 0:
                continue
            w = powers[i] * powers[j]
            den += w
            rungs = log(hi / lo) / _LOG_PHI  # how many phi-rungs apart
            k = round(rungs)
            if k >= 1 and abs(rungs - k) <= tol:
                num += w
    return float(num / den) if den > 0 else 0.0


def _cluster_periods(items: Sequence[tuple[float, float]], *, rel_tol: float = 0.06):
    """Merge (period, power) pairs whose periods are within rel_tol into distinct lines (power-summed)."""
    clusters: list[list[float]] = []  # [period_weighted_sum, power_sum]
    for period, power in sorted(items):
        placed = False
        for c in clusters:
            centre = c[0] / c[1] if c[1] > 0 else period
            if abs(period - centre) <= rel_tol * centre:
                c[0] += period * power
                c[1] += power
                placed = True
                break
        if not placed:
            clusters.append([period * power, power])
    out = [(c[0] / c[1] if c[1] > 0 else 0.0, c[1]) for c in clusters]
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def _spectrum(velocity: np.ndarray, *, max_peaks: int = 4, rel_floor: float = 0.25):
    """Return (periods, powers) of up to max_peaks dominant spectral lines of the velocity."""
    v = velocity - velocity.mean()
    if v.size < 8 or not np.any(np.abs(v) > 1e-12):
        return [], []
    spec = np.abs(np.fft.rfft(v))
    freqs = np.fft.rfftfreq(v.size, d=1.0)  # cycles per step
    spec[0] = 0.0  # drop DC
    if spec.max() <= 0:
        return [], []
    # local maxima above a relative floor
    floor = spec.max() * rel_floor
    peaks = []
    for i in range(1, spec.size - 1):
        if spec[i] >= floor and spec[i] >= spec[i - 1] and spec[i] >= spec[i + 1] and freqs[i] > 0:
            peaks.append((spec[i], 1.0 / freqs[i]))
    # the last bin can be a peak too
    if spec.size >= 2 and spec[-1] >= floor and spec[-1] >= spec[-2] and freqs[-1] > 0:
        peaks.append((spec[-1], 1.0 / freqs[-1]))
    peaks.sort(reverse=True)
    peaks = peaks[:max_peaks]
    powers = [float(p) for p, _ in peaks]
    periods = [float(per) for _, per in peaks]
    return periods, powers


def score_breathing(
    rows: Iterable[Mapping[str, Sequence[float]]],
    *,
    velocity_alpha: float = 0.05,
    default_purity: float = 1.0,
    flat_eps: float = 1e-4,
    min_confidence: float = 0.1,
) -> BreathReport:
    """Score a timeseries of driven cycles for quasiperiodic golden-ratio breathing.

    ``rows`` is an iterable of maps ``{key: (fill, phase)}`` (or ``(fill, phase, purity)``),
    one map per timestep, with ``fill`` and ``phase`` in ``[0, 1]``. Returns a
    :class:`BreathReport`.
    """
    bgp = BlochGeometricPhase()
    tape = BerryTape()
    # BerryTape() builds its own BerryTicker (default EMA alpha); set the smoothing directly.
    tape.ticker._velocity_alpha = velocity_alpha

    velocities: list[float] = []
    phase_hist: dict[str, list[float]] = {}  # per-key accumulated γ over time (for per-cycle spectra)
    flat_steps = 0
    steps = 0

    for row in rows:
        steps += 1
        for key, vals in row.items():
            fill, phase = float(vals[0]), float(vals[1])
            purity = float(vals[2]) if len(vals) > 2 else default_purity
            bgp.update(key, _bloch(fill, phase, purity))
        for key, gamma in bgp.phases.items():
            phase_hist.setdefault(key, []).append(gamma)
        tape.tick(list(bgp.phases.values()))
        velocities.append(tape.ticker.velocity)
        if abs(tape.ticker.velocity) < flat_eps:
            flat_steps += 1
        tape.stamp_custom("breath", "cluster", detail=f"t={steps}")

    report = BreathReport(steps=steps)
    if steps == 0:
        return report

    report.flatness = flat_steps / steps
    active = tape.returns(min_confidence=min_confidence)
    report.return_count = len(active)
    report.returns = [
        {
            "phase_delta": round(r.phase_delta, 6),
            "occurrences": r.occurrences,
            "confidence": round(r.confidence, 4),
        }
        for r in active
    ]

    # Multi-cycle spectrum: take EACH key's own dominant γ-velocity period, then cluster across keys so
    # castes sharing a period reinforce one line and distinct periods stay distinct. This surfaces "many
    # cycles of different varieties" that a single global-sum FFT would hide behind the loudest shared period.
    key_lines: list[tuple[float, float]] = []
    for series in phase_hist.values():
        arr = np.asarray(series, dtype=float)
        if arr.size < 3:
            continue
        vel = np.diff(arr)
        periods_k, powers_k = _spectrum(vel, max_peaks=1)
        if periods_k:
            key_lines.append((periods_k[0], powers_k[0]))
    distinct = _cluster_periods(key_lines)
    report.dominant_periods = [p for p, _ in distinct]
    report.peak_powers = [w for _, w in distinct]
    report.golden_ratio_score = _golden_ratio_score(report.dominant_periods, report.peak_powers)
    n_peaks = len(distinct)
    report.multiplicity = min(1.0, max(0.0, (n_peaks - 1) / 2.0))

    has_returns = 1.0 if report.return_count > 0 else 0.0
    report.breathing_score = (1.0 - report.flatness) * (
        0.30 * has_returns + 0.30 * report.multiplicity + 0.40 * report.golden_ratio_score
    )
    return report
