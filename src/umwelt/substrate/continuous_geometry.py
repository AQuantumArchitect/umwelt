"""Continuous differential geometry of a process trajectory — the bin-free reformulation.

Everything in the axial program so far reads a process from DISCRETE 10-min bins: ProcessCycle accumulates
per-tick phase increments, directedness is a finite sum over a sliding window. That works, but the bin width is
an arbitrary knob that BIASES the answer — coarser bins average motion away, smoothing out reversals, so a path
looks straighter (more "restful", more "progressing") than it is. The readout shouldn't depend on how finely we
chose to sample.

The differential-geometric view removes the knob. A process is a smooth curve γ(t) on a manifold:
  • the cycle coordinate lives on the circle S¹ (phase θ);
  • the belief qubit lives on the Bloch sphere S² (the fiber);
  • together they form a fiber bundle, and the holonomy of its natural (Berry) connection IS the winding.

The invariants below are functionals of the CURVE, not of its sampling:
  • winding number    W = (1/2π)∮ dθ        — topological; how many loops the process closed;
  • arc length        L = ∮ |dθ|            — total variation of the path (how much it wandered);
  • directedness      D = |∮dθ| / ∮|dθ|     — net displacement over arc length ∈[0,1]; REPARAMETERIZATION-
                                              INVARIANT (depends only on the curve's image, not its timing) —
                                              the continuous limit of process-vs-thrash;
  • Berry holonomy    Γ = ½∮(1−cos ϑ) dφ    — the connection's curvature integral over the Bloch fiber (the
                                              geometric phase a loop accumulates).

These converge as the sampling refines (above Nyquist they stop moving — the mark of a true continuous
quantity). The discrete ProcessCycle is the bin-width discretization of exactly this; continuous_geometry is
the limit it should be read against. Research module — not wired into build_house.
"""
from __future__ import annotations

import numpy as np


def _differential(theta, wrapped: bool = False) -> np.ndarray:
    """dθ, the continuous differential of the phase trajectory.

    The inputs here are CUMULATIVE (already-unwrapped) phase curves — event_driven_phase and the motion→phase
    maps integrate θ continuously, so a long still stretch is a large positive step and must be kept as-is
    (a +2π advance is one full restful loop, NOT zero). So the default is the raw difference.

    `wrapped=True` is for inputs that are genuinely WRAPPED angles in (−π,π] (e.g. a qubit's arg(ρ₀₁) read
    tick-by-tick), where the shortest-path branch is the right reconstruction of dθ between samples."""
    d = np.diff(np.asarray(theta, float))
    return (d + np.pi) % (2 * np.pi) - np.pi if wrapped else d


def path_invariants(theta) -> dict:
    """Continuous geometric invariants of a phase trajectory θ (samples of a curve on S¹)."""
    d = _differential(theta)
    net = float(d.sum())                 # ∮dθ
    arc = float(np.abs(d).sum())         # ∮|dθ|
    return {
        "winding": net / (2 * np.pi),                    # topological winding number
        "arc_length_loops": arc / (2 * np.pi),           # total variation, in loops
        "directedness": abs(net) / arc if arc > 1e-12 else 0.0,   # reparameterization-invariant
    }


def berry_holonomy(theta, polar: float) -> float:
    """Geometric phase Γ = ½∮(1−cos ϑ)dφ for a path that winds in azimuth φ=θ at fixed polar angle ϑ — the
    holonomy of the Bloch-sphere connection (solid angle / 2). With ϑ varying, pass a polar array."""
    d = _differential(theta)
    pol = np.full(len(d), polar) if np.isscalar(polar) else np.asarray(polar, float)[1:]
    return float(0.5 * np.sum((1.0 - np.cos(pol)) * d))


def event_driven_phase(event_times, t0: float, t1: float, *, advance_rate: float, kick: float) -> np.ndarray:
    """Grid-FREE sleep phase from a sparse movement-event stream — the honest continuous flow (no bins, no
    resampling). Between events the subject is still, so the phase advances at `advance_rate` × the gap
    (descending into / dwelling in sleep); each event is a movement that kicks the phase back by `kick` (an
    arousal). A restful night = long still gaps → clean advance; a restless night = dense events → the phase
    wanders. `event_times` are seconds in [t0, t1]; returns the phase sampled at the events + the endpoints."""
    theta, cur, last = [0.0], 0.0, t0
    for t in sorted(event_times):
        cur += advance_rate * (t - last)      # still since the last event → advance through the cycle
        theta.append(cur)
        cur -= kick                           # this movement → an arousal, phase reverses
        theta.append(cur)
        last = t
    cur += advance_rate * (t1 - last)         # the final still stretch to morning
    theta.append(cur)
    return np.array(theta)


def converged(theta_fine, factors=(1, 2, 4, 8, 16)) -> dict:
    """Directedness of the SAME curve subsampled at coarsening factors — the continuous quantity is the value
    these converge to as factor→1; the spread is the bin-width bias. (Coarser sampling chords across wiggles →
    underestimates arc length → OVERestimates directedness: a path looks straighter than it is.)"""
    out = {}
    for f in factors:
        out[f] = round(path_invariants(np.asarray(theta_fine)[::f])["directedness"], 3)
    return out
