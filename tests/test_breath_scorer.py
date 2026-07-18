"""Pins the breath scorer's core claim: it rewards quasiperiodic golden-ratio breathing
and penalises a flat (still) tape, ranking golden-ratio cycle pairs above resonant ones."""

from __future__ import annotations

from math import pi, sin

import pytest

from umwelt.clocks.breath_scorer import score_breathing
from umwelt.clocks.phi_clock import PHI


def _tape(fn, n=1600):
    return [fn(t) for t in range(n)]


def _flat(t):
    return {"a": (0.5, 0.5)}


def _single(t, P=120.0):
    return {"a": (0.5 + 0.3 * sin(2 * pi * t / P), 0.5 + 0.3 * sin(2 * pi * t / P + pi / 2))}


def _two_cycle(P1, P2):
    def fn(t):
        return {
            "a": (0.5 + 0.3 * sin(2 * pi * t / P1), 0.5 + 0.3 * sin(2 * pi * t / P1 + pi / 2)),
            "b": (0.5 + 0.3 * sin(2 * pi * t / P2), 0.5 + 0.3 * sin(2 * pi * t / P2 + pi / 2)),
        }
    return fn


def test_flat_tape_scores_zero():
    r = score_breathing(_tape(_flat))
    assert r.flatness > 0.9          # no phase motion
    assert r.return_count == 0        # nothing recurs
    assert r.breathing_score < 0.05   # a still world does not breathe


def test_breathing_beats_flat():
    breathing = score_breathing(_tape(_single)).breathing_score
    flat = score_breathing(_tape(_flat)).breathing_score
    assert breathing > flat
    assert breathing > 0.2


def test_golden_ratio_pair_beats_resonant_pair():
    golden = score_breathing(_tape(_two_cycle(120.0, 120.0 * PHI)))
    resonant = score_breathing(_tape(_two_cycle(120.0, 240.0)))  # 2:1, a brittle resonance
    # The golden-ratio (KAM / maximally non-resonant) pair is the design target.
    assert golden.golden_ratio_score > resonant.golden_ratio_score
    assert golden.breathing_score > resonant.breathing_score
    assert golden.breathing_score > 0.6


def test_golden_beats_a_lone_cycle():
    golden = score_breathing(_tape(_two_cycle(120.0, 120.0 * PHI))).breathing_score
    lone = score_breathing(_tape(_single)).breathing_score
    assert golden > lone   # many cycles at golden ratios > one cycle


def test_scorer_is_pure():
    a = score_breathing(_tape(_two_cycle(120.0, 120.0 * PHI))).breathing_score
    b = score_breathing(_tape(_two_cycle(120.0, 120.0 * PHI))).breathing_score
    assert a == b


if __name__ == "__main__":  # allow `python tests/test_breath_scorer.py`
    raise SystemExit(pytest.main([__file__, "-q"]))
