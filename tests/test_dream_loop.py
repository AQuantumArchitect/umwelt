"""Guards the dream-loop seam (meerkat/brain/dream_loop.py) — how dreaming wires into the hindbrain.

b9.1 ships it WIRED BUT INERT (UMWELT_DREAM off): the gate must short-circuit before any clone/replay
cost, and the (possibly expensive) cassette build must be skipped on the common path. The rest gate
(should_dream) is tested in test_dreaming; here we guard the integration's gating + cheapness.
"""
from __future__ import annotations

from umwelt.foresight import dream_loop


def test_disabled_by_default_returns_none_without_building_cassette():
    calls = {"n": 0}

    def cassette_fn():
        calls["n"] += 1
        return [({"x": 1.0}, 0)]

    # default (UMWELT_DREAM unset) → disabled
    assert dream_loop.maybe_dream(None, cassette_fn, solar_phase=0.55) is None
    assert calls["n"] == 0                       # cassette never built — zero cost on the common path


def test_enabled_but_not_resting_skips_without_cassette(monkeypatch):
    monkeypatch.setattr(dream_loop, "DREAM_ENABLED", True)
    calls = {"n": 0}

    def cassette_fn():
        calls["n"] += 1
        return [({"x": 1.0}, 0)]

    # awake (dawn phase, no low-surprise signal) → not resting → no dream, no cassette build
    assert dream_loop.maybe_dream(None, cassette_fn, solar_phase=0.05, live_surprise=0.9) is None
    assert calls["n"] == 0


def test_enabled_and_resting_but_empty_cassette_returns_none(monkeypatch):
    monkeypatch.setattr(dream_loop, "DREAM_ENABLED", True)
    # resting (siesta) but no events to dream on → None, never reaches the engine
    assert dream_loop.maybe_dream(None, lambda: [], solar_phase=0.55) is None


def test_cooldown_blocks_a_second_dream_in_the_window(monkeypatch):
    """b9.7 graduation: after a session, the cooldown holds off the next one — a rest window dreams ONCE,
    not every idle cycle."""
    monkeypatch.setattr(dream_loop, "DREAM_ENABLED", True)
    monkeypatch.setattr(dream_loop, "DREAM_COOLDOWN_S", 3600.0)
    monkeypatch.setattr(dream_loop, "_last_dream", {"t": 100.0})   # a session ran at t=100
    calls = {"n": 0}

    def cassette_fn():
        calls["n"] += 1
        return [({"x": 1.0}, 0)]

    # t=200 is within the 3600s cooldown → no dream, cassette never built (even though resting)
    assert dream_loop.maybe_dream(None, cassette_fn, solar_phase=0.55, now=200.0) is None
    assert calls["n"] == 0


def test_status_reports_the_flags():
    s = dream_loop.status()
    assert set(s) == {"enabled", "consolidate", "n_dreams", "surprise_rest", "cooldown_s",
                      "topology", "topology_lookback_days"}
    assert s["enabled"] is False                 # default-off floor
    assert s["topology"] is False                # topology sub-organ default-off too
