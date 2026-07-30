"""ObservationTrust — the learned per-leaf reliability coordinate (was untested).

A sensor whose readings keep DISAGREEING with the settled belief is noisy and should
earn a LOW alpha (distrust); a sensor whose readings sit on the belief earns a HIGH
alpha (trust). This is the engine's native version of the gym's "discrimination":
does a signal actually track reality. These tests pin the monotonicity, clipping, and
the readable snapshot that host.api.beliefs / the hearth /beliefs endpoint now surface.
"""
from umwelt.learning.observation_trust import ObservationTrust


def test_reliable_leaf_earns_higher_alpha_than_noisy():
    ot = ObservationTrust()
    leaf_ok, leaf_noisy = ("a", "level"), ("b", "level")
    # reliable: obs sits on the belief (innovation ~0) many times
    for _ in range(50):
        ot.learned_alpha(leaf_ok, obs_z=0.8, belief_z=0.8)
    # noisy: obs swings far from the belief every time
    for i in range(50):
        ot.learned_alpha(leaf_noisy, obs_z=(1.0 if i % 2 else -1.0), belief_z=0.0)
    a_ok = ot.snapshot()["a.level"]["alpha"]
    a_noisy = ot.snapshot()["b.level"]["alpha"]
    assert a_ok > a_noisy, (a_ok, a_noisy)
    assert a_ok > 0.8 and a_noisy < 0.5, (a_ok, a_noisy)


def test_alpha_is_clipped_to_bounds():
    ot = ObservationTrust(alpha_min=0.10, alpha_max=0.97)
    # perfectly consistent -> alpha saturates at alpha_max, never above
    for _ in range(100):
        a = ot.learned_alpha(("x", "level"), 0.5, 0.5)
    assert a <= 0.97 + 1e-9
    # violently inconsistent -> alpha floors at alpha_min, never below
    for i in range(100):
        a = ot.learned_alpha(("y", "level"), 1.0 if i % 2 else -1.0, 0.0)
    assert a >= 0.10 - 1e-9


def test_snapshot_is_readable_shape():
    ot = ObservationTrust()
    ot.learned_alpha(("n", "r"), 0.3, 0.2)
    snap = ot.snapshot()
    assert "n.r" in snap
    assert set(snap["n.r"]) == {"innov_ema", "alpha"}
    assert 0.0 <= snap["n.r"]["alpha"] <= 1.0


def test_deterministic_given_same_stream():
    a, b = ObservationTrust(), ObservationTrust()
    stream = [(0.1, 0.0), (0.2, 0.1), (-0.3, 0.2), (0.4, -0.1)]
    for (o, be) in stream:
        a.learned_alpha(("k", "level"), o, be)
        b.learned_alpha(("k", "level"), o, be)
    assert a.snapshot() == b.snapshot()
