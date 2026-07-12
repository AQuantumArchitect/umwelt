"""Trust-web source isolation — the leave-one-out learning pin.

The first foreign world measured the consensus-label failure: with two co-equal
feeds, a mid-run corruption lowered BOTH reliabilities (blame spread through the
contaminated label). Leave-one-out labels fix what is fixable:

  • ≥3 sources: the corrupted feed isolates — its reliability falls decisively
    below its healthy peers', who keep scoring against each other;
  • exactly 2 sources: genuinely symmetric (no referee exists) — both fall together,
    and the actionable signal is the fused confidence falling with the disagreement.
"""
from __future__ import annotations

import random

from umwelt.foresight.trust_web import TrustWeb


def _stream(n: int, seed: int = 3):
    """A latent ±1 regime with three noisy views of it."""
    rng = random.Random(seed)
    sign = 1.0
    for i in range(n):
        if i % 25 == 0:
            sign = -sign
        yield sign, rng


def _drive(web: TrustWeb, n: int, corrupt: str | None, corrupt_from: int,
           sources=("a", "b", "c")) -> None:
    for i, (sign, rng) in enumerate(_stream(n)):
        inputs = {}
        for s in sources:
            z = max(-1.0, min(1.0, sign * 0.8 + rng.gauss(0.0, 0.15)))
            if s == corrupt and i >= corrupt_from:
                z = -z
            inputs[s] = (z, 0.9, True)
        web.fuse(inputs)
        if len(inputs) >= 2:                    # mirrors the _fuse_leaves wiring
            labels = web.loo_labels(inputs)
            if labels:
                web.learn(inputs, labels)


def test_three_sources_isolate_the_corrupted_feed():
    web = TrustWeb()
    _drive(web, 200, corrupt="b", corrupt_from=100)
    r = {s: web.reliability(s) for s in ("a", "b", "c")}
    assert r["b"] < 0.45, f"corrupted feed not distrusted: {r}"
    assert r["a"] > r["b"] + 0.3 and r["c"] > r["b"] + 0.3, (
        f"healthy peers did not stay clearly above the corrupted feed: {r}")


def test_three_sources_healthy_run_all_stay_trusted():
    web = TrustWeb()
    _drive(web, 200, corrupt=None, corrupt_from=0)
    r = {s: web.reliability(s) for s in ("a", "b", "c")}
    assert all(v > 0.6 for v in r.values()), f"healthy feeds lost trust: {r}"


def test_two_sources_fall_together_and_confidence_says_so():
    """The 2-source symmetric limit, stated as a feature: blame cannot be assigned
    without a referee, but the leaf's fused confidence falls with the disagreement —
    the honest downstream signal."""
    web = TrustWeb()
    _drive(web, 200, corrupt="b", corrupt_from=100, sources=("a", "b"))
    r_a, r_b = web.reliability("a"), web.reliability("b")
    assert r_a < 0.7 and r_b < 0.7, f"disagreement must cost both: {r_a:.2f} {r_b:.2f}"
    assert abs(r_a - r_b) < 0.25, "no referee exists — blame should stay near-symmetric"
    # the actionable signal: fused confidence collapses under live disagreement
    _, conf_disagree = web.fuse({"a": (0.8, 0.9, True), "b": (-0.8, 0.9, True)})
    _, conf_agree = web.fuse({"a": (0.8, 0.9, True), "b": (0.8, 0.9, True)})
    assert conf_disagree < 0.5 * conf_agree


def test_loo_labels_exclude_self():
    web = TrustWeb()
    inputs = {"a": (0.9, 0.9, True), "b": (-0.6, 0.9, True), "c": (0.9, 0.9, True)}
    labels = web.loo_labels(inputs)
    # b's label comes from a+c only: strongly positive, untouched by b's own dissent
    assert labels["b"] > 0.8
    # a's label includes b's dissent: pulled toward the middle
    assert labels["a"] < labels["b"]
    # peers in flat contradiction mint NO label (min_conf) — a lone-source input none
    assert "x" not in web.loo_labels({"x": (0.9, 0.9, True)})