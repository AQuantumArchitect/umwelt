"""TrustWeb — the per-leaf conditional-trust fuser. One operator that unifies
sensor-health rerouting, forecast ensembling, and brain-chaining (FORESIGHT.md §1)."""
from __future__ import annotations

from umwelt.foresight.trust_web import TrustWeb


# ── day-1: fusion == today's confidence-weighted observation ───────────────

def test_day1_single_source_passes_through():
    # untracked, untrained: one full-confidence source → fuse returns it verbatim
    w = TrustWeb()
    z, conf = w.fuse({"sensor_a": (0.6, 1.0, True)})
    assert z == 0.6
    assert conf == 1.0          # min(1, total_w=1) × agreement(1) = 1


def test_day1_is_confidence_weighted_average():
    # two sources, no learning → confidence-weighted average; small spread → near-confident
    w = TrustWeb()
    z, conf = w.fuse({"a": (0.4, 1.0, True), "b": (0.8, 1.0, True)})
    assert abs(z - 0.6) < 1e-9   # equal weight average
    assert conf > 0.9            # corroborated, mild spread (0.4 vs 0.8) → ~0.96
    # perfectly-agreeing sources → fully confident
    _z, conf_same = w.fuse({"a": (0.6, 1.0, True), "b": (0.6, 1.0, True)})
    assert conf_same == 1.0


def test_all_down_is_a_noop():
    # nothing live → (0, 0): the belief free-evolves (confidence contract preserved)
    w = TrustWeb()
    w.fuse({"a": (0.5, 1.0, True)})            # a becomes "seen"
    z, conf = w.fuse({"a": (0.5, 1.0, False)})  # now a is down, nothing else
    assert (z, conf) == (0.0, 0.0)


def test_zero_confidence_source_contributes_nothing():
    w = TrustWeb()
    z, conf = w.fuse({"a": (0.9, 0.0, True), "b": (0.2, 1.0, True)})
    assert z == 0.2              # a (conf 0) dropped → b only
    assert conf > 0.0


def test_disagreement_lowers_fused_confidence():
    w = TrustWeb()
    _z, conf_agree = w.fuse({"a": (0.5, 1.0, True), "b": (0.5, 1.0, True)})
    _z, conf_disagree = w.fuse({"a": (-0.9, 1.0, True), "b": (0.9, 1.0, True)})
    assert conf_disagree < conf_agree   # honest uncertainty when trusted sources clash


# ── learning: reliability is earned on realized outcomes ───────────────────

def test_reliability_rewards_the_better_predictor():
    w = TrustWeb(lr=0.2)
    # ground truth is always ~ +0.8; A tracks it, B is wrong
    for _ in range(60):
        inp = {"A": (0.8, 1.0, True), "B": (-0.5, 1.0, True)}
        w.fuse(inp)
        w.learn(inp, label_z=0.8)
    assert w.reliability("A") > 0.8
    assert w.reliability("B") < 0.4
    # the fused estimate now leans hard toward the reliable source
    z, _conf = w.fuse({"A": (0.8, 1.0, True), "B": (-0.5, 1.0, True)})
    assert z > 0.4


def test_forecast_ensembling_tracks_the_skilled_brain():
    # three "forecast brains"; only C is consistently right → fusion converges to C
    w = TrustWeb(lr=0.2)
    for _ in range(80):
        inp = {"A": (0.0, 1.0, True), "B": (-0.6, 1.0, True), "C": (0.7, 1.0, True)}
        w.fuse(inp)
        w.learn(inp, label_z=0.7)
    z, _ = w.fuse({"A": (0.0, 1.0, True), "B": (-0.6, 1.0, True), "C": (0.7, 1.0, True)})
    assert abs(z - 0.7) < 0.25
    assert w.reliability("C") > w.reliability("A") > w.reliability("B")


# ── the keystone: health rerouting — "trust A more when B is out" ──────────

def test_compensation_reroutes_trust_when_a_peer_is_down():
    # The conditional structure compensation captures: A is MEDIOCRE in general (noisy
    # when B is present) but RELIABLE specifically when B is out → A should be trusted
    # more in B's absence than its global reliability would suggest.
    w = TrustWeb(lr=0.2)
    truth = 0.8
    for i in range(160):
        if i % 2 == 0:                      # B present, A noisy-wrong
            inp = {"A": (0.1, 1.0, True), "B": (0.8, 1.0, True)}
        else:                               # B down, A carries the signal perfectly
            inp = {"A": (0.8, 1.0, True), "B": (0.8, 1.0, False)}
        w.fuse(inp)
        w.learn(inp, label_z=truth)

    # A's global reliability is dragged down by its noisy live-rounds…
    assert w.reliability("A") < 0.9
    # …but it earns positive compensation for B's absence (better-than-baseline when B out)
    assert w.compensation("A", "B") > 0.1

    # Effect: with B down, A's effective weight is LIFTED above its raw reliability, so the
    # belief stays firmly pinned even though A alone is only mediocre on paper.
    z_down, conf_down = w.fuse({"A": (0.8, 0.5, True), "B": (0.8, 1.0, False)})
    assert z_down == 0.8
    assert conf_down > 0.5 * w.reliability("A")   # the compensation lift is real


def test_dead_source_never_injects_a_phantom():
    # a source marked not-live must contribute nothing even with high stated confidence
    w = TrustWeb()
    z, conf = w.fuse({"ghost": (0.99, 1.0, False), "real": (-0.3, 1.0, True)})
    assert z == -0.3


# ── persistence: the learned web is heritage ───────────────────────────────

def test_snapshot_round_trip():
    w = TrustWeb(lr=0.1)
    for _ in range(20):
        inp = {"A": (0.5, 1.0, True), "B": (0.5, 1.0, False)}
        w.fuse(inp); w.learn(inp, 0.5)
    snap = w.snapshot()
    w2 = TrustWeb()
    w2.load(snap)
    assert w2.reliability("A") == w.reliability("A")
    assert w2.compensation("A", "B") == w.compensation("A", "B")
    assert w2.seen == w.seen
    assert w2.fuse({"A": (0.5, 1.0, True)}) == w.fuse({"A": (0.5, 1.0, True)})
