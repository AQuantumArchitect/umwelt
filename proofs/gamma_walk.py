"""The gamma walk — what decay is FOR, measured per signal class in this repo's gate.

docs/DECAY_NOTES.md carries the second-domain verdict (103 real accumulator series
off a foreign game's LLM-playtester fleet): decay as a VALUE model lost 0/103 to
persistence, while decay as an UNCERTAINTY model (staleness 1−|b| predicting the
coming |error|) won 79/103, with the calibration-optimal γ* at 3e-3–1e-2 /s. That
note's open question is the EVENT class: ±1 signals (sightings, outcomes) may want
the OU/relaxation prior that accumulators reject. This proof walks the same
dissociation on the deterministic synthetic gridworld day — both signal classes, in
the gate, every run: the taxonomy's first measurement living in this repo itself.

Three arms, all off ONE deterministic day (examples/gridworld/world):

    events               sight_<cell> — ±1 sightings (boot report + transition
                         narration + a seeded heartbeat): the event class
    accumulators         resource_<cell> — a continuous level riding the day's
                         ambient drive, at its native REGULAR 600 s cadence
    accumulators-sparse  the same levels through a seeded sparse wire (keep-p
                         thinning, the sparse_reports idiom) — irregular gaps,
                         the shape of the foreign fleet's session tapes

Two scores per arm, across a γ sweep (belief decays toward 0 — "unknown" — between
observations; full α-blend on observation, so γ=0 IS the persistence baseline by
construction):

    VALUE        prequential RMSE of the pre-observation belief against the next
                 observation. If decay helps forecasting, value-γ* > 0.
    CALIBRATION  Pearson corr of pre-observation staleness (1−|b|) with the
                 realized |persistence error|. If decay is a good uncertainty
                 model, calibration-γ* > 0 with the corr peaking there.

MEASURED (this stream, seeded, deterministic — the pins below hold every run):

    accumulators (both arms): value-γ* = 0 — persistence wins the value model and
        EVERY γ > 0 loses (dense: 0.108 at γ=0 vs 0.111+ beyond; sparse: 0.179 vs
        0.198+). The foreign fleet's 0/103 verdict, replicated in the gate.
    accumulators, dense arm: calibration is γ-INVARIANT (+0.096 at every γ that
        leaves staleness varying) — under a constant gap, staleness is an affine
        transform of |last value| and Pearson cannot see γ at all. The honest
        negative: identifying a calibration timescale REQUIRES gap variance; a
        regular-cadence feed has no γ* to find.
    accumulators, sparse arm: with irregular gaps the foreign dissociation appears
        whole — calibration peaks INSIDE the sweep at γ* = 3e-4 /s (corr +0.302 vs
        +0.223 at γ=0), while value-γ* stays 0. γ* sits below the foreign band
        (3e-3–1e-2 /s) because it tracks the gap scale: these gaps are ~2000 s
        where the fleet's were minutes — the calibration timescale is a property
        of the FEED's cadence, not a universal constant.
    events: the OU/relaxation prior EARNS ITS KEEP as a value model — value-γ* =
        1e-4 > 0 beats persistence (0.520 vs 0.532 RMSE, a modest ~2% but the SIGN
        is the finding): a decayed belief hedges the flips persistence bets
        against at full confidence. This completes the taxonomy: events accept
        mean-reversion in the value model, accumulators refuse it at every γ.
        Event calibration is positive at every γ > 0 but weak (≤ +0.03): this
        wire narrates transitions the moment they happen, so staleness carries
        little flip information here — measured and reported, not claimed.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest

from examples.gridworld.world import agent_walk, gridworld_spec, synthesize_rows

DAYS = 3.0
GAMMAS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)   # /s
SPARSE_P = 0.3        # sparse-wire keep probability (the sparse_reports idiom)
SPARSE_SEED = 0


# ── the deterministic day, split into its two signal classes ────────────────────────

def day_series() -> "tuple[dict, dict]":
    """One deterministic synthetic day (same generator the blank-slate proof
    replays), split by signal class: (events, accumulators), each
    {sensor_id: [(t_seconds, value ∈ [-1, 1]), ...]} sorted by time.

    Events are the sight_* wire values (0/1 → ±1). Accumulators are the resource_*
    levels through the spec's OWN declared normalizer geometry (range 0..10 →
    [-1, 1]) — the belief scale a binding would actually put them on."""
    spec = gridworld_spec()
    rows = synthesize_rows(spec, agent_walk(days=DAYS))
    t0 = datetime.fromisoformat(rows[0][0])
    events: dict[str, list] = {}
    accums: dict[str, list] = {}
    for ts, sid, val, _meta in rows:
        t = (datetime.fromisoformat(ts) - t0).total_seconds()
        v = float(val)
        if sid.startswith("sight_"):
            events.setdefault(sid, []).append((t, 2.0 * v - 1.0))
        elif sid.startswith("resource_"):
            accums.setdefault(sid, []).append((t, (v - 5.0) / 5.0))
    return events, accums


def sparse_wire(series: dict, *, keep_p: float = SPARSE_P,
                seed: int = SPARSE_SEED) -> dict:
    """A seeded sparse wire over a dense series: keep each observation with
    probability keep_p — irregular gaps, the sparse reality the foreign session
    tapes actually had (and the same idiom as gridworld's sparse_reports)."""
    rng = np.random.default_rng(seed)
    out: dict[str, list] = {}
    for sid, obs in series.items():
        keep = rng.random(len(obs)) < keep_p
        out[sid] = [o for o, k in zip(obs, keep) if k]
    return out


# ── one prequential pass: value score + calibration score at one γ ──────────────────

def prequential(series: dict, gamma: float) -> dict:
    """Walk every series once, belief decaying toward 0 between observations and
    fully α-blending on each one (α=1, so γ=0 reproduces persistence EXACTLY —
    the baseline lives inside the sweep, not beside it).

    Per observation (after the first of each series):
        value error        (b_pred − v)²  where b_pred = b·e^(−γ·Δt)
        persistence error  (v_prev − v)²
        staleness          1 − |b_pred|   (the uncertainty claim)
    Returns pooled value/persistence RMSE and corr(staleness, |persistence error|).
    """
    val_se: list[float] = []
    pers_se: list[float] = []
    staleness: list[float] = []
    pers_ae: list[float] = []
    for obs in series.values():
        b = t_prev = v_prev = None
        for t, v in obs:
            if b is not None:
                b_pred = b * math.exp(-gamma * (t - t_prev))
                val_se.append((b_pred - v) ** 2)
                pers_se.append((v_prev - v) ** 2)
                staleness.append(1.0 - abs(b_pred))
                pers_ae.append(abs(v_prev - v))
                b = v                                       # α=1 blend
            else:
                b = v                                       # first report seeds
            t_prev, v_prev = t, v
    s, e = np.asarray(staleness), np.asarray(pers_ae)
    corr = 0.0                                              # constant staleness (γ=0
    if s.std() > 1e-12 and e.std() > 1e-12:                 # on ±1 events; γ→∞ on
        corr = float(np.corrcoef(s, e)[0, 1])               # anything) predicts
    return {"gamma": gamma,                                 # nothing — honest zero
            "value_rmse": float(np.sqrt(np.mean(val_se))),
            "persistence_rmse": float(np.sqrt(np.mean(pers_se))),
            "calibration_corr": corr,
            "n": len(val_se)}


def gamma_walk(gammas=GAMMAS) -> dict:
    """The full walk: three arms × the γ sweep. Returns
    {arm: {"sweep": [row, ...], "value_gamma": γ*, "calibration_gamma": γ*}}."""
    events, accums = day_series()
    arms = (("events", events),
            ("accumulators", accums),
            ("accumulators-sparse", sparse_wire(accums)))
    out = {}
    for name, series in arms:
        sweep = [prequential(series, g) for g in gammas]
        out[name] = {
            "sweep": sweep,
            "value_gamma": min(sweep, key=lambda r: r["value_rmse"])["gamma"],
            "calibration_gamma": max(
                sweep, key=lambda r: r["calibration_corr"])["gamma"],
        }
        print(f"\n{name} ({sweep[0]['n']} scored observations)")
        hdr = f"{'gamma/s':>9s} {'VALUE rmse':>11s} {'CALIB corr':>11s}"
        print(hdr + f"   (persistence rmse {sweep[0]['persistence_rmse']:.4f})")
        print("-" * len(hdr))
        for r in sweep:
            print(f"{r['gamma']:>9g} {r['value_rmse']:>11.4f} "
                  f"{r['calibration_corr']:>+11.3f}")
        print(f"value-γ* = {out[name]['value_gamma']:g}, "
              f"calibration-γ* = {out[name]['calibration_gamma']:g}")
    return out


# ── the pytest face: pin the taxonomy's shape, per class ────────────────────────────

@pytest.fixture(scope="module")
def walk():
    return gamma_walk()


def test_gamma_zero_is_exactly_persistence(walk):
    # the baseline lives INSIDE the sweep: at γ=0 the value model IS persistence
    for arm in walk.values():
        g0 = arm["sweep"][0]
        assert g0["gamma"] == 0.0
        assert g0["value_rmse"] == pytest.approx(g0["persistence_rmse"], abs=1e-12)


def test_accumulators_reject_decay_as_a_value_model_on_both_arms(walk):
    """The foreign fleet's 0/103 verdict, replicated in the gate: on the level
    class, decay-toward-unknown loses the value model to persistence at EVERY
    γ > 0 — dense cadence and sparse wire alike."""
    for name in ("accumulators", "accumulators-sparse"):
        arm = walk[name]
        assert arm["value_gamma"] == 0.0, name
        g0 = arm["sweep"][0]
        for r in arm["sweep"][1:]:
            assert r["value_rmse"] > g0["value_rmse"], (name, r)


def test_dense_cadence_cannot_identify_a_calibration_gamma(walk):
    """The honest negative this stream forced: under a CONSTANT gap, staleness is
    an affine transform of |last value|, so the Pearson calibration score is
    γ-invariant — there is no γ* to find. Identifying a calibration timescale
    requires gap variance (which the sparse arm then supplies)."""
    sweep = walk["accumulators"]["sweep"]
    g0 = sweep[0]
    assert g0["calibration_corr"] > 0.05          # the |value|→error signal exists…
    for r in sweep[1:]:
        if r["calibration_corr"] != 0.0:          # …but γ can't move it at all
            assert r["calibration_corr"] == pytest.approx(
                g0["calibration_corr"], abs=0.01), r
    # at the sweep's top γ staleness saturates to a constant → honest zero
    assert sweep[-1]["calibration_corr"] == 0.0


def test_sparse_accumulators_buy_decay_as_uncertainty_only(walk):
    """The foreign dissociation, whole, in the gate: irregular gaps make staleness
    informative — calibration peaks INSIDE the sweep at γ* > 0, clearly above the
    γ=0 corr — while the value model still wants γ = 0 (pinned above)."""
    arm = walk["accumulators-sparse"]
    sweep = arm["sweep"]
    g0 = sweep[0]
    assert arm["calibration_gamma"] > 0.0
    # an interior peak, not a sweep-edge artifact
    assert arm["calibration_gamma"] < GAMMAS[-1]
    best = max(sweep, key=lambda r: r["calibration_corr"])
    assert best["calibration_corr"] > 0.25                       # a real signal
    assert best["calibration_corr"] > g0["calibration_corr"] + 0.05
    # the measured timescale on THIS stream (γ* tracks the ~2000 s gap scale;
    # the foreign band 3e-3..1e-2 /s tracked minutes-scale gaps — a property of
    # the feed's cadence, not a universal constant)
    assert arm["calibration_gamma"] == pytest.approx(3e-4)


def test_events_buy_decay_as_a_value_model_completing_the_taxonomy(walk):
    """The open question DECAY_NOTES seeded, answered on this stream: the ±1 event
    class ACCEPTS the OU/relaxation prior that accumulators reject — a decayed
    belief beats persistence prequentially (γ* = 1e-4, ~2% RMSE: modest, but the
    SIGN is the finding — it hedges the flips persistence bets against at full
    confidence). Event calibration is positive at every γ > 0 but weak (≤ +0.03):
    this wire narrates transitions the moment they happen, so staleness carries
    little flip information — the honest measurement, not a claim. If a future
    stream contradicts the value win, this pin moves to the negative, per the
    repo's culture."""
    arm = walk["events"]
    sweep = arm["sweep"]
    g0 = sweep[0]
    assert arm["value_gamma"] > 0.0                         # decay EARNS value here
    best_v = min(sweep, key=lambda r: r["value_rmse"])
    assert best_v["value_rmse"] < g0["value_rmse"] * 0.99   # a real win, not a tie
    # at γ=0 staleness on ±1 signals is constant → calibration honestly reads 0
    assert g0["calibration_corr"] == 0.0
    for r in sweep[1:]:
        assert r["calibration_corr"] > 0.0, r               # positive throughout…
    best_c = max(sweep, key=lambda r: r["calibration_corr"])
    assert best_c["calibration_corr"] < 0.1                 # …but honestly weak


if __name__ == "__main__":
    gamma_walk()
