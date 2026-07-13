"""The estimator-ladder walk — persistence vs α-blend vs Belavkin on sparse reports.

Ported from the origin deployment's promotion experiment (meerkat
experiments/ladder_walk.py; docs/THEORY.md §4): same substrate (one CumulantCluster),
same dynamics, same event stream — only the MEASUREMENT RULE changes, and the belief
sees only the bins where a node actually REPORTED (the sparse reality of the wire),
carrying the gaps on dynamics + correlations. The rungs:

    persistence   hold the last report (the floor)
    L0  blend     observe_qubit: α-blend toward the report; correlations DECORRELATE
    L4  belavkin  measure_qubit: log-odds shift, Wonham gain, and the cumulant
                  CROSS-UPDATE — one node's report moves the correlated peers
    L4η belavkin  + the learned per-leaf strength (ObservationTrust innovation-EMA —
                  the online η estimator; UMWELT_LEARN_COLLAPSE's machinery)

× a coupling axis: nodes independent vs exchange-coupled on the adjacency pairs —
exchange (σxσx+σyσy) actually transfers z between nodes, which ZZ alone cannot (it
conserves populations). The target is the NEXT bin's truth (one-step-ahead nowcast);
each rule's strength dial is calibrated on the TRAIN split and scored on the held-out
last 30%, three slices: ALL / GAP (no report this bin) / TRANSITION (the truth flips
next bin). Lower RMSE is better; the floor to beat is persistence.

PORTING DECISION: the origin harness also loaded its recorded 24-day cassette
(load_reports / a forward-filled truth PSV); that tape format and the tape itself stay
with the origin. Here the stream is the deterministic synthetic gridworld walk from
examples/gridworld/world.py — the machinery is identical, the data is not, and no number
measured here speaks for the origin's home.

VERDICT OF RECORD (the origin deployment, 2026-07-04 — real 24-day cassette, 6908
five-minute bins, held-out last 2073; NOT re-measurable from this repo's synthetic
stream and quoted here as the decision it produced):

    contender    couplings   ALL      GAP      TRANSITION
    persistence  independent 0.1346   0.1210   0.8359
    l0           independent 0.1349   0.1215   0.8292
    l4           independent 0.3034   0.2997   0.9000
    l0           coupled     0.5226   0.5613   0.7399   ← transition winner

    → the real home was persistence-dominated: nothing beat hold-the-last-report
      overall, the α-blend tied it, and the full Belavkin filter LOST — the flag was
      DENIED by its own experiment and ships default-OFF (UMWELT_BELAVKIN=0). On a
      transition-dominated synthetic walk the rung's upside exists (the origin
      measured L4η-coupled 1.365 vs persistence 1.987 at transitions) — the regime,
      not the law, decides.

What THIS file pins, on the synthetic gridworld stream: the whole ladder runs and
scores; the rungs are genuinely different estimators; and the transition/gap slices
behave like slices should. It decides nothing about any deployment — it keeps the
deciding machinery alive and honest.
"""
from __future__ import annotations

import numpy as np
import pytest

from examples.gridworld.world import agent_walk, binned_truth, grid_adjacency, grid_cells, sparse_reports
from umwelt.learning.observation_trust import ObservationTrust
from umwelt.substrate.cumulant_cluster import CumulantCluster

# the ladder's world: the 2×2 gridworld (4 nodes, 4 adjacency pairs)
NODES = grid_cells(2, 2)
PAIRS = grid_adjacency(2, 2)
STRENGTHS = (0.3, 0.6, 1.2, 2.4)   # the dial sweep; each rule keeps its train-split best
J_EX = 0.8           # exchange coupling on the adjacency pairs (couplings-ON arm)
DT_SCALE = 4.0       # evolution per bin
BIN_S = 300.0
Z_CAP = 0.98         # the pole floor (purity floor): repeated one-sided evidence drives the
                     # log-odds unbounded, the Wonham gain (1−z²)→0 and the belief goes DEAF
                     # at the pole (and the cross-update's 1/v gain explodes). THEORY.md §6
                     # names this pathology + this mitigation. Applied to every contender
                     # uniformly (harmless for the blend, essential for Belavkin).


def gridworld_stream(*, days: float = 3.0, mean_dwell_s: float = 1200.0,
                     report_p: float = 0.35, walk_seed: int = 11, report_seed: int = 0):
    """The deterministic synthetic stream: one agent walks the 2×2 grid; each node
    reports its ±1 state sparsely. Returns (reports, truth), both (T, len(NODES))."""
    walk = agent_walk(2, 2, seed=walk_seed, days=days, mean_dwell_s=mean_dwell_s)
    truth = binned_truth(walk, NODES, bin_s=BIN_S)
    return sparse_reports(truth, report_p=report_p, seed=report_seed), truth


def _cluster(coupled: bool) -> CumulantCluster:
    c = CumulantCluster("walk", list(NODES), gamma=0.0, dt=0.01,
                        role_modes={r: "unitary" for r in NODES},
                        connectivity=[(a, b) for a, b in PAIRS])
    if coupled:
        xy = {}
        for a, b in PAIRS:
            xy[(NODES.index(a), NODES.index(b))] = (J_EX, J_EX)   # exchange: z transfers
        c.set_couplings(xy=xy)
    c.e1[:, 2] = -1.0                                             # start: every node vacant
    c._sync_e2_product()
    return c


def _run(rule: str, reports: np.ndarray, coupled: bool, strength: float) -> np.ndarray:
    """Drive one contender over the tape; return its per-bin belief-z estimates (T, R)."""
    T, R = reports.shape
    c = _cluster(coupled)
    trust = ObservationTrust() if rule == "l4eta" else None
    est = np.zeros((T, R))
    for t in range(T):
        c.step(dt_scale=DT_SCALE)
        c.clamp_physical()
        for k in range(R):
            z = reports[t, k]
            if np.isnan(z):
                continue
            # the pole floor BEFORE each measurement: keeps the Wonham variance
            # v = 1−z² off zero, so the cross-update gain conn/v stays bounded
            np.clip(c.e1[:, 2], -Z_CAP, Z_CAP, out=c.e1[:, 2])
            if rule == "l0":
                c.observe_qubit(k, (0.0, 0.0, float(z)), alpha=min(1.0, strength))
            elif rule == "l4":
                c.measure_qubit(k, float(z), strength=strength)
            elif rule == "l4eta":
                base = trust.learned_alpha((NODES[k],), float(z), float(c.e1[k, 2]))
                c.measure_qubit(k, float(z), strength=strength * base)
        # physicality: pair correlators are bounded (|⟨σσ⟩| ≤ 1); without this the
        # near-pole cross-update can pump e2 and the coupled evolution runs away
        np.clip(c.e1[:, 2], -Z_CAP, Z_CAP, out=c.e1[:, 2])
        np.clip(c.e2, -1.0, 1.0, out=c.e2)
        est[t] = c.e1[:, 2]
    return est


def _persistence(reports: np.ndarray) -> np.ndarray:
    T, R = reports.shape
    est = np.zeros((T, R))
    last = np.full(R, -1.0)
    for t in range(T):
        for k in range(R):
            if not np.isnan(reports[t, k]):
                last[k] = reports[t, k]
        est[t] = last
    return est


def _slices(est, truth, reports, lo, hi):
    """RMSE over bins [lo, hi) predicting the NEXT bin's truth, on three slices:
    all / gap (no report this bin) / transition (the truth flips next bin)."""
    target = truth[1:]                                # est[t] predicts truth[t+1]
    e = (est[:-1] - target)[lo:hi]
    gaps = np.isnan(reports[:-1])[lo:hi]
    trans = (np.abs(np.diff(truth, axis=0)) > 1e-6)[lo:hi]

    def rmse(mask=None):
        x = e if mask is None else e[mask]
        return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")

    return {"rmse_all": round(rmse(), 4),
            "rmse_gap": round(rmse(gaps), 4),
            "rmse_transition": round(rmse(trans), 4),
            "n_gap": int(gaps.sum()), "n_transition": int(trans.sum())}


def ladder_walk(reports=None, truth=None, split_frac=0.7):
    """Walk the whole ladder over a (reports, truth) pair (default: the deterministic
    synthetic gridworld stream); calibrate each rule's dial on the train split, score
    the held-out tail. Returns {(rule, couplings_tag, dial): slice_scores}."""
    if reports is None or truth is None:
        reports, truth = gridworld_stream()
    T = reports.shape[0]
    split = int(T * split_frac)
    print(f"ladder walk on synthetic gridworld stream ({T} bins); strength calibrated "
          f"on first {split} bins, scored on the held-out last {T - split}")
    results = {}
    for coupled in (False, True):
        tag = "coupled" if coupled else "independent"
        results[("persistence", tag, "-")] = _slices(
            _persistence(reports), truth, reports, split, T - 1)
        for rule in ("l0", "l4", "l4eta"):
            best_s, best_train = None, None
            for s in STRENGTHS:                       # calibrate the dial on TRAIN
                train = _slices(_run(rule, reports, coupled, s), truth, reports, 0, split)
                if best_train is None or train["rmse_all"] < best_train:
                    best_train, best_s = train["rmse_all"], s
            est = _run(rule, reports, coupled, best_s)
            results[(rule, tag, best_s)] = _slices(est, truth, reports, split, T - 1)
    hdr = (f"{'contender':>13s} {'couplings':>12s} {'dial':>6s} "
           f"{'ALL':>8s} {'GAP':>8s} {'TRANSITION':>11s}")
    print(hdr)
    print("-" * len(hdr))
    for (rule, tag, s), sc in results.items():
        print(f"{rule:>13s} {tag:>12s} {str(s):>6s} {sc['rmse_all']:>8.4f} "
              f"{sc['rmse_gap']:>8.4f} {sc['rmse_transition']:>11.4f}")
    return results


# ── the pytest face: the machinery is pinned, no deployment verdict is claimed ──────

@pytest.mark.slow
def test_ladder_runs_and_the_rungs_are_different_estimators():
    reports, truth = gridworld_stream()
    results = ladder_walk(reports, truth)
    # every contender ran on both coupling arms and produced finite scores on all slices
    rules = {r for (r, _, _) in results}
    assert rules == {"persistence", "l0", "l4", "l4eta"}
    assert len(results) == 8
    for key, sc in results.items():
        for slice_name in ("rmse_all", "rmse_gap", "rmse_transition"):
            assert np.isfinite(sc[slice_name]), (key, slice_name, sc)
        assert sc["n_gap"] > 0 and sc["n_transition"] > 0
    # the rungs are genuinely different estimators: their belief trajectories diverge
    l0 = _run("l0", reports, False, 1.2)
    l4 = _run("l4", reports, False, 1.2)
    assert float(np.max(np.abs(l0 - l4))) > 0.05
    # and the coupling axis is real: exchange moves a never-reporting node's belief
    masked = reports.copy()
    masked[:, 0] = np.nan                              # node 0 goes silent
    ind = _run("l0", masked, False, 1.2)
    cpl = _run("l0", masked, True, 1.2)
    assert float(np.max(np.abs(ind[:, 0] - cpl[:, 0]))) > 0.05, (
        "exchange coupling failed to transfer belief into the silent node")


def test_slices_score_against_next_bin_truth():
    # a hand-checkable miniature: persistence over a 1-transition tape
    truth = np.array([[-1.0], [-1.0], [1.0], [1.0]])
    reports = np.array([[-1.0], [np.nan], [1.0], [np.nan]])
    est = _persistence(reports)
    sc = _slices(est, truth, reports, 0, 3)
    assert sc["n_transition"] == 1
    # persistence is blind exactly at the flip: est[1]=-1 vs truth[2]=+1 → error 2
    assert sc["rmse_transition"] == pytest.approx(2.0)
    # and perfect off the flip
    assert sc["rmse_all"] == pytest.approx(np.sqrt((0 + 4 + 0) / 3.0), abs=1e-6)
