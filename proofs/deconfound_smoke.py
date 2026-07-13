"""De-confounding SMOKE — the causal self-tagging mechanism, on a synthetic loop.

The trap, in gridworld terms: give the system an ANTICIPATORY beacon policy (light the
cell the agent is about to enter — modeled at its strongest, an oracle one-bin-ahead
forecast). The beacon state is re-observed every bin. A naive online learner then
absorbs a correlation that is causally FALSE — "beacons come on, then the agent
arrives" reads as beacons PREDICTING (summoning) the agent — when the truth is the
policy's own foresight. The de-confounding mechanism is the shipped router
(umwelt.learning.learning_router: echo_likelihood + per-channel gate; the graph-derived
confounding SURFACE that names which role an actuator self-causes is
umwelt.learning.confounding, pinned separately in tests/test_confounding.py).

Arms, identical in everything but the attribution:

  naive     every observation trains the learner at full weight.
  tagged    attribution d = author-of-record (echo_likelihood for the transient +
            standing-state match), and the confounded FEATURE's gradient is scaled by
            LearningRouter.gate(d).world. Nothing else differs.
  nopolicy  the same learner in a world where the system never acts (the agent's own
            manual beacon use only) — the TRUE association strength.

THIS IS A SMOKE, NOT AN EFFECT-SIZE CLAIM: it pins that the mechanism produces
different learned weights naive vs tagged on a synthetic actuation loop, with the
tagged arm landing nearer the no-policy truth. The measured effect sizes — the origin
deployment's 24-day real-tape A/B (meerkat experiments/deconfound_ab.py, 2026-07-09):
naive credits its own anticipatory lights at 10.8× the true association and the
shipped router cuts the bias 79%, with the naive arm alone degrading when its policy
silences — belong to that origin run and are NOT re-claimed from synthetic data.
"""
from __future__ import annotations

import numpy as np

from examples.gridworld.world import agent_walk, binned_truth, grid_cells
from umwelt.learning.learning_router import LearningRouter
from umwelt.learning.regressor import OnlineRegressor

CELLS = grid_cells(2, 2)
BIN_S = 300.0          # a dispatch is observed next bin (age ≈ BIN_S)
ECHO_TOL = 0.5         # |observed − dispatched| tolerance (binary beacons → generous)
ECHO_RECENT_S = 900.0  # dispatch echo window: 3 bins
LR = 0.05
P_MANUAL = 0.04        # the agent touches a beacon ~1/25 bins/cell
SPLIT = 0.7            # policy active on the first 70%; silenced after (the shift test)


def _occupancy() -> np.ndarray:
    """Ground-truth ±1 occupancy over the 2×2 grid — the deterministic gridworld walk."""
    walk = agent_walk(2, 2, seed=11, days=3.0, mean_dwell_s=1200.0)
    return binned_truth(walk, CELLS, bin_s=BIN_S)


def run_arm(arm: str, occ: np.ndarray, seed: int = 0):
    """One learner per cell: [own_occ, beacon] → next-bin occupancy, learning ONLINE
    over the whole tape. Policy: anticipatory (oracle one-bin foresight) during the
    policy era, silenced after SPLIT. The tagged arm gates each cell's beacon feature
    by the shipped router when the observation is the echo of our own dispatch."""
    T, R = occ.shape
    manual_rng = np.random.default_rng(1234)   # the agent's hand — identical across arms
    router = LearningRouter()
    regs = [OnlineRegressor(n_targets=1, target_ids=(f"{c}_occ",), lr=LR, l2=1e-4)
            for c in CELLS]
    split = int(T * SPLIT)
    w_beacon = np.zeros((T, R))
    sqerr = np.full((T - 1, R), np.nan)
    beacons = np.full((T, R), -1.0)
    last_dispatch = np.full(R, np.nan)          # what we last commanded (the echo record)
    dispatch_age = np.full(R, np.inf)
    for t in range(T - 1):
        # ── the system's policy acts (writes the world) — only during the policy era ──
        if t < split and arm != "nopolicy":
            # ANTICIPATORY: pre-light the cell the agent is about to enter
            # (oracle 1-bin foresight — the strongest-confounding case, by design)
            cmd = np.where(occ[t + 1] > 0, 1.0, -1.0)
            changed = cmd != beacons[t]
            beacons[t:, :] = np.where(changed, cmd, beacons[t])
            last_dispatch = np.where(changed, cmd, last_dispatch)
            dispatch_age = np.where(changed, 0.0, dispatch_age + BIN_S)
        else:
            dispatch_age = dispatch_age + BIN_S    # SILENCED: no more dispatches
        # ── the agent acts (identical across arms): occasionally sets a beacon to
        #    match true occupancy — beacons they touch carry REAL information ──
        manual = manual_rng.random(R) < P_MANUAL
        if manual.any():
            mval = np.where(occ[t] > 0, 1.0, -1.0)
            flip = manual_rng.random(R) < 0.2          # imperfect habits: 20% contrarian
            mval = np.where(flip, -mval, mval)
            beacons[t:, :] = np.where(manual, mval, beacons[t])
        # ── the learner predicts next-bin occupancy, then learns ────────────────
        for k in range(R):
            x = np.array([occ[t, k], beacons[t, k]])
            p = regs[k].predict(x)                       # None before lazy-init
            pred = float(p[0]) if p is not None else 0.0
            sqerr[t, k] = (pred - occ[t + 1, k]) ** 2
            # attribution: are WE the author of the beacon's current state?
            # (echo_likelihood covers the transient; author-of-record covers the
            # standing state — the matured forecast attribution's deterministic analog)
            if arm == "tagged":
                echo = router.echo_likelihood(
                    observed=float(beacons[t, k]),
                    dispatched=(None if np.isnan(last_dispatch[k]) else float(last_dispatch[k])),
                    age_s=float(dispatch_age[k]),
                    tol=ECHO_TOL, recent_s=ECHO_RECENT_S,
                )
                standing = (not np.isnan(last_dispatch[k])
                            and beacons[t, k] == last_dispatch[k])
                d = max(echo, 1.0 if standing else 0.0)
            else:
                d = 0.0
            # PER-CHANNEL gate (the confounding-surface semantics: the actuator
            # confounds the role it projects onto — the beacon channel — not the
            # sighting-fed occupancy channel): learning proceeds at full rate, but the
            # confounded feature's gradient is scaled by the world weight (d=1 → the
            # beacon contributes nothing to this update; the clean channel still learns).
            gate = router.gate(d)
            x_train = x.copy()
            x_train[1] *= gate.world
            regs[k].update(x_train, np.array([occ[t + 1, k]]), lr=LR)
            if regs[k].W is not None:
                w_beacon[t + 1, k] = float(regs[k].W[0, 1])   # W is (n_targets, feat_dim)
    return w_beacon, sqerr, split


def deconfound_smoke() -> dict:
    occ = _occupancy()
    out = {}
    for arm in ("naive", "tagged", "nopolicy"):
        w, e, split = run_arm(arm, occ)
        out[arm] = {
            "w_beacon": float(np.mean(np.abs(w[split - 1]))),   # |w| at policy silencing
            "rmse_pre": float(np.sqrt(np.nanmean(e[:split]))),
            "rmse_post": float(np.sqrt(np.nanmean(e[split:]))),
            "split": split,
        }
        print(f"  {arm:>8s}: |w_beacon|={out[arm]['w_beacon']:.4f}  "
              f"rmse pre={out[arm]['rmse_pre']:.4f} post={out[arm]['rmse_post']:.4f}")
    return out


def test_tagging_mechanism_moves_the_learned_weight():
    """The smoke: naive and tagged learners land at DIFFERENT beacon weights on the
    same tape, with tagged nearer the no-policy truth — the mechanism works. Effect
    sizes are the origin run's business (module docstring), not this test's."""
    out = deconfound_smoke()
    w_naive, w_tagged, w_truth = (out[a]["w_beacon"] for a in ("naive", "tagged", "nopolicy"))
    assert w_naive > w_tagged, (
        f"tagging did not reduce the self-credit weight (naive {w_naive:.4f} "
        f"vs tagged {w_tagged:.4f})")
    assert abs(w_tagged - w_truth) < abs(w_naive - w_truth), (
        f"tagged ({w_tagged:.4f}) did not land nearer the no-policy truth "
        f"({w_truth:.4f}) than naive ({w_naive:.4f})")
    for arm in out:
        assert np.isfinite(out[arm]["rmse_pre"]) and np.isfinite(out[arm]["rmse_post"])
