# Decay notes — what gamma means, measured on a second foreign world

**Status: second-domain evidence, 2026-07-14.** The origin deployment's
ladder-walk left gamma's purpose unsettled: on a persistence-dominated home,
nothing beat hold-the-last-report, and decay-toward-uncertainty never earned
its keep as a forecaster. The question ("what IS decay for?") was walked
again on a richer environment: 103 real observation series from SpaceWheat's
LLM-playtester fleet (wallet lines off seat tapes — accumulator-like signals
with real regime changes), prequential scoring, persistence as the honest
bar. Harness: the game repo's `🍄/🧪/hive/gamma_walk.py`; committed results
alongside it.

## The dissociation

| decay as a... | verdict |
|---|---|
| **value model** (belief decays toward "unknown", forecast reads the belief) | loses **0/103** vs persistence; beats never-decay on only 10/103. The origin verdict replicates in a second domain. |
| **uncertainty model** (value held; staleness `1−\|belief\|` predicts the coming error's magnitude) | wins **79/103**: mean corr(staleness, \|error\|) = **+0.285** at the best γ vs −0.002 at γ=0. |

The calibration-optimal γ* concentrates at **3e-3–1e-2 /s** across seats — a
real, reproducible timescale (confidence half-life ≈ 1–4 minutes of play),
where the value-optimal γ* is just "zero, please".

## The reading

For accumulator-class signals the right prior is **Wiener, not
Ornstein–Uhlenbeck**: the mean persists under identity dynamics while the
variance grows — γ is a *variance growth rate*, not a mean-reversion rate.
The engine's dissipative decay toward I/2 imposes the OU prior on
everything; accumulators are the class it wrongs.

In this engine's own ladder vocabulary: such signals want the **phasor
rung's geometry** — value carried where dephasing shrinks confidence without
reverting the estimate — rather than amplitude damping toward the mixed
state. Concretely: a role-mode (or read policy) where the *forecast value*
is the last committed observation and the *forecast confidence* is the
decayed radius, with γ set (or learned) to the measured calibration
timescale.

## What's owed

- E2 (live): decay-variant world specs A/B'd on live LLM runners — success
  metric (task progress), not MAE. Decay may earn behaviorally (stale
  beliefs forcing re-scouting) even where it loses prequentially.
- E3: learnable `gamma_diss` trained across sessions — does the fiber
  converge to the measured calibration γ*? (The origin's
  `UMWELT_LEARN_COLLAPSE` machinery, pointed at decay instead of alpha.)
- The event/regime signal classes (±1 outcomes, occupancy) were NOT in this
  tape — the OU prior may be exactly right there. The taxonomy
  (accumulator vs event vs regime → which decay semantic) is the open
  design question this note exists to seed.

If this note and [CLAIMS.md](../CLAIMS.md) ever disagree, the ledger wins —
no ledger row is claimed here; this is cross-repo evidence with the data and
harness living in the game's repository.
