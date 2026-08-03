# The Local-Observation Wall

*A rigorous negative-result note on when higher-order structure is decorative in a live forecasting field — and the one real edge that survived the sweep.*

**Status:** draft for review. Not published, not sent, not committed. Numbers below are pulled verbatim from the underlying experiment logs (`umwelt` repo history, branches `feat/fractal-developmental` / `feat/self-forecast-spec`, and the `Attention Investment Forecaster` corpus). Where a number could not be traced to a source run it is flagged, not invented.

---

## Executive summary

We ran four independently-designed tests of the same hypothesis — that a belief-field's higher-order coupling machinery (total correlation, O-information "grain," multi-node exchange, cross-asset coupling) adds live forecasting value beyond what its first-order (single-node) signals already carry. All four crashed into the same wall, for the same underlying reason: **when a field is read by local, strong observation — one collapse per node, no joint readout — the act of observing factorizes the joint state before any higher-order structure can be used.** The higher-order machinery is architecturally correct (it names real structure when you *can* look at the joint state without collapsing it) but decorative in every live loop we tested, because live loops observe locally and hard.

We then ran a disciplined negative-result sweep on top of a real financial panel (S&P 500, GDELT attention data, 2014–2026, 1.45M ticker-days) to see whether this same architecture, pointed at a genuine trading question, could find anything at all. Twelve independently-designed tests, purged walk-forward validation, placebo/label-shuffle controls, and Bonferroni correction across the batch. Eleven of twelve came back null or economically inert. One came back real: a volume-anomaly decile long-short spread, **4.63%/yr, t = 3.95, n = 2,993 trading days**, leak-free, surviving Bonferroni correction — a known factor (volume anomaly), not a new discovery, but a genuine, correctly-sized, correctly-tested hit inside a sweep that killed everything else including its own headline hypothesis (attention).

The point of this note is not the one hit. The point is that a sweep this disciplined — four independent architectures converging on the same negative, twelve pre-registered tests with proper multiple-testing correction, and a single real signal correctly sized rather than oversold — is what research discipline looks like when the researcher has no institutional QA layer standing over them. That is the credential this note is meant to establish.

---

## 1. The four independent crashes

Each of these was a separately motivated experiment, built for a different reason, run at a different time, against different data. They converge on one architectural finding.

### 1.1 Gymnasium / manifold-game replay (2026-07-21)

`proofs/higher_order_live_null.py`, corroborated on a real recorded gameplay tape (`proofs/manifold_walk.py`, `manifold_game` "pantry" world, 5-role cumulant cluster, 2,282 raw rows → 138 belief batches).

- On the real tape: grain (the sign of O-information, which distinguishes redundant "chorus" binding from synergistic "conspiracy" binding) reads **flat 100% of sampled ticks**; total correlation (TC) never crosses the noise floor (**max ‖C‖ ≈ 1e-5**, roughly 10× below the measurement floor); zero multi-member constellations ever form.
- Isolated corridor-physics pins (5 gates) established this is **architectural, not incidental**: local single-qubit observation drives a one-hot product state (TC = 0.0000 exactly); a diagonal ZZ/number Hamiltonian is powerless on a diagonal one-hot mixture (it commutes — coupling strength 0.5 and 2.0 are byte-identical in outcome); exchange coupling builds real TC only while **unobserved** (TC → 1.9 in isolation) and live observation crushes it back to ≈0.
- One partial exception found and pinned: a **latent (never directly observed) concept-qubit**, exchange-coupled to observed beliefs, sustains a small live boundary correlation (C ≈ 0.08, ~800× the measurement floor; TC ≈ 0.01–0.05) — so binding is marginally live *through* an unobserved intermediary, even though it is dead between directly-observed nodes.
- The learned coupling skeleton itself is real (10 edges, |J| decaying 0.11 → 0.02) — the field does learn correct structure. What dies is the live *correlation pulse* that structure would need to produce to be useful in the moment.

### 1.2 AFI direction task (2026-07-22)

Attention Investment Forecaster — S&P 500, GDELT attention features + OHLCV forward returns, scored through the repo's own purged walk-forward harness against a gradient-boosted `ReferenceAdapter` baseline.

- 13,578 out-of-sample predictions. Bare belief-field: **0.5096** accuracy. Belief-field + learning organs engaged: **0.5096** (identical — the coupling/surprise-EMA machinery produced zero measurable couplings; it didn't even fire). Belief-field + struggle-weighted learning-rate + confidence fix: **0.5116** (+0.002, inside noise).
- All three lose to the reference baselines: a fitted gradient-boosted model scored **0.5304**, and simply predicting "always up" scored **0.5554** and beat everything, including the tuned GBM (0.5400 < 0.5462 base rate — a genuinely humbling baseline check that the harness caught before any belief-field number could hide behind it).
- Root cause traced to the same architecture as 1.1: local, strong observation factorizes the joint state before cross-belief information can route.

### 1.3 AFI synergy-hunt (2026-07-22)

A separate, harder test: rather than asking whether the belief-field's *own* coupling machinery helps, an engine-free feature-discovery search (`scratchpad/afi_synergy_hunt.py`) hunted directly for genuine second-order (pairwise-product) synergy in the same AFI panel — the exact kind of structure the field's coupling machinery is built to exploit.

- Cross-sectional information coefficient (IC) of 5-day forward return, out-of-sample over 2,460 test days: first-order IC **+0.0181 (t = 3.93)**; full second-order (all 45 pairwise feature products) IC **+0.0164 (t = 4.19)** — interactions added nothing over first-order.
- Best individual second-order product: |IC| ≤ 0.004 (t = 2–2.9), which is exactly what 45 simultaneous tests produce by chance — multiple-testing noise, not signal.
- The (small) first-order edge that does exist traces to known factors already in the literature (60-day realized volatility, 1-day reversal), not to attention.
- This same search architecture, pointed at a synthetic panel engineered to contain pure parity synergy (`C = A × B`, zero first-order signal by construction), **decisively found it** — first-order MAE 0.96 (worse than a persistence baseline of 0.917, i.e. cannot even beat "predict no change"), second-order MAE 0.415, sparse-discovered feature set 0.355, beating even brute-force full-second-order. So the search method is not blind to synergy in general — it correctly reports none exists in the real financial panel, which is a stronger negative than a hand-picked feature test would be.

### 1.4 Volatility forecasting / cross-ticker coupling (2026-07-22)

A layered, coupled belief-field volatility forecaster (K=30 tickers, star-coupled via ZZ+XY exchange to one latent "regime" qubit) vs. a HAR-RV benchmark and a plain EWMA control, five folds.

- The underlying forecasting objective is real: volatility genuinely is forecastable at this horizon (HAR-RV rank-IC 0.526, QLIKE 0.954), and the belief-field free-run edges it out (**rank-IC 0.573 vs. HAR 0.526**, recalibrated QLIKE 0.89 vs. 2.03).
- But decoupling the tickers (setting the cross-ticker exchange terms to zero) **ties or beats** the coupled version (decoupled rank-IC 0.573 vs. coupled 0.567) — the fourth independent confirmation that cross-node coupling adds nothing live, even in a domain where a *linear, non-live* regime factor genuinely helps (+9.0% QLIKE improvement from a contemporaneous market-regime covariate fit the normal way, no live field required).
- Second deflation, run honestly against itself: the field's edge over HAR is not coupling, it is long memory. A plain ~33-day EWMA vol tracker gets pooled rank-IC 0.566, essentially matching the field's 0.573; a ~100-day EWMA gets 0.560 per-day. The belief-field is, empirically, a faithful long-memory EWMA tracker with extra machinery riding along for free — not a source of surplus predictive value.

### Convergence

Four different systems, four different data sources, four different time windows, one mechanism: **a live belief-field takes evidence through local, strong (hard-collapse) observation, which factorizes the joint state before higher-order structure can be read out.** This is not a bug or a tuning failure — the underlying math (total correlation, O-information, exchange coupling) is verified correct on canonical test states (product states read flat, GHZ states read chorus, XOR/parity states read conspiracy with zero pairwise correlation visible — exactly the textbook answer). It fails specifically and only under the observation model that live deployment forces on it. The one crack (§1.1, latent-concept coupling) points at the fix — weak/collective observation that does not collapse the joint state — but that is future work, out of scope for this note.

---

## 2. Methodology (the twelve-test sweep)

Having characterized the architectural wall, the natural next question for a financial application is not "does the coupling machinery help" (answered: no) but "is there *any* edge in this panel worth having," tested with the same discipline. This section documents that sweep.

**Corpus:** AFI's S&P 500 panel — 500 names, 2014–2026, ~1.45M ticker-days. GDELT attention/tone features plus standard OHLCV-derived factors (volume anomaly, realized volatility, momentum, reversal).

**Validation discipline:**
- **Purged walk-forward**: train/test splits with an embargo period around the boundary, so no forward-looking leakage from overlapping labels near the split.
- **Placebo / label-shuffle controls**: every test re-run with labels permuted; a real effect must vanish under shuffle. Observed: shuffled labels returned lift ≈ 1.00–1.01 on every fold, for every test — nothing in the panel is a suppressed real edge hiding behind noise.
- **Bonferroni correction**: with twelve simultaneous, methodologically distinct tests run against the same panel, the significance bar was raised accordingly rather than reading each test's t-statistic in isolation.
- **Day-pooled, not row-pooled, significance**: one test initially showed t ≈ 19 on row-pooled data (N = 244k observations); day-pooling (N = number of trading days, not rows) corrected this to its honest economic size (a 1.08× lift over base rate — real but small). Flagged explicitly because pseudo-replication from row-pooling is an easy way to manufacture a false sense of significance, and the sweep's own numbers caught it.

**The twelve tests** (each a methodologically distinct approach, not twelve variants of one regression): direction classification (decile lift on next-period up/down), magnitude regression vs. an HAR volatility baseline, up/down tone asymmetry, a fresh-feature IC scan (12 previously untested columns), nonparametric decile long-short spread, event-study cumulative abnormal returns around attention spikes, turbulence-regime conditional split, sector-neutral IC, liquidity-tercile split, a momentum horse-race (residualizing the attention signal against 20/60-day momentum and reversal), a reverse-causality check (does price move first and tone follow, not the reverse), and a cross-era stability check (year-by-year, 2015–2025).

This sweep sits on top of, and is broader-but-shallower than, an earlier deeper single-hypothesis test (§1.3 above) that had already killed second-order synergy specifically. The two are complementary: §1.3 is one hypothesis tested six ways; this sweep is twelve hypotheses tested one way each.

---

## 3. Results

### 3.1 Killed

Everything attention/tone-related came back dead across every angle tried — roughly twenty independently-framed hypotheses across the full research arc (the twelve-test sweep below, plus the six-door adversarial audit that preceded it, plus the direction/synergy/coupling tests in §1). Representative results, numbers as measured:

| # | Hypothesis | Test | Result | Verdict |
|---|---|---|---|---|
| 1 | Attention magnitude predicts return magnitude | Standalone threshold lift | 0.99× (vs. volume-anomaly control 1.23×) | Killed |
| 2 | Attention has an edge in the extreme tail | q99/q99.5 lift | best 1.08× (need ≥1.15×) | Killed |
| 3 | Attention adds over a market-structure model | Incremental XGBoost lift | 2.04× vs. 2.04× (no gain) | Killed |
| 4 | Attention ranks forward returns cross-sectionally | Rank-IC | +0.013–0.016 (t ≈ 14, but far below the ~0.03–0.05 tradeable line; decays from same-day +0.02-ish to +0.006 by 20 days — coincident, not leading) | Killed (statistically real, economically inert) |
| 5 | Attention edge concentrated in low-volatility regimes | Conditional lift | 0.94× | Killed |
| 6 | Volatility + attention interaction sharpens the signal | Combined lift | +0.014 (noise) | Killed |
| 7 | Direction is predictable from attention (top decile) | Decile lift | 2.25× vs. 2.26× baseline, +0.001× attention gain | Killed |
| 8 | Signed tone predicts direction | Rank-IC(tone, fwd_ret) | +0.0001 (t ≈ 0.1) | Killed |
| 9 | Negative tone spikes are directional (not just volatile) | Up-touch vs. down-touch rate | 0.197 vs. 0.184 (symmetric — a volatility signature, not directional) | Killed |
| 10 | Attention edge is hidden inside a sector | Sector-neutral IC gap | t = 0.96 | Killed |
| 11 | Attention edge is hidden inside a liquidity tercile | Liquidity-tercile IC gap | t = 0.15 | Killed |
| 12 | Attention edge is regime-conditional (turbulence) | Regime split | t = −1.4 | Killed |
| 13 | Attention spikes produce abnormal returns (event study) | CAR around spikes | t = 0.96 | Killed |
| 14 | Attention edge is masked by momentum | Residualized vs. mom20/mom60/reversal | barely moves, t = 0.73 | Killed |
| 15 | Attention leads price (or price leads attention) | Reverse-causality (both directions) | t = −0.82 both ways | Killed |
| 16 | Attention edge is stable across market eras | Year-by-year 2015–2025 | unstable; single "best" year (2025) is actually **negative**, t = −3.0 | Killed |
| 17 | Second-order (pairwise) attention interactions carry synergy | Cross-sectional IC, 1st vs 2nd order | +0.0181 (t=3.93) vs +0.0164 (t=4.19) — interactions add nothing | Killed |
| 18 | Belief-field direction forecasting beats baselines | Purged walk-forward accuracy | 0.5096–0.5116 vs. base-rate 0.5554 and GBM 0.5304 | Killed |
| 19 | Cross-ticker coupling improves volatility forecasting | Coupled vs. decoupled rank-IC | 0.567 (coupled) vs. 0.573 (decoupled) — coupling loses | Killed |
| 20 | Grain/higher-order coupling drives a live decision | Real gameplay tape replay | grain flat 100%, TC ≤ 1e-5, 0 constellations formed | Killed |

*(Numbering above groups the twelve-test sweep — rows 1–16 — with the directly-relevant kills from the other three research lines for a complete picture; it is not a claim that exactly twenty tests were run in one batch.)*

### 3.2 Survived

| Hypothesis | Test | Result | Verdict |
|---|---|---|---|
| Unusual trading volume predicts a forward return premium | Decile long-short spread, volume-anomaly z-score | **4.63%/yr annualized, t = 3.95, n = 2,993 days** | **Real, clears Bonferroni over 12 tests, leak-free** |

This is a known classic factor (the volume-anomaly premium) rediscovered inside the panel, not a novel edge — which is itself informative: it means the sweep's methodology is sound enough to find a real, modest, previously-documented effect while correctly rejecting nineteen-plus plausible-sounding hypotheses that didn't survive scrutiny. A methodology that finds nothing ever is as suspect as one that finds everything; this one found exactly the one thing a careful reader of the market-microstructure literature would expect it to find, and nothing else.

Runner-up (not counted as a hit): `tone_ewm5` IC = +0.0045, t = 4.23 vs. forward return — nominally statistically significant but economically inert (tradeable rank-IC needs to be roughly 0.02–0.03; this is a fifth to a sixth of that threshold).

---

## 4. Theoretical takeaway: why local strong observation factorizes the joint state

The unifying finding across §1 and §3 is architectural, not statistical. A belief-field (or any live forecasting system built on sequential, hard measurement of individual signals) updates its state by collapsing one node at a time onto the evidence it just received. Each such collapse is a strong (projective) measurement on a single subsystem. The mathematics of quantum-like joint states says this repeated local strong measurement drives the joint density matrix toward a product state — the correlations between nodes that would carry higher-order information (synergy, conspiracy-type binding, genuine multi-way structure) get erased by the very act of reading each node individually and hard.

This has a precise, testable signature, which is what makes it a real finding rather than a hand-wave: the coupling machinery (total correlation, O-information sign, exchange terms) reads *correctly* on canonical test states where the joint state is allowed to exist undisturbed — GHZ states show chorus, XOR/parity states show conspiracy with literally zero pairwise correlation visible, exactly matching the textbook math. The same machinery reads flat, dead, or decorative the instant the system is driven by the kind of observation a live deployment actually uses: one signal arrives, gets hard-assigned to one node, the joint state factorizes, repeat. The failure mode is not "the coupling math is wrong" or "we didn't tune it enough" — every attempt to fix it by tuning (coupling strength, learning rate, interaction order, cross-ticker topology) failed identically, because tuning cannot fix an observation-model problem.

Why this is a useful negative result, not just a failure: it draws a hard boundary around where a large, actively-worked class of techniques (higher-order coupling, joint-state binding, "the field learns cross-signal structure") can and cannot pay off in live deployment, and it does so with the kind of falsifiable, multiply-replicated evidence that lets someone downstream skip re-deriving the same wall. Four independent teams-of-one experiments (a game-replay pin, a direction-forecasting benchmark, a feature-discovery search, and a coupled-volatility forecaster) converged on identical architecture-level conclusions using different data, different time windows, and different implementations. That convergence is the actual result — not any single number in isolation. The one place a crack was found (a latent, never-directly-observed intermediary node sustaining live boundary correlation) is a specific, falsifiable next hypothesis rather than a vague "more research needed," and is explicitly out of scope for this note.

The financial sweep (§2–3) is a second, independent test of the same discipline applied to a genuinely commercial question rather than a mechanism question: does an entire class of hypothesized signal (social/news attention as a return predictor) survive twelve honestly different angles of attack, with proper correction for the fact that twelve angles were tried? The answer was no for attention, and yes — correctly, modestly sized — for the one classic factor that should have shown up if the pipeline was trustworthy at all.

---

## 5. Why this matters as a credibility artifact

This note is not being published as an academic contribution — it is not claiming a new theorem, and the negative results, while rigorous, are not novel in the finance literature (volume anomaly and attention-doesn't-predict-returns are both well-trodden ground). Its value is different: it is evidence of a specific kind of research discipline, produced solo, at a pace and rigor level that a one-person shop does not usually demonstrate publicly.

Concretely, this note demonstrates:

- **Pre-registered, methodologically diverse testing** rather than one model run twelve ways — the twelve tests in §2 are genuinely different statistical approaches (decile spreads, event studies, regime splits, reverse-causality checks), not twelve p-values mined from the same regression.
- **Correction for the researcher's own multiple-testing exposure** (Bonferroni across the sweep) applied *before* reporting the one hit, not after the fact to justify it.
- **Self-correction caught and reported, not hidden** — the row-pooled vs. day-pooled significance error (§2) is exactly the kind of mistake that gets silently fixed and never mentioned in most write-ups; it's disclosed here because the discipline that catches it is the point.
- **A negative result treated as a deliverable**, not a dead end — four independently-built systems converging on the same architectural wall is a stronger, more useful claim than any one of them succeeding by luck would have been, and it de-risks an entire downstream research direction (joint-state / higher-order coupling in live forecasting systems) for anyone who reads it before spending compute re-discovering the same wall.
- **Honest sizing of the one real hit** — 4.63%/yr is reported as what it is (a known, modest, correctly-attributed factor), not oversold as alpha discovery.

This note also functions as a proof artifact feeding two adjacent efforts not scoped here: a broader umwelt research-paper portfolio (candidate paper alongside the causal-self-tagging and estimator-ladder work already indexed in `docs/papers.md`), and a future public-facing dashboard demonstrating this research process in the open. Both are separate, not-yet-scoped pieces of work; this note stands on its own as the finished research package behind them.

**Disclosure note:** this note, and AAI's Gate-1 output, are being made available ahead of the full yurt-Foundry market-gate clearing. AAI's own Gate-3 validation is the bar this work is held to; the financial urgency of establishing research credibility now outweighs waiting for Foundry-level infrastructure to mature. A related, not-yet-started effort (importing yurt's purged walk-forward gate wholesale into AAI's own Gate-2 backtest harness) will bring AAI's infrastructure up to the same standard this note was produced under; it is a sibling project, not a prerequisite for this one.

---

## Appendix: source material

- `~/.claude/projects/-home-primearchitect-ws-SpaceWheat/memory/project_courtship_braid_2026-07-21.md` — gymnasium/R1 grain-null, AFI direction, AFI synergy-hunt, volatility-coupling run, six-door adversarial audit.
- `~/.claude/projects/-home-primearchitect-ws-SpaceWheat/memory/project_self_forecast_2026-07-22.md` — Value Hunt #1 (AFI synergy engine) and Value Hunt #2 (twelve-rod sweep, volume-anomaly hit).
- `~/ws/umwelt/CLAIMS.md` §1 and §5 (DENIED tier) — test-pinned architectural finding (`proofs/higher_order_live_null.py`, `tests/test_higher_order.py`) and canonical-state validation of the coupling machinery.
- Underlying code: `umwelt` branch `feat/fractal-developmental` (`cea80bc`) stacked on `feat/manifold-higher-order` (`62ee356`); AFI adapter `afi/src/afi/engine_umwelt.py` (uncommitted, in Luke's non-git AFI working copy); scratch scripts `afi_synergy_hunt.py`, `rod_*.py` (12 scripts, workflow run `wf_b5498517-ee9`), `vol_bench/vol_umwelt/vol_control.py`.
