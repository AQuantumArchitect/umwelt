# CLAIMS — the honest-claims ledger

Every claim this repo makes, sorted by how bulletproof it is when challenged.
**Keep it current or delete it; if this ledger and any other doc disagree, the ledger
wins.** Each row: the claim, the evidence pointer, the status. Docs are the one place
the meerkat origin is cited freely — the engine source itself is domain-free
(enforced by `tests/test_vocabulary_lint.py`).

The lineage in one line: this library was extracted from **meerkat**, a belief-field
brain that ran a real instrumented home; claims measured there are marked as the
origin's and are not re-claimed from this repo's synthetic data.

---

## 1. MEASURED / TEST-PINNED (in this repo — runs in the gate, every time)

| claim | evidence | status |
|---|---|---|
| The full gate is green: 107 tests, zero failures, including every proof below. | `python3 -m pytest -q` over `tests/` + `proofs/` | PINNED |
| **The blank-slate theorem**: a max-entropy, unlocated engine boots a gridworld it has never seen, replays a deterministic synthetic day through the production ingest path, and provably comprehends it — every declared binding drove the field, beliefs track the agent's ground-truth walk at checkpoints, the learnable fiber drifts off its priors, the coordinate is honest end to end (nowhere → grounded fix → nowhere on a fresh boot), and the learned state survives save/load and keeps working. | `proofs/blank_slate.py` | PINNED |
| The engine source knows no domain: no houses, rooms, astronomy, vendors — a banned-vocabulary lint over all of `src/umwelt/`. | `tests/test_vocabulary_lint.py` | PINNED |
| No place token without an anchor: a spec with no anchor can never mint a location, codec or not. | `proofs/blank_slate.py::test_no_place_token_without_an_anchor` | PINNED |
| **Place is provenance, not radius**: an ungrounded anchor can never mint a place token, no matter how far the dynamics drift its qubit — the regression found by the first real foreign replay (13 days of real drift crossed a radius gate the 1-day synthetic proof never reaches). | `tests/test_gauge_place_provenance.py`; [docs/FIELD_NOTES.md §2](docs/FIELD_NOTES.md) | PINNED |
| Cumulant-closure fidelity **on this repo's synthetic gridworld stream**: 100.00% decision parity vs full-ρ on ZZ couplings (z divergence ≈ 0); divergence shows honestly under exchange stress (max 0.02 at J=0.6, parity 99.97%). | `proofs/fidelity_harness.py` | PINNED |
| The estimator ladder runs end to end and its rungs are genuinely different estimators; exchange coupling really transfers belief into a silent node; slice scoring is next-bin prequential. | `proofs/ladder_walk.py` | PINNED |
| The causal self-tagging mechanism works: on a synthetic anticipatory-actuation loop, the naive learner credits its own policy more than the router-gated learner, and the gated learner lands nearer the no-policy truth. (Mechanism only — effect sizes belong to the origin row below.) | `proofs/deconfound_smoke.py` | PINNED |
| The confounding surface is graph-derived and uniform: every actuator confounds exactly the learned role it projects onto — no per-actuator code. | `tests/test_confounding.py` | PINNED |
| `graph_state` is a provable superset of the transparency snapshot (params/clusters/summary regrouped field-for-field), self-describing (every organ typed), cheap (never calls `engine.context()`), and topology-agnostic (a foreign-shaped graph still projects; failures degrade, never 500). | `tests/test_graph_state.py`, `tests/test_transparency.py` | PINNED |
| Shadow-first egress: spec outputs decide visibly and dispatch NOTHING until the app opts in; operator overrides move the tendril's learned geometry. | `tests/test_egress_tendrils.py` | PINNED |
| Save/load preserves the field canon hash byte-for-byte. | `tests/test_engine_blank_boot.py`, `proofs/blank_slate.py` | PINNED |
| Trust-web off = last-wins with prior-init, so day-1 equals a confidence-weighted average (no silent behavior change when the web is dark). | `tests/test_trust_web.py` | PINNED |

## 2. MEASURED ON THE ORIGIN DEPLOYMENT (meerkat — real data, cited, not re-claimed here)

| claim | evidence | status |
|---|---|---|
| **The de-confounding A/B**: on the origin's real 24-day presence cassette, a naive online learner credited its own anticipatory lights at **10.8×** the true association strength (0.481 vs the no-policy world's 0.044); the shipped router's tagged arm landed at 2.2× — a **79% bias reduction** — and the naive arm was the only one that got WORSE when its policy silenced (+0.067 while every honest arm improved). | meerkat `experiments/deconfound_ab.py` (verdict of record in its docstring, 2026-07-09) | ORIGIN-MEASURED |
| **The ladder-walk verdict**: on the origin's 6908 real bins, persistence 0.1346 ≈ α-blend 0.1349 ≪ Belavkin 0.3034 overall — the full Belavkin filter was denied by its own experiment (see the DENIED tier). Cross-node coupling earned at transitions (−11.5%). | meerkat `experiments/ladder_walk.py`; [docs/THEORY.md §5](docs/THEORY.md) | ORIGIN-MEASURED |
| Cumulant-closure fidelity **on real data**: 100% actuator-decision parity vs full-ρ on the production coupling class over 6908 real bins (z divergence ≤ 0.0104); 97.21% under exchange stress. | meerkat `experiments/fidelity_harness.py` | ORIGIN-MEASURED |
| The lineage ran **live for ~18 months on a $100 ARM board** (an RDK X5-class SBC) as the origin home's resident brain — the engine's cost envelope is a measured deployment fact, not a benchmark projection. | the meerkat deployment record (its docs/STATUS.md lineage) | ORIGIN-MEASURED |

## 3. MEASURED ON THE FIRST FOREIGN WORLD (a real smart-home export, 2026-07 — data and rig private, cited, not re-claimed here)

The first time this engine met real data it was not built around: a 13-day Home
Assistant export of a real family home, replayed blank on x86 through the production
ingest path. The data and rig stay private permanently; the transferable lessons live
in [docs/FIELD_NOTES.md](docs/FIELD_NOTES.md).

| claim | evidence | status |
|---|---|---|
| **The blank-slate theorem held on real foreign data**: blank boot (learnedness 0.054 floor), all 45 declared bindings drove the field, the fiber drifted (0.054 → 0.129), the gauge read `.nowhere.` end to end, and the field canon hash survived save → fresh boot → load. Held-out comprehension: belief AUC **0.94 / 0.87 / 0.79** in the three well-sensed rooms vs 0.60–0.67 persistence-of-own-inputs baselines, natural sign; the under-sensed floor reported inconclusive. | [docs/FIELD_NOTES.md](docs/FIELD_NOTES.md) (headline) | FOREIGN-MEASURED |
| **The dissipative-role law**: a role fed only one polarity of evidence saturates if declared unitary (dephasing never relaxes populations — every belief pinned at z ≈ +0.95 for 13 days, scores riding 4th-decimal ripples under an inverted sign). Declaring the role dissipative (evidence lands hard, relaxes toward uncertainty on a learnable timescale) restored real dynamics and the natural sign. The synthetic proof gate is structurally blind to this: the gridworld feeds both polarities. | [docs/FIELD_NOTES.md §1](docs/FIELD_NOTES.md) | FOREIGN-MEASURED |
| **Topology growth found true structure on real data**: driven per batch over a blank growth replay, the learned-topology machinery grew 7 couplings within the first 36 h of world-time and pruned none over 11 days — every edge independently corroborated by offline co-movement statistics and by engine-free transition counts. Replay, not live: the live win stays owed in the tier below. | [docs/FIELD_NOTES.md §5](docs/FIELD_NOTES.md) | FOREIGN-MEASURED |

## 4. DESIGNED / EVALUATION OWED (say so out loud)

| claim | evidence | status |
|---|---|---|
| Berry-phase decision authority: geometric phase as a process clock that can GATE a downstream decision (a choice that flips because of winding). The tape machinery exists; the decision demo is owed. | `umwelt/clocks/berry_tape.py`; origin topology pins (loop → γ=−π) stayed with the origin | EVALUATION OWED |
| The trust web improves fused readings over last-wins on live data. Day-1 parity is pinned (tier 1); the live A/B that shows the web EARNING its keep is owed. | `umwelt/foresight/trust_web.py`, `UMWELT_TRUST_WEB` | EVALUATION OWED |
| The domain examples beyond gridworld: the smarthome vocabulary module is real, runnable code (`examples/smarthome/` — role modes, normalizers, the SolarDriver); the recommender / butler / sentiment-market adapters are DESIGNED SKETCHES with their synthetic demos owed (each README says so itself). | `examples/` | EVALUATION OWED |
| Dream-loop topology growth (surprise-minimizing coupling search) helps a live deployment. The mechanism is test-pinned, and the first real-data *replay* evidence now exists (tier 3: 7 grown couplings, 0 spurious); the **live** win is still not demonstrated. | `tests/test_dream_topology.py`, `tests/test_dream_loop.py`; [docs/FIELD_NOTES.md §5](docs/FIELD_NOTES.md) | EVALUATION OWED |

## 5. DENIED (by their own experiments — kept because negative results with numbers are the discipline)

| claim | evidence | status |
|---|---|---|
| ~~The Belavkin filter should ship as the default estimator~~ — **DENIED**: it lost to the α-blend (and to bare persistence) on the origin's own promotion experiment. `UMWELT_BELAVKIN` ships default-OFF with a measured justification; the reference implementation stays because the confidence-as-η theorem stands. | meerkat `experiments/ladder_walk.py`; [docs/THEORY.md §5](docs/THEORY.md) | DENIED, SHIPS OFF |
| ~~Offload the field step to the origin board's NPU (BPU)~~ — **DENIED**: measured on the origin hardware, the A55 CPU beat the BPU 7–12× at field sizes (d≤32). The accelerator path here is an inert shim (`foresight/bpu_forecast.py` falls back to CPU); the compile path was kept at the origin for large-model work only. | the origin's BPU kernel benchmark (meerkat lineage) | DENIED |
| ~~The origin's "3.4×" region-merge speedup~~ — **did not survive re-measurement** (~1.1× after rework). Kept here deliberately as the example of the honesty discipline catching its own number; no merge-transform speedup is claimed by this repo (the fold transform isn't even ported). | the origin's b6.0→b7.0 record (meerkat `docs/PORTFOLIO.md` §3) | DENIED / CORRECTED |
