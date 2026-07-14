# Path: umwelt today → a generalized Fledgeling core

*A roadmap, not a claim. This document maps how the current belief-field engine
could become the reusable **simulation intelligence substrate** under
[Fledgeling](http://www.peripheralarbor.com/fledgeling/) (`f.tryop.com`) —
without pretending it already is that substrate, and without swallowing the whole
game into one library.*

If this page and [CLAIMS.md](../CLAIMS.md) disagree on what is proven, the ledger
wins. Everything below that is not already in the ledger is **design intent**.

---

## 1. North star (one sentence)

A **Fledgeling core** is a domain-agnostic runtime that holds **live, partial,
honest beliefs** about nested, player-mutable worlds — so agents (and the player
as an SI) can observe, plan, act, and learn **without mistaking their own
footprint for world truth** — while game content, rendering, and facet rules live
outside the core.

umwelt today is a strong candidate for the **belief / observe / act-with-echo**
slice of that core. It is not a candidate for the whole of Fledgeling.

---

## 2. Where we are (current project state)

### 2.1 What ships and is gate-pinned

| Capability | Home | Status |
|---|---|---|
| World as data (`DomainSpec`: nodes, bridges, bindings, outputs, drivers) | `umwelt.spec` | PINNED |
| Blank boot → ingest → belief ease → save/load hash | `boot`, `engine`, proofs | PINNED |
| Domain-free engine (vocabulary lint) | `tests/test_vocabulary_lint.py` | PINNED (intent) |
| Causal self-tagging (graph-derived confounding surface) | `learning/confounding`, router | Mechanism PINNED; effect sizes ORIGIN |
| Shadow-first decisions (tendrils) | `membranes/egress` | PINNED |
| Trust-web fusion (day-1 parity; isolation when referee exists) | `foresight/trust_web` | Mechanism PINNED; live skill OWED |
| Local daemon + client + playground | `umweltd` | PINNED parity |
| One CI domain (gridworld fog-of-war as weak measurement) | `examples/gridworld` | PINNED |
| Experimental rant → gated spec (forge) | `umweltforge` | Pipeline PINNED; quality OWED |

### 2.2 What is measured only at the origin / private foreign world

- ~18 months live home deployment (meerkat)
- De-confound A/B (10.8× → ~79% bias cut)
- First foreign Home Assistant replay (AUCs vs persistence; dissipative-role law)
- Estimator ladder: full Belavkin **denied** as default; α-blend / persistence hard to beat

These inform design. They are **not** re-claimed as Fledgeling evidence.

### 2.3 What is explicitly not ready

- Multi-domain production beyond home extraction
- Berry-phase **decision authority** (tape exists; flip-a-choice demo OWED)
- Dream-loop / topology growth **live** wins (replay evidence only)
- Plain-English authoring quality at scale
- Any game loop, narrative layer, multiplayer, voxels, or multi-scale physics

### 2.4 Architectural center of gravity today

```
sensors / feeds  →  bindings (η, normalizers)  →  belief field (graph of roles)
                                                      ↓
actions / recs   ←  shadow tendrils ←──── self-tagging echo on learn
```

Time is **drivers + ingest cadence**, not nested game ticks or retro-sim.
Topology is a **declared tree + bridges**, not fractal scale nesting.
The default successful estimator on origin data is closer to **classical blend /
persistence** than to full open-quantum filtering — the quantum language is a
superset formalism with mixed ablations, not the product default.

---

## 3. What “Fledgeling core” means (target, scoped)

Fledgeling (public vision) is a family of facets and a long-horizon sandbox:

| Facet / theme | Core need (simulation intelligence) |
|---|---|
| Fog / discovery / Glyph-class puzzles | Partial observation, confidence, scouting as buying η |
| Warmth / attention | Directed awareness, source reliability, catalysts |
| Bread Winner / economics | Stock / scarcity beliefs; policy that doesn’t train on own market moves |
| Anteciplace / retro-sim | Plausible prior context (hard; mostly **out of core v1**) |
| Uplift / multi-scale voxels | Nested abstraction (hard; **out of core v1**) |
| Lucid / dreams | Counterfactual replay (partial overlap with dream machinery) |
| Gerbil / SI progression | Agency, sub-routines, competence gating, time-vs-attention |

### 3.1 In-scope for “generalized Fledgeling core” (this roadmap)

Call this **FL-core**: the reusable substrate any facet can host.

1. **Belief graph runtime** — nodes/roles/bridges as data; blank or seeded profiles
2. **Partial observation law** — every report has η; silence ≠ false certainty
3. **Self-action hygiene** — act → world change → learn only through tagged echo
4. **Shadow → earned autonomy** — decide visibly; dispatch only when competence allows
5. **Multi-agent / multi-SI observations** — many observers, shared world, private beliefs
6. **Game-time membrane** — ticks, turns, and time-contraction as first-class cadence
7. **Public baselines** — every new capability beats persistence / last-wins on synthetic **Fledgeling-shaped** streams (no private house cassettes required)
8. **Thin host API** — library-first; daemon optional; **no quantum vocabulary required at the game boundary**

### 3.2 Explicitly out of scope for FL-core

- Rendering, audio, input, networking
- LLM dialogue and content generation (may sit **beside** the core)
- Voxel physics, geology, plant simulation
- Full retro-sim / history generation
- “The game designs itself”
- Claiming consciousness, AGI, or a drop-in Dwarf Fortress brain

Facets compose **on top of** FL-core. The core does not implement Warmth or Bread
Winner; it makes their **belief and agency contracts** cheap and honest.

---

## 4. Gap map (current → FL-core)

```
                    umwelt today                    FL-core target
                    ────────────                    ──────────────
World shape         house-like tree + bridges       same graph model, game idioms
Observation         sensors / normalizers           events, scouting, speech, UI probes
Action              device tendrils + shadow        intents, sub-routines, SI acts
Learning            online fiber + confound gate    same + multi-actor attribution
Time                drivers + ingest_hold           game tick, turn, era, pause, FF
Agents              one engine ≈ one mind           N minds over 1 world (or N worlds)
API surface         DomainSpec + engine.ingest      + GameHost / Observer / Intent
Evidence            home + gridworld + private HA   public fledgeling fixtures in CI
Language            Bloch / η / tendril internal    plain belief/confidence at boundary
```

| Gap | Severity for Fledgeling | Notes |
|---|---|---|
| Single-mind assumption | High | One `BeliefEngine` ≈ one umwelt; NPCs need many |
| Shared world, private beliefs | High | Need world state vs per-agent field separation |
| Actor identity on actions | High | Confounding must key on **who** acted, not only graph_node |
| Game clock / turn membrane | High | Wall-clock and solar drivers are the wrong default |
| Procedural topology mutation | Medium | Growth exists experimentally; player-mutable maps are the product |
| Abstraction / LOD of belief | Medium | Nested voxels need nested or rolled-up roles later |
| Retro-sim / prior synthesis | High (but later) | Different research problem; don’t block v1 |
| Quantum metaphor at game API | Medium (social/tech tax) | Keep substrate; wrap the host face |
| Content / narrative | N/A | Never the engine’s job |

---

## 5. Design principles for the path

1. **Steal contracts, don’t force the house shape.** Partial observation, self-tagging, shadow autonomy, blank-boot proofs stay. Room/sensor vocabulary does not.
2. **Public synthetic fixtures or it didn’t happen.** Fledgeling gates must run offline in CI with no private meerkat/HA data.
3. **Beat a dumb baseline.** Persistence, last-wins, and “always believe the player’s last act” are the control arms.
4. **Library-first.** `umweltd`/forge are optional hosts; the game embeds the library.
5. **Plain face, rich guts.** External types: `Belief`, `Confidence`, `Intent`, `Observation`. Internal qubits optional and invisible.
6. **One facet spike before generalization.** Prefer a working fog/attention micro-game over a grand multi-facet architecture.
7. **Ledger every promotion.** New FL claims get CLAIMS.md rows (or a sibling `CLAIMS_FLEDGELING.md`) with the same DENIED discipline.

---

## 6. Phased path

Each phase ends with a **gate**: demos + tests + a baseline bake-off. Do not start
the next phase until the gate is green or the phase is explicitly **parked** with a
reason.

### Phase 0 — Alignment (no new architecture)

**Goal:** Agree what FL-core is *for* and what the first bake-off is.

| Work | Done when |
|---|---|
| This document reviewed by Fledgeling + umwelt maintainers | Written sign-off or issues filed |
| Pick **one** vertical spike (recommended below) | Named in §7 |
| List 5–10 “belief questions” the spike must answer in play | Checklist in the spike README |
| Freeze non-goals for 90 days | §3.2 accepted |

**Exit:** A short ADR or issue: “Spike X is the path; Y is deferred.”

---

### Phase 1 — Fledgeling-shaped domain, no engine changes

**Goal:** Prove the **existing** engine can host a game-like world without forks.

**Deliverable:** `examples/fledgeling_fog/` (name flexible) — a small graph world:

- Nodes: places (cells / rooms / corridors), maybe one resource role
- Observation: scout/probe events with η; unobserved beliefs relax (dissipative law)
- Action: optional “mark safe / claim resource” in **shadow** first
- Driver: **turn or tick** via `DriverSpec` or synthetic timestamps (not solar)
- Demo: ASCII or JSON timeline; CI proof cloned from blank_slate shape

**Bake-off:** belief vs persistence-of-own-inputs on held-out “is the agent near?”
style labels (same spirit as FIELD_NOTES, fully synthetic).

**Engine changes:** none required; vocabulary lives in the example package only.

**Exit gate:**

- [ ] `python -m umwelt.spec.validate` green on the fledgeling example
- [ ] Proof: blank boot, bindings drive field, beliefs track ground truth, save/load
- [ ] README states what is *not* proven (no multi-agent, no narrative)
- [ ] Optional: side-by-side MAE/AUC table vs persistence in the proof output

**If this fails:** stop. The gap is product-market for *this* substrate on game graphs,
not missing forge features.

---

### Phase 2 — Host API: game face over the engine

**Goal:** A boundary a Fledgeling host can depend on without speaking DomainSpec daily.

```
GameHost
  register_world(spec | fledgeling_ir)
  observe(observer_id, channel, value, confidence, t)
  intend(actor_id, intent) → Decision (shadow | live)
  beliefs(observer_id, query) → {node.role → (value, confidence)}
  step(t) / step_turn(n)
```

| Work | Notes |
|---|---|
| `umwelt.fledgeling` or `umwelt.host` package | Thin adapter; engine remains domain-free |
| Map Intent → OutputSpec / tendril dispatch | Keep shadow default |
| Map Observation → binding + η | Scout, hear-say, UI inspect as channels |
| Hide Bloch z in default responses | Export calibrated scalars + confidence |
| Clock helper for turn-based and FF | Builds on TIME.md: cadence ≠ universe time model |

**Exit gate:**

- [ ] Host API used by the Phase 1 example (example no longer calls `engine.ingest` directly for the happy path)
- [ ] Contract tests: η=0 no-op; shadow dispatch no world side effect; kill/reload state
- [ ] No banned domain words leaked into `src/umwelt/` core (lint still green)

---

### Phase 3 — Multi-mind: shared world, private umwelten

**Goal:** N agents, one ground truth (or one shared classical ground), N belief fields.

```
                ┌── Agent A field (private)
 Ground / scene ┼── Agent B field (private)
                └── Player-SI field (private)
         ↑ observations (partial, noisy, role-limited)
         ↓ intents (tagged by actor_id)
```

| Work | Difficulty |
|---|---|
| `WorldSession` owning ground + map of engines or isolated clusters | Medium–high |
| Per-observer binding masks (what each mind is allowed to sense) | Medium |
| Actor-keyed confounding (`actor_id` × graph surface) | High — extend, don’t replace, graph-derived surface |
| Competence / autonomy per agent per intent type | Medium (warden ideas, game-facing) |
| Cost envelope: many small fields vs one big field | Measure early |

**Exit gate:**

- [ ] Synthetic scenario: two agents, same corridor; A sees B, B does not; beliefs diverge correctly
- [ ] A’s action does not inflate B’s world-model of “spontaneous” change without an observation path
- [ ] Baseline: shared-global-belief cheat loses on a “privacy of mind” assertion suite

**Park criterion:** If multi-engine cost explodes, document a single-field multi-partition design before proceeding.

---

### Phase 4 — Agency loop: sub-routines, attention budget, earned automation

**Goal:** Match Fledgeling’s “player is an SI” fantasy at the **control** layer, not the story layer.

| Concept (Fledgeling) | Core mechanism |
|---|---|
| Sub-routine | Named policy that consumes “attention” and emits intents on a schedule |
| Time contraction | When free attention is low, host may FF until surprise / gate fires |
| Attention / Warmth (lite) | Trust-web + explicit attention weights on channels; catalysts as high-η priors |
| Earned automation | Shadow → live promotion by measured competence (existing tendril + ledger patterns) |

**Exit gate:**

- [ ] Demo: player teaches a patrol sub-routine; after N successes it may auto-intend in shadow; promotion is explicit
- [ ] Surprise / rest gate can pause FF (reuse dream/rest ideas carefully — no claim of “dreaming improves play” until measured)
- [ ] CLAIMS row: automation does not reintroduce self-confounding

---

### Phase 5 — Facet kits (optional modules, not core bloat)

Only after Phases 1–3 are real. Each kit is a **package of specs + normalizers +
baselines**, not a rewrite of the engine.

| Kit | Depends on | First demo |
|---|---|---|
| `fog` | Phases 1–2 | Scout / capture-the-flag bot (gridworld evolution) |
| `attention` | Phase 2–3 | Warmth-lite: two sources, one corrupted, isolation |
| `market` | Phase 2 + self-tag | Bread-Winner-lite: recommend/trade without self-demand poison |
| `dream` | Phase 4 | Counterfactual cassette that never actuates (existing dreaming, game-clocked) |
| `scale` | Research | Nested node fold / LOD beliefs — **do not schedule until asked by Uplift** |

**Exit gate per kit:** public synthetic cassette + baseline beat + README honesty tier.

---

### Phase 6 — Product shape inside Fledgeling (integration)

**Goal:** FL-core is a dependency of a real Fledgeling host repo, not a parallel essay.

| Work | Notes |
|---|---|
| Versioned package (`umwelt-engine` or split `umwelt-fledgeling`) | Semver; 0.x until Phase 3 gate |
| Host adapter in Fledgeling tree | Game owns content; core owns belief contracts |
| Optional umweltd for tools / editors | Not required in the player binary |
| Shared CLAIMS or dual ledger | Keep DENIED culture across repos |
| Drop or quarantine quantum names at the host boundary | Internal modules may keep them |

**Exit gate:**

- [ ] Fledgeling build runs FL-core tests or a vendor copy of the fog proof
- [ ] Designers can author a small place-graph without reading THEORY.md
- [ ] One playable loop (even tiny) where turning FL-core off makes the SI dumber in a measured way

---

## 7. Recommended first spike (Phase 1 concrete)

**Name:** *Fog Corridor* (working title)

**Why this one:** Closest to what the engine already proves (gridworld + η + dissipative
roles); maps to Fledgeling “discovery / traversal / partial map”; needs no multi-scale
or economy.

**Loop:**

1. Seeded agent walks a small graph (or player issues move intents).
2. Sightings arrive with η; distant nodes stay uncertain and forget.
3. Optional: place a “claim” tendril in shadow when belief(node.safe) exceeds threshold.
4. Score: belief calibration vs ground truth; compare to “last seen freezes forever.”

**Success:** Engine-on beats freeze-baseline; designers understand the belief printout
without Bloch spheres.

**Failure (useful):** Engine-on ≈ baseline → keep self-tagging for later agency work,
but do not market umwelt as Fledgeling’s perception layer yet.

---

## 8. Milestone timeline (indicative, not a promise)

Assuming part-time dedicated work and no major substrate rewrites:

| Phase | Rough horizon | Output |
|---|---|---|
| 0 Alignment | days | ADR + spike choice |
| 1 Fog domain | 1–3 weeks | `examples/…` + CI proof + bake-off |
| 2 Host API | 2–4 weeks | Stable observe/intend/beliefs/step |
| 3 Multi-mind | 1–2 months | WorldSession + privacy assertions |
| 4 Agency | 1–2 months | Sub-routines + attention budget |
| 5 Facet kits | ongoing | Optional packages |
| 6 Host integration | overlaps 3–5 | Dependency in Fledgeling tree |

If Phase 1 fails the bake-off, **do not** spend Phase 2–4 time on packaging.

---

## 9. Risk register

| Risk | Mitigation |
|---|---|
| Quantum rhetoric blocks adoption | Plain host API; THEORY stays internal |
| House-shaped APIs leak (`zone`, solar, appliance) | Vocabulary lint + fledgeling example as counterexample |
| Multi-engine cost | Budget test at N=8, N=32 agents early in Phase 3 |
| Scope creep into Uplift/retro-sim | Hard non-goals (§3.2); separate research tracks |
| Private-data claims smuggled as Fledgeling proof | Public fixtures only for FL gates |
| Forge used as “game designer AI” too early | Forge stays optional; manual/AI-assisted specs until quality measured |
| Self-confounding returns via multi-actor bugs | Actor-keyed tags + adversarial tests (“I moved the crate; you must not learn gravity flipped”) |

---

## 10. Relationship to existing docs

| Doc | Role on this path |
|---|---|
| [CLAIMS.md](../CLAIMS.md) | Truth about what is proven *today* |
| [SPEC.md](SPEC.md) | How to declare worlds until Host API exists |
| [NEW_DOMAIN.md](NEW_DOMAIN.md) | Checklist for Phase 1 example |
| [FIELD_NOTES.md](FIELD_NOTES.md) | Dissipative-role law, adapter honesty — apply to game sensors |
| [TIME.md](TIME.md) | Cadence vs clocks; Phase 2 game tick builds here |
| [THEORY.md](THEORY.md) | Estimator ladder; do not require host authors to read it |
| [FORGE.md](FORGE.md) | Optional authoring; not on the critical path to FL-core |
| [SERVICE.md](SERVICE.md) | Optional tooling host |

---

## 11. Decision log (fill as you go)

| Date | Decision | Outcome |
|---|---|---|
| (pending) | First spike = Fog Corridor? | |
| (pending) | Host package name / monorepo vs dependency | |
| (pending) | Multi-mind: N engines vs partitioned field | |
| (pending) | Quantum names at boundary: hide / rename / keep | |

---

## 12. One-page summary

```
NOW     Belief engine extracted from a home brain; gridworld + house evidence;
        strong observe/act hygiene; multi-domain mostly sketches.

NEXT    Phase 1: public game-like fog domain + baseline bake-off.
        Phase 2: plain GameHost API.
        Phase 3: multi-mind shared world.
        Phase 4: sub-routines / attention / earned automation.

LATER   Facet kits (attention, market, dream); optional scale research.

NEVER   (as this core) Full Fledgeling game, voxels, narrative AGI, unearned
        quantum mystique as a product feature.

GATE    If it isn't in CI on synthetic Fledgeling-shaped data, it isn't FL-core yet.
```

---

*End of roadmap. Implementation PRs should link the phase gate they close; if a PR
only advances metaphor or docs without a gate, it does not move this path.*
