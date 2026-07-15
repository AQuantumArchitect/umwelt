# Path: umwelt today → a generalized Fledgeling core

*A roadmap **and status note**. This document maps how the belief-field engine
becomes the reusable **simulation intelligence substrate** under
[Fledgeling](http://www.peripheralarbor.com/fledgeling/) (`f.tryop.com`) —
without pretending it already *is* that product, and without swallowing the whole
game into one library.*

If this page and [CLAIMS.md](../CLAIMS.md) disagree on what is proven, the ledger
wins. Phases 1–5 below ship **in this monorepo** with public synthetic gates;
Phase 6 (host-repo integration) does not.

**Status snapshot (2026-07):** Phases **1–5 implemented and CI-gated** on synthetic
Fledgeling-shaped fixtures. Phase **0** (maintainer ADR) informal. Phase **6** not
started. Next product work is integration into a real Fledgeling host, not more
metaphor.

---

## 1. North star (one sentence)

A **Fledgeling core** is a domain-agnostic runtime that holds **live, partial,
honest beliefs** about nested, player-mutable worlds — so agents (and the player
as an SI) can observe, plan, act, and learn **without mistaking their own
footprint for world truth** — while game content, rendering, and facet rules live
outside the core.

umwelt is a strong candidate for the **belief / observe / act-with-echo** slice of
that core. It is not a candidate for the whole of Fledgeling.

---

## 2. Where we are (current project state)

### 2.1 What ships and is gate-pinned

| Capability | Home | Status |
|---|---|---|
| World as data (`DomainSpec`: nodes, bridges, bindings, outputs, drivers) | `umwelt.spec` | PINNED |
| Blank boot → ingest → belief ease → save/load hash | `boot`, `engine`, proofs | PINNED |
| Domain-free engine (vocabulary lint) | `tests/test_vocabulary_lint.py` | PINNED (intent) |
| Causal self-tagging (graph-derived confounding surface) | `learning/confounding`, router | Mechanism PINNED; effect sizes ORIGIN |
| Actor-keyed intent log (extends surface; does not replace it) | `learning/confounding.record_actor_intent` | PINNED (mechanism) |
| Shadow-first decisions (tendrils) | `membranes/egress` | PINNED |
| Trust-web fusion (day-1 parity; isolation when referee exists) | `foresight/trust_web` | Mechanism PINNED; live skill OWED |
| Local daemon + client + playground | `umweltd` | PINNED parity |
| Gridworld fog-of-war (weak measurement) | `examples/gridworld` | PINNED |
| **Fog corridor (FL Phase 1 domain)** | `examples/fledgeling_fog` | PINNED (synthetic) |
| **Host API game face (Phase 2)** | `umwelt.host` (`GameHost`) | PINNED (contract tests) |
| **Multi-mind session (Phase 3)** | `umwelt.host.WorldSession` | PINNED (privacy suite) |
| **Agency loop (Phase 4)** | `umwelt.host.agency_loop` | PINNED (tests + demo) |
| **Facet kits fog / attention / market / dream (Phase 5)** | `umwelt.kits.*` | PINNED (cassette baselines) |
| Experimental rant → gated spec (forge) | `umweltforge` | Pipeline PINNED; quality OWED |

### 2.2 What is measured only at the origin / private foreign world

- ~18 months live home deployment (meerkat)
- De-confound A/B (10.8× → ~79% bias cut)
- First foreign Home Assistant replay (AUCs vs persistence; dissipative-role law)
- Estimator ladder: full Belavkin **denied** as default; α-blend / persistence hard to beat

These inform design. They are **not** re-claimed as Fledgeling evidence.

### 2.3 What is explicitly not ready

- **Phase 6:** FL-core as a dependency inside a real Fledgeling host repo / playable product loop
- Multi-domain production beyond home extraction + synthetic CI domains
- Berry-phase **live** decision authority (replay demo PINNED; production OWED)
- Dream-loop / topology growth **live** wins (replay evidence only; kit never actuates)
- Plain-English authoring quality at scale
- Narrative layer, multiplayer, voxels, multi-scale physics, retro-sim
- Nested LOD / `scale` kit (explicitly unscheduled)

### 2.4 Architectural center of gravity today

```
GameHost / WorldSession          (plain face — optional for non-game apps)
   observe(η) / intend / beliefs / step_turn
                ↓
sensors / feeds  →  bindings (η, normalizers)  →  belief field (graph of roles)
                                                      ↓
actions / recs   ←  shadow tendrils ←──── self-tagging + actor_id echo on learn
```

Time for FL demos is **tick / turn** via `DriverSpec` + host `step` / `step_turn`
(not solar). Topology remains a **declared tree + bridges**. Internal qubits stay
invisible at the host boundary (`Belief.value` + `Belief.confidence`).

---

## 3. What “Fledgeling core” means (target, scoped)

Fledgeling (public vision) is a family of facets and a long-horizon sandbox:

| Facet / theme | Core need (simulation intelligence) | FL-core status in-repo |
|---|---|---|
| Fog / discovery / Glyph-class puzzles | Partial observation, confidence, scouting as buying η | **Phase 1 + fog kit** |
| Warmth / attention | Directed awareness, source reliability, catalysts | **attention kit** (lite) |
| Bread Winner / economics | Stock / scarcity; no self-demand poison | **market kit** (lite) |
| Anteciplace / retro-sim | Plausible prior context | out of core v1 |
| Uplift / multi-scale voxels | Nested abstraction | out of core v1 (`scale` kit not scheduled) |
| Lucid / dreams | Counterfactual replay | **dream kit** (never actuates) |
| Gerbil / SI progression | Agency, sub-routines, competence gating | **Phase 4 agency loop** |

### 3.1 In-scope for “generalized Fledgeling core” (this roadmap)

Call this **FL-core**: the reusable substrate any facet can host.

1. **Belief graph runtime** — nodes/roles/bridges as data; blank or seeded profiles ✅
2. **Partial observation law** — every report has η; silence ≠ false certainty ✅
3. **Self-action hygiene** — act → world change → learn only through tagged echo ✅ (+ actor_id)
4. **Shadow → earned autonomy** — decide visibly; dispatch only when competence allows ✅ (host + agency loop)
5. **Multi-agent / multi-SI observations** — many observers, shared world, private beliefs ✅ (`WorldSession`)
6. **Game-time membrane** — ticks, turns, and time-contraction as first-class cadence ✅ (host step + agency FF/surprise)
7. **Public baselines** — synthetic Fledgeling-shaped streams in CI ✅
8. **Thin host API** — library-first; daemon optional; plain boundary ✅ (`umwelt.host`)

### 3.2 Explicitly out of scope for FL-core

- Rendering, audio, input, networking
- LLM dialogue and content generation (may sit **beside** the core)
- Voxel physics, geology, plant simulation
- Full retro-sim / history generation
- “The game designs itself”
- Claiming consciousness, AGI, or a drop-in Dwarf Fortress brain

Facets compose **on top of** FL-core. Kits supply specs + baselines; they do not
rewrite the engine.

---

## 4. Gap map (current → FL-core)

```
                    umwelt (pre–FL path)              FL-core in this repo (Phases 1–5)
                    ───────────────────              ────────────────────────────────
World shape         house-like tree + bridges        + corridor / place graphs (fog)
Observation         sensors / normalizers            + scout channels, η, host.observe
Action              device tendrils + shadow         + Intent → Decision (shadow|live)
Learning            online fiber + confound gate     + actor_id intent log (graph surface stays)
Time                drivers + ingest_hold            + tick driver + step / step_turn + FF pause
Agents              one engine ≈ one mind            + WorldSession (N engines, masks)
API surface         DomainSpec + engine.ingest       + GameHost plain face
Evidence            home + gridworld + private HA    + public fledgeling fixtures in CI
Language            Bloch / η / tendril internal     + Belief(value, confidence) at boundary
```

| Gap | Severity | Status |
|---|---|---|
| Single-mind assumption | High | **Addressed** — `WorldSession` map of private engines |
| Shared world, private beliefs | High | **Addressed** — classical `GroundState` + per-mind fields + channel masks |
| Actor identity on actions | High | **Addressed** — `record_actor_intent` / actor-keyed confounded_now (extends graph surface) |
| Game clock / turn membrane | High | **Addressed** — tick `DriverSpec` + host `step` / `step_turn`; agency FF/surprise |
| Quantum metaphor at game API | Medium | **Addressed** — default beliefs face hides Bloch |
| Multi-engine cost | Medium | **Measured** — N=8 / N=32 probe on `WorldSession.measure_cost` (partition design not required yet) |
| Procedural topology mutation | Medium | Still experimental / product-owned |
| Abstraction / LOD of belief | Medium | Deferred (`scale` kit unscheduled) |
| Retro-sim / prior synthesis | High (later) | Out of v1 |
| Host-repo integration | High (product) | **Phase 6 — open** |
| Content / narrative | N/A | Never the engine’s job |

---

## 5. Design principles for the path

1. **Steal contracts, don’t force the house shape.** Partial observation, self-tagging, shadow autonomy, blank-boot proofs stay. Room/sensor vocabulary does not.
2. **Public synthetic fixtures or it didn’t happen.** Fledgeling gates run offline in CI with no private meerkat/HA data.
3. **Beat a dumb baseline.** Persistence, last-wins, and “always believe the player’s last act” are the control arms. Report honest losses too.
4. **Library-first.** `umweltd`/forge are optional hosts; the game embeds the library.
5. **Plain face, rich guts.** External types: `Belief`, `Intent`, `Observation`, `Decision`. Internal qubits optional and invisible.
6. **One facet spike before generalization.** Fog Corridor first; kits after.
7. **Ledger every promotion.** New FL claims get [CLAIMS.md](../CLAIMS.md) rows with the same DENIED discipline.

---

## 6. Phased path

Each phase ends with a **gate**: demos + tests + a baseline bake-off. Do not start
the next phase until the gate is green or the phase is explicitly **parked** with a
reason.

### Phase 0 — Alignment (no new architecture)

**Goal:** Agree what FL-core is *for* and what the first bake-off is.

| Work | Status |
|---|---|
| This document | Written; spike = Fog Corridor |
| Vertical spike named | **Done** — §7 / `examples/fledgeling_fog` |
| Belief questions in spike README | **Done** |
| Freeze non-goals | §3.2 still the freeze list |
| Maintainer written ADR | **Informal** — this status section stands in until dual-repo sign-off |

**Exit:** Spike chosen and non-goals frozen. Formal dual-maintainer ADR still optional.

---

### Phase 1 — Fledgeling-shaped domain ✅ SHIPPED

**Goal:** Prove the **existing** engine can host a game-like world without forks.

**Shipped:** [`examples/fledgeling_fog/`](../examples/fledgeling_fog/)

- Nodes: `place_*` corridor + `agent_near` / `safe` roles
- Observation: `scout_{place}` with η; tick driver (`period_s=60`, not solar)
- Action: `claim_safe` shadow output
- Demo: JSON timeline via host API — `python examples/fledgeling_fog/demo.py`
- Bake-off: `python examples/fledgeling_fog/bakeoff.py` (engine vs freeze; report honest metrics)
- Proof: `proofs/fledgeling_fog_blank.py` (happy path uses `GameHost`, not raw `engine.ingest`)

**Exit gate:**

- [x] `python -m umwelt.spec.validate examples.fledgeling_fog.world:FOG_SPEC` green
- [x] Proof: blank boot, bindings drive field, beliefs track ground truth, save/load
- [x] README states what is *not* proven (no multi-agent, no narrative)
- [x] Side-by-side metrics vs freeze in bake-off output (accuracy win observed; MAE may not win — printed honestly)

---

### Phase 2 — Host API: game face over the engine ✅ SHIPPED

**Goal:** A boundary a Fledgeling host can depend on without speaking DomainSpec daily.

```
GameHost  (umwelt.host)
  register_world(spec)
  observe(observer_id, channel, value, confidence, t)
  observe_many(...)
  intend(actor_id, intent) → Decision (shadow | live)
  beliefs(observer_id, query) → {node.role → Belief(value, confidence)}
  step(t) / step_turn(n)
  save / load
```

| Work | Shipped as |
|---|---|
| Thin host package | `src/umwelt/host/` (`api.py`, `session.py`, `agency_loop.py`) |
| Intent → tendril / shadow | `GameHost.intend` |
| Observation → binding + η | `observe` / `observe_many` (η≤0 no-op) |
| Hide Bloch in default face | `Belief.value = (z+1)/2`, `confidence = \|r\|` |
| Turn cadence | `step` / `step_turn` |

**Exit gate:**

- [x] Phase 1 demo happy path uses host API (`tests/test_host_api.py` structural check)
- [x] Contract tests: η=0 no-op; shadow no world side effect; kill/reload
- [x] Vocabulary lint still green

---

### Phase 3 — Multi-mind: shared world, private umwelten ✅ SHIPPED

**Goal:** N agents, one ground truth (or one shared classical ground), N belief fields.

```
                ┌── Agent A field (private GameHost)
 Ground / scene ┼── Agent B field (private GameHost)
                └── … (channel masks per mind)
         ↑ observations (partial, masked)
         ↓ intents (tagged by actor_id)
```

**Shipped:** `WorldSession` — classical `GroundState` + map of private engines;
per-observer `channel_mask`; `record_actor_intent` + `actor_confounded_now` extend
the graph-derived confounding surface. Cost probe: `measure_cost(n_agents)`.

**Exit gate:**

- [x] Two-agent corridor: asymmetric observation → beliefs diverge; mask rejects foreign channels
- [x] A’s action does not inflate B without an observation path
- [x] Shared-global-belief cheat loses privacy-of-mind suite (`tests/test_multimind_privacy.py`)

**Park criterion:** Multi-engine cost measured at N=8 / N=32; partition design **not**
required yet (privacy gates hold).

---

### Phase 4 — Agency loop: sub-routines, attention budget, earned automation ✅ SHIPPED

**Goal:** Match Fledgeling’s “player is an SI” fantasy at the **control** layer.

| Concept | Shipped mechanism |
|---|---|
| Sub-routine | `SubRoutine` — named policy, schedule, attention cost |
| Time contraction | `TimeContraction` — FF when attention low; **pause on surprise / rest** |
| Earned automation | `PromotionGate.min_successes` gates **shadow auto-intend**; **live** only after explicit `promote()` |
| Self-confound hygiene | Auto intents always carry `actor_id`; shadow default → no live dispatch |

**Demo:** `python examples/fledgeling_fog/agency_demo.py`  
**Tests:** `tests/test_agency_loop.py` (includes: 1 success does **not** auto-intend when N=3)

**Exit gate:**

- [x] Patrol sub-routine auto-intends in **shadow** only after N successes; promotion explicit
- [x] Surprise / rest gate pauses FF / agency tick
- [x] Automation does not reintroduce untagged self-confounding (tests pin; CLAIMS row added)

---

### Phase 5 — Facet kits ✅ SHIPPED (lite kits; `scale` unscheduled)

Optional modules under `src/umwelt/kits/` — specs + baselines + README honesty, not engine rewrites.

| Kit | Home | Gate |
|---|---|---|
| `fog` | `umwelt.kits.fog` | Scout cassette + freeze baseline (`run_fog_baseline`) |
| `attention` | `umwelt.kits.attention` | Warmth-lite: two sources, one corrupted; isolation beats naive |
| `market` | `umwelt.kits.market` | Bread-Winner-lite: shadow recommend + actor tag vs self-demand poison |
| `dream` | `umwelt.kits.dream` | Counterfactual cassette on clone host; **zero** live dispatch; live field untouched |
| `scale` | — | **Not scheduled** until Uplift asks |

**Exit gate per kit:** public synthetic cassette + baseline comparison + README honesty tier — **met** (`tests/test_facet_kits.py`).

---

### Phase 6 — Product shape inside Fledgeling (integration) ⬚ OPEN

**Goal:** FL-core is a dependency of a real Fledgeling host repo, not only this monorepo.

| Work | Notes |
|---|---|
| Versioned package (`umwelt-engine` or split `umwelt-fledgeling`) | Semver; still 0.x |
| Host adapter in Fledgeling tree | Game owns content; core owns belief contracts |
| Optional umweltd for tools / editors | Not required in the player binary |
| Shared CLAIMS or dual ledger | Keep DENIED culture across repos |
| Drop or quarantine quantum names at the host boundary | Host face already plain; internal modules keep substrate names |

**Exit gate:**

- [ ] Fledgeling build runs FL-core tests or a vendor copy of the fog proof
- [ ] Designers can author a small place-graph without reading THEORY.md
- [ ] One playable loop (even tiny) where turning FL-core off makes the SI dumber in a measured way

---

## 7. First spike (Phase 1 concrete) — landed

**Name:** *Fog Corridor* — [`examples/fledgeling_fog/`](../examples/fledgeling_fog/)

**Loop (as shipped):**

1. Seeded agent walks a 6-place corridor.
2. Scout sightings arrive with confidence; host path only on the demo happy path.
3. Optional shadow `claim_safe` tendril.
4. Score: bake-off vs freeze baseline; blank-slate proof tracks occupancy.

**Success criterion (product):** Engine-on beats freeze on place-argmax accuracy in the
public bake-off (MAE may still favor freeze — metrics are printed honestly; do not
paper over).

---

## 8. Milestone timeline (indicative)

| Phase | Status | Output |
|---|---|---|
| 0 Alignment | Informal | This doc + Fog Corridor choice |
| 1 Fog domain | **Done** | `examples/fledgeling_fog` + proof + bake-off |
| 2 Host API | **Done** | `umwelt.host.GameHost` |
| 3 Multi-mind | **Done** | `WorldSession` + privacy suite |
| 4 Agency | **Done** | `agency_loop` + agency demo |
| 5 Facet kits | **Done (lite)** | `umwelt.kits.{fog,attention,market,dream}` |
| 6 Host integration | **Open** | Dependency in Fledgeling tree |

---

## 9. Risk register

| Risk | Mitigation / note |
|---|---|
| Quantum rhetoric blocks adoption | Plain host API shipped; THEORY stays internal |
| House-shaped APIs leak | Vocabulary lint + fledgeling example as counterexample |
| Multi-engine cost | Measured N=8/N=32; watch if product needs dozens of rich fields |
| Scope creep into Uplift/retro-sim | Hard non-goals (§3.2); `scale` unscheduled |
| Private-data claims as Fledgeling proof | Public fixtures only for FL gates |
| Self-confounding via multi-actor bugs | Actor tags + privacy / agency tests |
| Packaging chrome without bake-off | Phase 1 bake-off ran before kits; continue DENIED discipline |

---

## 10. Relationship to existing docs

| Doc | Role on this path |
|---|---|
| [CLAIMS.md](../CLAIMS.md) | Truth about what is proven *today* (includes FL rows) |
| [SPEC.md](SPEC.md) | How to declare worlds (still the authoring surface under the host) |
| [NEW_DOMAIN.md](NEW_DOMAIN.md) | Checklist; gridworld + **fledgeling_fog** as templates |
| [FIELD_NOTES.md](FIELD_NOTES.md) | Dissipative-role law, adapter honesty — apply to game sensors |
| [TIME.md](TIME.md) | Cadence vs clocks; host tick builds here |
| [THEORY.md](THEORY.md) | Estimator ladder; host authors need not read it |
| [FORGE.md](FORGE.md) | Optional authoring; not on the critical path to FL-core |
| [SERVICE.md](SERVICE.md) | Optional tooling host |

### Code map (quick)

| Concern | Path |
|---|---|
| Fog domain + demos | `examples/fledgeling_fog/` |
| Host face | `src/umwelt/host/` |
| Facet kits | `src/umwelt/kits/` |
| Fog blank proof | `proofs/fledgeling_fog_blank.py` |
| Contract / privacy / agency / kits tests | `tests/test_host_api.py`, `test_multimind_privacy.py`, `test_agency_loop.py`, `test_facet_kits.py` |

### Commands (from repo root)

```bash
python -m umwelt.spec.validate examples.fledgeling_fog.world:FOG_SPEC
python examples/fledgeling_fog/demo.py
python examples/fledgeling_fog/bakeoff.py
python examples/fledgeling_fog/agency_demo.py
python -m pytest proofs/fledgeling_fog_blank.py tests/test_host_api.py \
  tests/test_multimind_privacy.py tests/test_agency_loop.py tests/test_facet_kits.py -q
```

---

## 11. Decision log

| Date | Decision | Outcome |
|---|---|---|
| 2026-07 | First spike = Fog Corridor | Shipped `examples/fledgeling_fog` |
| 2026-07 | Host package name | `umwelt.host` (not `umwelt.fledgeling`) |
| 2026-07 | Multi-mind design | **N engines** per mind + channel masks; cost measured; partition deferred |
| 2026-07 | Quantum names at boundary | **Hide** — calibrated value + confidence only on default face |
| 2026-07 | Auto-intend threshold | Shadow auto-intend only after `PromotionGate.min_successes` (not after 1 success); live requires explicit `promote()` |

---

## 12. One-page summary

```
NOW     Belief engine + gridworld/house evidence + FL-core Phases 1–5 in-repo:
        fog corridor, GameHost, WorldSession multi-mind, agency loop, four facet kits.
        All FL gates are public synthetic CI — not origin/HA effect sizes.

NEXT    Phase 6: real Fledgeling host dependency; playable loop where FL-core off
        makes the SI measurably dumber; optional package split / semver 0.x polish.

LATER   scale / LOD research (only if Uplift asks); richer facet content in game tree.

NEVER   (as this core) Full Fledgeling game, voxels, narrative AGI, unearned
        quantum mystique as a product feature.

GATE    If it isn't in CI on synthetic Fledgeling-shaped data, it isn't FL-core yet.
        Phases 1–5 meet that bar in this monorepo; product integration does not yet.
```

---

*Implementation that closes a phase gate should update §2, §6 checkboxes, §11, and
CLAIMS.md in the same change. Docs-only metaphor without a gate does not move this path.*
