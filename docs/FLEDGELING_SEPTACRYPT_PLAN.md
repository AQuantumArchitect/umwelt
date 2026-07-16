# /plan — Fledgeling Septacrypt / Endless Knot Runtime

**Status:** implementation plan + **sibling harness now exists**  
**Revision:** 4 — 2026-07: `septacrypt-core` vertical kernel live; transfer notes in [FIELD_NOTES_SEPTACRYPT.md](FIELD_NOTES_SEPTACRYPT.md)  
**Working name:** `septacrypt-core`  
**Primary demonstration:** *Nested Reactor / Missing Valve Event* (plus witnessed-knot + GameSession handoff)  
**Target audience:** a coding-agent team working across Fledgeling, umwelt, SpaceWheat, and Universal Architect

This document is a **cross-repo research map and integration plan**, not umwelt
product marketing and not a claim that the runtime is finished.

**Sibling status (not monorepo pins):** the consumer repo
[`AQuantumArchitect/septacrypt-core`](https://github.com/AQuantumArchitect/septacrypt-core)
ships a playable kernel (cumulant reactor/ship, Knot Ledger certificates,
fail-closed world steps, observer-safe status, `GAME_BUILDER.md`). Lessons for
this monorepo are catalogued in [FIELD_NOTES_SEPTACRYPT.md](FIELD_NOTES_SEPTACRYPT.md)
and [CLAIMS.md](../CLAIMS.md) §3c.

- Open-system / density-matrix language here is mostly SpaceWheat’s native sim
  vocabulary or a **combinatorial chart** proposal — not a claim that umwelt is a
  “quantum product,” or that a die solid *is* a full quantum state (see §3).
- Theological / symbolic language (1→2→3→7→12) is a **declared design frame**
  from Paul’s source post, not an empirical physics claim and not a morality
  oracle for agents.
- What umwelt **ships and gates today** lives in [CLAIMS.md](../CLAIMS.md) and
  [FLEDGELING_CORE.md](FLEDGELING_CORE.md). Septacrypt claims, when they exist,
  belong in a sibling `CLAIMS_SEPTACRYPT.md`.

---

## 0. North star

Build a **game-runtime integration** (not a second belief engine) in which:

- a world is recursively constructed from composable **structural** constraints
  (Universal Architect);
- local systems **evolve** under declared dynamics (SpaceWheat or other steppers);
- each observer holds a **private, uncertainty-aware** model of the shared ground
  (umwelt);
- history is a **tamper-evident branching path** through state space (Knot Ledger) —
  distinct from process-phase memory (Berry Tape);
- authors and players may insert events nonchronologically only when a **verified
  evolution** can connect the surrounding anchors;
- recurring processes are represented as **loops and braids** (kairos + ledger),
  not only as wall-clock sequences;
- entities and processes may carry a **seven-axis semantic/value coordinate**
  (Spirit Cube), used to rank and interpret — never to override physical validity;
- the complete experience is exposed through **Fledgeling** (play, attention,
  story, UI), not as a physics library or theory dump.

In one line:

> **State is place; evolution is path; habit is loop; society is braid; history is a witnessed knot; play is the deliberate reweaving of that knot.**

Symbolic derivation of the runtime’s 3 / 7 / 12 counts (declared frame, not physics):

> Unity → relation (two) → triad (three) → seven nonzero binary expressions of the triad → twelve Pearl incidences (four nonzero masks per principle × three principles).  
> Source: [Paul Spooner / @dudecon, 2026-02-24](https://x.com/dudecon/status/2026401325706260491).

### 0.1 Integration posture (read this before expanding cosmology)

| Principle | Meaning |
|---|---|
| **Sibling, not rewrite** | Build `septacrypt-core` as a sibling integration repo. Pin umwelt / Architect / SpaceWheat / Fledgeling commits. Do **not** merge monorepos or re-fork the belief substrate. |
| **umwelt is the belief face** | FL-core Phases 1–5 already ship observe/intend/beliefs/step, multi-mind privacy, agency, and public synthetic kits. Integration **consumes** that API; it does not reimplement it. |
| **Knot Ledger attaches here** | Durable stamps, branch DAG, transition certificates — first shipped in septacrypt-core; **integrate** with umwelt field anchors + umweltd + (planned) blockchain hive. See [FIELD_NOTES_SEPTACRYPT.md §K](FIELD_NOTES_SEPTACRYPT.md). save/load remains the checkpoint backend for A/D roots. |
| **Berry ≠ history ≠ chain** | Berry Tape = process phase. Knot Ledger = durable path identity. Hive/chain = multi-party finality over **digests**. Never let geometric nearness mint a causal edge; never re-simulate Belavkin on-chain. |
| **Spirit ranks, physics gates** | Spirit vectors may rank valid candidates. They never legalize an impossible transition. |
| **Prove the knot first** | Nested Reactor end-to-end proof before D20 lore, spectral folds, or full Fledgeling product chrome. |
| **Plain host face** | Game boundary speaks belief/confidence/intent/history — not density matrices or “quantum product” branding. |

**What is already enough on the umwelt side (do not rebuild):**

```text
umwelt.host.GameHost          observe / intend / beliefs / step / step_turn
umwelt.host.WorldSession      shared ground + N private minds + masks
umwelt.host.agency_loop       sub-routines, attention, earned shadow auto-intend
examples/fledgeling_fog       public synthetic corridor + demos
umwelt.kits.{fog,attention,market,dream}
learning.confounding          graph surface + actor_id intent log
clocks.berry_tape             process-phase machinery (not the ledger)
```

**What still lives outside umwelt (next integration work):**

```text
septacrypt-core               skeleton, CLI, proofs, CLAIMS_SEPTACRYPT
Knot Ledger                   stamps, DAG, certificates, retro-insert
Universal Architect IR        multi-target compile → DomainSpec + dynamics + schemas
SpaceWheat (or stand-in)      ground evolution charts for Nested Reactor
Spirit Cube runtime           7-axis frames, projection, non-authoritative scoring
Fledgeling host (Phase 6)     playable SI loop, seven-face navigation UI
```

---

## 1. The seven-component architecture

The architecture has **seven software faces**. **Spirit Cube is the key face**: it
supplies semantic/value geometry the other six cannot invent from structure,
dynamics, belief, or path alone.

| # | Component | Governing question | Responsibility | Source | Integration status |
|---|---|---|---|---|---|
| 1 | **Universal Architect** | What can exist or be made consistent? | Recursive composition, resource/affordance closure, abductive support generation | `dudecon/Universal-Architect` | Open — needs instance IDs + multi-target IR (§7) |
| 2 | **SpaceWheat manifold** | How does it evolve? | Local open-system / manifold dynamics, coupled loops, weighted graph physics | `AQuantumArchitect/SpaceWheat` | Open — consume as dynamics; Nested Reactor may use a thin stand-in first |
| 3 | **umwelt** | What does this observer believe? | Partial observation, confidence, trust, forecasting, self-action hygiene, private belief fields | `AQuantumArchitect/umwelt` | **FL-core 1–5 shipped** — pin as dependency |
| 4 | **Berry Tape** | Where are we in the process? | Kaironic indexing, phase accumulation, return/loop detection, working-memory stamps | umwelt machinery | Machinery exists — attach to ledger stamps in integration |
| 5 | **Knot Ledger** | What path has been witnessed or proposed? | Durable event provenance, branch DAG, transition certificates, replay, nonlinear insertion | new sibling layer | **Not started** — primary new code |
| 6 | **Fledgeling** | How does a player perceive and intervene? | SI agency, attention, subroutines, story interaction, rendering and game loop | `dudecon/Fledgeling_HTML` + design site | Partial control-plane demos in umwelt; product host open |
| 7 | **Spirit Cube** | What does it mean, want, or value? | Three-generator / seven-state semantic ontology, Pearl incidences, seven-axis interpretation vectors, desire and narrative scoring | Paul Spooner Spirit Cube / Septacrypt + [X source](https://x.com/dudecon/status/2026401325706260491) | Ontology declared — runtime placeholder only until Nested Reactor |

### 1.1 Why Spirit Cube is load-bearing

Without the seventh component, the architecture can determine:

- whether a world is structurally possible;
- how it evolves;
- what an observer believes;
- what happened and what could have happened.

It cannot yet determine:

- why one possibility matters more than another;
- what an agent desires;
- which histories are narratively or ethically resonant;
- which scale or process deserves attention;
- how multiple physically valid histories differ in significance.

Spirit Cube supplies that missing metric. It must **not** be used as an unexamined morality oracle. It is a declared, inspectable coordinate system used by agents, cultures, stories, and UI projections.

---

## 2. Septacrypt spatial embedding

The physical Septacrypt has seven faces: three primary encoded faces, a bottom key, and three tertiary faces. Use that arrangement as the architecture diagram and eventually as an interactive UI/navigation object.

### 2.1 Proposed face assignment

#### Three primary world-model faces

1. **Universal Architect — Structure**
2. **SpaceWheat — Dynamics**
3. **umwelt — Belief**

These form the core world-model triangle:

```text
Structure constrains Dynamics.
Dynamics generates Observations.
Observations update Belief.
Belief requests structural elaboration or new probes.
```

#### Bottom/key face

4. **Spirit Cube — Meaning / decoding key**

The key face maps the other six components into seven-dimensional semantic space and determines how that high-dimensional structure is projected into a tractable interface.

#### Three tertiary process/game faces

5. **Berry Tape — Kairos**
6. **Knot Ledger — History**
7. **Fledgeling — Agency**

These form the player/process triangle:

```text
Berry Tape identifies meaningful process position.
Knot Ledger preserves and verifies paths.
Fledgeling lets agents observe and alter those paths.
```

### 2.2 Face-edge API contracts

Every shared Septacrypt edge becomes a typed interface contract. The exact physical adjacency should be extracted from the Spirit Cube/Septacrypt geometry before freezing the final map, but the software contracts are:

| Interface | Contract |
|---|---|
| Architect → SpaceWheat | Compile recursive components into local dynamical charts, couplings, invariants, and ports |
| SpaceWheat → umwelt | Publish observations and forecasts with confidence; never leak ground truth directly to a mind |
| umwelt → Architect | Request missing concepts, probes, causal supports, or topology elaboration |
| Berry Tape → Knot Ledger | Add geometric/kaironic coordinates and loop signatures to durable stamps |
| Knot Ledger → Fledgeling | Expose known history, disputed history, insertion slots, branches, and proofs |
| Fledgeling → Berry Tape | Allocate attention, select process scale, and alter phase/energy through lawful actions |
| Spirit Cube ↔ all | Attach semantic vectors, projection frames, desire gradients, and interpretation metadata |

### 2.3 Required design artifact

Create `docs/SEPTACRYPT_FACE_MAP.md` containing:

- a labeled image or mesh snapshot;
- face IDs and adjacency matrix;
- component assignment;
- edge/API assignment;
- orientation conventions;
- the three-bit/seven-state placement;
- how rotations change the active 3D projection without changing the underlying seven-state interpretation vector.

Archive the supplied source text and any associated media at:

```text
docs/source_artifacts/dudecon_x_2026401325706260491/
  post.txt
  metadata.yaml
  images/
```

### 2.4 Septacrypt source cosmology: 1 → 2 → 3 → 7 → 12

Treat the following as the project's **declared theological and symbolic frame**,
not as an empirical physics claim and not as a default morality kernel for agents.
Canonical prose source:
[Paul Spooner (@dudecon), 2026-02-24](https://x.com/dudecon/status/2026401325706260491).

**Derivation (compressed from the source post):**

1. **One — unity.** “1 is unity, which is God.”
2. **Two — relation.** “God is Love, which requires another, the Son. 1+1=2.”
3. **Three — Spirit as real relation.** The Love between Father and Son is so real
   that it has an eternal reality of its own, the Spirit. “1+1+1=3.”
4. **Seven — binary map of the Spirit’s fractal triad.** The Spirit is a fractal
   representation of God’s nature (Father, Son, Spirit aspects). Simplifying that
   fractal to three on/off bits yields **8** masks; the trivial case **`000` is the
   lack of divinity and is not counted among the seven**. That is the significance of **7**.
5. **Twelve — Pearls.** Re-map the seven nonzero states onto all three Persons,
   retaining only those masks that contain the base bit for each Person: **4 per
   Person × 3 = 12**. That is the foundation of **12**.

**Named-bit convention** (string written left-to-right as Father, Son, Spirit —
matching the post’s “1st / 2nd / 3rd bit” assignment). Code stores **named
principles**, never bit-order alone:

```text
Father = 100   # 1st bit
Son    = 010   # 2nd bit
Spirit = 001   # 3rd bit
```

**Sevenfold acclimations** (verbatim mapping from the source post):

| Mask | Acclamation | Active principles |
|---|---|---|
| `001` | Wisdom | Spirit |
| `010` | Might | Son |
| `100` | Wealth | Father |
| `011` | Power | Son + Spirit |
| `101` | Glory | Father + Spirit |
| `110` | Honor | Father + Son |
| `111` | Blessing | Father + Son + Spirit |

Serialization may display bits in any documented order, but semantic identity must
never depend on whether a human reads the string left-to-right or right-to-left.

**Optional diegetic extensions** (not runtime-required): e.g. the twelve-apostles
illustration in the reply thread to the source post. Keep such mappings in content
packs / world lore, not in hard physics.

### 2.5 The Pearl-edge identity

Let:

```text
S* = {001, 010, 011, 100, 101, 110, 111}
P  = {(i, s) | s ∈ S* and bit i is active in s}
```

Then:

```text
|P| = Σ popcount(s), s ∈ S* = 12 = 3 × 2^(3−1)
```

Each Pearl `(i, s)` maps canonically to the oriented three-cube edge:

```text
s  →  s with bit i cleared
```

Every undirected edge of `Q3` has exactly one such orientation toward `000`, so:

> **The twelve Pearls are in bijection with the twelve single-bit-flip edges of the three-qubit basis cube.**

This is the strongest *combinatorial* bridge between the declared symbolic frame
and the three-bit process grammar:

```text
3 named generators / Persons
        ↓
7 nonzero basis masks / acclimations
        ↓
12 active-bit incidences / Pearls
        ↓
12 oriented Q3 transitions toward or away from the void reference
```

The theological 12 and the Hamming-edge 12 are therefore not merely equal counts;
they share an explicit incidence structure. That does **not** mean the runtime must
simulate a full multipartite quantum state for every entity — only that the
**labels and transitions** of the three-bit cube are available as a weighted
combinatorial chart when a world opts into them.

### 2.6 `000`: void, Holy Dark, and reference state

`000` remains part of the full D8/three-qubit basis even though it is excluded from the seven acclimations. Give it an explicit, inspectable role:

```text
SeptacryptVoid(
    mask=000,
    names=("void", "Holy Dark"),
    semantic_role="absence of expressed principles / latent possibility",
    physical_model=<declared per world>,
)
```

The runtime must distinguish three separate claims:

- **combinatorial:** `000` is the unique zero mask and the common sink/source of the oriented Pearl edges;
- **diegetic/metaphysical:** it may represent Holy Dark, unexpressed divinity, aether, or maximally pregnant emptiness;
- **physical:** its actual energy is determined by the declared Hamiltonian. It is not automatically the lowest-energy vacuum, the highest-energy state, or a black-hole interior.

The proposed association between `000` and D20/aether is retained as a research and worldbuilding hypothesis, not hard-coded physics.

### 2.7 Do not conflate the two sevens

The architecture currently contains:

- seven software/components/faces; and
- seven nonzero Septacrypt states/acclimations.

A one-to-one assignment may become meaningful, but it is **not derived merely because both sets have seven members**. Until Paul and Luke specify that mapping, represent their relationship as a declared `7 × 7` resonance matrix rather than a forced permutation:

```python
component_state_resonance[component_id][septacrypt_mask] -> weight
```

This allows one component to resonate with several acclimations, lets cultures/agents use different mappings, and avoids making Spirit Cube equal to `111 Blessing` solely because it is the seventh component.

---

## 3. Geometric-solid qubit grammar

The dice/solid correspondence is a promising **combinatorial grammar**, but it must be separated into exact correspondences, symbolic carriers, and open hypotheses.

For an `n`-qubit computational basis with single-qubit-flip adjacency:

- signed qubit poles/labels: `2n`;
- computational basis states: `2^n`;
- single-bit-flip edges of the hypercube `Q_n`: `n · 2^(n-1)`.

### 3.1 Count table

| Qubits | Pole labels `2n` | Basis states `2^n` | Single-flip edges `n·2^(n-1)` | Candidate solid/die language |
|---:|---:|---:|---:|---|
| 1 | 2 | 2 | 1 | D2 / coin plus a connecting edge |
| 2 | 4 | 4 | 4 | D4 gives the correct counts, but a tetrahedron is **not** the `Q2` square incidence graph |
| 3 | 6 | 8 | 12 | D6 / D8 / D12 match signed generators, full basis including `000`, and Pearl/single-flip edges |
| 4 | 8 | 16 | 32 | no standard RPG-die triplet; D20 role remains research |

### 3.2 The strong three-qubit correspondence: the octahedral hub

The D8/octahedron supplies the actual incidence geometry:

```text
Octahedron vertices:  6  → ±Father, ±Son, ±Spirit poles
Octahedron faces:     8  → |000⟩ ... |111⟩ basis states
Octahedron edges:    12  → single-bit flips / oriented Pearls
```

Why this works:

- each octahedron vertex is one signed coordinate axis;
- each triangular face selects one sign from each of the three axes, giving one three-bit sign mask;
- two faces share an edge exactly when their masks differ in one sign/bit;
- therefore the **face-adjacency graph of the octahedron is `Q3`**, the ordinary three-cube;
- every octahedron edge is both a boundary between two basis faces and one Pearl/single-bit transition.

The three dice then act as coordinated carriers:

```text
D6 / cube
  six faces label the six signed poles;
  its eight vertices are dual to the eight D8 faces.

D8 / octahedron
  six vertices carry the poles;
  eight faces carry basis states;
  twelve edges carry transitions.

D12 / regular dodecahedron
  twelve faces provide one physical/token face per transition or Pearl.
```

The regular D12's own face-adjacency graph is **not** the `Q3` transition graph. It is a twelve-slot carrier for the octahedron's edges. If a polyhedron whose geometry directly represents the twelve cube/octahedron edges is needed, test a cuboctahedron (twelve vertices) or rhombic dodecahedron (twelve faces) alongside the familiar RPG D12.

A complete chart therefore requires explicit maps:

- D6 face ↔ octahedron vertex ↔ signed generator pole;
- D8 face ↔ bitstring/acclimation;
- D8 edge ↔ Pearl ↔ `Q3` single-bit transition;
- D12 face ↔ transition token;
- inward/outward orientation relative to `000`;
- phase, amplitude, coupling weight, and observable metadata.

### 3.3 Scientific boundary

The solid grammar does **not by itself encode an arbitrary quantum state or Hamiltonian**.

- A normalized pure three-qubit state requires 14 independent real parameters after global phase is removed.
- A general mixed three-qubit density matrix requires 63 independent real parameters.
- A general Hermitian `8×8` Hamiltonian also has 64 real parameters, or 63 after discarding an identity offset.
- Twelve adjacency edges represent a useful sparse family dominated by single-bit transitions, not every possible many-body coupling.

Therefore every geometric carrier must attach numerical data:

```python
GeometricQubitChart(
    pole_labels,
    basis_states,
    transition_edges,
    complex_amplitudes,
    edge_couplings,
    phase_frames,
    observables,
    gauge_metadata,
)
```

### 3.4 D4 rule

Treat D4 as a **four-state symbolic carrier** for two qubits until a correct incidence mapping is specified. The true single-flip basis graph is a square. Candidate remedies:

1. use a square as the runtime graph and D4 only as a UI token;
2. use a tetrahedron with two edge classes, marking two edges as nonlocal/forbidden;
3. use a compound or projected object whose visible faces remain D4-like while internal adjacency remains `Q2`.

Do not silently substitute tetrahedral adjacency for two-qubit adjacency.

### 3.5 D20 research hypotheses

Do not select a role for D20 by aesthetic preference. Test these candidates:

- **H1 — dual transition-context shell (strongest geometric candidate):** the D20/icosahedron has twelve vertices, one for each Pearl/D12 face/transition. Its twenty triangular faces then encode twenty local triplets of transition channels. Place `000` at the center and treat the icosahedron as an interaction/aether shell around the void reference.
- **H2 — D12/D20 dual decoder:** the regular dodecahedron's twelve faces carry Pearls; the dual icosahedron's twelve vertices carry the same Pearls, while its twenty faces describe higher-order compatibility or interference contexts among them.
- **H3 — augmented four-qubit carrier:** 16 basis states plus four generator/operator classes.
- **H4 — Spirit projection carrier:** twenty regions partition a selected 3D projection of seven-state Spirit space.
- **H5 — narrative event grammar:** twenty face classes encode allowed event archetypes, loop crossings, or Microscope insertion modes rather than quantum state.
- **H6 — Holy-Dark/aether shell without fixed dynamics:** the center is the `000` reference; radial distance and face activity represent latent/excited possibility, but energy ordering remains declared by the world's Hamiltonian.
- **H0 — no canonical quantum role:** D20 remains visual/narrative geometry.

Gate any promotion on a measurable advantage: compression, prediction, interpretability, transition fidelity, or gameplay legibility.

### 3.6 Geometry proof package

Create:

```text
src/septacrypt_core/geometry/
  counts.py
  qgraph.py
  dice_maps.py
  polyhedral_incidence.py
  gauge.py
  d20_hypotheses.py

proofs/
  prove_d2.py
  prove_d4_boundary.py
  prove_d6_d8_d12.py
  prove_pearl_edge_bijection.py
  compare_d20_hypotheses.py
```

`prove_d6_d8_d12.py` must verify that octahedron faces label all eight masks and that octahedron face adjacency equals `Q3`; it must separately mark D12 as a twelve-face label carrier rather than pretending dodecahedral adjacency is `Q3`. `prove_pearl_edge_bijection.py` must prove that the twelve active principle/state incidences map one-to-one onto the twelve oriented `Q3`/octahedron edges.

---

## 4. Runtime ontology

### 4.1 Shared ground versus private worlds

```text
WorldSession
  ├── GroundState                 authoritative simulated state
  ├── KnotLedger                 authoritative and hypothetical history DAG
  ├── BerryIndex                 process coordinates and loop signatures
  ├── ArchitectRegistry          structural and causal templates
  ├── SpiritFrameRegistry        seven-dimensional semantic frames
  └── Minds
       ├── keith → private umwelt
       ├── dwayne → private umwelt
       └── player_si → private umwelt
```

No mind reads `GroundState` directly. It receives `Observation` objects through declared channels.

### 4.2 Core types

```python
@dataclass(frozen=True)
class EntityRef:
    entity_id: str
    lineage_id: str
    parent_id: str | None
    scale_path: tuple[str, ...]
    schema_version: str

@dataclass(frozen=True)
class SpiritVector:
    wisdom: float
    might: float
    wealth: float
    power: float
    glory: float
    honor: float
    blessing: float
    frame_id: str
    confidence: float

@dataclass(frozen=True)
class Observation:
    observer_id: str
    source_id: str
    target: EntityRef
    channel: str
    value: object
    confidence: float
    chronological_time: float
    branch_id: str

@dataclass(frozen=True)
class Intent:
    actor_id: str
    target: EntityRef
    action_type: str
    parameters: dict
    spirit_gradient: SpiritVector | None
    shadow: bool = True

@dataclass(frozen=True)
class TransitionCertificate:
    dynamics_version: str
    pre_state_root: str
    post_state_root: str
    event_digest: str
    chronological_interval: tuple[float, float]
    residual: float
    tolerance: float
    rng_commitment: str | None
    replay_cassette: str
    affected_surface: tuple[str, ...]
    hidden_conditions: tuple[str, ...]

@dataclass(frozen=True)
class KnotStamp:
    stamp_id: str
    parent_ids: tuple[str, ...]
    branch_id: str
    event_kind: str
    actor_id: str | None
    observer_id: str | None
    chronological_time: float
    berry_coordinate: dict
    scale_address: tuple[str, ...]
    pre_state_root: str
    post_state_root: str
    transition_certificate_id: str
    truth_mode: str
    confidence: float
    spirit_vector: SpiritVector | None
```

### 4.3 Truth modes

At minimum:

```text
observed
acted
referee_confirmed
inferred
retro_generated
counterfactual
dreamed
rumored
disputed
redacted
```

Truth mode and confidence are independent. A confidently held rumor remains a rumor.

---

## 5. Knot Ledger: the durable temporal topology

### 5.1 Separation from Berry Tape

| Berry Tape | Knot Ledger |
|---|---|
| bounded working memory | durable append-only history |
| fast loop/return detection | branch ancestry and merge provenance |
| geometric/kaironic index | chronological replay and state commitments |
| approximate process similarity | transition witnesses and causal attribution |
| may be pruned | must be recoverable and auditable |

### 5.2 Data structure

Use a content-addressed Merkle DAG rather than one chain:

```text
A ─ B ─ C ─ D
        ├─ E ─ F
        └─ G ─ H
```

A branch is an immutable checkpoint plus an event overlay. Avoid full engine copies for every hypothesis.

```text
shared checkpoint
  + immutable prefix
  + branch-local event overlay
  + copy-on-write changed clusters
```

### 5.3 State roots

Do not hash raw floating-point arrays or raw eigenvectors as identities. Build a canonical state root from:

- topology Merkle root;
- stable entity IDs and schema versions;
- quantized role expectations;
- selected reduced density matrices in a canonical basis;
- gauge-invariant projectors rather than phase-sensitive eigenvectors;
- parameter-fiber root;
- RNG state/commitment;
- dynamics and compiler versions.

Retain exact checkpoints separately for replay. The root is a commitment, not a substitute for state storage.

### 5.4 Transition verification

The core verifier answers:

```python
verify_transition(pre_state, event_segment, post_constraints) -> TransitionResult
```

It returns:

- valid/invalid;
- post-state candidate;
- residual and tolerance;
- required hidden conditions;
- random samples consumed;
- affected causal surface;
- replay cassette;
- certificate.

For a channel `Φ`:

```text
post ≈ Φ(event_segment, pre)
```

Exact inverse unitary evolution is permitted as an optimization only when the interval is certified closed, unitary, fully specified, and free of measurement/topology mutation. General retro-simulation remains forward-scored abduction.

---

## 6. Microscope-style nonlinear history insertion

### 6.1 User operation

Given established stamps `A` and `D`, propose an event or segment between them:

```text
A ───────── D

proposal:
A ─ B ─ C ─ D
```

### 6.2 Insertion algorithm

1. Resolve the chronological interval and affected scale.
2. Load the nearest checkpoint at or before `A`.
3. Construct a local causal cone from affected entities and graph surfaces.
4. Compile the proposed event into executable intents/exogenous inputs.
5. Ask Universal Architect to fill missing structural or causal requirements.
6. Generate a diverse beam of candidate segments.
7. Replay each candidate forward from `A`.
8. Compare the resulting state to the constraints committed at `D`.
9. Score candidates by:
   - dynamical residual;
   - structural constraint satisfaction;
   - number/complexity of hidden additions;
   - contradiction penalties;
   - source evidence;
   - Spirit-vector resonance under the active cultural/agent frame.
10. Present:
   - valid insertion;
   - valid but uncertain alternatives;
   - required supports;
   - incompatible proposal;
   - counterfactual fork option.
11. Commit selected stamps and transition certificates.

### 6.3 Search discipline

History completion is combinatorial. Enforce:

- local causal cones;
- scale budgets;
- maximum added latent entities;
- beam width and diversity penalty;
- baseline candidate `unknown cause`;
- explicit stopping when evidence cannot discriminate;
- no LLM-authored fact becomes ground truth without deterministic gates.

---

## 7. Universal Architect extension

Universal Architect is the **constraint and abductive compiler**, not the quantum simulator.

### 7.1 Preserve the existing strengths

- tiny declarative syntax;
- recursive composition;
- hierarchical flattening;
- resource-vector balancing;
- minimal supplier addition;
- exact rational arithmetic where applicable.

### 7.2 Required extensions

1. **Instance identity** — `valve-17`, not only `Valve × 1`.
2. **Typed resources** — conserved quantities, capacities, predicates, evidence, permissions, affordances.
3. **Alternative producers** — top-k, weighted, diverse candidates rather than one greedy closure.
4. **Vector costs** — physical, narrative, coincidence, complexity, ethical/cultural, and uncertainty costs.
5. **Temporal rules** — events consume/produce conditions across intervals.
6. **Legal cycles** — distinguish recursive-definition errors from dynamical feedback loops.
7. **Compiler output** — emit an intermediate representation, not only flattened text.

### 7.3 Architect IR

```python
CompositeNode(
    entity_type="coolant_loop",
    children=(...),
    requirements=(...),
    productions=(...),
    invariants=(...),
    observations=(...),
    intents=(...),
    causal_templates=(...),
    local_roles=(...),
    exposed_roles=(...),
    spirit_priors=(...),
)
```

Compiler targets:

- umwelt `DomainSpec`;
- SpaceWheat/local dynamics specification;
- Knot Ledger event schemas;
- causal candidate rules;
- Fledgeling affordances/UI vocabulary;
- Spirit Cube semantic metadata.

---

## 8. Nested scale charts and transitions

### 8.1 Atlas, not one universe-sized state object

Each recursively composed subsystem owns a local chart (SpaceWheat may use open-system
state; umwelt minds stay separate). Parents see a reduced logical interface.

```text
component charts
   ↓ fold
machine logical chart
   ↓ fold
facility logical chart
   ↓ fold
town / ecosystem chart
```

### 8.2 Initial fold hierarchy

Implement and compare in this order:

1. declared semantic reductions;
2. graph-Laplacian modes;
3. Hamiltonian low-energy/relevant subspaces;
4. Lindbladian slow/metastable modes;
5. adaptive density-matrix eigenspaces.

Do not start with adaptive eigenvectors. They are gauge-sensitive and unstable under degeneracy.

### 8.3 Stable spectral fold

For selected subspace isometry `V`:

```text
H_eff   = V† H V
ρ_eff   = normalize(V† ρ V)
O_eff   = V† O V
```

Expose projectors/subspaces, not raw eigenvector byte identities.

Every fold requires:

- upward state projection;
- downward intent lifting;
- error estimate;
- versioned basis/frame;
- conservation/invariant checks;
- ability to unfold for diagnosis.

### 8.4 Scale identity

Entity identities survive fold/unfold:

```text
station/reactor/coolant/valve-17
```

Store:

- persistent entity ID;
- lineage ID;
- parent/child relation;
- projection version;
- source-to-logical operator map;
- logical-to-affordance intent map.

---

## 9. Spirit Cube semantic layer

### 9.1 Three generators, seven acclimations, twelve Pearls

The default `SeptacryptFrame` has three named binary generators:

```text
Father, Son, Spirit
```

Their seven nonzero composites provide the semantic axes:

1. `001` Wisdom
2. `010` Might
3. `100` Wealth
4. `011` Power
5. `101` Glory
6. `110` Honor
7. `111` Blessing

The twelve Pearls are not twelve additional semantic axes. They are the twelve incidences connecting an active generator to a composite state, and therefore the twelve oriented transition channels of the three-cube.

### 9.2 Keep three representations distinct

Do not collapse these into one datatype:

| Representation | Meaning |
|---|---|
| Three named bits | Generative principles and binary membership |
| Eight basis masks | Full three-qubit symbolic basis, including `000` |
| Seven-axis `SpiritVector` | Interpretable intensities, affinities, or evidence over the nonzero acclimations |
| Twelve Pearls | Generator-to-state incidences / oriented transition channels |
| Quantum state `ρ` | Actual physical or belief state, potentially including phase and coherence |

A `SpiritVector` may be derived from observables over `ρ`, supplied as semantic metadata, or maintained as an observer-dependent belief. It is not automatically identical to the diagonal of a density matrix.

These names are a declared default frame, not forced universal constants. The core should support alternate `SpiritFrame`s while retaining the Septacrypt frame as the canonical Fledgeling configuration.

### 9.3 Fractal application

Attach `SpiritVector`s to:

- agents;
- places;
- institutions;
- artifacts;
- actions;
- causal histories;
- loops/processes;
- cultures and interpretive frames.

A person, town, or history may have both:

- current vector;
- desired gradient;
- believed vector per observer;
- public/reputed vector;
- confidence and provenance.

### 9.4 Seven-state-to-3D projection

Spirit Cube is a view/controller over a seven-state interpretation vector, not a lossy replacement for the underlying three-bit basis or quantum state.

```python
ProjectedSpirit = Projection3D(
    frame_id,
    basis_3x7,
    origin_7,
    scale,
    active_slice,
)
```

Requirements:

- rotations/remappings preserve the underlying 7D vector;
- the active projection is ledgered, so historical UI interpretations can be reproduced;
- multiple agents may use different projections;
- projection distortion is measurable and shown where relevant;
- no axis is silently discarded.

### 9.5 Use in candidate scoring

Physical validity is a hard gate. Spirit score is a transparent preference/ranking layer:

```text
candidate_score =
    hard_validity_gate
    × evidence_likelihood
    × structural_prior
    × agent/culture_spirit_preference
    × diversity_term
```

Never allow a high Spirit score to legalize an impossible transition.

---

## 10. Fledgeling host contract

Provide a plain game-facing API. Substrate / combinatorial-chart vocabulary stays
internal unless a designer explicitly opts into a debug face.

### 10.1 Already shipped in umwelt (consume, do not reimplement)

```python
# umwelt.host — FL-core Phases 2–4 (public synthetic gates in this monorepo)
from umwelt.host import GameHost, WorldSession
from umwelt.host.agency_loop import AgencyLoop, SubRoutine, PromotionGate

host = GameHost()
host.register_world(spec)
host.observe(observer_id, channel, value, confidence, t)
host.beliefs(observer_id, query)          # Belief(value, confidence)
host.intend(actor_id, intent)             # Decision(shadow|live)
host.step() / host.step_turn(n)

session = WorldSession().register_world(spec)
session.add_mind(observer_id, channel_mask=...)
```

### 10.2 Integration face (septacrypt-core — still to build)

Compose umwelt minds with ledger / spirit / retro-insert. Suggested shape:

```python
session.observe(observer_id, channel, value, confidence, t)   # → umwelt mind
session.beliefs(observer_id, query)
session.intend(actor_id, intent)
session.step(dt_or_turn)
session.history(query, observer_id=None)                      # → Knot Ledger
session.propose_insertion(anchor_a, anchor_d, proposal)
session.verify_insertion(candidate_id)
session.commit_insertion(candidate_id)
session.branch(from_stamp, label)
session.project_spirit(observer_id, projection_id)            # → Spirit Cube
session.inspect_knot(process_query, scale)
```

Fledgeling owns:

- rendering;
- input;
- dialogue/presentation;
- player attention and permissions;
- subroutine teaching UI;
- scenario content.

**umwelt** owns belief contracts (partial observation, shadow/live, multi-mind privacy).  
**septacrypt-core** owns history verification, branch/replay, structural compilation
hooks, dynamics ports, and Spirit coordinates/provenance.

---

## 11. Repository strategy

Start as a **sibling integration repository**. Do not destabilize umwelt or Universal
Architect until boundaries are proven. Do **not** reimplement FL-core inside
`minds/` — wrap `umwelt.host` instead.

```text
septacrypt-core/                    # NEW sibling repo (or workspace package)
  pyproject.toml
  src/septacrypt_core/
    architect/                      # IR + compile targets (may call out-of-tree UA)
    geometry/                       # Q_n, dice maps, Pearl edges (combinatorial)
    dynamics/                       # chart ports; SpaceWheat or thin stand-in
    ledger/                         # Knot Ledger — primary new code
      stamp.py
      dag.py
      roots.py
      checkpoint.py                 # wraps umwelt engine.save/load initially
      replay.py
    kairos/                         # Berry index attachment (uses umwelt berry tape)
    minds/                          # thin facade over umwelt.host.WorldSession
      session.py
      observer.py
      attribution.py
    spirit/                         # SpiritVector frames — non-authoritative at first
    host/
      api.py                        # §10.2 composition face
      fledgeling_adapter.py
  examples/
    nested_reactor/
  proofs/
  tests/
  docs/
    ARCHITECTURE.md
    GEOMETRIC_GRAMMAR.md
    SEPTACRYPT_FACE_MAP.md
    SPIRIT_FRAME.md
    KNOT_LEDGER.md
    CLAIMS_SEPTACRYPT.md
    source_artifacts/
      dudecon_x_2026401325706260491/
```

Pin dependency commits during the spike:

```text
umwelt @ <commit>                 # FL-core 1–5 green on this pin
SpaceWheat @ <commit>             # optional for first Nested Reactor stand-in
Universal-Architect @ <commit>
Fledgeling_HTML @ <commit>
```

**Integration rules**

1. No integration claim is promoted without a public fixture and a repeatable proof.
2. umwelt vocabulary lint stays green — no domain / house / Fledgeling lore in
   `src/umwelt/`.
3. If multi-engine mind cost becomes a product issue, document a partition design in
   the sibling repo; do not silently change umwelt’s privacy semantics.
4. Prefer **hand-authored Nested Reactor** before Architect auto-compile is ready.

---

## 12. Primary vertical slice — Nested Reactor / Missing Valve Event

### 12.1 World

```text
Repair Station
  └── Reactor
       ├── Power subsystem
       ├── Fuel subsystem
       └── Coolant loop
            ├── Pump
            ├── Valve-17
            ├── Pipes
            └── Temperature sensor
```

### 12.2 Local symbolic geometry

```text
Pump:        ⚙️ working ↔ 💥 failed
Valve-17:    🟢 open ↔ 🔴 closed
Sensor:      👁️ accurate ↔ 🌫️ drifting
Coolant:     ❄️ cooling ↔ 🔥 heating
Reactor:     ⚛️ stable ↔ ☢️ runaway
```

Use a three-bit combinatorial chart for the pump/valve/sensor microstate and test the D6/D8/D12 mapping (attached weights optional; not a full multipartite state).

### 12.3 Scenario

1. Compile the station from Universal Architect declarations.
2. Boot an authoritative ground and private minds for Keith and Dwayne.
3. Execute a valve-close event at `t1`; Dwayne observes it, Keith does not.
4. Reactor temperature rises at `t2`.
5. Keith observes heat and incomplete telemetry.
6. Retro solver produces candidates:
   - valve closed;
   - pump failure;
   - sensor drift;
   - unknown/unmodeled cause.
7. Each candidate is replayed forward from the checkpoint before `t1`.
8. umwelt holds Keith’s confidence distribution over candidates.
9. Keith probes the command log or valve position.
10. Beliefs update.
11. Keith emits a repair intent in shadow mode.
12. The player proposes a Microscope-style inserted maintenance event between `t0` and `t2`.
13. Universal Architect supplies any missing prerequisites.
14. Transition verifier accepts, rejects, or forks the insertion.
15. Berry/Knot view shows the original and counterfactual loops.
16. Spirit Cube displays how the histories differ under Keith’s and the station culture’s value frames.

### 12.4 Exit gates

- [ ] D6/D8/D12 bijection, Pearl-edge identity, and `Q3` adjacency tests pass.
- [ ] Replay from checkpoint plus event tail reproduces the same state root.
- [ ] Different branch produces a different state root but shares prefix stamps.
- [ ] Keith and Dwayne hold correctly divergent beliefs.
- [ ] Actor-keyed self-confounding prevents Keith’s own repair from being learned as spontaneous world behavior.
- [ ] At least three candidate histories are generated and forward-scored.
- [ ] Ground-truth candidate ranks first after discriminating evidence.
- [ ] Valid nonlinear insertion reaches the committed endpoint within tolerance.
- [ ] Invalid insertion explains the violated constraints.
- [ ] Spirit score changes ranking among physically valid alternatives but never overrides validity.
- [ ] A second route to a visually similar reactor state has a distinct loop/path signature.
- [ ] Full demo runs deterministically from a seed in CI.

---

## 13. Phased implementation

### Phase 0 — Source capture and ADR alignment

**Deliverables**

- archive screenshots/mesh metadata for Spirit Cube;
- archive [https://x.com/dudecon/status/2026401325706260491](https://x.com/dudecon/status/2026401325706260491) text, acclamation table, and any reply-thread media (e.g. 12-apostles illustration as optional lore);
- extract Septacrypt face adjacency;
- write ADRs for sibling integration, component count, face mapping, named-bit orientation, `000` semantics, truth modes, geometry claims, and **pin umwelt FL-core as belief dependency**;
- freeze 90-day non-goals (§16).

**Gate:** all source artifacts are locally referenceable and the seven-component map is signed off.

### Phase 1 — Geometric grammar proof

**Deliverables**

- `Q_n` generator;
- D2/D4/D6-D8-D12 mapping package;
- proof of the three-qubit bijections;
- proof that twelve Pearls are the twelve oriented `Q3` edges;
- explicit D4 limitation test;
- D20 hypothesis harness, no promoted interpretation.

**Gate:** incidence, Pearl-edge, and round-trip tests pass; documentation distinguishes theology, combinatorics, semantic interpretation, and physical state representation.

### Phase 2 — Knot Ledger minimum viable DAG

**Deliverables**

- canonical state roots;
- stamps and transition certificates;
- append-only DAG;
- checkpoint plus overlay replay;
- branch creation;
- Berry coordinate attachment.

**Gate:** deterministic replay, tamper detection, shared-prefix branch proof.

### Phase 3 — Shared ground / private minds

**Status in umwelt monorepo (2026-07):** largely **done** for the belief face —
`WorldSession`, channel masks, actor-keyed intent log, privacy suite, multi-engine
cost probe. See [FLEDGELING_CORE.md](FLEDGELING_CORE.md) Phase 3 and
`tests/test_multimind_privacy.py`.

**Remaining for septacrypt-core**

- wire Nested Reactor observers (Keith / Dwayne) through the same contracts;
- truth-mode projections for ledger-facing queries (rumor vs fact vs inference);
- optional: richer per-intent competence surfaces per agent.

**Gate:** two-agent reactor privacy test on the Nested Reactor fixture (reuse umwelt
assertions; do not re-prove the fog corridor alone as “done”).

### Phase 4 — Universal Architect IR/compiler

**Deliverables**

- instance IDs;
- typed affordance resources;
- alternative producer search;
- causal templates;
- DomainSpec and dynamics compilation.

**Gate:** Nested Reactor compiles from declarative source without hand-building the runtime graph.

### Phase 5 — Retro insertion and branch scoring

**Deliverables**

- local causal cone builder;
- top-k candidate generation;
- forward verifier;
- endpoint constraint matching;
- insertion/fork UX payloads.

**Gate:** missing-valve scenario works end-to-end and beats an unweighted/random candidate baseline.

### Phase 6 — Spirit Cube semantic projection

**Deliverables**

- seven-axis frame;
- agent/culture-specific projections;
- 7D-to-3D projection registry;
- semantic gradients for desires and actions;
- candidate preference scoring.

**Gate:** two physically valid histories remain physically tied but are ranked differently by two declared value frames, with the reason inspectable.

### Phase 7 — Septacrypt/Fledgeling host

**Deliverables**

- seven-face navigation prototype;
- current face/edge context drives the UI query mode;
- timeline insertion interaction;
- knot/loop visualization;
- shadow action and learned subroutine demo.

**Gate:** a non-engine developer can play the Nested Reactor loop without reading umwelt theory or manipulating density matrices.

### Phase 8 — Spectral folding and scale performance

**Deliverables**

- semantic reduction baseline;
- spectral fold candidates;
- copy-on-write branch state;
- N-agent and N-branch benchmarks;
- fold/unfold fidelity reports.

**Gate:** spectral approach must outperform or compress the semantic baseline without breaking action parity or endpoint verification. Otherwise it remains experimental.

---

## 14. Tests and evidence ladder

### 14.1 Mathematical tests

- density matrices remain Hermitian, normalized, and positive within tolerance;
- projectors are gauge-invariant;
- geometry mappings are bijective;
- transition edges match Hamming-distance-one adjacency;
- fold/unfold error is reported;
- state roots are stable under canonical serialization but change under meaningful state changes.

### 14.2 Causal tests

- self-action is tagged by actor, intent, target, and affected surface;
- geometric nearness never creates a causal edge by itself;
- interventions distinguish correlation from cause in synthetic fixtures;
- hidden requirements introduced by Architect are explicit in certificates.

### 14.3 Epistemic tests

- observers receive only permitted channels;
- confidence zero is a no-op;
- rumors, inferences, and ground facts remain distinct;
- source reliability can isolate a faulty reporter when a referee exists;
- save/reload preserves private beliefs.

### 14.4 History tests

- insertions cannot mutate existing stamps;
- a valid insertion connects both anchors;
- invalid insertion reports why;
- branch prefix sharing is correct;
- merge/reconciliation never erases incompatible provenance;
- RNG commitments reproduce stochastic events.

### 14.5 Spirit tests

- projection changes do not mutate underlying 7D vectors;
- agent frames are versioned;
- semantic scores are decomposable and explainable;
- impossible candidates remain impossible regardless of preference;
- no default axis is silently privileged by projection scale.

### 14.6 Performance tests

Early budgets:

```text
8 minds / 1 ground
16 active history candidates
1,000 durable stamps
50-node nested reactor/station graph
sub-100 ms common belief/history query
sub-2 s top-8 local retro search in the small fixture
```

These are targets, not promises. Record actual numbers in the claims ledger.

---

## 15. Major risks and mitigations

| Risk | Mitigation |
|---|---|
| Solid symbolism is mistaken for complete quantum representation | publish parameter counts and incidence proofs; require attached weights/phases |
| D4 topology is silently wrong | hard test and explicit symbolic-only status |
| D20 acquires lore before utility | hypothesis bake-off; allow H0/no role |
| Full density matrices explode | local charts, cumulants, reduced interfaces, measured folds |
| Eigenvectors become unstable identities | use projectors/subspaces and versioned frames |
| Floating-point hashes are brittle | canonical quantization plus exact checkpoint storage |
| Berry proximity is mistaken for causation | separate geometric and causal links in schema |
| Retro search explodes | local cones, beam limits, latent-entity budgets, unknown-cause baseline |
| LLM invents ground truth | claims are observations; deterministic verifier gates writes |
| Seven Spirits become an imposed morality | declared replaceable frames, observer/culture specificity, transparent scoring |
| Integration damages mature umwelt behavior | sibling repository and pinned dependency commits; wrap host API |
| Metaphor outruns evidence | `CLAIMS_SEPTACRYPT.md`; every phase ends with public proofs and denied results |
| Rebuilding FL-core inside septacrypt | pin umwelt; minds/ is a facade only |
| Theology treated as runtime physics | keep §2.4 as declared frame; Spirit never overrides validity |

---

## 16. Non-goals for the first 90 days

- full Fledgeling game;
- arbitrary universe-scale density matrix;
- consciousness or AGI claims;
- automatic generation of an entire coherent civilization history;
- multiplayer networking;
- production cryptographic consensus;
- proving the Seven Spirits are universal physical dimensions;
- assigning D20 a final interpretation;
- using spectral folding before a semantic-reduction baseline exists;
- merging all repositories into one monorepo.

---

## 17. First execution sprint

### Sprint objective

Prove the **smallest complete knot**: a three-bit reactor subsystem (combinatorial
chart), a durable branchable **Knot Ledger**, and one valid nonlinear event insertion —
**on top of** pinned umwelt FL-core, not by rewriting it.

### Ordered tasks

1. Create `septacrypt-core` skeleton; pin umwelt (FL-core 1–5 green) + other deps.
2. Archive Spirit Cube materials and the [source X post](https://x.com/dudecon/status/2026401325706260491) under `docs/source_artifacts/`.
3. Implement `QubitCombinatorics(n)` returning poles, basis states, and Hamming-one edges.
4. Implement and test D6/D8/D12 mappings for `n=3` (combinatorial grammar, not “full quantum state”).
5. Define `KnotStamp`, `TransitionCertificate`, and canonical state-root interfaces.
6. Wrap umwelt `engine.save` / `load` / `field_canon_hash` as the initial checkpoint backend.
7. Implement branch overlays over an immutable event prefix.
8. Hand-author the first Nested Reactor world; do **not** wait for the Architect compiler.
9. Drive ground + private minds via `umwelt.host.WorldSession` (Keith / Dwayne masks).
10. Execute and replay valve-close → overheating; assert divergent beliefs.
11. Insert a maintenance event between anchor states and verify endpoint reachability.
12. Attach Berry coordinates to stamps (kairos ≠ causal edge).
13. Add a placeholder seven-axis `SpiritVector` (acclamation names from §2.4) without using it for control.
14. Produce `proofs/nested_reactor_knot.py` and a machine-readable result report.
15. Update `CLAIMS_SEPTACRYPT.md` with proven, owed, and denied rows.

### Sprint completion definition

A single command:

```bash
python -m proofs.nested_reactor_knot
```

must produce:

```text
[PASS] D6/D8/D12 mapping
[PASS] deterministic checkpoint replay
[PASS] branch prefix integrity
[PASS] valid event insertion reaches endpoint
[PASS] invalid insertion rejected with reasons
[PASS] Berry coordinate attached
[PASS] Spirit vector persisted but non-authoritative
```

---

## 18. Proposed CLI

```bash
septacrypt compile examples/nested_reactor/world.yaml
septacrypt run nested-reactor --seed 42
septacrypt inspect state --observer keith
septacrypt inspect knot --process reactor.cooling
septacrypt branch --from <stamp> --name repaired-early
septacrypt propose-event --between <A> <D> event.yaml
septacrypt verify-candidate <candidate>
septacrypt commit-candidate <candidate>
septacrypt project-spirit --observer keith --projection default
septacrypt prove nested-reactor
```

---

## 19. Key architectural decisions to record

Create ADRs before implementation drifts:

1. `ADR-001`: sibling integration repository (`septacrypt-core`), not monorepo merge.
2. `ADR-002`: seven components and Septacrypt face assignment.
3. `ADR-003`: shared ground / private umwelten — **consume `umwelt.host.WorldSession`**.
4. `ADR-004`: Merkle DAG plus checkpoints, not inverse physics as history foundation.
5. `ADR-005`: geometric solids are weighted combinatorial carriers, not complete state by shape alone.
6. `ADR-006`: Spirit vectors rank valid possibilities but cannot override physical validity.
7. `ADR-007`: raw eigenvectors are not persistent identities.
8. `ADR-008`: Berry adjacency and causal adjacency are separate relations.
9. `ADR-009`: public synthetic fixtures are required for every promoted claim.
10. `ADR-010`: umwelt FL-core is the belief face; do not reimplement observe/intend/beliefs in core.
11. `ADR-011`: named-bit Father/Son/Spirit convention and 7 acclamations from the 2026-02-24 source post.

---

## 20. Final architecture in one diagram

```text
                         SEPTACRYPT
              seven-face spatial/user embedding
                              │
                      SPIRIT CUBE / KEY
         7 acclimations + 12 Pearls (declared frame)
         ranks meaning — never overrides validity
                              │
       ┌──────────────────────┴──────────────────────┐
       │                                             │
PRIMARY WORLD TRIANGLE                    TERTIARY PROCESS TRIANGLE

Universal Architect                      Berry Tape
structure / affordance closure            kairos / loops / returns
       │                                             │
       ▼                                             ▼
SpaceWheat manifold  ───────────────►     Knot Ledger   ◄── NEW (sibling)
local dynamics and coupled processes      paths / stamps / branches / proofs
       │                                             │
       ▼                                             ▼
umwelt (FL-core 1–5 ✓)                    Fledgeling
private belief + host face                attention / agency / story / play

Cross-cutting combinatorial grammar (opt-in charts):
D2 / D4 / D6-D8-D12 / open D20  — carriers with attached weights, not full state by shape
```

**Build order (integration, not cosmology expansion):**

```text
1. Pin umwelt FL-core
2. Knot Ledger MVP + Nested Reactor hand world
3. Private minds on reactor fixture (reuse WorldSession)
4. Retro-insert + certificates
5. Spirit vectors non-authoritative
6. Architect IR + Fledgeling host polish
7. Folds / D20 only after semantic baselines
```

The architecture is complete enough to integrate. Unresolved research (deliberately
bounded):

- exact Spirit Cube / Septacrypt spatial projection into UI;
- incidence-preserving solid mappings beyond three bits;
- the role, if any, of D20;
- which spectral folds beat declared semantic reductions;
- how far retro-generation scales before authorial constraints dominate.

Proceed by proving the Nested Reactor knot, not by expanding the cosmology first.

---

## 21. Reference inputs

- Fledgeling overview: https://www.peripheralarbor.com/fledgeling/
- Fledgeling HTML prototype: https://github.com/dudecon/Fledgeling_HTML
- Universal Architect: https://github.com/dudecon/Universal-Architect
- umwelt: https://github.com/AQuantumArchitect/umwelt
- umwelt FL-core status: [FLEDGELING_CORE.md](FLEDGELING_CORE.md)
- SpaceWheat: https://github.com/AQuantumArchitect/SpaceWheat
- Spirit Cube model: https://sketchfab.com/3d-models/spirit-cube-bdcfaf40b1e447e69b8db95b465ff39c
- **Canonical 1→2→3→7→12 derivation (archive this):** https://x.com/dudecon/status/2026401325706260491
- Septacrypt: https://peripheralarbor.com/gallery/Projects/SeptaCrypt/



---

## Appendix A — normalized source statement

Canonical narrative source:
[https://x.com/dudecon/status/2026401325706260491](https://x.com/dudecon/status/2026401325706260491)
(Paul Spooner / @dudecon, 2026-02-24). Archive verbatim under
`docs/source_artifacts/dudecon_x_2026401325706260491/`.

Compressed chain used by the runtime:

```text
unity → relation (two) → triad (three)
      → seven nonzero binary composites of the triad
      → twelve Pearls (4 nonzero masks per principle × 3)
```

**Source mapping of acclimations** (do not silently rename):

```text
001 Wisdom | 010 Might | 100 Wealth
011 Power  | 101 Glory | 110 Honor | 111 Blessing
```

Normalized implementation form (named principles; bit order documented):

```yaml
# string form FSS = Father Son Spirit left-to-right
principles:
  father: 0b100
  son: 0b010
  spirit: 0b001

states:
  0b000: {name: Holy Dark, class: void_reference}  # "lack of divinity" — not one of the seven
  0b001: {name: Wisdom}    # Spirit
  0b010: {name: Might}     # Son
  0b100: {name: Wealth}    # Father
  0b011: {name: Power}     # Son + Spirit
  0b101: {name: Glory}     # Father + Spirit
  0b110: {name: Honor}     # Father + Son
  0b111: {name: Blessing}  # Father + Son + Spirit
```

Pearls are **generated** (4 per principle × 3), never hand-authored ad hoc:

```python
pearls = [
    (principle, state)
    for state in NONZERO_STATES
    for principle in PRINCIPLES
    if state & principle.mask
]
assert len(pearls) == 12
```

Each Pearl also yields an oriented basis transition:

```python
edge = (state, state ^ principle.mask)  # clears the active bit toward void
```

This generated representation is the single source of truth for the mapping table,
D12 transition labels, Spirit Cube sectors, and tests. Optional lore mappings are content-pack data, not required for the Nested Reactor gate.
