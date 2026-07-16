# FIELD NOTES — septacrypt-core consumer harness (2026-07)

**Status:** informational transfer from a **sibling consumer**, not an umwelt capability claim.  
**Harness:** [`AQuantumArchitect/septacrypt-core`](https://github.com/AQuantumArchitect/septacrypt-core) @ `81b2e53` (and nearby).  
**Audience:** umwelt core (library + `umweltd` service + FL host face).

This note is the same *kind* of document as [FIELD_NOTES.md](FIELD_NOTES.md) and the
umwelt-market engine asks: **what a foreign/product harness taught us about the
engine**, without re-claiming product metrics as monorepo pins.

Septacrypt is **not** a second belief engine. It is a game-runtime integration that
sits on cumulant clusters, Berry machinery, and (intended) the host/multi-mind face.

It also built a **Knot Ledger** — typed cassettes, transition certificates, content-
addressed stamps, fail-closed commits. That ledger is **in scope for umwelt
integration**, especially as umwelt merges with a **blockchain-based hive
coordination surface**. This note therefore does two jobs:

1. Transfer cognition/service *discipline* from the consumer.
2. Give the core team a **head start on where to connect the Knot Ledger** to
   engine, daemon, and on-chain hive rails (see §K).

Mythic vocabulary (Septacrypt, Pearls, Holy Dark, Endless Knot) is **domain data in
the consumer**. Engine source remains domain-free ([CLAIMS.md](../CLAIMS.md) vocabulary
lint). Do not import lore strings into `src/umwelt/`. The ledger’s *neutral* contracts
(stamp, certificate, cassette, residual) are fine to host under domain-free names.

---

## Headline

A playable vertical kernel can already:

1. Evolve real `CumulantCluster` state with ZZ coupling and Belavkin measurement.
2. Stamp history only when a **replay cassette** reproduces committed endpoints.
3. Fail **closed** on bad certificates / branch heads (no silent re-anchor).
4. Separate **UI focus** from **which physics advances**.
5. Separate **observer-facing narration** from **authoritative ground**.

Those five are engine-relevant. The rest of this document translates them into
**umwelt-as-cognition** and **umwelt-as-service** asks.

---

## 1. Cognition — composite world steps must be atomic

### What broke in the consumer

When “active zone” (the UI selection) determined which cluster received `step()`,
changing the camera changed the ship’s physical trajectory. Soft cross-zone bridges
were applied **outside** the certified cassette and then absorbed by re-anchoring —
history that looked continuous but could not be honestly replayed.

### Transferable law

> **Presentation selectors must not select which subsystems evolve.**  
> One world turn = snapshot → evolve all zones → apply versioned coupling rules →
> single composite commitment.

### Asks for umwelt (cognition / host)

| Ask | Why | Suggested home |
|-----|-----|----------------|
| **Atomic multi-cluster turn** | Games and multi-region homes both have concurrent zones | `GameHost.step_turn` / field tick already multi-cluster — document and test that **no cluster is skipped** because of focus |
| **Versioned bridge / transfer rules** | Soft observes between zones need a named rule id in the event log | `DomainSpec` bridges already exist; ensure **every applied transfer is loggable** with rule version |
| **World-level canon hash** | Consumers need one hash over all clusters + param bundles + seed profile | Extend or document composition of `field_canon_hash` for multi-zone worlds |
| **No out-of-band substrate mutation** | STIR-like “kick” must be a typed param/field event | Host/API forbid direct `cluster._h` writes; only ingest / intend / registered param paths |

**Not an ask:** joint 9-qubit full-ρ ship Hamiltonian. Soft / 2-local coupling is enough
when the **cassette** names the rule.

---

## 2. Cognition — fail-closed history (cassette + residual)

### What the consumer built

```text
checkpoint A
→ apply typed cassette on isolated working copy
→ checkpoint D
→ independent replay A→cassette
→ residual ≤ tolerance
→ mint TransitionCertificate
→ append stamp only if branch head + pre/post continuity hold
→ commit live state atomically
else: world + ledger head unchanged
```

Berry path signatures index **process geometry**; they never mint causal edges.
Certificates own durable path identity.

### Transferable law

> **Every authoritative mutation is a typed, serializable event.**  
> **When certification is on, mutations fail closed.**  
> Symbolic graph validity ≠ dynamical realizability.

### Asks for umwelt (cognition)

| Ask | Why | Notes |
|-----|-----|--------|
| **Event taxonomy for field mutations** | Ingest is strong; param/H/coupling/measure record should share a single “cassette” vocabulary for consumers | Align with `umwelt.events` schema; export a stable event kind enum |
| **Replay residual API** | Consumers reimplemented residual on e1/e2 | Optional: `engine.replay(events) -> residual_report` against a snapshot |
| **Berry vs ledger boundary** | Already correct in spirit ([FLEDGELING_SEPTACRYPT_PLAN.md](FLEDGELING_SEPTACRYPT_PLAN.md)) | Keep BerryTape as process clock; durable branching stays **out** of umwelt or as an optional sidecar contract |
| **RNG commitment** | Measurement outcomes must be recorded for residual-zero replay | Host measure paths should record `record_z` (or equivalent) in the event log |

**Ask:** treat the Knot Ledger as an **optional durable coordination layer** (see §K),
not as a rewrite of `events.db` or BerryTape.

---

## 3. Cognition — multi-mind is not “private e1 + shared mythos”

### What broke in the consumer

Private copies of first-moment `e1` still leaked **ground** Q3 masks, lore, and tension
through `status()`. A LOOK auto-nudged other observers (telepathy). That is not an
umwelt.

### Transferable law

> **Observer-facing payloads must be belief-derived unless a fact was intentionally
> public or observed.**  
> Other minds update only through **typed report / channel** events with provenance.

umwelt already pins multi-mind privacy on the host face
(`tests/test_multimind_privacy.py`, WorldSession). The consumer rediscovered the same
law the hard way on a thinner stack (direct clusters, no DomainSpec).

### Asks for umwelt (cognition / host)

| Ask | Why | Notes |
|-----|-----|--------|
| **Default status is private-first** | Game builders will otherwise poll ground | Document `beliefs()` vs admin/debug substrate reads; keep `/beliefs` on umweltd as debug |
| **Typed inter-mind reports** | “Rumor” without channel is a bug | Extend or document channel masks + observation paths; no automatic partial sync |
| **Honest belief kind labels** | `e1` snapshot ≠ complete private engine | Surface already richer; export a `belief_kind` / estimator rung in host JSON |
| **Shadow-first remains default** | Consumer briefly shipped `auto_live=True` | Agency loop already pins promotion gates — keep consumer docs pointing at FL Phase 4 |

---

## 4. Service — umweltd as the shared brain for sibling runtimes

Septacrypt today often attaches to **hand-built clusters** because DomainSpec
registration is a larger step. That is fine for a vertical kernel; it is **wrong** as
the long-term product path.

### Transferable law

> **Library and daemon must remain the single source of substrate truth.**  
> Consumers should own: domain adapters, ledgers, narrative, UI.  
> They should not own: alternate cumulant integrators or silent field forks.

### Asks for umwelt-as-service

| Ask | Why | Suggested surface |
|-----|-----|-------------------|
| **Stable multi-world catalog + pin** | Consumers need `umwelt @ <commit>` + world name | Already: worlds catalog; publish **install pin guidance** in SERVICE.md |
| **Cassette / event batch ingest** | Certified segments are batches with residual checks | `POST /events` already batch; add optional `client_request_id` + idempotency notes for harnesses |
| **Snapshot cursor = consumer checkpoint** | Knot anchors map to daemon snapshots | Document mapping: consumer stamp ↔ `field_canon_hash` + event cursor |
| **Shadow recommendations over the wire** | Game AI and SI should stay ghost until promote | `/recommendations` exists — call out game SI use |
| **No lore in workers** | Mythos stays in septacrypt / Fledgeling | Vocabulary lint already enforces domain-free engine |
| **Exportable world physics hash for harnesses** | Active-zone-invariant hash tests | Expose composite hash in `/health` or `/state` meta |

### Explicit non-goals for the service

- Shipping Septacrypt emoji / theology in the daemon.
- Replacing GameHost with GameSession naming.
- Replacing `events.db` (ingest log) with the Knot DAG (different job — see §K).

---

## K. Knot Ledger → umwelt → blockchain hive (connection map)

### K.0 What the Knot Ledger *is* (neutral contracts)

Shipped today in septacrypt-core (`src/septacrypt_core/ledger/`, `world/transaction.py`):

| Contract | Meaning |
|----------|---------|
| **KnotEvent / cassette** | Ordered, fully serializable ops (measure with `record_z`, world_evolve, set_fields, bridges, report, promote, …) |
| **TransitionCertificate** | pre/post content hashes, event_digest, residual, tolerance, rng_commitment, replay_cassette, dynamics_version, affected_surface |
| **KnotStamp** | Content-addressed node: parent_ids, branch_id, berry_coordinate, scale_address, pre/post roots, cert id, truth_mode |
| **KnotLedger** | CAS append, expected branch head, parent pre==parent post continuity |
| **CertifiedTransaction** | Fail-closed: working copy → residual replay → cert → stamp → atomic commit, or no mutation |

Honest limits (preserve them):

- pre/post “roots” are **content hashes**, not Merkle trees yet (tree-hash is the
  natural upgrade for on-chain partial proofs).
- Berry coordinates are **path signatures**, not causal edges.
- Symbolic Q3 “weave” paths are **not** certificates.

### K.1 Three layers that must stay distinct

Hive coordination will fail if these get collapsed:

```text
┌─────────────────────────────────────────────────────────────┐
│  C. HIVE / CHAIN  (multi-party coordination surface)        │
│     commitments, votes, role grants, settlement, forks      │
│     ← publish: stamp_id, cert_id, hashes, branch tips       │
│     ← never: full density matrices, raw sensor floods       │
└──────────────────────────▲──────────────────────────────────┘
                           │ attest / anchor / challenge
┌──────────────────────────┴──────────────────────────────────┐
│  B. KNOT LEDGER  (durable witnessed history)                │
│     stamps + certificates + branch DAG                      │
│     “this cassette is dynamically realizable A→D”           │
└──────────────────────────▲──────────────────────────────────┘
                           │ bind pre/post to field anchors
┌──────────────────────────┴──────────────────────────────────┐
│  A. UMWELT FIELD  (belief / cumulant substrate)             │
│     DomainSpec, ingest, clusters, BerryTape, host minds     │
│     events.db + snapshot.pkl + field_canon_hash (umweltd)   │
└─────────────────────────────────────────────────────────────┘
```

| Layer | Owns | Does *not* own |
|-------|------|----------------|
| **A Field** | Uncertain beliefs, estimation, shadow decisions | Branch identity, multi-party settlement |
| **B Knot** | Certified segments, branch tips, residual proofs | Full multi-mind estimators (refs only) |
| **C Hive/chain** | Who may propose/promote, economic/coordination finality | Re-running Belavkin in the EVM |

**BerryTape stays in A.** It indexes process phase / loop geometry.  
**Knot stays in B.** It indexes durable path identity and branches.  
**Chain stays in C.** It indexes social/economic finality over *digests* of B.

### K.2 Concrete attachment points in *this* monorepo

#### A — Library / host (cognition)

| Hook | Current artifact | Knot use |
|------|------------------|----------|
| `engine.field_canon_hash()` | Byte-stable field commitment | Stamp `pre_state_root` / `post_state_root` (or composite world hash when multi-zone) |
| `engine.save` / `load` | Checkpoint backend | Materialize A/D anchors; never the only history |
| `umwelt.events.Event` + SQLite log | Ingest tape | **Source** of raw observations that cassettes may *summarize*; not a substitute for cassettes |
| `GameHost.observe` / `intend` / `step_turn` | FL host face | Emit or wrap as KnotEvents; promote_routine ↔ shadow→live |
| `WorldSession` multi-mind | Private engines | Observer id on stamps; reports as typed events; never stamp another mind’s private state as public ground |
| `AgencyLoop` / tendrils | Shadow-first egress | Promotion events must be ledgered before live dispatch is hive-visible |
| `BlochGeometricPhase` / `BerryTape` | Process clock | Fill `KnotStamp.berry_coordinate`; never use phase nearness as a parent edge |

#### B — umweltd (service)

Worker contract today ([SERVICE.md](SERVICE.md)):

```text
events.db  (append-only ingest truth)
snapshot.pkl + cursor.txt  (cache)
POST /events → append then ingest
POST /snapshot → field_canon_hash + cursor
```

**Recommended extension surface** (design, not implemented here):

```text
$UMWELTD_HOME/worlds/<name>/
  events.db              # unchanged — field ingest
  snapshot.pkl           # unchanged — field cache
  cursor.txt
  knot/                  # NEW optional plugin store
    stamps.db            # or content-addressed objects
    certs.db
    branches.json        # branch_id → head stamp_id
```

HTTP (sketch — keep domain-free names):

| Route | Role |
|-------|------|
| `POST /worlds/<n>/knot/commit` | Body: cassette + expected_branch_head → server runs residual against current snapshot path, returns stamp_id + cert or 409 fail-closed |
| `GET /worlds/<n>/knot/head?branch=` | Branch tip |
| `GET /worlds/<n>/knot/stamp/<id>` | Stamp + cert digests |
| `GET /worlds/<n>/knot/proof/<id>` | Compact payload for chain: hashes, digest, residual, dynamics_version |
| `POST /worlds/<n>/knot/challenge` | Re-verify a published stamp against local replay (hive dispute) |

**Ordering rule (non-negotiable):**

```text
1) append any raw ingest events that the cassette depends on  (events.db first)
2) field applies / residual-checks on working state
3) knot commit (stamp) only if residual OK
4) optional: publish digest to hive/chain
5) snapshot field + knot heads together (or knot head points at snapshot cursor)
```

Never: chain finality before residual. Never: field mutation after a failed cert without explicit typed reanchor.

#### C — Blockchain / hive coordination surface

Publish **only digests and authorizations**, not the field:

```json
{
  "schema": "umwelt.knot.anchor.v1",
  "world_id": "urn:umwelt:world:…",
  "stamp_id": "stamp_…",
  "cert_id": "cert_…",
  "pre_hash": "…",
  "post_hash": "…",
  "event_digest": "…",
  "residual": 0.0,
  "tolerance": 1e-9,
  "dynamics_version": "…",
  "branch_id": "main",
  "parent_ids": ["stamp_…"],
  "berry_path_signature": "…",
  "field_canon_hash": "…",
  "events_cursor": "iso-ts-or-seq",
  "proposer": "did:… or agent id",
  "authorization": "sig or hive vote ref"
}
```

| Hive concern | Knot/field binding |
|--------------|-------------------|
| **Proposal** | Cassette + expected parent head (off-chain compute, on-chain intent hash) |
| **Validation** | Any replica with the world snapshot runs residual; posts pass/fail |
| **Branching** | Parallel `branch_id`s = rival histories; hive chooses tip policy |
| **Merge** | Requires multi-parent stamp + cert that both parents’ posts are inputs (consumer DAG already has parent_ids; history linearization is separate) |
| **Role / SI promotion** | On-chain grant references `promote_routine` stamp id + evidence hash |
| **Settlement** | Pays out on `post_hash` + cert_id, not on a UI narrative string |
| **Privacy** | Private mind fields never land on-chain; only public ground anchors or encrypted commitments |
| **Data availability** | Full cassette may live in IPFS/object store; chain holds digests |

**Merkle upgrade path (when hive needs partial proofs):**

1. Keep content-hash stamps working today.  
2. Replace `generate_state_hash(whole)` with a **state tree** (per-zone / per-role leaves).  
3. Certificates gain inclusion proofs for `affected_surface` only.  
4. Chain verifies inclusion + residual claim without full world blob.

### K.3 Reference implementation to lift

Prefer **porting neutral modules** over re-deriving:

| septacrypt-core path | Suggested umwelt home (future) |
|----------------------|--------------------------------|
| `ledger/events.py` | `umwelt.knot.events` or `umwelt.history.events` |
| `ledger/certificate.py` | `umwelt.knot.certificate` |
| `ledger/dag.py` | `umwelt.knot.ledger` |
| `ledger/roots.py` | share with field canon hashing utilities |
| `world/transaction.py` | `umwelt.knot.transaction` (depends on engine snapshot API) |
| `world/snapshot.py` | bridge to `field_canon_hash` + multi-cluster compose |

Vocabulary lint: keep names **knot / history / certificate / cassette** — not Septacrypt mythos.

Consumer remains the **first product user** of the package; hive is the **second**.

### K.4 Minimal integration sequence (suggested)

1. **Read-only bridge:** umweltd snapshot returns `{field_canon_hash, cursor}` already — document as Knot A/D anchors.  
2. **Library package:** extract neutral knot types into umwelt (or shared `umwelt-knot` wheel) with residual against `engine` snapshots.  
3. **Fail-closed host path:** optional `GameHost` flag `certified_steps=True` wraps step_turn in CertifiedTransaction.  
4. **Daemon knot store:** `knot/` directory + commit/proof routes.  
5. **Hive adapter:** thin publisher that posts `umwelt.knot.anchor.v1` and listens for challenges → `POST …/knot/challenge`.  
6. **Merkleization** when partial on-chain proofs become load-bearing.

### K.5 Invariants for hive (copy into hive ADR)

1. Field mutation without a KnotEvent is forbidden in certified mode.  
2. Chain never re-simulates the cumulant integrator; it verifies digests + optional residual attestations.  
3. Branch head mismatch → reject (no silent re-anchor).  
4. Shadow/live promotion is a ledgered event before hive-visible autonomy.  
5. Private umwelten are not public stamps.  
6. Berry similarity does not authorize a parent link.  
7. Mythos/UI strings are never part of the hash preimage for settlement.

---

## 5. Packaging & install friction (consumer pain)

The first septacrypt `pyproject` pointed at `file:///home/primearchitect/ws/umwelt`
and blocked third-party install. Fixed in consumer by sibling editable + docs.

### Ask

Document the **supported consumer install** in one place:

```text
pip install -e ../umwelt          # or git+https://...@<pinned-sha>
pip install -e ../septacrypt-core
```

Optional: publish a `docs/CONSUMER_PIN.md` (or section in SERVICE.md) with:

- minimum Python version  
- how to pin `aa12f94…`-class commits  
- that CI for siblings should checkout umwelt as a path dependency  

---

## 6. Mapping table (mythos → engineering — for agents)

Do not delete consumer vocabulary; do not put it in the numerical kernel.

| Project term | Engineering contract |
|---|---|
| Septacrypt | Three-bit state grammar + semantic vocabulary (consumer) |
| Pearl | Directed state-transition incidence (consumer geometry) |
| Spirit | Semantic scoring layer (downstream of physics) |
| Knot | Branchable certified event history (integrate with umwelt + hive; §K) |
| Berry journey | Gauge-aware trajectory signature (umwelt BerryTape / BlochGeometricPhase) |
| Endless Knot | Coupled multiscale process graph (product vision) |
| Holy Dark / `000` | Distinguished null/reference state (consumer lore) |
| Umwelt | Observer-specific uncertain state estimator (**this repo**) |
| Weave | Propose symbolic **or** certified history segment |
| Acclimation | Named nonzero three-bit state (consumer) |
| Mythos | Presentation vocabulary derived from state (consumer UI) |

**Instruction for coding agents:** build neutral infrastructure beneath domain adapters.
Do not “rationalize away” the adapter vocabulary.

---

## 7. Priority order for core team pickup

1. **Document + test multi-cluster atomic turn** (no focus-selected physics).  
2. **Knot attachment ADR** — adopt §K layers; decide package home (`umwelt.knot` vs sibling wheel).  
3. **umweltd snapshot ↔ stamp anchor** mapping doc (field_canon_hash + cursor).  
4. **Fail-closed residual API** sketch against engine snapshots (library).  
5. **Hive digests** — freeze `umwelt.knot.anchor.v1` JSON for the chain team.  
6. **Keep shadow-first + multi-mind privacy defaults obvious** in host + SERVICE docs.  
7. **Consumer install pin** one-pager.  
8. **Do not** put Septacrypt lore into `src/umwelt/`; **do** port neutral knot contracts.

---

## 8. Pointers

| Resource | Location |
|---|---|
| Consumer repo | https://github.com/AQuantumArchitect/septacrypt-core |
| Hardening commit | `81b2e53` — fail-closed composite ledger |
| Handoff surface | consumer `GAME_BUILDER.md` |
| Prior plan in this monorepo | [FLEDGELING_SEPTACRYPT_PLAN.md](FLEDGELING_SEPTACRYPT_PLAN.md) |
| FL-core status | [FLEDGELING_CORE.md](FLEDGELING_CORE.md) |
| Service shape | [SERVICE.md](SERVICE.md) |
| Honest claims ledger | [CLAIMS.md](../CLAIMS.md) |

---

## 9. What this note is *not*

- Not a claim that septacrypt’s Monte Carlo win rates are engine metrics.  
- Not a claim that umwelt already implements TransitionCertificates.  
- Not a request to rename host APIs to GameSession.  
- Not theology, D20 interpretation, or new cosmology.

It is a **field note from a product-shaped sibling**: where the substrate held, where
integration shortcuts lied, and which small engine/service contracts would make the
next sibling cheaper and more honest.
