# FIELD NOTES — septacrypt-core consumer harness (2026-07)

**Status:** informational transfer from a **sibling consumer**, not an umwelt capability claim.  
**Harness:** [`AQuantumArchitect/septacrypt-core`](https://github.com/AQuantumArchitect/septacrypt-core) @ `81b2e53` (and nearby).  
**Audience:** umwelt core (library + `umweltd` service + FL host face).

This note is the same *kind* of document as [FIELD_NOTES.md](FIELD_NOTES.md) and the
umwelt-market engine asks: **what a foreign/product harness taught us about the
engine**, without re-claiming product metrics as monorepo pins.

Septacrypt is **not** a second belief engine. It is a game-runtime integration that
sits on cumulant clusters, Berry machinery, and (intended) the host/multi-mind face.
It also built a **Knot Ledger** (typed cassettes, transition certificates, fail-closed
commits) that umwelt does not own — and should not swallow wholesale. What *should*
cross back is the discipline.

Mythic vocabulary (Septacrypt, Pearls, Holy Dark, Endless Knot) is **domain data in
the consumer**. Engine source remains domain-free ([CLAIMS.md](../CLAIMS.md) vocabulary
lint). Do not import lore strings into `src/umwelt/`.

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

**Not an ask:** absorb Knot Ledger into the monorepo. Sibling integration is the plan.

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

- Hosting Knot Ledger DAG as a first-class umweltd feature (keep sibling).
- Shipping Septacrypt emoji / theology in the daemon.
- Replacing GameHost with GameSession naming.

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
| Knot | Branchable certified event history (sibling ledger) |
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
2. **Keep shadow-first / multi-mind privacy defaults obvious** in host + SERVICE docs.  
3. **Event/cassette residual story** for consumers that re-verify history.  
4. **Consumer install pin** one-pager.  
5. **Optional:** world composite hash endpoint for harness acceptance tests.  
6. **Do not** merge Knot Ledger or Septacrypt lore into `src/umwelt/`.

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
