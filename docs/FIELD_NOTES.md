# FIELD NOTES — the first real foreign world (2026-07)

umwelt's first contact with reality it was not built around: a 13-day Home Assistant
export of a real family home the engine had never seen, replayed blank through the
production ingest path on a plain x86 machine. **The data, the rig, and the extracted
household model are private and live outside this repo permanently** — what crossed
back is exactly two things: one bug fix (`53fe99a`, pinned by
`tests/test_gauge_place_provenance.py`) and the lessons below. Numbers quoted here are
cited from that private rig the same way origin-deployment numbers are cited in
[CLAIMS.md](../CLAIMS.md): measured, not re-claimable from this repo.

**The headline:** every blank-slate assertion held on real data — booted blank
(learnedness at the 0.054 floor), all 45 declared bindings drove the field, the
learnable fiber drifted (0.054 → 0.129), the gauge read `.nowhere.` before, during,
and after (this is the run that caught the place bug), and the field canon hash
survived save → fresh boot → load byte-for-byte. Scored against held-out occupancy
channels the engine never saw, the belief beat the persistence of its own raw inputs
in every well-sensed room: **0.94 / 0.87 / 0.79 AUC vs 0.60–0.67 baselines**, natural
sign, with the one under-sensed floor reported honestly as inconclusive.

The rest of this document is the part that matters for the next foreign world —
whatever domain it comes from. None of these lessons are about houses.

---

## 1. The dissipative-role law

**A role fed only one polarity of evidence must be declared dissipative, or it
saturates.**

How it happened: real presence-style detectors emit an OFF that is a re-arm timeout,
not an observation of absence (median 326 s in this dataset). The honest adapter drops
those (absence of evidence is not evidence of absence) — which means the bound role
only ever hears "present." On a **unitary** role, dephasing never relaxes populations,
so the first sighting pinned every belief at z ≈ +0.95 and *nothing in 13 days of
dynamics ever brought it down*. The belief tape was flat (std ≈ 0.002, every node
identical), and — the treacherous part — AUC scoring still produced numbers that
looked excellent, riding 4th-decimal recency ripples near saturation, under an
**inverted** sign.

Symptoms to check for on any new world:

- a belief tape whose std is orders of magnitude below the evidence amplitude;
- sibling nodes with indistinguishable trajectories;
- sign calibration coming out inverted (evidence says +, scoring wants −);
- scores that survive only at implausible decimal precision.

The fix: declare the role **dissipative** in the spec's role modes. Evidence still
lands hard (z ≈ +0.95); the belief then relaxes toward maximal uncertainty on a
learnable per-node timescale (`gamma_diss` lives on the fiber, so the world tunes its
own forgetting). Measured end-to-end after the fix: sighting → +0.95, four quiet
hours → +0.09, re-sighting → re-lights. Signs turned natural and the scores above are
real belief dynamics.

**Why the proof gate never sees this:** the gridworld proof feeds both polarities —
the agent's absence from a cell is itself observed. No synthetic proof role is ever
one-sided. Real detectors are one-sided all the time. When authoring a spec, audit
every binding: *what does this role's evidence stream look like when the world is in
the state nothing reports?* If the answer is "silence," the role is dissipative.

## 2. Place is provenance, not radius

The one engine bug real data found (fixed in `53fe99a`): the gauge projection minted a
place token whenever the anchor qubit's Bloch radius crossed a threshold. Thirteen
days of real field dynamics drifted an **ungrounded** anchor past that gate — the run
ended with a geohash for a location that was never given. The 1-day synthetic proof
never drifts that far, so the gate was structurally blind to it.

The rule the fix encodes: **a coordinate may only be named from evidence, never from
state geometry.** Grounding is provenance; radius is just where the dynamics happen to
have pushed a qubit. Pinned forever by `tests/test_gauge_place_provenance.py`.

The general form: any projection that turns internal state into an external *claim*
(a place, an identity, a label) must check where that state came from, not what it
looks like.

## 3. Adapter honesty

The adapter is where a foreign world's semantics get translated, and it is where
silent lies enter. The discipline that worked:

- **Map event semantics before binding.** A transition in the raw stream is not
  necessarily an observation (see §1's re-arm OFFs). Read the sensor's manual, or
  measure its timing signature, before deciding what an event *means*.
- **Count every dropped row by (entity, reason).** 125,770 rows in, every single one
  either emitted or accounted for. Silent loss is forbidden; a drop table is the
  difference between "the adapter is honest" and "we hope."
- **Normalizer bounds are measured, never magic.** Analog ranges came from the
  export's own long-term statistics via robust percentiles (5th/95th of daily
  extremes — one glitch day reading 212 °F must not set a range).
- **Real exports truncate and gap.** Per-channel coverage windows were recorded at
  census time, and every downstream score was computed inside each channel's live
  window only. A channel that goes dark mid-run poisons any claim that ignores it.

## 4. The flow corrects the map

The declared topology is a hypothesis, and the world's metadata will lie to you: this
dataset's labels mis-placed four sensors (an exterior camera named like an interior
room, a mislabeled door/lock pair, an analog channel attached to the wrong node's
device, a room on the wrong floor). All four were caught not by reading metadata
harder but by the *behavior*: transition mining and the field's own learned couplings
kept disagreeing with the declared placement until the spec moved. One fix alone
lifted a room's held-out AUC from 0.77 to 0.87 — mis-binding doesn't just mislabel, it
pollutes the belief.

The loop that emerges is the useful pattern: **declare → replay → let the learned
structure argue with the declaration → correct → re-replay.** The spec converges on
the world's real shape because the field pushes back.

## 5. Topology growth on real data

The learned-topology machinery (`substrate/web_topology.py`, the tier-3
"dream-loop topology growth" claim), driven per batch over a blank growth replay of
the same stream: it grew **7 couplings within the first 36 hours of world-time and
pruned none over the remaining 11 days** — and every one of the 7 was independently
corroborated by offline co-movement statistics and by raw transition counts computed
without the engine. Nothing it grew was spurious; the strongest structure it found
(a 3-node co-movement triangle at r = 0.88–0.93) is the same structure both other
methods rank first.

This is replay, not live, so the tier-3 "live win" stays owed — but it is the first
real-data evidence that the growth mechanism finds *legible, true* structure rather
than noise.

## 6. What a real deployment asks for next

Generic gaps, in the order the real world surfaced them:

1. **Derived aggregate nodes.** A root node's own belief saturates when the world is
   almost always in one state (this world was "occupied" 95% of the time); detecting
   the rare complement needs an explicit derived node (a `reduce` over regions), not a
   deeper reading of the root. Same lesson as §1 in structural form: rare-state
   detection must be built, not hoped for.
2. **A live event-bus → engine bridge.** The adapter defines the mapping; a replay rig
   becomes a deployment the moment the rows arrive over a socket instead of a file.
3. **A GPU `SubstrateBackend`.** A 13-day house replays in minutes on CPU; the seam
   exists for the worlds that won't.

---

*Provenance: private rig, 2026-07-11; umwelt-engine 0.1.0.dev0 at `53fe99a`. The
household's extracted model was delivered to its owner and appears nowhere public.
This document intentionally contains no identifying details.*
