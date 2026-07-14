# Hive relay — field notes from the first multi-LLM deployment

**Status: real-data replay; runs offline from a committed tape.** On 2026-07-14 a
relay of haiku-class LLM agents played SpaceWheat through its player-parity seat,
coordinated by one supervising model, with the game's in-engine umwelt port ("the
Witness") projecting a shared belief graph into every agent's observations. This is
the first use of the engine as a **hive-enabling structure** — many LLMs coordinating
on a complex task through one world state — and these are the honest field notes,
with the coordination data committed as `relay_tape.json`.

```bash
python3 examples/hive_relay/demo.py   # agent reports as sensors; the trust web prices them
```

## What worked (keep building on these)

1. **Comprehension inheritance across agent generations.** The belief field rides
   the game's save files, so each relay leg inherited its predecessors'
   *comprehension*, not just their inventory. A fresh agent never started blind.
   For the hive design: `engine.save/load` (or umweltd snapshots) between agent
   invocations IS the shared brain — it already exists and it already works.
2. **The compact graph projection is LLM-legible.** The <2KB graph_state (3dp,
   coherence-gated, emoji glyphs) fit small-model context and was parsed reliably
   across ~15 legs. The projection shape needs no redesign for LLM consumers.
3. **Attention beliefs are the coordination channel.** The most-used signal was
   `coverage` — agents navigate by *where nobody has looked* more than by expected
   payoff. In a many-writer hive, that is the anti-duplicate-work channel: shared
   coverage beliefs are how N agents avoid doing the same job.

## What the deployment found (the tape in this directory)

**Confabulation is the central enemy of LLM coordination, and it is not rare.**
Across nine verified legs, agents' structured self-reports were wrong 4/9 times on
the headline question ("did this leg advance the story?") — three over-claims, one
*under*-claim (an agent failed to recognize its own breakthrough). One agent reported
checkpoint banks that never existed; the next agent loaded the phantom and burned an
entire leg in a wrong world state. Every wrong report was fluent, well-formatted,
and confident.

The supervisor's manual fix — verify every claim against a manifest diff before
acting — is exactly this engine's trust-web contract, and the demo shows it on the
real tape:

- **Agent reports are sensor readings at honest η, never writes.** A claim ingests
  at the reporter's earned confidence; it moves shared belief only as far as its
  η allows (the confidence contract, doing coordination work).
- **Referees are just more sensors.** A directory listing, a manifest diff, a test
  result: cheap, boring, η≈1.0. With one referee present, the fused state tracks
  truth and the confabulator's weight collapses to what it earned.
- **With ≥3 heterogeneous reporters, no privileged oracle is needed** — the
  leave-one-out form isolates the confabulator against its peers' consensus alone
  (the engine's pinned isolation claim, now exercised on real multi-LLM data).

## What didn't earn its keep yet (honest)

- **Reading beliefs ≠ acting on them.** The A/B run (belief graph vs screen text)
  tied on the outcome metric; the graph arm's own debrief: *"used it as
  confirmation rather than strategy."* Value appeared only when the agent's
  instructions said HOW to decide from the signals. Design implication: the hive
  projection should carry **salience/recommendations** (the `recommendations()`
  surface), not just state — or the coordinator must ship a decision vocabulary
  with every agent.
- **The surprise tape hasn't caught anything live yet** — purpose-built probes found
  the deep bugs first. It remains the right shape for unattended anomaly detection;
  unproven in anger.
- **Diff-stable gauges beside checkpoints** ("what did this agent's shift teach the
  field") are banked on every leg and have never been read operationally. An unused
  instrument is a design smell or a missing habit — undecided which.

## The recommended hive architecture (from this deployment's scars)

```
agents (N, small)  ──claims at η<1──►  one umwelt world (umweltd)
referees (cheap)   ──truth at η≈1──►   ├─ trust web prices every reporter
                                       ├─ beliefs/coverage = shared task state
coordinator (1)    ◄──graph_state──────┘  + recommendations for the next agent
        └──── snapshots between invocations = the hive's continuity
```

The pieces all exist: umweltd multi-writer ingest, per-source trust, the LLO
referee-free mode, graph_state, snapshots. What this deployment adds is the
evidence that the contract is *necessary*: uncalibrated agent writes corrupted the
shared state 4 times in 9 on real work, and the calibration machinery priced it
correctly from the tape alone.
