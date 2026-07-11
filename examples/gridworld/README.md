# Gridworld — the proof-gate domain

**Status: runs in CI.** This is the domain the library proves itself on: fog-of-war is
*literally* weak measurement. Seeing a cell is an observation at the scout's confidence;
scouting is buying η; unobserved cells' beliefs decohere back toward the prior instead of
freezing at the last sighting. The blank-slate theorem (`proofs/blank_slate.py`) and the
estimator-ladder harness (`proofs/ladder_walk.py`) both run on this world.

```bash
python3 examples/gridworld/demo.py     # boot blank, replay a synthetic day, watch belief ease
```

The spec + deterministic day generator live in `proofs/_gridworld.py` (single source —
this example is a viewer over the proof harness, not a fork of it). The demo prints the
belief field as a per-cell heatmap at checkpoints: watch the occupied cell's belief rise
on a sighting and *ease* — not snap — back toward uncertainty as the agent moves on.

Owed here (see CLAIMS.md): the Berry-phase decision demo — looping around a region vs
scouting out-and-back leave different geometric phase (the topology is test-pinned in
`tests/test_berry_geometric`-lineage tests; the *decision authority* demo is the open
gate), and the capture-the-flag bot built on OutputSpec tendrils.
