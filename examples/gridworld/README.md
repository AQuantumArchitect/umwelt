# Gridworld — the new-domain template AND the proof-gate domain

**Status: runs in CI.** This is both things at once, deliberately: the domain the
library proves itself on, and the copy-paste starting point for a new domain — see
[docs/NEW_DOMAIN.md](../../docs/NEW_DOMAIN.md) for the full checklist. Because
`proofs/blank_slate.py`, `proofs/ladder_walk.py`, and `proofs/deconfound_smoke.py` all
import [`world.py`](world.py) directly rather than owning their own copy, the example
you'd copy is never allowed to drift from what the gate actually proves comprehension
on.

```bash
python3 examples/gridworld/demo.py     # boot blank, replay a synthetic day, watch belief ease
```

Fog-of-war is partial observation with confidence: seeing a cell is an observation at
the scout's η; scouting buys reliability; unobserved cells' beliefs *ease* back toward
uncertainty instead of freezing at the last sighting. The demo prints the belief field
as a per-cell heatmap at checkpoints: watch the occupied cell's belief rise on a
sighting and ease — not snap — as the agent moves on.

## What's here, and what to copy

- [`world.py`](world.py) — `gridworld_spec()` (the `DomainSpec`: nodes, bridges,
  bindings, shadow outputs, one harmonic driver), `agent_walk()` (seeded synthetic
  ground truth), `synthesize_rows()` (the wire-shaped event stream). This is the file
  to fork: copy its shape, swap the grid/agent for your own topology and generator.
- [`demo.py`](demo.py) — a thin viewer: boot blank, replay, print. Copy this to get a
  first look at your own domain's belief dynamics.
- The **proof** for this domain is `proofs/blank_slate.py::
  test_blank_gridworld_engine_comprehends_a_synthetic_day` — steal its shape
  (blank floor witnessed → every binding drove the field → beliefs track ground truth
  → the fiber drifted → the gauge stayed honest → save/load round-trips) for your own
  domain's proof; `proofs/` isn't shipped in the installed package, so vendor the
  harness the way `umwelt-market` did.

Owed here (see CLAIMS.md): a path-topology decision demo on this domain (looping vs
out-and-back geometric phase is test-pinned elsewhere; live decision authority still
owed), and the capture-the-flag bot built on OutputSpec tendrils.
