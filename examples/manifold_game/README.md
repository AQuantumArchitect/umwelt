# Manifold game — the second foreign world, and the Berry-decision demo

**Status: real-data replay; runs offline from committed tapes.** The domain is
[SpaceWheat](https://github.com/AQuantumArchitect/SpaceWheat), a quantum farming
game whose biomes are density matrices evolved by a native Lindblad engine — a
game literally about navigating manifolds. Everything under `data/` was recorded
from the game's **player-parity LLM-playtester seat** (rows are only what a
player could read off the screen) and exported by the game's
`🍄/🛠️/export_umwelt_tape.py` / `🍄/🧪/berry_tape_drive.py`. Nothing here talks
to the game; this is replay through the production ingest path.

The pairing runs the other way too: SpaceWheat grew an in-engine port of this
library's architecture ("the Witness", `Core/Witness/` in its repo) — weak-
measurement observation of player-visible events on its own native substrate,
graph_state-v1 projection, gauges, the dissipative-role law — so its LLM
playtesters navigate the game by belief graph instead of screen-text
archaeology. This example is that bridge's return traffic.

```bash
python3 examples/manifold_game/demo.py            # blank boot → replay → honest prequential score
python3 examples/manifold_game/berry_decision.py  # a choice that flips because of winding
```

## What's here

- [`world.py`](world.py) — `manifold_spec()`: a pantry of dissipative wallet
  beliefs + a story-progress belief; normalizer bounds are data
  (`data/bounds.json`), refit from the train split only (adapter honesty,
  FIELD_NOTES §3). Sparse tape → `ingest_hold_s` (docs/TIME.md).
- [`demo.py`](demo.py) — replay a real session, score belief vs persistence
  prequentially on the held-out tail, and SAY SO when the tape has no signal
  (a stuck session scores as an honest null, and one of the committed tapes
  is exactly that — it's the session that caught a game bug).
- [`berry_decision.py`](berry_decision.py) — **the Berry-phase decision demo.**
  Two real tapes off the game's native solid-angle integrator: a driven loop
  and a same-duration pole-hugging control. A finding the tapes forced (kept,
  not hidden): the game's world never sits still — its boot deliberately
  kicks stationary states — so even the control winds slowly. The honest
  contrast is RATE: one decision rule, reading only accumulated γ, opens the
  loop's harvest gate ~8× sooner, and **at a fixed time budget the two
  processes yield opposite choices** — stamped onto this repo's
  `BerryStamper` as it goes.

## Provenance, honestly

- The wallet tape is a REAL LLM-playtester session. Real sessions include the
  one where the tester was stuck on a game bug — that tape is flat, the demo
  reports it as a null, and that is the correct reading of it.
- The berry tapes come from the game's own Berry-phase register (Bloch
  polyline, L'Huilier solid angle per slice, ripeness at |γ| ≥ 2π) — the same
  machinery its players harvest by. This repo's origin pin (loop → γ=−π,
  out-and-back ≈ 0) stayed at the origin; these tapes are the first FOREIGN
  geometry through the berry-tape API.
- What this does NOT show yet: a live deployment where the Berry gate makes a
  consequential decision in production. The claim moves from "designed" to
  "demonstrated on real foreign geometry, replayed"; live authority is still owed.
