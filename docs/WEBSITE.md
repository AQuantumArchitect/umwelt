# The web surface — what actually exists, and the contract behind it

*(This file is cited by `projection/graph_state.py` and `projection/transparency.py`. It
existed only as a dangling reference until 2026-08-03; this is the thin, honest version —
what is real today, what the contract is, and what is deliberately still dark.)*

## What exists today

Two served pages, both from the supervisor (`umweltd`, default `:7071`; the yurt hearth
runs `:7073`):

- **`/ui` — the playground** (`src/umweltd/playground.py`). One self-contained HTML page,
  zero build step, zero external assets (air-gapped-LAN safe): pick a world, watch beliefs
  ease in near-real-time, push readings against the world's declared `/bindings`, read the
  shadow `/recommendations`. This is the only interactive web surface in the repo.
- **`/docs` — the docsite** (`src/umweltd/docsite.py`). Renders the curated `DOC_REGISTRY`
  ladder; `python -m umweltd.docsite --export <dir>` writes it static.

There is **no** `ui/transparency.html` and **no** `/api/transparency` route in this repo.
Those names come from the origin deployment (Meerkat) where `model_snapshot` feeds a
`/transparency` page; here the same data is served as part of **`GET /state`**.

## The contract

`projection/graph_state.py` is the single source of truth for any frontend:

- **One projection, every surface.** `GET /state` returns `graph_state(engine)`: the whole
  topology (nodes + bridge edges), each node's organs, plus global organs. The transparency
  view (`model_snapshot`) is a provable *subset* of it (`tests/test_graph_state.py`).
- **Self-describing organs.** Every organ carries a `type` from `KNOWN_ORGAN_TYPES`; a
  frontend maps `type → renderer` and never needs to know the page. A new organ type that
  isn't added to the registry fails the test — nothing can silently render as nothing.
- **Cheap reads only.** Never `engine.context()` (density-matrix rebuild); safe to poll.

Global organ types today: `agency`, `competence`, `run_gauge`, `body` (app-registered),
`ingest` (unmatched-signal legibility), `summary`, and — added 2026-08-03 —
`trust_web` (per-leaf learned observation-trust: which sources the system currently
distrusts) and `forecast_comprehension` (per-cluster foresight geometry:
disparity / downstream / meta_surprise; observe-only by contract). The last two appear
only when an engine actually carries the organ — honest absence, never a faked zero.

For the *human* read of any world without a browser: `umweltctl brief --world <name>`
(`projection/brief.py`) — glyph field lines, movers, staleness, shadow decisions, all
composed from the worker's own wire payloads. `--json` prints the same source dict.

## Deliberately still dark (catalogued, not built)

- **Tape readers** — `surprise_tape` / `stream_tape` accumulate with no query surface
  (only `berry_tape.snapshot()` surfaces, inside the engine state dict).
- **The provenance shelf / brain-lineage DAG** — `gauge_name.py` computes sortable names,
  `substrate/shelf.py` holds the DAG; no browser exists.
- **The clocks subsystem** (adaptive_clock, phi_clock, cadence_dial, breath_scorer) — no
  render surface.

Each of these is a real, deliberate gap: building a viewer is a separate decision, not an
oversight. If you find yourself needing one, that's the signal to promote it.
