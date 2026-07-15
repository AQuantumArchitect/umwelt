# Changelog

Notable changes, loosely following [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
See [CLAIMS.md](CLAIMS.md) for the full evidence ledger behind any claim made here —
if the two ever disagree, CLAIMS.md wins.

## [Unreleased]

### Fledgeling core (FL-core Phases 1–5)
- **Host API** (`umwelt.host`): thin game face — `GameHost.register_world` /
  `observe` / `intend` / `beliefs` / `step` / `step_turn`; calibrated
  `Belief(value, confidence)` on the default face (no substrate vectors); η=0 no-op;
  shadow intents leave no live world side effects (`tests/test_host_api.py`).
- **Multi-mind** (`WorldSession`): shared classical ground + N private engines,
  per-observer channel masks, actor-keyed intent logging on the confounding
  surface (`tests/test_multimind_privacy.py`).
- **Agency loop** (`umwelt.host.agency_loop`): sub-routines, attention budget,
  FF pause on surprise/rest, shadow auto-intend only after
  `PromotionGate.min_successes`, explicit live promotion
  (`tests/test_agency_loop.py`, `examples/fledgeling_fog/agency_demo.py`).
- **Fog corridor domain** (`examples/fledgeling_fog/`): public synthetic place
  graph, tick driver (not solar), host-API demo + freeze bake-off + blank-slate
  proof (`proofs/fledgeling_fog_blank.py`).
- **Facet kits** (`umwelt.kits.{fog,attention,market,dream}`): optional cassette
  + baseline + README honesty; dream path never actuates
  (`tests/test_facet_kits.py`).
- Roadmap status: [docs/FLEDGELING_CORE.md](docs/FLEDGELING_CORE.md) (Phase 6
  host-repo integration still open). Ledger rows in CLAIMS.md.

### Engine
- `umwelt.spec.validate` — the deterministic spec gate, one reusable command
  (`python -m umwelt.spec.validate module:ATTR`): topology, strict binding
  registration (surfacing exactly what the boot path's membrane guard swallows),
  blank boot, a synthetic exercise proving every binding drives the field (driver-
  role bindings exempt — the ingest path routes them out of band), and a save/load
  round-trip. The shadow law is enforced (`shadow=False` fails unless waived).
- Actor-keyed confounding helpers (`record_actor_intent`, `actor_confounded_now`)
  extend the graph-derived surface for multi-mind hygiene — they do not replace it.

### umweltd
- `spec_path` manifest knob: worlds whose spec module lives outside the installed
  packages (a forge workspace) boot and event-source-recover through the manifest
  alone; the worker prepends it to `sys.path` before the spec ref imports.
- **The playground** (`/ui`): a self-contained browser dashboard served by the
  supervisor — live per-role belief bars, a push-readings panel over the new
  `GET /worlds/<n>/bindings` endpoint, the shadow-decision feed, raw state. Loads
  without auth (static, no world data); its API calls carry the visitor's key.
- **The docs site** (`/docs`): project docs rendered server-side from the checkout
  (plain-terms overview first); `python -m umweltd.docsite --export DIR` writes the
  same pages as a standalone static site. `UMWELTD_UI=off` disables both surfaces.
- `umweltctl bindings --world <w>` and `UmweltClient.bindings()`; `umweltctl
  create --spec-path` exposes the manifest knob from the operator CLI.
- SERVICE.md gained a "Sharing on your LAN" recipe (API key, WSL2 port-forward
  note, trust model).

### umweltforge — the embedded compiler + warden (EXPERIMENTAL)
- `umwelt-forge new <name> --rant "..."`: an embedded coding agent (Claude Agent
  SDK, optional `forge` extra) authors the `DomainSpec` module in a scoped
  workspace; the pipeline independently re-runs the deterministic gate in a fresh
  subprocess and registers the world only on green — a lying agent provably cannot
  register a broken world (test-pinned, offline, no API key in the repo gate).
- `umwelt-forge warden tick <name>`: cron-able one-shot inspection of a running
  world under **earned autonomy** — per-world, per-change-type dials, everything
  defaulting to propose-only; `topology_change` can never auto-apply; the warden
  cannot promote itself (policy hash-checked around the session); every proposal,
  apply, and failure lands on an append-only competence ledger.
- Authoring quality on real rants is unmeasured — evaluation owed (see CLAIMS.md).

### Examples
- `examples/mirror/` — the daemon's self-portrait: a world whose five sensors
  are umweltd's own `/health` telemetry (including the byte-size of its own
  event log, which grows because of the readings that measure it), fed back in
  through `POST /events`. The first sitting witnessed a sibling world's
  operator-initiated outage as a single `gh_alive = 0` reading and recovered;
  it doubles as an end-to-end exercise of the whole service API.

## [0.1.0] - 2026-07-12

First tagged release.

### Engine
- The belief-field engine: `DomainSpec`-driven worlds, blank-slate boot, weak-
  measurement observation (confidence = η, η=0 provably a no-op), forecasting (trust
  web, free-run rollouts), and shadow-first output tendrils with causal self-tagging.
- Membrane cadence (`ingest_hold_s`) for sparse-feed worlds, kept structurally
  separate from compute time and from a domain's own in-universe clocks
  ([docs/TIME.md](docs/TIME.md)).
- Blank boot mixes analog-dissipative beliefs to max entropy — no boot transient.
- Leave-one-out trust-web learning isolates a corrupted source when a referee exists.
- Spec binding validation: a `BindingSpec` targeting a nonexistent node or an
  undeclared role now raises (and is always logged), instead of failing silently or
  warn-only.

### umweltd — the engine as a service
- Supervisor + per-world worker daemon: event-sourced boot (snapshot + log-tail
  replay), API-key auth (constant-time compare), TLS, webhook dispatch for non-shadow
  outputs, gauge discipline (REPLAY on recovery, LIVE once caught up).
- `umweltctl` operator CLI and a `Dockerfile`/`docker-compose.yml` for one-command
  startup.
- Structured logging plus a per-request JSON access log.
- A crash watchdog auto-restarts a world that dies unexpectedly, backing off after
  repeated crashes in a window rather than tight-looping; `UMWELTD_MAX_WORLDS` caps
  world count; `/health` reports per-world disk usage.

### Docs & onboarding
- [docs/NEW_DOMAIN.md](docs/NEW_DOMAIN.md): the one-page checklist for a new domain,
  indexing what used to be scattered across several docs (the dissipative-role law,
  the `gamma_diss` knob, `ingest_hold_s`, the vocabulary lint gate).
- `examples/gridworld/` promoted to a complete, standalone new-domain template —
  also the proof gate's own fixture, so the two can never drift apart.
- README leads with Install + a working Quickstart before the evidence/claims
  framing.

See [CLAIMS.md](CLAIMS.md) for what's measured-and-pinned versus designed-but-owed
versus denied.
