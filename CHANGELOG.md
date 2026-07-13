# Changelog

Notable changes, loosely following [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
See [CLAIMS.md](CLAIMS.md) for the full evidence ledger behind any claim made here —
if the two ever disagree, CLAIMS.md wins.

## [Unreleased]

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
