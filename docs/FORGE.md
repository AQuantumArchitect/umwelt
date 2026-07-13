# umwelt-forge — the embedded compiler + warden. EXPERIMENTAL.

The seam between "a plain-English description of a domain" and "a running umweltd
world". An embedded coding agent (Claude Agent SDK) authors the `DomainSpec` module;
a **deterministic gate decides** whether it's real; only a green gate registers the
world. The agent's own success claim is never trusted — the pipeline re-runs
`python -m umwelt.spec.validate` in a fresh subprocess after every session, and a
lying agent provably cannot register a broken world (test-pinned:
`tests/test_forge_pipeline.py`).

Status honesty: the *pipeline discipline* is test-pinned and runs offline in the
gate. The *authoring quality* — how often the agent produces a correct world from a
real rant — is unmeasured (CLAIMS.md: evaluation owed). Treat this as an
experimental v1.

## Requirements

```bash
pip install "umwelt-engine[forge]"       # adds claude-agent-sdk
export ANTHROPIC_API_KEY=sk-ant-...      # the embedded agent's auth
```

The repo's own test gate needs **neither** — every forge test runs against scripted
agents (`UMWELT_FORGE_AGENT=module:factory` is the same seam, available to you for
offline/CI use).

## The compile pipeline

```bash
umweltd &                                          # the daemon
umwelt-forge new greenhouse --rant "I have a small greenhouse with a fan and a
  heater, a temperature sensor and a humidity sensor, and I care about whether
  it's warm enough overnight..."
```

What happens:

1. A workspace is created (see layout below); the rant, the distilled authoring
   guide (`GUIDE.md`), and an all-watch warden policy land in it.
2. The agent authors `world_greenhouse.py` (defining `SPEC`) in the workspace,
   running the gate itself as it iterates.
3. The pipeline **independently** re-runs the gate in a fresh subprocess. Red →
   the failure report is fed back verbatim and the agent tries again (default 3
   attempts). Still red → the workspace is kept for a human; nothing registers.
4. Green → `create_world(name, spec="world_greenhouse:SPEC",
   spec_path=<workspace>)`. The manifest's `spec_path` means every future respawn
   (watchdog, restart) imports the module from the workspace — the world is
   self-sufficient from here on.

Every output in a forged world is `shadow=True` — the gate fails anything else.
The world decides visibly and dispatches nothing until you promote it.

## Workspace layout

Default root `~/.umwelt/forge` (override: `UMWELT_FORGE_ROOT` or `--root`).

```
<root>/<world>/
    rant.txt              the description, verbatim
    GUIDE.md              the authoring law the agent worked from
    world_<name>.py       the generated spec module — the daemon imports from here
    attempts/             every attempt's module + validation report (audit trail)
    warden/
        policy.json       the earned-autonomy dials
        ledger.jsonl      the append-only competence ledger
        ticks/<ts>/       per-tick context, findings.json, per-proposal diffs
        staging/          candidate modules under validation
```

## The warden

One-shot inspection of a running world — cron-able:

```bash
umwelt-forge warden tick greenhouse
# e.g. crontab: 17 4 * * * umwelt-forge warden tick greenhouse
```

The tick gathers the world's live surfaces (health, state incl. the
unmatched-signal gap, shadow recommendations, ledger tail) into `context.json`,
runs a **read-only agent session** (Read/Write only, no shell) whose entire output
contract is one `findings.json`, and ledgers every proposal.

### Earned autonomy

Authority is a per-world, per-change-type dial — everything starts at `watch`
(propose-only), the engine's own shadow law applied to the warden itself:

| change-type | scope | auto-apply? |
|---|---|---|
| `normalizer_tune` | normalizer params only | promotable |
| `param_tune` | `NodeSpec.params` (`gamma_diss*`) | promotable |
| `binding_add` | new binding, evidenced in the unmatched surface | promotable |
| `binding_remove` | drop a binding | promotable |
| `topology_change` | any node/bridge/sector change | **never** — human only |

```bash
umwelt-forge warden status greenhouse            # dials + competence summary
umwelt-forge warden promote greenhouse param_tune
umwelt-forge warden demote greenhouse param_tune
umwelt-forge warden accept greenhouse p3         # human verdict on a watched proposal
umwelt-forge warden reject greenhouse p4
```

A change-type dialed to `run` lets the tick auto-apply a proposal — but only
through the same staging gate: the proposed module validates in a fresh subprocess
first; red means ledgered `validation_failed` and an untouched world. `accept` runs
the identical path with `by: "cli"` on the ledger.

The warden cannot promote itself: promotion is CLI-only, `policy.json` is
hash-checked around the agent session (a session that rewrites it is reverted and
ledgered `policy_tampered`, and that tick is treated as all-watch), and
`topology_change` is clamped to watch even if the file on disk says otherwise. All
test-pinned in `tests/test_warden_tick.py`.

## Defined behaviors worth knowing

- **An apply restarts the world.** `stop` snapshots (SIGTERM), `start` re-imports
  through `spec_path` and replays the event-log tail — *under the new spec*. That
  is the event-sourcing contract (the log is truth): a `binding_remove` lands old
  readings for that id as unmatched during the tail replay; a `normalizer_tune`
  reinterprets the tail from the snapshot cursor forward.
- **The stop→start window 503s pushers.** A data pusher hitting the supervisor
  proxy during the restart gets a 503; retry is the pusher's job.
- **`spec_path` is arbitrary code execution by design** — the exact trust already
  granted to the `spec` ref itself (both import a module into the worker). The
  workspace is the trust boundary; don't point a world at a directory you don't
  control.

## Cost & offline notes

A compile run is a few agent-loop turns plus 1–3 gate subprocesses — order
$0.05–0.15 per attempt at current API pricing; a warden tick is usually cheaper.
The `--agent module:factory` flag / `UMWELT_FORGE_AGENT` env swaps in any
`ForgeAgent`-shaped implementation (see `umweltforge/agent.py`) — the repo's own
tests are the reference for scripted agents.
