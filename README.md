# umwelt

**A belief-field engine.** Describe a world as data; feed it observations; it holds a
live, honest, uncertain comprehension of that world — and forecasts from it.

## Install

```bash
git clone <this repo> && cd umwelt
pip install -e .          # not on PyPI yet — local/editable install only
```

## Quickstart — a tiny world

```python
from datetime import datetime, timedelta, timezone
from umwelt.spec.schema import DomainSpec, NodeSpec, BridgeSpec, BindingSpec, DriverSpec
from umwelt.boot import build_engine

spec = DomainSpec(
    name="tiny-world",
    nodes=(
        NodeSpec("hall", parent=None, kind="root", roles=("occupied",)),
        NodeSpec("den", parent="hall", roles=("occupied", "warmth"),
                 role_modes={"occupied": "unitary", "warmth": "dissipative"}),
    ),
    bridges=(BridgeSpec("hall", "den", shared_roles=("occupied",), kind="open"),),
    bindings=(
        BindingSpec("den_motion", zone="den", role="occupied", normalizer="binary",
                    force_observe=True),
        BindingSpec("den_temp", zone="den", role="warmth",
                    normalizer={"type": "regime", "center": 21, "width": 4}),
    ),
    drivers=(DriverSpec("day", period_s=86400.0),),
)

engine = build_engine(spec=spec)              # boots BLANK: max-entropy, located nowhere
t = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
for i in range(10):
    engine.ingest(sensor_readings={"den_motion": 1.0, "den_temp": 18.2},
                  now=t + timedelta(seconds=10 * i))

z = engine.field.clusters["den"].role_bloch("occupied")[2]
print(f"den occupied: z={z:+.2f}")            # belief, not a flag — with uncertainty attached
engine.save("engine_state.pkl")               # canonical, hash-stable, diffable
```

This is the gate-pinned path: the same construction the proof suite drives. Starting
your own domain? [docs/NEW_DOMAIN.md](docs/NEW_DOMAIN.md) is the checklist, and
[examples/gridworld/](examples/gridworld) is a complete one to copy.

## Why this exists, and what it's already measured

Its de-confounding mechanism was measured on 24 days of real home data: a naive learner
credits the system's own actions at **10.8×** their true strength; the self-tagging
learner cuts that bias **79%**. Every claim in this README links to a row in
[CLAIMS.md](CLAIMS.md) — including the claims we measured and **rejected**.

> An [umwelt](https://en.wikipedia.org/wiki/Umwelt) is a world as modeled by an organism.
> That is what this library builds: give it a domain's umwelt as a declarative spec and
> it grows a live world-model it can forecast from and act through.

## The 60-second mental model

```
DomainSpec (a world as data: nodes, roles, bridges, bindings, outputs, drivers)
    → the belief field         coupled qubit clusters on a graph; evolves continuously
    → observe = partial collapse   confidence IS measurement strength (η=0 ⇒ provable no-op)
    → forecast                  the same dynamics run forward; a sensor is a forecast at horizon 0
    → act                       committed tendrils at the device edge — self-tagged, so the
                                engine never learns its own reflection as world signal
```

- **Spec** — a frozen manifest ([docs/SPEC.md](docs/SPEC.md)). A world is data; the
  engine runs any spec. Proven by the blank-slate gate: an unconfigured engine boots from
  an arbitrary spec and comprehends a synthetic day (`proofs/blank_slate.py` — it runs in
  CI, it is not a slogan).
- **Field** — each node holds a small belief cluster plus a learnable parameter fiber;
  couplings are learned; the whole mind is one self-describing graph projection
  (`umwelt.projection.graph_state`).
- **Observe** — a weak measurement. Belief *eases* between sparse reports; it does not
  snap. A zero-confidence observation provably does nothing ([docs/THEORY.md](docs/THEORY.md)).
- **Forecast** — one learned fuser (the trust web) handles sensor health, forecast
  ensembling, and engine chaining, prior-initialized so day-1 behavior is unchanged
  ([docs/forecasting.md](docs/forecasting.md)).
- **Act** — outputs are data too (`OutputSpec`): shadow by default, they decide visibly
  and dispatch nothing until you opt in; operator corrections move their learned
  geometry. Actions carry a decaying echo over a graph-derived confounding surface, so
  world-model learning discounts what the engine itself caused.

## The novel parts

Ordered by how bulletproof the evidence is — details and provenance in [CLAIMS.md](CLAIMS.md).

**1. Causal self-tagging — the system knows what it caused.** An anticipatory actuator
poisons its own world model ("we acted, the world changed, so we learned the world does
that"). The confounding surface is derived entirely from the world graph — an actuator
confounds exactly the learned roles its state projects onto, no per-device code — and a
per-channel gate discounts learning by the action's decaying echo. Measured on the origin
deployment's real data: 10.8× naive self-crediting, cut 79% by tagging. The mechanism is
quantum-independent (`umwelt.learning.confounding`, `learning_router`;
[docs/papers.md](docs/papers.md); figure: `docs/figures/deconfound_ab.png`).

**2. Confidence is measurement efficiency — as a theorem, not a convention.** Every
observation carries η ∈ [0,1]; the update law is a weak quantum measurement in which η=0
makes the innovation vanish identically. A garbage parse, a flaky sensor, a hedged
forecast — none can move a belief faster than its admitted confidence allows
([docs/THEORY.md](docs/THEORY.md), test-pinned).

**3. The trust web — one operator, three problems.** A sensor is a forecast with horizon
0, so sensor-health rerouting, forecast ensembling, and engine chaining are one learned
per-leaf fuser — prior-initialized so turning it on changes nothing until it has evidence
([docs/forecasting.md](docs/forecasting.md)).

**4. Provable non-training.** Every learnable parameter lives on a gauge-tracked fiber;
snapshots are deterministic and diff-stable. A frozen subsystem produces an *empty git
diff* across days of operation — "this did not train on you" as a checkable property,
not a promise (figure: `docs/figures/empty_diff_figure.png`).

**5. Worlds as data.** The spec seam plus the blank-slate proof: nodes, measurements,
decisions, and time are all declarative; the engine ships zero domain vocabulary — a lint
test fails the build if a domain word ever appears in engine source.

*Further out, honestly hedged:* Berry-phase process-time (path-topology memory — the
geometry is test-pinned: a loop accumulates γ=−π, an out-and-back ≈0; the *decision*
demo is owed) and free-energy reward channels (framework shipped; effect sizes owed).

## The ledger

[CLAIMS.md](CLAIMS.md) sorts every claim by tier — measured-and-test-pinned here,
measured on the origin deployment, measured on the first foreign world,
designed-but-owed, and **DENIED**: the full Belavkin
filter lost to a two-line α-blend on real data (0.3034 vs 0.1349 MAE) and ships OFF; a
headline speedup re-measured from 3.4× to 1.1× and was corrected. If this README and the
ledger ever disagree, the ledger wins.

## Case study: 18 months in a real house

The engine was extracted from **meerkat**, a home-comprehension system that has run for
18 months on a $100 ARM board in a real apartment with a real resident — real sensors,
real actuators, a ~1,450-test release gate, and autonomy earned per-output (mostly Watch
mode; Run is competence-gated). One real day of the belief field easing between sparse
sensor ticks: `docs/figures/belief_field_day.png`. Meerkat remains the flagship
deployment and the source of the measured evidence.

## Second contact: the first foreign world

In July 2026 the extracted engine met real data it was not built around — a 13-day
smart-home export from a house nobody on this project had ever instrumented. It booted
blank, replayed through the production ingest path, held every blank-slate assertion,
and beat the persistence of its own inputs in every well-sensed room (belief AUC
0.94/0.87/0.79 vs 0.60–0.67 baselines). The data stays private; the transferable
lessons — including the **dissipative-role law** (a role fed only one polarity of
evidence must forget, or it saturates) and the place-provenance fix the run forced —
are in [docs/FIELD_NOTES.md](docs/FIELD_NOTES.md).

## What this is not

- Not yet multi-domain in production: one domain (the home) is deployed; the adapters in
  [examples/](examples/) are designed sketches with synthetic demos owed.
- "Quantum" always means *classically simulated open-quantum-system dynamics* — and the
  ablations are mixed: the production estimator is a cumulant closure that beat the full
  density matrix on economics, and persistence baselines are genuinely hard to beat.
- A 0.x API. The origin deployment remains the source of truth until it imports this
  library and its full gate stays green — that back-port is the 1.0 trigger.
- Not an ML framework, not an LLM, not a drop-in Kalman replacement.

## Where this is going

| Domain | Why this engine | Status |
|---|---|---|
| [Gridworld bot](examples/gridworld/) | Fog-of-war IS weak measurement; scouting buys η | Proof-gate domain (runs in CI) |
| [Resilience recommender](examples/resilience-recommender/) | The recommender feedback loop is the 10.8× trap | Sketch |
| [Butler](examples/butler/) | LLM parses at their honest η; non-training as privacy | Sketch |
| [Sentiment ↔ market](examples/sentiment-market/) | Trust-web fusion; ships its own baselines | Sketch |
| [Smart home](examples/smarthome/) | The origin — 18 months live | Deployed (meerkat) |

## The engine as a service

`umweltd` (in `src/umweltd/`, [docs/SERVICE.md](docs/SERVICE.md)) runs worlds as a
local daemon: one worker process per world, an events.db write-ahead log, snapshots,
and boot = snapshot + log-tail replay through the production ingest path. Its founding
claim runs in the gate: **the daemon adds nothing and loses nothing** — wire replay
hash-equals library replay, and a killed worker recovers its exact state
(`tests/test_daemon_parity.py`). Start it with `umweltd` (or `docker compose up`); talk
to it with `umweltd.client.UmweltClient` or the `umweltctl` CLI. Harness repos stay
offline-first; the daemon is the live deployment shape.

## Origin & license

Extracted from meerkat (private) at its flagship-b10.1 release; every curated cut is
ledgered in `tools/RENAMES.md`. Apache-2.0.
