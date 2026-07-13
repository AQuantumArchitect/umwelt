# Authoring a DomainSpec — a world as data

The engine runs ANY spec; no domain is a code path. This is the onboarding guide: author
a spec → boot blank → replay the domain's life → prove comprehension (the same path the
proof gate walks in `proofs/blank_slate.py`, which is also the best worked example).

## The five declarations

A `DomainSpec` (`umwelt.spec.schema`) is a frozen manifest of five things:

### 1. Nodes — the topology
A parent-referenced tree of `NodeSpec`s. Exactly one root (`parent=None`). Each node
declares its `roles` (the qubit axes it holds beliefs on), a `kind`
(`root | region | environment | clock | anchor | signal | actuator | entity | component | synthetic`
— domain dialects like `zone`/`sensor`/`person` are accepted as aliases), optional
`role_modes` (`{role: "unitary" | "dissipative"}` — event-kicked vs continuously-driven;
unregistered roles default to dissipative, the safe choice). **Read
[docs/FIELD_NOTES.md §1](FIELD_NOTES.md) before picking this** — a role fed only one
polarity of evidence (most sensor-derived roles) saturates permanently if left
`"unitary"`; that is the dissipative-role law, found the hard way on real data.
Optional `params` (`{name: (default, sigma, lo, hi)}` — this node's learnable priors;
the meaningful keys are `gamma_diss` / `gamma_diss_{role}`, the dissipative relaxation
rate — a bare `"gamma"` key is inert and does nothing, see `NodeSpec.params` in
`schema.py`), and optional `reduce` (`"max" | "mean" | "or"` — a derived belief
synthesized from the children's shared role).

### 2. Bridges — the lateral structure
`BridgeSpec(source, target, shared_roles, kind)` — `open | gated | wall` set the coupling
prior (1.0 / 0.7 / 0.3, then learned); a non-empty `role_map` makes the bridge a directed
tendril edge (region → actuator).

### 3. Bindings — the measurement vocabulary
`BindingSpec(sensor_id, zone=<node>, role=..., normalizer=...)` — every signal that will
ever arrive, each with a DECLARATIVE normalizer (`"binary"`,
`{"type": "regime", "center": 21, "width": 4}`, …) resolving through the registry in
`umwelt.spec.normalizers`; register domain idioms with `register_normalizer`. The
measurement model is explicit: `strength` (k) and `efficiency` (η) declare a weak
measurement; `collapse_alpha` is the folded legacy form. Signals that arrive but are
DELIBERATELY unbound go in `ignored` with a reason — the ingest gap should read
"0 actionable, N explained", never a mystery.

### 4. Outputs — the decisions
`OutputSpec(name, node, role, kind, decode, codomain, gates, coupling, readback_sensor,
dispatch)` — each becomes a live tendril at boot: the engine reads the node/role
continuously, pumps a slow committed belief (rise=`coupling.coupling`,
linger=`coupling.decay`), decodes through the decoder registry (`"sticky"` binary with
purity-derived hysteresis, `"linear"` onto the codomain; `register_decoder` for domain
shapes), gates (enable param / rate limit / device-unit deadband), and emits an Action.
**`shadow=True` is the default and the law**: a new output decides visibly and dispatches
nothing until the app flips it. `readback_sensor` names the channel on which operator
corrections arrive — overrides collapse the belief toward the revealed preference AND
move the learned rise/fall geometry.

### 5. Drivers — the time
`DriverSpec(name, node, role, type, period_s, rest_window)` — the domain's clock(s). The
engine anchors the named qubit toward the driver's phase each tick; the phase is fixed
physics, its comprehension is learned. `"harmonic"` is built in; register an ephemeris,
an exchange session calendar, or a game tick with `register_driver`
(`umwelt.clocks.drivers`). Anchor nodes you don't declare are materialized at boot.

## Booting it

```python
from umwelt.boot import build_engine
engine = build_engine(spec=MY_SPEC)          # or "my_pkg.specs:MY_SPEC", or UMWELT_SPEC env
result = engine.ingest(sensor_readings={...}, now=t)
```

The engine boots BLANK — max-entropy beliefs, unlocated, nothing assumed
(`engine.seed_profile == "blank"`). Anchors (`spec.anchors`) are grounded explicitly by
the app via `engine.ground_anchor(name, value, codec=...)` — unanchored means unanchored.

If your feed arrives sparsely (daily bars, not a dense sensor poll), declare
`DomainSpec.ingest_hold_s` — see [docs/TIME.md](TIME.md) for why this is cadence
plumbing, not a time model, and why the domain's own clock (if it has one) belongs in
a `DriverSpec` instead.

Vocabulary you register (role modes, normalizers, decoders, drivers) lives in YOUR
package, imported at boot — `src/umwelt/` itself is checked by
`tests/test_vocabulary_lint.py` to contain zero domain words, so nothing you add there
will ever land upstream by accident.

## Proving it

Steal the proof gate's shape (`proofs/blank_slate.py`): blank floor witnessed → every
binding drove the field (no dead vocabulary) → beliefs track your synthetic ground truth
→ the fiber drifted off its priors → the gauge coordinate stays honest → save/load
round-trips. A domain whose spec passes that harness is onboarded.
