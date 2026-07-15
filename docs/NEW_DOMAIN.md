# Starting a new domain — the checklist

This is the index, not a duplicate. Every step below already has a canonical home
elsewhere in this repo; this page just puts them in the order you'll actually hit
them and tells you which doc/file to open at each step. It exists because the first
external consumer (umwelt-market) had to rediscover several of these the hard way —
see the "traps" callouts, each one a real thing that shipped once before it was caught.

## 0. Start from a template, not a blank page

Copy [`examples/gridworld/`](../examples/gridworld) — it is a complete, standalone,
CI-running domain (spec + vocabulary idioms + a runnable demo + the proof shape) and
the only example in this repo that is not a fragment. Read its README first.

If your domain's vocabulary-registration needs are closer to a sensor-driven world
(role modes, normalizers, a custom driver), also skim
[`examples/smarthome/`](../examples/smarthome) — narrower, but the worked reference
for that idiom specifically.

## 1. Write your `DomainSpec`

The five declarations (nodes, bridges, bindings, outputs, drivers) are documented in
[docs/SPEC.md](SPEC.md) — read it top to bottom once, it's short. Two traps SPEC.md
now links inline but are easy to miss on a skim:

- **The dissipative-role law** ([FIELD_NOTES.md §1](FIELD_NOTES.md)): any role fed
  only one polarity of evidence (most real sensors) must be `role_modes={role:
  "dissipative"}` or it saturates and never comes back down. The synthetic proof gate
  can't catch this — it feeds both polarities by construction.
- **The gamma trap**: `NodeSpec.params={"gamma": ...}` does nothing. The real
  learnable relaxation knob is `gamma_diss` (or `gamma_diss_{role}` per-role) —
  see `NodeSpec.params` in `schema.py`.

If your feed arrives sparsely (daily bars, not a dense poll), declare
`DomainSpec.ingest_hold_s` — [docs/TIME.md](TIME.md) is the short read on why that's
cadence plumbing, not a time model, and why your domain's actual clock (if it has
one) belongs in a `DriverSpec` instead.

### Worlds of reports

Some worlds are fed **discrete claims**, not continuous readings: agent messages,
session digests, test verdicts, manifest diffs. The first multi-LLM hive deployment
([examples/hive_relay/README.md](../examples/hive_relay/README.md)) ran exactly this
shape and left a scar worth naming: dissipative thermal-drive bindings barely move on
isolated events, and with sessions hours apart a spec whose gammas satisfy
`gamma × gap ≫ 1` relaxes every belief back to uncertainty between batches — no error
anywhere, just a mute world. (`python -m umwelt.spec.validate` now warns on this
combination when `ingest_hold_s` is declared.)

For report-shaped signals, bind them as **observations**, not drive:

- `force_observe=True` on the binding — a claim lands as a collapse event, moving the
  belief once and letting it *hold* (and be priced by the trust web), instead of
  needing sustained drive to matter.
- `collapse_alpha` (or `strength`/`efficiency`) as the reporter's honest η — how far
  one claim may move shared belief. The hive tape's lesson: agent self-reports were
  wrong 4/9 times, so uncalibrated writes corrupt shared state; η-weighted
  observations are the confidence contract that priced it correctly.
- Keep dissipative bindings (and their gammas) for quantities that are genuinely
  continuous *within* a session; give slow relaxation timescales to anything that
  must survive the gaps *between* sessions.

## 2. Register your vocabulary

Role modes, normalizers, decoders, drivers — registered at import in YOUR package,
never in `src/umwelt/` (`tests/test_vocabulary_lint.py` enforces a zero-domain-word
engine; if you ever see that test fail on a PR against this repo, the fix is to move
the word into your own package, not to widen the allow-list lightly).
`examples/smarthome/vocabulary.py` is the worked pattern:
`register_role_mode`, `register_normalizer`, a custom `DriverSpec` type.

## 3. Boot blank and check the ingest gap

Before anything else, run the one-command spec gate — it packages every check below
(topology, strict binding registration, blank boot, "every binding drove the field",
save/load round-trip) and exits nonzero with the exact failure named:

```bash
python -m umwelt.spec.validate your_module:SPEC        # add --json for tooling
```

Then boot it yourself:

```python
from umwelt.boot import build_engine
engine = build_engine(spec=MY_SPEC)
result = engine.ingest(sensor_readings={...}, now=t)
```

A binding whose `zone` or `role` doesn't exist in your spec raises loudly at direct
registration, but the boot path is membrane-guarded — a bad binding is *skipped with a
warning* so it can't break the others. That's why the gate above exists: it re-runs
every binding strictly and fails on what boot would only log (`"spec binding %s
skipped"` names the exact bad binding in a running world's logs).

## 4. Prove it

Steal the proof gate's shape — literally: `examples/gridworld/` includes one, and
`proofs/blank_slate.py` is the reference the gate itself runs. Blank floor witnessed
→ every binding drove the field (no dead vocabulary) → beliefs track your synthetic
ground truth → the fiber drifted off its priors → the gauge coordinate stays honest →
save/load round-trips. `proofs/` is not shipped in the installed package — vendor the
~120-line ladder harness into your own repo (`CumulantCluster`/`ObservationTrust`
import fine from the installed package).

## 5. Adapter honesty, once you're on real data

[FIELD_NOTES.md §3](FIELD_NOTES.md) — account every dropped row by (channel, reason);
derive normalizer bounds from robust percentiles of your TRAIN split only; score only
inside coverage windows; a no-event day is an absence, not a zero reading.

## Where each piece of knowledge actually lives (so you know where to look next)

| Question | Doc |
|---|---|
| What are the five DomainSpec declarations? | [docs/SPEC.md](SPEC.md) |
| Unitary vs dissipative, adapter honesty, real-data lessons | [docs/FIELD_NOTES.md](FIELD_NOTES.md) |
| `dt` vs `ingest_hold_s` vs a domain's own clock | [docs/TIME.md](TIME.md) |
| Running umwelt as a daemon (`umweltd`) | [docs/SERVICE.md](SERVICE.md) |
| What's proven vs designed vs denied | [CLAIMS.md](../CLAIMS.md) |
| A complete worked domain | [examples/gridworld/](../examples/gridworld) |
| Vocabulary-registration idiom | [examples/smarthome/](../examples/smarthome) |
