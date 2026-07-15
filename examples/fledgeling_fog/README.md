# Fledgeling Fog Corridor — Phase 1 public synthetic domain

**Status: CI / public fixtures only.** No multi-agent. No narrative. No private
home-assistant or meerkat cassettes.

A small corridor graph of **places**. A scout walks; each place gets
`scout_{place}` observations with confidence η. Unobserved places ease back
toward uncertainty instead of freezing forever. Optional `claim_safe` is a
**shadow** output (decides visibly, dispatches nothing).

Time is a **tick** driver (`period_s=60`), not solar.

## Run

```bash
# spec gate
python -m umwelt.spec.validate examples.fledgeling_fog.world:FOG_SPEC

# demo (host API happy path — no raw engine.ingest)
python examples/fledgeling_fog/demo.py

# bake-off: belief vs freeze baseline
python examples/fledgeling_fog/bakeoff.py
```

## Belief questions this spike answers

1. Where is the agent likely now?
2. How sure are we about each place?
3. Does silence make distant places forget (dissipative ease)?
4. Does shadow claim fire without world side effects?
5. Does save/load keep comprehension?

## What's here

| File | Role |
|---|---|
| [`world.py`](world.py) | `FOG_SPEC` / `fog_corridor_spec`, walk, synthetic stream |
| [`demo.py`](demo.py) | Host-API timeline printout |
| [`bakeoff.py`](bakeoff.py) | Engine vs persistence-of-last-input metrics |
| Proof | `proofs/fledgeling_fog_blank.py` |

## What is *not* proven

- Multi-agent / multi-mind privacy (Phase 3)
- Narrative, dialogue, or content generation
- Live game integration (Phase 6)
- That full open-quantum filtering beats classical blend (estimator ladder is separate)
- Any origin-deployment effect sizes

## Honesty tier

**Synthetic CI gate.** Success = blank boot + bindings drive field + beliefs track
walk + save/load + bake-off metrics printed (win *or* honest loss).
