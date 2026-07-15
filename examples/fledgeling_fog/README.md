# Fledgeling Fog Corridor — public synthetic FL-core domain

**Status: CI / public fixtures.** Roadmap: [docs/FLEDGELING_CORE.md](../../docs/FLEDGELING_CORE.md)
(Phases **1–4** exercise this tree; Phase **5** kits live under `src/umwelt/kits/`).

A small corridor graph of **places**. A scout walks; each place gets
`scout_{place}` observations with confidence η. Unobserved places ease back
toward uncertainty instead of freezing forever. Optional `claim_safe` is a
**shadow** output (decides visibly, dispatches nothing).

Time is a **tick** driver (`period_s=60`), not solar. The demo happy path uses
`umwelt.host.GameHost` only (no raw `engine.ingest`).

## Run

```bash
# spec gate
python -m umwelt.spec.validate examples.fledgeling_fog.world:FOG_SPEC

# demo (host API happy path)
python examples/fledgeling_fog/demo.py

# bake-off: belief vs freeze baseline
python examples/fledgeling_fog/bakeoff.py

# Phase 4 agency: patrol earns shadow auto-intend after N successes
python examples/fledgeling_fog/agency_demo.py
```

## Belief questions this spike answers

1. Where is the agent likely now?
2. How sure are we about each place?
3. Does silence make distant places forget (dissipative ease)?
4. Does shadow claim fire without world side effects?
5. Does save/load keep comprehension?
6. (Agency demo) Can a patrol sub-routine auto-intend in shadow only after N successes?

## What's here

| File | Role |
|---|---|
| [`world.py`](world.py) | `FOG_SPEC` / `fog_corridor_spec`, walk, synthetic stream |
| [`demo.py`](demo.py) | Host-API timeline printout (Phase 1–2) |
| [`bakeoff.py`](bakeoff.py) | Engine vs persistence-of-last-input metrics |
| [`agency_demo.py`](agency_demo.py) | Sub-routine + promotion + surprise pause (Phase 4) |
| Proof | `proofs/fledgeling_fog_blank.py` |
| Host / multi-mind / agency | `src/umwelt/host/` |
| Related kit baselines | `src/umwelt/kits/fog` (and attention / market / dream) |

## What is *not* proven

- Narrative, dialogue, or content generation
- Live Fledgeling host integration / playable product loop (Phase 6)
- That the optional Belavkin filter beats classical blend (it ships OFF; see THEORY.md)
- Any origin-deployment effect sizes
- That facet kits improve real play skill (synthetic cassette metrics only)

Multi-mind privacy and agency automation **are** gated in-repo (`tests/test_multimind_privacy.py`,
`tests/test_agency_loop.py`) — this README’s domain package alone is still the fog
corridor fixture, not the full multiplayer game.

## Honesty tier

**Synthetic CI gate.** Success = blank boot + bindings drive field + beliefs track
walk + save/load + bake-off metrics printed (win *or* honest loss) + host/agency
contract tests green.
