# ADR: Knot-ledger attachment — sibling wheel, not `umwelt.knot`

**Status:** proposed (septacrypt sprint, 2026-07) — FIELD_NOTES_SEPTACRYPT §7 item 2.

## Question

Where do the witnessed-history primitives (stamp, transition certificate,
cassette, replay residual) live long-term: inside this repo as `umwelt.knot`,
or as a sibling package?

## Decision: sibling wheel

The ledger stays in the consumer stack (today `septacrypt-core`; later a
neutral `knot-ledger` wheel if a second consumer appears). `umwelt.knot` is
NOT created now.

Reasons:

1. **The engine is the substrate authority, not the history authority.**
   umwelt owns cluster dynamics, measurement, and snapshot/save-load. What a
   host considers a committable "turn," which events are typed, and what a
   certificate witnesses are host-side product decisions.
2. **Vocabulary lint stays trivially green.** The knot layer's value language
   (branches, truth modes, witness stamps) is one lint bug away from leaking
   into workers if it lives in-tree.
3. **One consumer is not a product.** Extracting a shared wheel before a
   second consumer exists would freeze contracts on one data point.

## What umwelt SHOULD own (the neutral contracts)

The attachment surface, already exercised by septacrypt-core:

| Contract | umwelt side | Consumer side |
|---|---|---|
| Physical payload | `cluster.snapshot()` / `load()` (stable dict shape) | content-hash into state roots |
| Deterministic replay | `step(dt_scale)` + seeded measurement path | independent replay + residual check |
| Anchor identity | `field_canon_hash` / snapshot cursor (see SNAPSHOT_STAMP_ANCHOR.md) | `pre/post_state_root` in stamps |

Any change to snapshot shape or step determinism is a breaking change to
certificate replay — treat those as semver-relevant surfaces.

## Revisit when

- a second consumer wants stamps/certificates (extract `knot-ledger` wheel), or
- the hive/chain work needs `umwelt.knot.anchor.v1` frozen in-tree (then only
  the anchor JSON schema moves here, not the ledger).
