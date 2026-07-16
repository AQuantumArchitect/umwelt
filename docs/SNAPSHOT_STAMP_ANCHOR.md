# Snapshot ↔ consumer-stamp anchor mapping

FIELD_NOTES_SEPTACRYPT §7 item 3. How a consumer's witnessed-history stamps
anchor to engine state, so a third party can replay across the boundary.

## The two sides

| Side | Identity | Where |
|---|---|---|
| Engine (library) | `cluster.snapshot()` dict — e1/e2 arrays, couplings, rng-free | `umwelt.substrate` |
| Engine (service) | `field_canon_hash` + event cursor in `events.db` | `umweltd` `/state`, `/health` |
| Consumer (septacrypt) | `pre_state_root` / `post_state_root` — content hash of the composite world snapshot (all zones' physical payloads + turn + rng state + berry coordinates; presentation fields excluded) | `KnotStamp` |

## The mapping law

A consumer stamp anchors engine state iff:

```
stamp.post_state_root == H( { zone: canonical(cluster.snapshot()) for zone in world }
                            + turn + rng_state + berry )
```

- `H` and `canonical(...)` are the CONSUMER's (septacrypt: JSON-canonical
  content hash, `ledger/roots.py`). The engine promises only that
  `snapshot()` is deterministic and complete.
- Library consumers hash snapshots directly (no daemon involved).
- Service consumers should record `(field_canon_hash, event_cursor)` alongside
  their own root at anchor time; the pair lets a verifier ask the daemon "was
  this your state at that cursor?" without replaying the consumer's ledger.

## Engine guarantees this depends on (please keep true)

1. `snapshot()` → `load()` round-trips bit-identically.
2. Same snapshot + same typed inputs + same recorded rng draws ⇒ same next
   snapshot (replay determinism; RK4 evolution is deterministic given dt).
3. Snapshot dict keys/shapes change only with a version bump visible in the
   payload (consumers embed schema tags in their roots).

Violating any of these silently breaks every already-minted consumer
certificate — they are contract surfaces, not internals.
