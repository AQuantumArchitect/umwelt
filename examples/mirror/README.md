# mirror — umweltd's self-portrait

The daemon pointed at the only instrument available: itself. Every sensor in
[`world_mirror.py`](world_mirror.py) is a field umweltd already serves over its
own HTTP surface:

| sensor | source | meaning |
|---|---|---|
| `sup_pulse` | `GET /health` | the supervisor answered at all |
| `gh_alive` | `GET /health` → `worlds[].running` | a sibling world's worker is up |
| `gh_bulk` | `GET /worlds/<sibling>/health` | the sibling's event-log byte size |
| `self_alive` | `GET /health` | the mirror's **own** worker is up |
| `self_bulk` | `GET /worlds/mirror/health` | its **own** event log — which grows *because* these readings are appended to it |

[`pusher.py`](pusher.py) closes the loop: poll those surfaces, post the values
back in through `POST /events`. The result is the engine holding a live belief
field about the daemon it runs inside — read it back with `GET /state`, or
watch it in the playground at `/ui`.

## Run it

```bash
export UMWELTD_API_KEY=pick-something
umweltd &                                          # plus any sibling world you like

# gate the spec, then register it (spec_path lets it boot from this directory)
python -m umwelt.spec.validate world_mirror:SPEC --json   # run from this dir, or set PYTHONPATH
umweltctl create mirror --spec world_mirror:SPEC --spec-path "$PWD"

# 40 rounds of self-observation, one every 2 s
python pusher.py 40 2

umweltctl belief --world mirror --node mirror_worker --role alive
```

No sibling world? Point `WATCHED_WORLD` at any registered world, or delete the
`greenhouse_worker` node + its two bindings from the spec.

## What the first sitting showed (2026-07-13)

Run against a live daemon with a forge-authored greenhouse world as the sibling
(45 rounds, 225 readings, zero unmatched):

- All beliefs settled calm/alive (`pulse +0.95`, both workers `alive +0.95`,
  `bulk ≈ +1.0` — low log-bulk reads as calm under the energy convention).
- **It witnessed an outage.** The sibling was stopped through the same API
  mid-run; the mirror's write-ahead log caught exactly one `gh_alive = 0.0`
  reading at the stop window, and the belief eased back to calm within two
  rounds of the restart.
- **The recursion is measurable.** `self_bulk` grew 0 → 20,480 bytes over the
  sitting — the log thickening precisely because of the readings that measure
  it. The engine filed it as a calm, steady swelling.
- The one declared output (`attention_advice`) decided visibly over
  `GET /recommendations` and dispatched nothing — shadow is the law.

This doubles as an end-to-end exercise of the service API: create (with
`spec_path`), bindings, events, state, beliefs, recommendations, snapshot,
stop/start — all through the front door, nothing measured from inside.
