# umweltd — the engine as a service

The engine stays a library; `umweltd` (in `src/umweltd/`) wraps it as a local daemon
so every harness — and eventually every SaaS — talks to one brain surface.

## Shape

```
harness / SaaS app ──HTTP──▶ supervisor (:7071) ──proxy──▶ worker (world A)
                                   │                        worker (world B)
                                   └── catalog: $UMWELTD_HOME/worlds/<name>/
```

- **One world = one worker process.** Vocabulary registries are process-global, so a
  world imports exactly one domain's vocabulary; workers give crash/CPU isolation for
  free.
- **The log is the truth, the snapshot is a cache.** Every posted event is appended to
  the world's `events.db` (the umwelt.events schema) *before* ingest; snapshots save
  the engine plus a cursor; boot = load snapshot + replay the log tail through the
  production ingest path. This is the cassette pattern run live.
- **Shadow-first over the wire.** `GET /recommendations` is the ghost layer; nothing
  dispatches unless the app supplies a dispatcher (a later phase adds webhook dispatch,
  still opt-in per output).

## API

Supervisor (`python -m umweltd.supervisor`, default `:7071`):

| route | verb | body → result |
|---|---|---|
| `/health` | GET | supervisor + per-world liveness |
| `/worlds` | GET/POST | catalog / `{"name","spec":"module:ATTR","vocabulary":"module:fn"?}` |
| `/worlds/<n>/stop` `/start` | POST | lifecycle (stop snapshots on the way out) |
| `/worlds/<n>/*` | any | proxied to the worker |

Worker (behind the proxy):

| route | verb | body → result |
|---|---|---|
| `/health` | GET | `{world, step, last_event_ts, seed_profile}` |
| `/events` | POST | `{"events":[[ts,sid,value,meta|null],...]}` → append + ingest |
| `/state` | GET | the canonical `graph_state` projection |
| `/beliefs?node=&role=` | GET | one raw-Bloch belief read |
| `/recommendations` | GET | the shadow layer |
| `/snapshot` | POST | save + cursor → `{"field_canon_hash"}` |

Client: `umweltd.client.UmweltClient` (stdlib-only).

## Contracts and caveats (Phase S1)

- **Parity is the founding claim**: wire replay hash-equals library replay, and
  kill/respawn recovers the exact state (`tests/test_daemon_parity.py`, pinned).
- Batch boundaries are per-request: one `POST /events` = one
  `replay_sensor_batches` pass. A live pusher sends one request per flush window.
- Localhost only, no auth — Phase S2 (single-tenant cloud) adds an API key + TLS.
- Sparse-cadence worlds under-drive the field (the stream-density lesson from the
  market run); until the engine addresses it, the pusher owns the republish burst.
- The context gauge is not yet stamped per-world (live vs replay mode) — Phase S2.
