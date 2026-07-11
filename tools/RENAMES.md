# Cross-phase rename ledger

Renames made during extraction that LATER phases must apply when porting consumers
(the meerkat source still uses the old name — grep for it when porting).

## Modules
| meerkat | umwelt |
|---|---|
| meerkat/brain/world_graph.py | substrate/graph.py |
| meerkat/brain/world_model.py | substrate/ground.py |
| meerkat/brain/substrate.py | substrate/backend.py |
| meerkat/brain/house_spec.py | spec/schema.py (split: normalizers→spec/normalizers, role tables→spec/roles) |
| meerkat/brain/reservoir.py | engine.py (P2) |
| meerkat/brain/bootstrap.py | boot.py (P2) |
| meerkat/core/util.py | _util.py |
| meerkat/core/models.py + event_replay.py | events.py |

## Fiber param keys (root bundle) — consumers: reservoir/engine, calibration, trust web
| meerkat key | umwelt key |
|---|---|
| celestial_alpha | driver_alpha |
| celestial_hebbian_lr | driver_hebbian_lr |
| celestial_anticipation_ema | driver_anticipation_ema |
| celestial_trust_floor | driver_trust_floor |
| location_ground_alpha | anchor_ground_alpha |

## Functions / API
| meerkat | umwelt |
|---|---|
| gauge.solar_phase(field) | projection.gauge.driver_phase(field, node, role) |
| gauge.is_siesta(phase) / SIESTA_WINDOW | projection.gauge.in_rest_window(phase, window) / DEFAULT_REST_WINDOW |
| event_replay.hindbrain_lag_seconds | events.replay_lag_seconds |
| reservoir.ingest(celestial_observations=…) | engine.ingest(driver_observations=…) (P2) |
| reservoir.ground_location(lat, lon) | engine.ground_anchor(name, value, codec) (P2) |
| HouseSpec | DomainSpec (+ new fields outputs/drivers/anchors/channel_maps/params) |
| BridgeSpec.kind "door" | "gated" (alias map accepts "door") |
| NodeSpec.kind "zone" | "region" (alias map accepts "zone") |
| field.house_view() | REMOVED (app-side rendering) |
| emoji.ROLE_EMOJI house maps | register_role_emoji()/register_node_icon() (maps → smarthome example) |
| gauge_name._KNOWN_PLACES austin | register_known_place() (empty by default) |

## Cut from the root fiber (return via OutputSpec gates/coupling or app registration)
dimmer_*, ac_*, fan_flush_enabled, sleep_beta_steer_alpha, light_tendril_enabled,
light_qubit_commit_enabled, light_commit_{coupling,decay,hysteresis_scale},
device_tendril_enabled, device_{on_at,off_at,position_deadband}, presence_* (16),
light_baseline_drift, light_presence_floor.
Per-node Austin catalog (bedroom…closet, resident_*, rdk, exterior*) → NodeSpec.params.

## Env vars
MEERKAT_* → UMWELT_* (mechanical, done by tools/extract.py). Notables:
MEERKAT_HOUSE_SPEC → UMWELT_SPEC; MEERKAT_ARCHIVE_ROOT → UMWELT_ARCHIVE_ROOT;
reservoir_state.pkl → engine_state.pkl (UMWELT_STATE_PATH).
| field_unify.to_manifold(zone=) / manifold_from_pickle(zone=) | node= kwarg |
| ground.occupied/active + occupied_zones/active_zones | NodeState.asserted(role) / asserted_nodes(role) |
| ground.COLLAPSE_EMOJI maps | register_collapse_emoji() (empty by default) |
