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
| sensors/bridge.py (module) | membranes/ingress.py |
| bindings.apply_spec_bindings(bridge, spec=None) | ingress.apply_spec_bindings(bridge, spec) — spec REQUIRED, no default domain |
| bridge.zones (property) | bridge.bound_nodes |
| bridge.fresh_presence_zones(ttl) | bridge.fresh_nodes(role, ttl_seconds) — role now a parameter |
| bridge.hebbian_celestial_update | bridge.hebbian_driver_update |
| bridge SENSOR_PRESETS (11-entry table) | register_sensor_preset() (empty by default) |
| bridge _ROLE_EVENT_TYPE table | register_event_type() (empty by default; unregistered → "sensor") |
| bridge upgrade_to_param_norms ROLE_PARAM_MAP | register_range_param_prefixes() (empty by default) |
| bridge CONTINUOUS_ROLES = {"environment"} | SensorBridge.CONTINUOUS_ROLES (empty set; domain adds its continuous roles) |
| SensorBinding.is_event_driven (_EVENT_DRIVEN_ROLES set) | role_input_mode(role) == "unitary" (registry-derived) |
| bridge trust_webs / attach_trust_web / _fuse_leaves / trust_web_snapshot | CUT (trust_web/qubit_trust_web not yet ported; observe_targets is last-wins) |
| bridge sensor_kind() | CUT (domain-vocabulary UI helper) |
| register() merged_zone_role choke point | CUT (merge transform not yet in umwelt; re-add at engine port if merge lands) |
| MEERKAT_CONF_BRAKE_GAMMA / MEERKAT_COLLAPSE_ALPHA | UMWELT_CONF_BRAKE_GAMMA / UMWELT_COLLAPSE_ALPHA |
| QuantumReservoir | BeliefEngine (engine.py keeps `QuantumReservoir = BeliefEngine` alias) |
| build_reservoir(...) | boot.build_engine(...) — spec REQUIRED (arg / "module:ATTR" / UMWELT_SPEC); no `actuators=` flag (tendrils = P3 OutputSpec) |
| reservoir._anchor_solar_clock + celestial_targets machinery | engine._anchor_drivers(now, explicit) over injected `engine.drivers` |
| reservoir.set_celestial_targets(targets, forecast_labels) | engine.set_driver_targets(...) |
| reservoir.celestial_anticipation / celestial_forecast_labels / celestial_targets | engine.driver_anticipation / driver_forecast_labels / driver_targets |
| reservoir.location_bloch()/location_latlon()/location_pin_target()/_anchor_earth_gear | engine.anchor_bloch/anchor_value/anchor_pin_target/delocate_anchor (+ thin location_* compat over the "geo" anchor) |
| calibration.celestial_{forecast,anticipation,anticipation_snr} attrs, _calibrate_celestial, celestial_{enabled,interval,obs_sigma,stride}, stats "celestial_updates" | driver_* equivalents (_calibrate_drivers; stats "driver_updates") |
| reservoir.clock_snapshot() {"local_solar_time": …} | engine.clock_snapshot() {driver.name: …} (per injected driver) |
| observe path: bridge's energy flip (active → z = −1) | engine un-flips for NON-registered-observe roles (spec force_observe keeps normalizer's +pole = ground.asserted) |
| reservoir save keys location_grounded / ac_use_seed_baseline / phi_time | anchors_grounded (list) / DROPPED / DROPPED (loader maps legacy location_grounded → "geo") |
| load order: …→ hamiltonians → population → fractal_stack | …→ population → fractal_stack → hamiltonians/cumulant grafts LAST (stack load re-projects H; saved H_base must win for canon-hash roundtrip) |
| autonomy REGISTRY (4-row Austin catalog) | register_actuator_autonomy() (empty by default) |
| coupling_learn.measured_contrast(a_type="presence", b_type="motion_score") | a_type/b_type REQUIRED kwargs (stream types are domain vocabulary) |
| berry_tape.stamp_collapse(zone=…) | clocks/berry_tape.stamp_collapse(node=…) |
| field_unify.to_manifold(zone=)/reservoir.as_manifold(zone=) | node= kwarg (see Env table note) |

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
