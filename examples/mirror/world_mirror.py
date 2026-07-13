"""The mirror world: umweltd observing itself through its own API surface.

Every sensor here is a field the daemon already serves over HTTP:
  sup_pulse   <- GET /health            (supervisor answered at all: 1/0)
  gh_alive    <- GET /health            (worlds.greenhouse.running)
  gh_bulk     <- GET /worlds/greenhouse/health  (events_db_bytes)
  self_alive  <- GET /health            (worlds.mirror.running -- itself)
  self_bulk   <- GET /worlds/mirror/health      (its OWN events_db_bytes,
                 which grows because these very readings are appended)
"""

from umwelt.spec.schema import BindingSpec, DomainSpec, DriverSpec, NodeSpec, OutputSpec

SPEC = DomainSpec(
    name="mirror",
    nodes=(
        NodeSpec("daemon", parent=None, kind="root", roles=()),
        NodeSpec("supervisor", parent="daemon", roles=("pulse",),
                 role_modes={"pulse": "dissipative"},
                 params={"gamma_diss": (0.08, 0.01, 0.0, 1.0)}),
        NodeSpec("greenhouse_worker", parent="daemon", roles=("alive", "bulk"),
                 role_modes={"alive": "dissipative", "bulk": "dissipative"},
                 params={"gamma_diss": (0.08, 0.01, 0.0, 1.0)}),
        NodeSpec("mirror_worker", parent="daemon", roles=("alive", "bulk"),
                 role_modes={"alive": "dissipative", "bulk": "dissipative"},
                 params={"gamma_diss": (0.08, 0.01, 0.0, 1.0)}),
    ),
    bindings=(
        BindingSpec("sup_pulse", zone="supervisor", role="pulse",
                    normalizer="binary", force_observe=True),
        BindingSpec("gh_alive", zone="greenhouse_worker", role="alive",
                    normalizer="binary", force_observe=True),
        BindingSpec("gh_bulk", zone="greenhouse_worker", role="bulk",
                    normalizer={"type": "range", "lo": 0.0, "hi": 262144.0}),
        BindingSpec("self_alive", zone="mirror_worker", role="alive",
                    normalizer="binary", force_observe=True),
        BindingSpec("self_bulk", zone="mirror_worker", role="bulk",
                    normalizer={"type": "range", "lo": 0.0, "hi": 262144.0}),
    ),
    outputs=(OutputSpec("attention_advice", node="supervisor", role="pulse"),),
    drivers=(DriverSpec("day", period_s=86400.0),),
    ingest_hold_s=30.0,
)
