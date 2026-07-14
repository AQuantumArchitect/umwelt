"""examples/manifold-game/world — a real foreign game as an umwelt world.

The domain: SpaceWheat, a quantum farming game whose biomes are density
matrices evolved by a native Lindblad engine — a game literally about
navigating manifolds. The tape under data/ is a REAL session recorded from
the game's LLM-playtester seat (player-parity surface: the rows are only
what a player could read off the screen), exported by the game's
`export_umwelt_tape.py`. Nothing here talks to the game; this is replay.

Spec shape (see docs/SPEC.md):
  pantry   — one dissipative role per dynamic wallet resource (a wallet level
             is one-polarity evidence: the dissipative-role law applies,
             docs/FIELD_NOTES.md §1).
  story    — `progress`: the campaign's cumulative story-flag count.
  ingest_hold_s — the tape is sparse (one snapshot every few player turns);
             membrane cadence, not a time model (docs/TIME.md).

Normalizer bounds are DATA (data/bounds.json), derived from the train split
only (adapter honesty, FIELD_NOTES §3) by `demo.py --refit`.
"""
from __future__ import annotations

import json
from pathlib import Path

from umwelt.spec import roles as role_registry
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec

HERE = Path(__file__).resolve().parent

# Wallet sensors worth a belief axis: role → sensor id (the exporter's
# ascii names for the game's emoji). Kept small — six qubits is a cluster.
PANTRY_ROLES = {
    "grain": "wallet_ear_of_rice",
    "bread": "wallet_bread",
    "folk": "wallet_busts_in_silhouette",
    "seed": "wallet_seedling",
    "compost": "wallet_fallen_leaf",
}

# The vocabulary idiom (examples/smarthome/vocabulary.py): a domain registers
# its roles; the engine ships zero domain words. ANALOG + dissipative is
# load-bearing twice over: wallet levels are one-polarity evidence (the
# dissipative-role law), and a continuous quantity must boot maximally MIXED,
# not at the cold pole — a blank engine claiming "the pantry is empty" would
# be a false certainty (CLAIMS.md, blank-mix row).
for _role in list(PANTRY_ROLES) + ["progress"]:
    role_registry.register_role_mode(_role, "dissipative", analog=True)


def load_bounds() -> dict:
    p = HERE / "data" / "bounds.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    # Conservative defaults; refit from a train split with demo.py --refit.
    return {sensor: {"lo": 0.0, "hi": 100.0} for sensor in PANTRY_ROLES.values()} | {
        "flags_fired": {"lo": 0.0, "hi": 43.0}}


def manifold_spec(bounds: dict | None = None) -> DomainSpec:
    bounds = bounds or load_bounds()

    def norm(sensor: str) -> dict:
        b = bounds.get(sensor, {"lo": 0.0, "hi": 100.0})
        return {"type": "range", "lo": float(b["lo"]), "hi": float(b["hi"])}

    nodes = (
        NodeSpec("farm", parent=None, kind="root", roles=()),
        NodeSpec("pantry", parent="farm", kind="entity",
                 roles=tuple(PANTRY_ROLES.keys()),
                 role_modes={role: "dissipative" for role in PANTRY_ROLES},
                 params={"gamma_diss": (0.02, 0.01, 0.0, 1.0)}),
        NodeSpec("story", parent="farm", kind="entity", roles=("progress",),
                 role_modes={"progress": "dissipative"},
                 params={"gamma_diss": (0.005, 0.01, 0.0, 1.0)}),
    )
    bindings = tuple(
        BindingSpec(sensor, zone="pantry", role=role, normalizer=norm(sensor))
        for role, sensor in PANTRY_ROLES.items()
    ) + (
        BindingSpec("flags_fired", zone="story", role="progress",
                    normalizer=norm("flags_fired")),
    )
    return DomainSpec(
        name="manifold-game",
        nodes=nodes,
        bindings=bindings,
        ingest_hold_s=120.0,
        ignored=(
            ("wallet_*", "static stores with no train-split dynamics get no axis"),
            ("measured_*", "per-biome sightings; a later, finer spec"),
            ("visible_*", "per-biome sightings; a later, finer spec"),
        ),
    )


SPEC = manifold_spec()


def load_rows() -> list:
    """The committed real tape: [[ts_iso, sensor_id, value_str, None], ...]."""
    return json.loads((HERE / "data" / "rows.json").read_text(encoding="utf-8"))
