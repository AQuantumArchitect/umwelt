"""The synergy world: a belief the forecast can ONLY foresee via a discovered higher-order feature.

Two periodic collapse sources A, B, and a leaf C that is their PARITY: C(t) = A(t) × B(t). This is
pure synergy (the conspiracy / conjunction case of the O-information work): no first-order feature
carries any signal about C — E[C | A] = A·E[B] = 0 and likewise for B — so a forecaster restricted
to single states/traces provably cannot beat persistence on C. Only the PRODUCT z_A × z_B predicts
it, and that product is one of the monomials the self-discovering higher-order context hallucinates
and whose weight the readout learns.

This is the forecast-layer proof that higher-order structure is USABLE when it is genuine synergy
(irreducible) — the capability the trace-only order-1 context could not reach. Contrast the
`contraption` world, whose single-parent dependency is a first-order-adjacent (order-2) product;
here the structure is irreducibly second-order.

Compile/validate: python -m umwelt.spec.validate examples.synergy.world:SYNERGY_SPEC
Run the proof:      python proofs/synergy_walk.py
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from umwelt.spec.schema import BindingSpec, DomainSpec, DriverSpec, NodeSpec

LEAVES = ("A", "B", "C")
FORECAST_LEAVES = tuple((n, "state") for n in LEAVES)   # A,B must be leaves so z_A·z_B is an atom-product

PA, PB = 10, 14   # source periods (ticks); coprime-ish so the parity is not trivially periodic-short

SYNERGY_SPEC = DomainSpec(
    name="synergy",
    nodes=(
        NodeSpec("field", parent=None, kind="root", roles=()),
        *(NodeSpec(n, parent="field", roles=("state",)) for n in LEAVES),
    ),
    bindings=tuple(
        BindingSpec(f"{n}_state", zone=n, role="state",
                    normalizer={"type": "range", "lo": -1.0, "hi": 1.0},
                    force_observe=True)
        for n in LEAVES
    ),
    drivers=(DriverSpec("day", period_s=86400.0),),
    forecast_leaves=FORECAST_LEAVES,
    forecast_horizons_min=(8.0, 16.0, 26.0),
    forecast_context_surprise=True,      # turn on the self-discovering higher-order context
    forecast_interaction_order=2,        # pairwise monomials → z_A·z_B (the synergy feature) is reachable
)


def synergy_stream(seed: int = 1, ticks: int = 1500, dt_min: float = 2.0,
                   t0: datetime | None = None):
    """Yield (now, readings): A,B are periodic ±1 (phase set by seed → held-out schedules); C=A·B."""
    rng = np.random.default_rng(seed)
    t0 = t0 or datetime(2026, 7, 22, 9, 0, 0)
    offA, offB = int(rng.integers(0, PA)), int(rng.integers(0, PB))
    for k in range(ticks):
        zA = 1.0 if ((k + offA) // PA) % 2 == 0 else -1.0
        zB = 1.0 if ((k + offB) // PB) % 2 == 0 else -1.0
        yield t0 + timedelta(minutes=dt_min * k), {
            "A_state": zA, "B_state": zB, "C_state": zA * zB,
        }
