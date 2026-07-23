"""The synergy capability: the forecast discovers a higher-order feature it cannot see first-order.

C = A × B (parity) — pure synergy. Held out, the order-2 monomial context (which can form z_A·z_B)
beats persistence, while the first-order context (raw atoms only) cannot. This pins the capability
that the earlier trace-only context lacked: usable, discovered higher-order structure. Deterministic
(fixed seeds), so the margins pinned are stable, not noise.
"""
import numpy as np

from examples.synergy.world import SYNERGY_SPEC, synergy_stream
from proofs.synergy_walk import run
from umwelt.boot import build_engine


def test_synergy_context_is_higher_order():
    """The spec attaches an order-2 monomial context over the A/B/C atom pool."""
    eng = build_engine(spec=SYNERGY_SPEC, population=False)
    assert eng.forecast_surface is not None
    assert eng.forecast_surface.interaction_order == 2
    # 3 leaves → 6 atoms (z+trace each); order-2 monomials = C(6,1)+C(6,2) = 6+15 = 21
    assert eng.forecast_surface.n_context == 21
    # C genuinely moves (parity flips), so persistence is a real baseline, not a frozen mirage
    z = []
    for now, readings in synergy_stream(seed=1, ticks=300):
        eng.ingest(sensor_readings=readings, now=now)
        c = eng.field.clusters["C"]
        z.append(float(c.qubit_bloch(c.role_index["state"])[2]))
    assert np.std(z) > 0.3


def test_higher_order_beats_persistence_where_first_order_cannot():
    """Held out: order-2 (z_A·z_B reachable) beats persistence on the parity leaf; order-1 does not."""
    r = run(train_seeds=(1, 2, 3), test_seeds=(101,), ticks=1000)
    assert r["svp_order2"] > 0.03, r                      # discovered synergy beats predict-last-value
    assert r["svp_order2"] > r["svp_order1"] + 0.10, r    # and clearly beats first-order (which can't)
