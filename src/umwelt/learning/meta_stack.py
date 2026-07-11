"""φ-clocked stratified stack of meta-learners over the parameter fiber.

The companion to the FractalStack. Where that learns *operators* (Hamiltonians)
on golden-ratio timescales, this learns the scalar *fiber* parameters on the
same ladder (phi_clock). The two towers share the clock and the presentation,
not the substrate.

    tier 0  — learns the WORLD/field params (sensor ranges, gamma, couplings,
              gamma_diss, projection_coupling, driver_alpha, forecast lr/l2/ema).
              Signal: the field's own surprise / tracking error / correlation.
    tier 1  — learns tier 0's OWN knobs (its channel φ-strides, obs_sigmas,
              targets). Signal: whether tier 0's tuning is reducing that surprise.
    tier d  — would learn tier d-1's knobs, a φ-step slower again.

Each tier runs a φ-slower clock than the tier it tunes (timescale separation:
a meta-learner must respond to the consequences of its nudges, not the noise).

**Honest depth cap.** A tier exists only where there is a real, non-circular
effectiveness signal. Today that is tiers 0 and 1. A tier 2 would need a genuine
"is tier 1 helping" signal; absent one it is left UNBUILT rather than faked —
turtles up only as far as the ground actually pushes back.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class MetaTier:
    """One stratum of the meta-tower.

    `run(host_step, *ctx)` does the tier's learning, gating internally on its
    own learnable φ-stride(s). `learns` is a human description of which fiber
    params it tunes; `strides` returns the tier's current φ-stride(s) — both for
    the unified tower view.
    """
    name: str
    learns: str
    run: Callable[..., None]
    strides: Callable[[], dict]


class MetaStack:
    """An ordered (shallow→deep) φ-clocked stack of MetaTiers."""

    def __init__(self, tiers: list[MetaTier]):
        self.tiers = tiers

    def step(self, host_step: int, *ctx) -> None:
        """Run every tier for one host (ingest) step; each self-gates on its stride."""
        for tier in self.tiers:
            tier.run(host_step, *ctx)

    def snapshot(self) -> list[dict]:
        """Tower view: depth-ordered tiers with what each learns and its strides."""
        return [
            {"depth": i, "name": t.name, "learns": t.learns, "strides": t.strides()}
            for i, t in enumerate(self.tiers)
        ]
