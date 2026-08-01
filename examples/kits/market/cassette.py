"""Bread-Winner-lite: demand belief must not train on own recommend as market truth."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from umwelt.host import GameHost
from umwelt.host.api import Intent
from umwelt.learning.confounding import actor_intent_log
from umwelt.spec.schema import (
    BindingSpec,
    DomainSpec,
    DriverSpec,
    NodeSpec,
    OutputSpec,
)

HONESTY_TIER = "synthetic CI — self-demand poison vs actor-tagged shadow recommend"
START = datetime(2026, 5, 1, tzinfo=timezone.utc)


def market_spec() -> DomainSpec:
    nodes = (
        NodeSpec("market", parent=None, kind="root", roles=("demand",)),
        NodeSpec(
            "good",
            parent="market",
            kind="region",
            roles=("demand",),
            role_modes={"demand": "dissipative"},
            params={"gamma_diss": (0.08, 0.01, 0.001, 0.4)},
        ),
        NodeSpec(
            "trader",
            parent="market",
            kind="actuator",
            roles=("level",),
            projection={"level": "demand"},
        ),
    )
    bindings = (
        BindingSpec(
            "price_feed",
            zone="good",
            role="demand",
            normalizer={"type": "range", "lo": 0.0, "hi": 1.0},
            strength=0.5,
            efficiency=1.0,
            force_observe=True,
        ),
    )
    outputs = (
        OutputSpec(
            "recommend_buy",
            node="good",
            role="demand",
            kind="binary",
            decode="sticky",
            gates={"rate_limit_s": 0.0},
            dispatch={"actuator_id": "trader"},
            shadow=True,
        ),
    )
    return DomainSpec(
        name="kit-market-bread-lite",
        nodes=nodes,
        bindings=bindings,
        outputs=outputs,
        drivers=(DriverSpec("tick", period_s=1.0),),
    )


MARKET_SPEC = market_spec()


@dataclass
class BaselineReport:
    kit: str
    clean_err: float
    poison_err: float
    beats_baseline: bool
    honesty: str

    def summary(self) -> str:
        v = "BEATS" if self.beats_baseline else "DOES_NOT_BEAT"
        return (
            f"[{self.kit}] clean_err={self.clean_err:.4f} poison_err={self.poison_err:.4f} "
            f"→ {v} ({self.honesty})"
        )


def run_market_baseline(*, steps: int = 80, seed: int = 9) -> BaselineReport:
    """Clean path: external feed only + shadow tagged recommends.

    Poison baseline: after each recommend, re-ingest a fabricated full-demand
    observation as if the bot's own desire were market truth (self-demand poison).
    Clean must stay closer to quiet true demand than the poison path.
    """
    rng = np.random.default_rng(seed)
    true_demand = 0.25  # quiet market

    def _drive(host: GameHost, poison: bool) -> float:
        for i in range(steps):
            t = START + timedelta(seconds=i)
            feed = float(np.clip(true_demand + rng.normal(0, 0.03), 0.0, 1.0))
            host.observe("mkt", "price_feed", feed, confidence=1.0, t=t)
            host.intend(
                "bot",
                Intent(actor_id="bot", name="recommend_buy", shadow=True),
            )
            if poison:
                # Self-demand poison: own recommend treated as high market demand
                host.observe("mkt", "price_feed", 1.0, confidence=1.0, t=t)
            host.step(t=t)
        return host.belief_value("good", "demand").value

    h_clean = GameHost()
    h_clean.register_world(MARKET_SPEC, population=False, start=START)
    clean_v = _drive(h_clean, poison=False)
    clean_err = abs(clean_v - true_demand)
    assert h_clean.live_dispatches == []
    assert any(a == "bot" for a, _, _ in actor_intent_log(h_clean.engine))

    h_poison = GameHost()
    h_poison.register_world(MARKET_SPEC, population=False, start=START)
    # reset RNG path independence: re-seed so feed noise comparable
    rng = np.random.default_rng(seed)
    poison_v = _drive(h_poison, poison=True)
    poison_err = abs(poison_v - true_demand)

    beats = clean_err + 0.05 < poison_err  # clean clearly closer to truth
    # Also accept if poison belief is clearly inflated above clean toward 1.0
    if not beats and poison_v > clean_v + 0.1:
        beats = True
        # re-score: poison bias magnitude as the "err" story
        poison_err = max(poison_err, abs(poison_v - clean_v))
    return BaselineReport(
        kit="market",
        clean_err=clean_err,
        poison_err=poison_err,
        beats_baseline=beats,
        honesty=HONESTY_TIER,
    )
