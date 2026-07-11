"""Learning router — gate world-model learning by the observe-vs-actuate regime.

The brain learns DIFFERENT things depending on whether an observation is clean
(exogenous — human/world-caused) or confounded (downstream — caused by our OWN
actuation). Learning the world model from a self-caused observation is circular: the
brain confirms a hypothesis it itself created (the "downstream-from-us" trap). Under
pure observation (the brain recommends, the human acts) the data is clean; under
actuation the belief→action→observation→belief loop closes on itself.

This router takes a per-cluster "downstream" attribution d ∈ [0,1] (how much of an
observed change is self-caused) and splits the learning gradient by channel:

    world_weight    = 1 - d    world / dynamics / preference — learn from CLEAN data
    actuator_weight = d        actuator / outcome model — the self-caused part is
                               informative about "did my action do what I predicted?"

Two attribution sources, combined by MAX (either saying "self-caused" discounts world
learning):
  - forecast-based — b2.1's ForecastComprehension `downstream` (statistical; matures as
    the brain learns its own dynamics H, so it reads ~0 on a fresh brain).
  - direct actuation-echo — when an observed device state matches our RECENT dispatch
    within tolerance, the observation is the echo of our own command (definitively
    downstream). `echo_likelihood()` computes this from the dispatch record; it does
    NOT depend on H maturity, so it bites immediately.

OBSERVE-FIRST: built to be SHADOW-MEASURED. `shadow_summary()` reports what the router
WOULD gate (the confounded fraction of learning) so it can be validated on live data
before it scales a single gradient. See the heritage-braid exploration — this is Stage 0
(the actuate-coordinate of the gauge-colored lineage, made operational).
"""
from __future__ import annotations

from dataclasses import dataclass


from umwelt._util import clamp01 as _clip01  # [0,1] clamp — one home (#313)


@dataclass(frozen=True)
class LearningGate:
    """Per-cluster learning weights for one tick. `world` scales world/dynamics/
    preference learning (1 = clean → full, 0 = fully confounded → none); `actuator`
    scales actuator-outcome learning (the self-caused part). `downstream` is the raw
    attribution; `regime` is a human-readable label."""
    downstream: float
    world: float
    actuator: float
    regime: str


def _regime(d: float) -> str:
    if d < 0.2:
        return "exogenous"
    if d > 0.8:
        return "downstream"
    return "mixed"


class LearningRouter:
    """Map per-cluster downstream attribution → per-channel learning gates. Pure +
    stateless; the reservoir feeds it the attribution each learning tick."""

    def __init__(self, *, world_floor: float = 0.0):
        # world_floor keeps a trickle of world-learning even on fully-downstream data
        # (so a leaf we always actuate never goes completely blind). 0 = hard gate.
        self.world_floor = _clip01(world_floor)

    def gate(self, downstream: float) -> LearningGate:
        d = _clip01(downstream)
        world = max(self.world_floor, 1.0 - d)
        return LearningGate(downstream=d, world=world, actuator=d, regime=_regime(d))

    def route_clusters(
        self,
        attribution: dict[str, float],
        echoes: dict[str, float] | None = None,
        actuate_level: float = 1.0,
    ) -> dict[str, LearningGate]:
        """attribution = {cluster: forecast-downstream∈[0,1]}; echoes (optional) =
        {cluster: actuation-echo∈[0,1]}. Combine by MAX → per-cluster gate.

        `actuate_level` = the agency qubit's |act⟩ ∈ [0,1] — the de-confounding read, made geometric.
        Self-causation is only possible to the extent the system is ACTING: while LISTENING (|act⟩→0)
        nothing observed is downstream-of-us, so we scale d by it. This is what stops a silenced system
        from mis-attributing an external agent's clean manual changes to its own (muted) actuation — the
        wind-up/confound trap. world-learning weight = 1 − |act⟩·d → full clean learning while listening."""
        echoes = echoes or {}
        a = _clip01(actuate_level)
        out: dict[str, LearningGate] = {}
        for name, d in attribution.items():
            combined = max(_clip01(d), _clip01(echoes.get(name, 0.0)))
            out[name] = self.gate(a * combined)
        # echo-only clusters (no forecast attribution) still get a gate
        for name, e in echoes.items():
            if name not in out:
                out[name] = self.gate(a * _clip01(e))
        return out

    @staticmethod
    def echo_likelihood(
        observed: float,
        dispatched: float | None,
        age_s: float | None,
        *,
        tol: float,
        recent_s: float,
    ) -> float:
        """How likely is this observation the ECHO of our own recent command — i.e.
        self-caused, confounded? Fresh AND within `tol` of what we dispatched → ~1;
        stale (age > recent_s) or far (|Δ| ≥ tol) → 0 (an independent world event).
        Linear in both freshness and closeness."""
        if dispatched is None or age_s is None or age_s < 0 or age_s > recent_s or recent_s <= 0:
            return 0.0
        delta = abs(float(observed) - float(dispatched))
        if delta >= tol or tol <= 0:
            return 0.0
        freshness = 1.0 - age_s / recent_s
        closeness = 1.0 - delta / tol
        return _clip01(freshness * closeness)

    @staticmethod
    def shadow_summary(gates: dict[str, LearningGate]) -> dict:
        """What the router WOULD gate this tick — for observe-first logging. Reports
        the mean confounded fraction (1 - world weight), the count, and the most
        confounded clusters, WITHOUT changing any learning."""
        if not gates:
            return {"n": 0, "confounded_fraction": 0.0, "downstream": []}
        confounded = [1.0 - g.world for g in gates.values()]
        ranked = sorted(gates.items(), key=lambda kv: -kv[1].downstream)
        return {
            "n": len(gates),
            "confounded_fraction": sum(confounded) / len(confounded),
            "downstream": [(name, round(g.downstream, 3), g.regime)
                           for name, g in ranked if g.downstream > 0.05][:5],
        }
