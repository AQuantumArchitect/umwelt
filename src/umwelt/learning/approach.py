"""Approach — the ONE approach law (dissolution M5, b9.36).

Every continuous actuator in the world solves the same little control problem: move the
fixture toward a target without waggle. The pieces — glide toward the target with
bounded slew, clamp to the device range, and don't dispatch inside a dead-band — were
re-derived inline wherever a decider needed them (the origin's device loops, its continuous-output
actuator, the device tendril's position path…). The DECIDERS are plural on purpose
(policy: what should the light want); the approach calculus under them is one law.

And notice what the law is: an α-blend toward a target with a commit threshold — the
manifold's observation law running in reverse, on the actuator side. Collapse pulls
belief toward evidence; the glide pulls the fixture toward belief. Same mathematics,
dual direction (the binary cousin, sticky_collapse, already lives on tendril.py).

Contract:
    step(target) → the level to DISPATCH, or None = hold.
      • the proposal moves from the last dispatched level toward `target` by at most
        `slew`, then clamps into [lo, hi];
      • it dispatches when it's the FIRST level or when it moved ≥ `dead_band` from
        the last dispatched level; otherwise it holds (returns None).
      • Consequence: the fixture may REST up to dead_band−1 from the exact target —
        that is what a dead-band means. (The historical inline copies carried an
        `or nxt == target` arrival exception, which re-dispatched on every
        integer-goal jitter — waggle, the exact failure this law exists to prevent.
        Writing the law once surfaced the flaw; the property test pins the fix.)
Properties pinned in tests/brain/test_approach.py: monotone approach, never overshoots,
no dispatch inside the dead-band, no waggle under a jittering target, bounds respected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from umwelt._util import clamp


@dataclass
class Approach:
    slew: float                     # max move per step toward the target
    dead_band: float = 0.0          # suppress dispatches smaller than this
    lo: float = 0.0                 # device range
    hi: float = 100.0
    quantize: bool = True           # round proposals to ints (device levels)
    last_sent: float | None = field(default=None)   # the level the device holds

    def propose(self, target: float) -> float:
        """The next level on the way to `target` (glide + clamp), without deciding
        whether to dispatch it."""
        t = float(target)
        if self.last_sent is None:
            nxt = t
        else:
            cur = self.last_sent
            nxt = max(cur - self.slew, min(cur + self.slew, t))
        if self.quantize:
            nxt = float(round(nxt))
        return clamp(nxt, self.lo, self.hi)

    def step(self, target: float) -> float | None:
        """Advance toward `target`; return the level to dispatch, or None = hold."""
        nxt = self.propose(target)
        if self.last_sent is None or abs(nxt - self.last_sent) >= max(self.dead_band, 1e-12):
            if nxt != self.last_sent:
                self.last_sent = nxt
                return nxt
            return None   # already holding exactly this level → nothing to send
        return None
