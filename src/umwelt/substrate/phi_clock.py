"""Golden-ratio timescale separation — the shared φ-clock for the learning towers.

Both the H-tower (FractalStack, learns Hamiltonians) and the meta-tower
(parameter-fiber meta-learning) climb on the same ladder: adjacent levels tick
on Fibonacci strides (8, 13, 21, 34, ...), so neighbouring cadences sit at the
golden ratio and don't resonate. A level that *tunes* another runs on a φ-slower
clock than the one it tunes — the core stability principle of stacked
meta-learning: the meta-learner must respond to the consequences of its nudges,
not the per-step noise.

Single source of truth for the stride ladder so the two towers stay in step.

`fib_strides_at(dt_factor, n)` is the ContextState-aware version: when
`context.dt_factor` is φ^k, the whole ladder slides k Fibonacci rungs up. At
dt_factor=1.0 it reduces to the legacy `fib_strides(n)`. This is where the
ContextState `dt_factor` axis lands — one function, all tiers read it.
"""
from __future__ import annotations

import math

PHI = (1 + 5 ** 0.5) / 2  # 1.618033988749895
_LOG_PHI = math.log(PHI)


def fib_strides(n: int) -> list[int]:
    """The first n strides of the ladder: 8, 13, 21, 34, 55, ...

    Starts at Fibonacci index 5 (=8), matching the H-tower's historical
    `fibs[level + 4]` choice in phi_scales (level 1 → 8, level 2 → 13).
    Equivalent to `fib_strides_at(1.0, n)`.
    """
    return fib_strides_at(1.0, n)


def fib_strides_at(dt_factor: float, n: int) -> list[int]:
    """Fibonacci strides offset by ⌊log_φ(dt_factor)⌋ along the ladder.

    The ContextState `dt_factor` axis enters the φ-clock here. Sliding
    `context_dt_factor` from 1.0 → φ shifts every tier one Fibonacci rung up
    in unison — the whole ladder moves, no special-cases per tier.

    dt_factor=1.0   → [8, 13, 21, 34, 55, ...]   (live, current behavior)
    dt_factor=φ     → [13, 21, 34, 55, 89, ...]  (one rung up)
    dt_factor=φ³    → [34, 55, 89, 144, 233, ...] (replay-fast)
    dt_factor=φ⁶    → six rungs up (sandbox / weekly-replay scale)

    dt_factor < 1.0 clamps to live — the φ-clock can't go slower than wallclock
    here (that's what wallclock IS).
    """
    offset = max(0, round(math.log(max(1.0, float(dt_factor))) / _LOG_PHI))
    fibs = [1, 1]
    while len(fibs) < n + 6 + offset:
        fibs.append(fibs[-1] + fibs[-2])
    return [fibs[i + 5 + offset] for i in range(n)]


def effective_stride(param, floor: int = 2) -> int:
    """Integer tick stride from a learnable ScalarParam.

    Mirrors FractalScale.effective_stride so a learnable, Fibonacci-seeded
    stride param drives the tick gate (`parent_step % effective_stride == 0`).
    floor defaults to 2 (the H-tower's choice — never the parent's own cadence);
    base-level meta channels pass floor=1 to allow every-step calibration.
    """
    return max(floor, round(float(param.value)))
