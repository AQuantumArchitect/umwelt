"""Compute self-scheduling — the metacognition seed (b9.7).

MIND.md: "the moment the decision-graph chooses WHAT runs on CPU vs BPU, and when, the system is reasoning
about its OWN cognition." This is that seed, made concrete. The BPU dispatch node (bpu_dispatch) can run a
matmul on cpu / the INT8 expansion / the BPU; until now the backend was a fixed choice. The scheduler picks
it PER WORKLOAD from a gauge-tracked policy and LEARNS which backend to trust from each dispatch's parity
result — so a backend that silently drifts from the CPU reference gets demoted by the system itself.

The policy today: heavy, latency-tolerant batches (the hindbrain's dream/consolidation matmuls — exactly
the compute the CPU master wants to offload) route to the accelerator WHEN it's available AND has stayed in
parity; everything else stays on the CPU. The BPU kernel isn't compiled yet, so in practice it's all CPU —
but the SEAM + the learned trust are the deliverable: when the `.bin` lands, heavy work self-routes to it
and the scheduler proves (via parity) that it should, before anything depends on it. That self-monitoring
is the metacognition: the brain scheduling its own thought and checking whether the schedule was sound.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# accelerators in preference order (CPU is the always-available fallback, never in this list)
_ACCELERATORS = ("bpu", "expansion")
# a workload is "heavy" (worth offloading — amortizes dispatch) when n_ops × dim² exceeds this
HEAVY_FLOPS = float(os.environ.get("UMWELT_SCHED_HEAVY", str(8 * 32 * 32)))
MIN_PARITY_RATE = float(os.environ.get("UMWELT_SCHED_MIN_PARITY", "0.9"))


@dataclass
class Workload:
    """What we're about to compute — the scheduler reasons over this, not the data itself."""
    n_ops: int = 1                 # matmuls in the batch
    dim: int = 1                   # matrix dimension (cost ~ n_ops·dim²)
    latency_tolerant: bool = True  # hindbrain/dream work = yes; live forebrain reflex = no (stay CPU)

    @property
    def cost(self) -> float:
        return float(self.n_ops) * float(self.dim) ** 2

    @property
    def heavy(self) -> bool:
        return self.cost >= HEAVY_FLOPS


@dataclass
class _Stat:
    n: int = 0
    ms_ema: float = 0.0
    parity_pass: int = 0
    parity_total: int = 0

    def observe(self, ms: float, parity: bool, *, prove: bool):
        self.n += 1
        self.ms_ema = ms if self.n == 1 else 0.7 * self.ms_ema + 0.3 * ms
        if prove:                                  # only count parity when it was actually measured vs CPU
            self.parity_total += 1
            self.parity_pass += int(parity)

    @property
    def parity_rate(self) -> float:
        return 1.0 if self.parity_total == 0 else self.parity_pass / self.parity_total


class ComputeScheduler:
    """Chooses a backend per workload and learns each backend's reliability from parity. The brain
    scheduling its own compute — and auditing the schedule. Wraps a bpu_dispatch.Dispatcher."""

    def __init__(self, dispatcher, *, heavy_flops: float = HEAVY_FLOPS, min_parity_rate: float = MIN_PARITY_RATE):
        self.dispatcher = dispatcher
        self.heavy_flops = float(heavy_flops)
        self.min_parity_rate = float(min_parity_rate)
        self.stats: dict[str, _Stat] = {}
        self.decisions: list[dict] = []            # the audit trail (the metacognition record)

    def _trusted(self, name: str) -> bool:
        """Untried accelerator → allowed a trial (optimistic); once it has a parity record, it must hold
        the floor. A drifting backend demotes itself."""
        s = self.stats.get(name)
        return s is None or s.parity_total == 0 or s.parity_rate >= self.min_parity_rate

    def choose(self, workload: Workload) -> tuple[str, str]:
        """Pick a backend + the reason (the explained decision). CPU for light or latency-critical work;
        a trusted, available accelerator for heavy latency-tolerant batches."""
        avail = self.dispatcher.backends_available()
        if not workload.latency_tolerant:
            return "cpu", "latency-critical → CPU (no dispatch jitter)"
        if workload.cost < self.heavy_flops:
            return "cpu", f"light (cost {workload.cost:.0f} < {self.heavy_flops:.0f}) → CPU"
        for cand in _ACCELERATORS:
            if avail.get(cand) and self._trusted(cand):
                return cand, f"heavy + {cand} available + in-parity → offload to {cand}"
            if avail.get(cand) and not self._trusted(cand):
                # the metacognition bite: it WANTED to offload but its own audit says don't
                return "cpu", f"heavy but {cand} drifted (parity {self.stats[cand].parity_rate:.2f}) → stay CPU"
        return "cpu", "heavy but no accelerator available → CPU"

    def run(self, pairs, workload: Workload):
        """Schedule + run a batch of matmuls, then learn from the result. Returns (results, rollup)."""
        backend, reason = self.choose(workload)
        results, rollup = self.dispatcher.batched_matmul(pairs, backend=backend)
        self.stats.setdefault(rollup.backend, _Stat()).observe(
            rollup.ms, rollup.parity, prove=(rollup.backend != "cpu"))
        self.decisions.append({"workload": workload.cost, "chose": backend, "ran": rollup.backend,
                               "reason": reason, "parity": rollup.parity, "ms": round(rollup.ms, 2)})
        if len(self.decisions) > 256:
            del self.decisions[0]
        return results, rollup

    def report(self) -> dict:
        """The self-scheduling state — what the brain believes about its own compute options."""
        return {
            "backends": {n: {"n": s.n, "ms_ema": round(s.ms_ema, 3), "parity_rate": round(s.parity_rate, 3),
                             "trusted": self._trusted(n)}
                         for n, s in self.stats.items()},
            "available": self.dispatcher.backends_available(),
            "heavy_flops": self.heavy_flops, "recent_decisions": self.decisions[-8:],
        }
