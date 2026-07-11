"""Attention as a graph comprehension — "should I listen?" is a belief, not a gate.

The decision to attend to an expensive input channel (a transcriber, a heavy parser, a
deep scan) is not a classical `if`; it is a BELIEF the field holds and an ACTUATION it
emits. The `attend` qubit's meaning is extracted from EVIDENCE THREADS IN CONJUNCTION —
exactly how location emerges from occupancy threads.

The threads are DOMAIN DATA, registered by the application (`register_attention_thread`):
the origin deployment registered a speech-tag thread and a voice-band spectral-energy
thread from its audio fibers. The engine ships only the `listen` thread — the agency
qubit's readiness pole — which biases how hard the conjunction attends.

When the evidence threads agree, the field observes `attend` toward +1; absent evidence
it relaxes toward −1. A consumer reads the continuous `attend` readout and ACTUATES the
expensive channel (run / skip) with hysteresis — so the work runs because the engine
attended, not because a gate fired. Gated UMWELT_ATTEND (default off).
"""
from __future__ import annotations

import time
from typing import Callable

ATTEND_CLUSTER = "_attention"

# The run/skip decision is the canonical sticky_collapse (a confidence-derived measurement,
# not a hand-set band) — shared with the commit tendril; defined once on the Tendril base.
# Kept under this name for attention's call sites + tests.
from umwelt.membranes.tendril import sticky_collapse as transcribe_hysteresis  # noqa: E402

# Evidence-thread registry: name → fn(engine, now) -> float in [0,1]. The application
# registers its domain threads; every registered thread joins the soft-AND conjunction.
_THREAD_PROVIDERS: dict[str, Callable] = {}


def register_attention_thread(name: str, fn: Callable) -> None:
    """Register an evidence thread for the attend conjunction: fn(engine, now) -> [0,1]."""
    _THREAD_PROVIDERS[name] = fn


def register_attention(reservoir):
    """The `attend` belief is a META-cognition (like agency, the act↔listen qubit) — a
    standalone 1-qubit density-matrix register held on the engine, NOT in field.clusters,
    so it adds zero feature-geometry weight (no readout reset) and isn't bridged into the
    world graph. Seeded at |ignore⟩ (z=-1)."""
    from umwelt.substrate.product_cluster import ProductQubitCluster
    c = ProductQubitCluster(ATTEND_CLUSTER, ["attend"])
    c.observe_qubit(0, (0.0, 0.0, -1.0), 1.0)               # start NOT attending
    reservoir._attention = c
    if not hasattr(reservoir, "_attend_on"):
        reservoir._attend_on = False
    return c


def attention_threads(reservoir, now: float | None = None) -> dict:
    """Read every registered evidence thread plus the built-in `listen` pole."""
    now = time.time() if now is None else now
    t: dict[str, float] = {}
    for name, fn in _THREAD_PROVIDERS.items():
        try:
            t[name] = round(float(fn(reservoir, now)), 4)
        except Exception:
            t[name] = 0.0
    listen = float(reservoir.agency.listen()) if getattr(reservoir, "agency", None) is not None else 0.5
    t["listen"] = round(listen, 4)
    return t


def update_attention(reservoir, now: float | None = None, alpha: float = 0.4) -> dict:
    """Observe `attend` from the threads IN CONJUNCTION, then read it back. The conjunction
    is a soft-AND (geometric mean) of the registered evidence threads, biased by the
    listen-readiness; absence relaxes attend toward |ignore⟩. Returns the threads + the
    attend level + the run/skip actuation."""
    c = getattr(reservoir, "_attention", None)
    if c is None:
        return {}
    t = attention_threads(reservoir, now)
    evidence_threads = [v for k, v in t.items() if k != "listen"]
    if evidence_threads:
        prod = 1.0
        for v in evidence_threads:
            prod *= max(v, 1e-6)
        evidence = prod ** (1.0 / len(evidence_threads))    # geometric mean = soft-AND
    else:
        evidence = 0.0                                       # no threads registered → relax
    target_z = 2.0 * evidence - 1.0
    a = alpha * (0.4 + 0.6 * t["listen"])                    # attend more readily in listen-mode
    c.observe_qubit(0, (0.0, 0.0, float(target_z)), float(a))
    bx, by, z = (float(v) for v in c.qubit_bloch(0))
    level = (z + 1.0) / 2.0
    # the ACTUATION: a continuous belief decoded to run/skip. Hysteresis is not a hand-set
    # band — it's the qubit's OWN CONFIDENCE: flip at the 0.5 measurement midpoint with a
    # half-width = (1−purity)·scale. A CONFIDENT attend (purity→1) flips sharply at 0.5 (a
    # clean measurement); an UNCERTAIN one (purity→0) widens the band so the channel doesn't
    # flap on noise. The stickiness is the belief's uncertainty, not a magic number;
    # `attend_hysteresis_scale` is the only (gauge) knob.
    purity = (bx * bx + by * by + z * z) ** 0.5
    rb = getattr(getattr(reservoir, "graph", None), "root", None)
    scale = float(rb.param_bundle.get("attend_hysteresis_scale", 0.5)) if (rb and rb.param_bundle) else 0.5
    on = transcribe_hysteresis(level, purity, getattr(reservoir, "_attend_on", False), scale)
    reservoir._attend_on = on
    return {"threads": t, "evidence": round(evidence, 4), "attend_z": round(z, 4),
            "level": round(level, 4), "confidence": round(purity, 4), "transcribe": on}
