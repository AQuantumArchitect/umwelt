"""Generative dreaming — the mind hijacked to learn faster while the body rests.

The hindbrain already REPLAYS real cassettes (the backlog → pickle; training_backbone / BrainRunner).
Dreaming is the GENERATIVE sibling (docs/MIND.md): when the world is at rest and there's nothing live
to learn from, feed the learning-stream SHADOW CASSETTES — synthetic / mutated / counterfactual
experience (change facts, replay at a different driver phase, relocate the actors, scramble the order) —
extract the GRADIENT, and DISCARD the episode. We forget dreams because the cassette was disposable
training data: once the learning is baked in, the synthetic trace is thrown away. The clone the dream
ran on is discarded too; only what GENERALIZES (a parameter nudge consistent across mutations) is kept.

This is hands-off by construction — a dream NEVER actuates (the body rests). It learns on a CLONE of
the live field; the gradient folds back to the live field only if `apply_gradient` is on (shadow by
default, per the opt-in-not-gated floor). Every dream is gauge-tracked: the gradient IS the gauge delta
(field_gauge before→after), so a dream that taught nothing leaves an empty diff — provably-not-learning,
the same certificate the gauged-ensemble uses for a frozen sheet.

It is also the first real CLIENT of the BPU node (bpu_dispatch): the dream replay's matmuls are exactly
the compute-heavy, latency-tolerant, batchy learning work the CPU master wants to dispatch to the
accelerator. The engine takes an optional dispatcher so a dream batch can be offloaded; parity is the
gauge that says the offloaded learning matches the CPU reference.

  Cassette  = [(sensor_readings: dict, now), ...]   (the BrainRunner.replay batch format)
  Mutation  = a pure counterfactual transform of a cassette (named, seeded)
  Dream     = mutate → clone → replay (learn on, actuate OFF) → gauge delta → discard episode
  Rest gate = dream when at rest (siesta window OR low live surprise) — emergent, not a hardcoded window
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field as _field
from typing import Callable

from umwelt.projection.gauge import in_rest_window

log = logging.getLogger("dreaming")


# ── the cassette + its counterfactual mutations ──────────────────────────────

Batch = tuple                      # (sensor_readings: dict, now)
Cassette = list                    # [Batch, ...]


def _readings(b: Batch) -> dict:
    return dict(b[0]) if isinstance(b[0], dict) else {}


def _rebatch(b: Batch, readings: dict) -> Batch:
    return (readings,) + tuple(b[1:])


def m_identity(cassette: Cassette, rng: random.Random) -> Cassette:
    """The control — replay verbatim. A dream that changed nothing should learn nothing new."""
    return [(_readings(b), *b[1:]) for b in cassette]


def m_perturb_values(cassette: Cassette, rng: random.Random, scale: float = 0.15) -> Cassette:
    """Counterfactual INTENSITY: jitter every numeric reading by ±scale. 'What if it had been a bit
    warmer / louder / brighter?' — teaches the field the neighbourhood of the real trajectory."""
    out = []
    for b in cassette:
        r = _readings(b)
        out.append(_rebatch(b, {k: (v * (1.0 + rng.uniform(-scale, scale)) if isinstance(v, (int, float))
                                     and not isinstance(v, bool) else v) for k, v in r.items()}))
    return out


def m_relocate_regions(cassette: Cassette, rng: random.Random) -> Cassette:
    """Counterfactual GEOGRAPHY: remap region-name substrings in reading keys onto a shuffled region
    set. 'What if the same day had happened in a different region?' — decorrelates a learned habit
    from one specific region. The region vocabulary is registered per-domain (register_dream_region);
    empty by default → a no-op passthrough (domain-free)."""
    regions = list(_REGION_VOCAB)
    if len(regions) < 2:
        return list(cassette)
    perm = regions[:]
    rng.shuffle(perm)
    swap = dict(zip(regions, perm))
    out = []
    for b in cassette:
        r = _readings(b)
        nr = {}
        for k, v in r.items():
            nk = k
            for region in regions:
                if region in k:
                    nk = k.replace(region, swap[region])
                    break
            nr[nk] = v
        out.append(_rebatch(b, nr))
    return out


def m_time_shift(cassette: Cassette, rng: random.Random) -> Cassette:
    """Counterfactual TIMING: shift every timestamp by a random whole number of hours. 'What if this
    had happened at a different time of day?' — the same events against a different driver-phase context.
    Only shifts when the timestamp supports arithmetic; otherwise leaves it (robust to opaque `now`)."""
    import datetime as _dt
    delta_h = rng.choice([-6, -4, -3, -2, 2, 3, 4, 6])
    out = []
    for b in cassette:
        now = b[1] if len(b) > 1 else None
        try:
            shifted = now + _dt.timedelta(hours=delta_h)
        except Exception:
            shifted = now
        out.append((_readings(b), shifted, *b[2:]))
    return out


def m_scramble(cassette: Cassette, rng: random.Random) -> Cassette:
    """REM-ish reordering: shuffle the batch order (keep each batch intact). Breaks the field's reliance
    on the exact sequence — it must learn the events, not the script."""
    idx = list(range(len(cassette)))
    rng.shuffle(idx)
    return [(_readings(cassette[i]), *cassette[i][1:]) for i in idx]


def m_dropout(cassette: Cassette, rng: random.Random, p: float = 0.3) -> Cassette:
    """Counterfactual SPARSITY: drop ~p of the batches. 'What if we'd seen less?' — robustness to a
    thinner sensor stream (a quieter night, a dead collector)."""
    kept = [b for b in cassette if rng.random() > p]
    return [(_readings(b), *b[1:]) for b in (kept or cassette[:1])]


# Region vocabulary the relocate mutation shuffles over — empty by default (domain-free). A domain
# registers its region tokens; with <2 registered, relocate is a no-op passthrough.
_REGION_VOCAB: list[str] = []


def register_dream_region(name: str) -> None:
    """Register a region token the relocate mutation may swap. Domain-supplied; empty by default."""
    if name not in _REGION_VOCAB:
        _REGION_VOCAB.append(name)


MUTATIONS: dict[str, Callable] = {
    "identity": m_identity, "perturb": m_perturb_values, "relocate": m_relocate_regions,
    "time_shift": m_time_shift, "scramble": m_scramble, "dropout": m_dropout,
}
# the generative kinds (exclude identity, the control) — what a real dream draws from
DREAM_KINDS = ("perturb", "relocate", "time_shift", "scramble", "dropout")


@dataclass
class ShadowCassette:
    """A disposable counterfactual cassette — the dream's training data. Discarded after the gradient
    is extracted (why we forget dreams)."""
    batches: Cassette
    mutation: str
    seed: int

    def __len__(self):
        return len(self.batches)


def mutate(cassette: Cassette, *, mutation: str, seed: int) -> ShadowCassette:
    rng = random.Random(seed)
    fn = MUTATIONS[mutation]
    return ShadowCassette(batches=fn(cassette, rng), mutation=mutation, seed=seed)


# ── the rest gate (emergent, not a hardcoded window) ─────────────────────────

def should_dream(*, solar_phase: float | None = None, live_surprise: float | None = None,
                 surprise_rest: float = 0.15) -> bool:
    """Dream when the world is at REST — there's nothing live to learn from. Two emergent signals
    (either suffices): the periodic rest window (gauge.in_rest_window), or low live surprise (the field
    is well-predicted, the body is quiet). Deliberately NOT a clock window — it falls out of the dynamics."""
    if solar_phase is not None and in_rest_window(solar_phase):
        return True
    if live_surprise is not None and live_surprise <= surprise_rest:
        return True
    return False


# ── the gradient: a dream's learning IS the gauge delta ──────────────────────

def _flatten_gauge(g: dict, prefix: str = "") -> dict:
    """Flatten a nested field_gauge snapshot into {dotted_key: float} so two snapshots subtract."""
    flat = {}
    for k, v in g.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten_gauge(v, key + "."))
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                if isinstance(x, (int, float)):
                    flat[f"{key}[{i}]"] = float(x)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[key] = float(v)
    return flat


def gradient(before: dict, after: dict) -> dict:
    """The learning signal: per-coordinate motion between two gauge snapshots, plus a summary.
    total_motion (L1) = how much the field moved; n_moved = how many coordinates turned. An empty
    diff = the dream taught nothing (provably-not-learning)."""
    fb, fa = _flatten_gauge(before), _flatten_gauge(after)
    deltas = {k: fa[k] - fb.get(k, 0.0) for k in fa if abs(fa[k] - fb.get(k, 0.0)) > 1e-9}
    total = sum(abs(d) for d in deltas.values())
    return {"deltas": deltas, "n_moved": len(deltas), "total_motion": round(total, 8)}


@dataclass
class DreamReport:
    """One dream's outcome — the gradient kept, the episode discarded."""
    mutation: str
    seed: int
    n_batches: int
    n_replayed: int
    total_motion: float
    n_moved: int

    def gauge(self) -> dict:
        return {"mutation": self.mutation, "n_batches": self.n_batches, "n_replayed": self.n_replayed,
                "total_motion": self.total_motion, "n_moved": self.n_moved}


@dataclass
class DreamSession:
    """A night of dreams — aggregate learning, episodes already discarded."""
    reports: list = _field(default_factory=list)
    consolidated: bool = False

    @property
    def total_motion(self) -> float:
        return round(sum(r.total_motion for r in self.reports), 8)

    @property
    def generalized(self) -> float:
        """Fraction of dreams that moved the field — a crude 'did it learn something repeatable' read.
        High = the counterfactuals taught a consistent lesson; ~0 = the field was already settled."""
        if not self.reports:
            return 0.0
        return round(sum(1 for r in self.reports if r.n_moved > 0) / len(self.reports), 4)

    def summary(self) -> dict:
        return {"n_dreams": len(self.reports), "total_motion": self.total_motion,
                "generalized": self.generalized, "consolidated": self.consolidated,
                "by_mutation": {r.mutation: r.total_motion for r in self.reports}}


# ── the engine ───────────────────────────────────────────────────────────────

class DreamEngine:
    """Run generative dreams on a CLONE of the resting field, keep the gradient, discard the episode.

    Injectable seams (so it wraps a real reservoir in production and a fake learner in tests):
      clone()            -> a fresh learner initialized to the current baseline state
      replay(learner, b) -> ingest the shadow cassette under learn=ON, actuate=OFF (returns #replayed)
      read_gauge(l)      -> a (nested) gauge snapshot, subtractable into a gradient
      apply_gradient(g)  -> optional: fold a consolidated gradient back into the LIVE field (opt-in)

    A dispatcher (bpu_dispatch) may be passed so the heavy replay matmul is offloaded to the BPU node;
    it's recorded on the session but the learning math is the learner's — the dispatcher only accelerates.
    """

    def __init__(self, *, clone: Callable, replay: Callable, read_gauge: Callable,
                 apply_gradient: Callable | None = None, dispatcher=None):
        self._clone = clone
        self._replay = replay
        self._read_gauge = read_gauge
        self._apply_gradient = apply_gradient
        self.dispatcher = dispatcher

    def dream_once(self, real_cassette: Cassette, *, mutation: str, seed: int) -> tuple[DreamReport, dict]:
        """One dream. Returns (report, gradient). The shadow cassette + the clone are dropped on return
        (the episode is forgotten); only the gradient summary survives."""
        import copy
        shadow = mutate(real_cassette, mutation=mutation, seed=seed)
        learner = self._clone()                          # learn on a copy — the body rests, live field untouched
        before = copy.deepcopy(self._read_gauge(learner))  # snapshot: read_gauge may return a live reference
        n = self._replay(learner, shadow.batches)        # learn=ON, actuate=OFF by construction of `replay`
        after = self._read_gauge(learner)
        grad = gradient(before, after)
        report = DreamReport(mutation=mutation, seed=seed, n_batches=len(shadow),
                             n_replayed=int(n), total_motion=grad["total_motion"], n_moved=grad["n_moved"])
        # `learner`, `shadow`, `before`, `after` fall out of scope here — the episode is discarded.
        return report, grad

    def dream(self, real_cassette: Cassette, *, n_dreams: int = 5, seed: int = 0,
              kinds: tuple = DREAM_KINDS, consolidate: bool = False) -> DreamSession:
        """A night of `n_dreams` counterfactual dreams over one real cassette. Aggregates the gradients;
        if `consolidate` AND an apply hook is wired, folds the AVERAGE gradient (what generalized across
        mutations, not any single synthetic episode) back into the live field. Episodes are discarded."""
        session = DreamSession()
        acc: dict = {}
        for i in range(n_dreams):
            kind = kinds[i % len(kinds)]
            report, grad = self.dream_once(real_cassette, mutation=kind, seed=seed + i)
            session.reports.append(report)
            for k, d in grad["deltas"].items():
                acc[k] = acc.get(k, 0.0) + d
        if consolidate and self._apply_gradient is not None and acc:
            avg = {k: v / n_dreams for k, v in acc.items()}   # the generalized gradient (mean over dreams)
            self._apply_gradient(avg)
            session.consolidated = True
        log.info("dream session: %d dreams, motion=%.4f, generalized=%.0f%%, consolidated=%s",
                 len(session.reports), session.total_motion, session.generalized * 100, session.consolidated)
        return session


def reservoir_dream_engine(reservoir, *, apply_gradient=False, dispatcher=None) -> DreamEngine:
    """Wire a DreamEngine to a live reservoir: clone via pickle round-trip (a throwaway copy per dream),
    replay through BrainRunner with learn=ON/actuate=OFF, read field_gauge. Heavy — used in production /
    the hindbrain loop, not the unit tests (which inject a fake learner). apply_gradient stays a no-op
    unless explicitly enabled (shadow-first); the live consolidation seam is left for the wiring step."""
    import copy
    from umwelt.projection.gauge import field_gauge
    from umwelt.learning.runner import BrainRunner

    def _clone():
        return BrainRunner(reservoir=copy.deepcopy(reservoir))

    def _replay(runner, batches):
        return runner.replay(batches)

    def _read(runner):
        f = getattr(runner.reservoir, "field", None)
        return field_gauge(f) if f is not None else {}

    return DreamEngine(clone=_clone, replay=_replay, read_gauge=_read,
                       apply_gradient=None if not apply_gradient else (lambda g: None),
                       dispatcher=dispatcher)
