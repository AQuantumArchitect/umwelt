"""Warmth-lite: two observation sources; one is corrupted; trust/isolation wins."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from umwelt.host import GameHost
from umwelt.spec.schema import BindingSpec, DomainSpec, DriverSpec, NodeSpec

HONESTY_TIER = "synthetic CI — two sources, one corrupted"
START = datetime(2026, 4, 1, tzinfo=timezone.utc)


def attention_spec() -> DomainSpec:
    """One signal node with reliable + noisy channels into the same role."""
    nodes = (
        NodeSpec("scene", parent=None, kind="root", roles=("signal",)),
        NodeSpec(
            "target",
            parent="scene",
            kind="region",
            roles=("signal",),
            role_modes={"signal": "dissipative"},
            params={"gamma_diss": (0.05, 0.01, 0.001, 0.3)},
        ),
    )
    bindings = (
        BindingSpec(
            "src_good",
            zone="target",
            role="signal",
            normalizer={"type": "range", "lo": 0.0, "hi": 1.0},
            strength=0.4,
            efficiency=1.0,
        ),
        BindingSpec(
            "src_bad",
            zone="target",
            role="signal",
            normalizer={"type": "range", "lo": 0.0, "hi": 1.0},
            strength=0.4,
            efficiency=0.15,  # low η — corrupted source
        ),
    )
    return DomainSpec(
        name="kit-attention-warmth-lite",
        nodes=nodes,
        bindings=bindings,
        drivers=(DriverSpec("tick", period_s=1.0),),
    )


ATTENTION_SPEC = attention_spec()


@dataclass
class BaselineReport:
    kit: str
    isolated_err: float
    naive_err: float
    beats_baseline: bool
    honesty: str

    def summary(self) -> str:
        v = "BEATS" if self.beats_baseline else "DOES_NOT_BEAT"
        return (
            f"[{self.kit}] isolated_err={self.isolated_err:.4f} "
            f"naive_err={self.naive_err:.4f} → {v} ({self.honesty})"
        )


def run_attention_baseline(*, steps: int = 80, seed: int = 5) -> BaselineReport:
    rng = np.random.default_rng(seed)
    truth = 0.8  # constant ground level

    # Isolated: only good source
    h_iso = GameHost()
    h_iso.register_world(ATTENTION_SPEC, population=False, start=START)
    for i in range(steps):
        t = START + timedelta(seconds=i)
        good = float(np.clip(truth + rng.normal(0, 0.05), 0, 1))
        h_iso.observe("si", "src_good", good, confidence=1.0, t=t)
        h_iso.step(t=t)

    # Naive: both sources at full confidence (ignores corruption)
    h_nv = GameHost()
    h_nv.register_world(ATTENTION_SPEC, population=False, start=START)
    for i in range(steps):
        t = START + timedelta(seconds=i)
        good = float(np.clip(truth + rng.normal(0, 0.05), 0, 1))
        bad = float(rng.uniform(0.0, 0.2))  # corrupted
        h_nv.observe_many(
            "si",
            {"src_good": good, "src_bad": bad},
            confidence={"src_good": 1.0, "src_bad": 1.0},
            t=t,
        )
        h_nv.step(t=t)

    # Trust-aware: bad source at low η
    h_tr = GameHost()
    h_tr.register_world(ATTENTION_SPEC, population=False, start=START)
    for i in range(steps):
        t = START + timedelta(seconds=i)
        good = float(np.clip(truth + rng.normal(0, 0.05), 0, 1))
        bad = float(rng.uniform(0.0, 0.2))
        h_tr.observe_many(
            "si",
            {"src_good": good, "src_bad": bad},
            confidence={"src_good": 1.0, "src_bad": 0.1},
            t=t,
        )
        h_tr.step(t=t)

    iso = h_iso.belief_value("target", "signal").value
    nv = h_nv.belief_value("target", "signal").value
    tr = h_tr.belief_value("target", "signal").value
    err_iso = abs(iso - truth)
    err_nv = abs(nv - truth)
    err_tr = abs(tr - truth)
    # Isolation/trust should beat naive full-trust of corrupted source
    beats = err_tr < err_nv or err_iso < err_nv
    return BaselineReport(
        kit="attention",
        isolated_err=min(err_tr, err_iso),
        naive_err=err_nv,
        beats_baseline=beats,
        honesty=HONESTY_TIER,
    )
