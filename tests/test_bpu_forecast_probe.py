"""Guards the b9.7.3 BPU forecast OFFLOAD wiring — BpuForecastProbe + make_bpu_forecast_probe.

The probe is the live-loop consumer: it picks a dim-matched dense cluster's ρ, rolls it forward on the
BPU under the frozen baked operator, and stows the parity gauge for the health spine. These lock the
two safety properties: the factory is None unless explicitly enabled+configured (flag-off = no channel,
byte-identical), and when a (faithful) accelerator is available the probe produces an ENRICHED record
that names the node it rolled. hrt_runner is mocked (no RDK).
"""
from __future__ import annotations

import numpy as np

from umwelt.foresight import bpu_forecast as bf

# ops-health telemetry isn't a separate umwelt module — bpu_forecast carries a no-op shim exposing
# the ENRICHED/DECOHERED verdict constants + stream_record. Use it directly.
dh = bf.dh


def _H(d=8, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d))
    return (0.5 * (a + a.T)).astype(np.float32)


class _Cluster:
    def __init__(self, rho, *, product=False, cumulant=False):
        self.rho = rho
        self.is_product = product
        self.is_cumulant = cumulant


class _Field:
    def __init__(self, clusters):
        self.clusters = clusters


def test_factory_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("UMWELT_BPU_FORECAST", raising=False)
    assert bf.make_bpu_forecast_probe() is None


def test_factory_none_when_enabled_but_no_model(monkeypatch):
    monkeypatch.setenv("UMWELT_BPU_FORECAST", "1")
    monkeypatch.delenv("UMWELT_BPU_FORECAST_MODEL", raising=False)
    assert bf.make_bpu_forecast_probe() is None  # enabled but unconfigured → inert, not a crash


def test_probe_picks_dim_matched_dense_cluster_and_enriches(monkeypatch):
    """A faithful accelerator (mock returns the CPU rollout) → ENRICHED record naming the dense node;
    product/cumulant clusters and dim-mismatched ones are skipped."""
    H = _H(8)
    fc = bf.BpuForecaster("op.bin", H, n_steps=4, dim=8)
    monkeypatch.setattr(bf.hrt_runner, "available", lambda: True)
    # faithful BPU: echo the CPU rollout of whatever ρ it's asked for
    monkeypatch.setattr(bf.hrt_runner, "infer",
                        lambda binp, inputs, **k: {fc.out_name: fc.rollout_cpu(inputs["rho0"])})
    monkeypatch.setattr(fc, "available", lambda: True)

    probe = bf.BpuForecastProbe(fc)
    rng = np.random.default_rng(0)
    dense = 0.5 * (rng.standard_normal((8, 8)) + rng.standard_normal((8, 8)).T)
    field = _Field({
        "param_fiber": _Cluster(np.eye(4), product=True),     # skipped (product)
        "small": _Cluster(np.eye(4)),                          # skipped (dim 4 != 8)
        "house": _Cluster(dense.astype(np.float32)),           # the target (dim 8)
    })
    rec = probe.run(field)
    assert rec is not None
    assert rec["verdict"] == dh.ENRICHED
    assert rec["gauge"]["node"] == "house"
    assert rec["gauge"]["cos"] > 0.999
    assert probe.last_record is rec


def test_probe_inert_when_runtime_unavailable(monkeypatch):
    H = _H(8)
    fc = bf.BpuForecaster("op.bin", H, n_steps=4, dim=8)
    monkeypatch.setattr(fc, "available", lambda: False)   # no hrt / no .bin
    probe = bf.BpuForecastProbe(fc)
    field = _Field({"house": _Cluster(np.eye(8, dtype=np.float32))})
    assert probe.run(field) is None
    assert probe.available(field) is False
