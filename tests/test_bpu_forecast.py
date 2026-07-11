"""Guards meerkat/brain/bpu_forecast.py — frozen-H rollouts on the BPU for the forecast brain.

The runtime is RDK-only, so these mock hrt_runner: they lock the CPU-reference dynamics (the same unitary
commutator the .bin computes — what makes the parity gauge meaningful), the honest availability gate (no
runtime → CPU, always a result), and the gauge verdict (a drifting accelerator → DECOHERED).
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


def _rho(d=8, seed=2):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d))
    return (0.5 * (a + a.T)).astype(np.float32)


def test_commutator_rollout_matches_manual():
    H, rho = _H(), _rho()
    out = bf.commutator_rollout(rho, H, n_steps=3, c=0.01)
    r = rho.astype(np.float64)
    for _ in range(3):
        r = r + 0.01 * (H @ r - r @ H)
    np.testing.assert_allclose(out, r)


def test_forecast_falls_back_to_cpu_when_unavailable(monkeypatch):
    monkeypatch.setattr(bf.hrt_runner, "available", lambda: False)
    H, rho = _H(), _rho()
    fc = bf.BpuForecaster("missing.bin", H, n_steps=4, dim=8)
    assert fc.available() is False
    np.testing.assert_allclose(fc.forecast(rho), fc.rollout_cpu(rho))


def test_with_parity_reports_cpu_when_unavailable(monkeypatch):
    monkeypatch.setattr(bf.hrt_runner, "available", lambda: False)
    fc = bf.BpuForecaster("missing.bin", _H(), n_steps=4, dim=8)
    _, g = fc.with_parity(_rho())
    assert g["backend"] == "cpu" and g["parity"] and g["cos"] == 1.0


def test_forecast_uses_bpu_when_available(monkeypatch, tmp_path):
    """When the runtime + .bin are present, forecast() runs on the BPU. Mock hrt to return the CPU
    rollout (a faithful accelerator) and assert the BPU path is taken with parity."""
    H, rho = _H(), _rho()
    binp = tmp_path / "op.bin"; binp.write_bytes(b"x")
    fc = bf.BpuForecaster(str(binp), H, n_steps=5, dim=8)
    monkeypatch.setattr(bf.hrt_runner, "available", lambda: True)
    monkeypatch.setattr(bf.hrt_runner, "infer",
                        lambda *a, **k: {fc.out_name: fc.rollout_cpu(rho)})
    assert fc.available() is True
    bpu, g = fc.with_parity(rho)
    assert g["backend"] == "bpu" and g["parity"]
    assert g["cos"] > 0.999
    np.testing.assert_allclose(fc.forecast(rho), bpu)


def test_gauge_record_decoheres_on_drift():
    fc = bf.BpuForecaster("x", _H(), n_steps=2, dim=8)
    bad = {"backend": "bpu", "n_steps": 2, "cos": 0.1, "rel_err": 0.9, "parity": False}
    rec = fc.gauge_record(bad)
    assert rec["verdict"] == dh.DECOHERED and "drifted" in rec["action"]
    good = {"backend": "bpu", "n_steps": 2, "cos": 1.0, "rel_err": 1e-4, "parity": True}
    assert fc.gauge_record(good)["verdict"] == dh.ENRICHED


def test_bpu_failure_falls_back_without_raising(monkeypatch):
    """A runtime hiccup mid-forward must never propagate a wrong answer — forecast() returns the CPU result."""
    H, rho = _H(), _rho()
    fc = bf.BpuForecaster("x", H, n_steps=3, dim=8)
    monkeypatch.setattr(bf.hrt_runner, "available", lambda: True)
    monkeypatch.setattr(fc, "rollout_bpu", lambda _r: (_ for _ in ()).throw(RuntimeError("boom")))
    np.testing.assert_allclose(fc.forecast(rho), fc.rollout_cpu(rho))
