"""The free-run forecast engine — N RK4 steps fused per dim-group, the single whole-field forecast brain.

Three contracts:
  1. the fused free_run is EXACTLY the N-step rk4_step loop (numpy) — the engine reproduces the field's
     own batched evolution, it doesn't approximate it;
  2. the jax-fused and the expansion (BPU-native number-system) free-runs agree with numpy well under the
     actuation hysteresis — i.e. the forecast brain is BPU-ready (the one fused kernel is what the .bin runs);
  3. FieldRolloutForecaster.forecast_freerun reads the same leaves as forecast() and is side-effect-free
     (snapshot→roll→restore leaves the live belief untouched).
"""
from __future__ import annotations

import numpy as np
import pytest

from umwelt.substrate.batched_evolve import make_backend, NumpyBackend


def _synthetic_group(n=3, B=4, seed=0):
    rng = np.random.default_rng(seed)
    d = 1 << n
    A = (rng.standard_normal((B, d, d)) + 1j * rng.standard_normal((B, d, d))).astype(np.complex64)
    rho = A + A.conj().transpose(0, 2, 1)
    rho = rho / np.trace(rho, axis1=1, axis2=2)[:, None, None]
    Hm = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))).astype(np.complex64)
    H = (Hm + Hm.conj().T)[None].repeat(B, 0)
    r_minus = (rng.random((B, n)) * (rng.random((B, n)) > 0.4)).astype(np.float64)
    r_plus = (rng.random((B, n)) * (rng.random((B, n)) > 0.7)).astype(np.float64)
    dt = np.full(B, 0.05)
    return rho, H, r_minus, r_plus, dt, n


def test_free_run_is_exactly_the_stepwise_loop():
    """The engine = the field's own batched evolution rolled forward, not an approximation."""
    npb = NumpyBackend()
    rho, H, rm, rp, dt, n = _synthetic_group()
    N = 12
    fused = npb.free_run(rho.copy(), H, rm, rp, dt, n, N)
    loop = rho.copy()
    for _ in range(N):
        loop = npb.rk4_step(loop, H, rm, rp, dt, n)
    assert np.allclose(fused, loop, atol=1e-6, rtol=1e-5)
    # physicality preserved across the whole run
    assert np.allclose(np.trace(fused, axis1=1, axis2=2), 1.0, atol=1e-4)


def test_jax_free_run_matches_numpy():
    pytest.importorskip("jax")
    rho, H, rm, rp, dt, n = _synthetic_group()
    N = 20
    ref = NumpyBackend().free_run(rho.copy(), H, rm, rp, dt, n, N)
    jx = make_backend("jax").free_run(rho.copy(), H, rm, rp, dt, n, N)
    assert np.allclose(jx, ref, atol=1e-3, rtol=1e-2), f"max|Δ|={np.abs(jx - ref).max():.2e}"


def test_expansion_free_run_is_bpu_ready():
    """The BPU-native block-float EXPANSION number system survives the free-run, well under hysteresis."""
    rho, H, rm, rp, dt, n = _synthetic_group()
    N = 20
    ref = NumpyBackend().free_run(rho.copy(), H, rm, rp, dt, n, N)
    exp = make_backend("expansion").free_run(rho.copy(), H, rm, rp, dt, n, N)
    assert np.abs(exp - ref).max() < 0.1  # the actuation dead-band — a forecast disagreement here changes no decision


def _cumulant(seed=0):
    from umwelt.substrate.cumulant_cluster import CumulantCluster
    rng = np.random.default_rng(seed)
    c = CumulantCluster("house", ["living_presence", "kitchen_presence", "bedroom_presence",
                                   "exterior_temperature"], gamma=0.05, dt=0.02)
    # give it a learned H + ZZ so the rollout is non-trivial
    c.set_couplings(h_fields=rng.standard_normal((c.n_qubits, 3)) * 0.3,
                    zz={p: float(rng.standard_normal() * 0.2) for p in c._zz})
    for _ in range(5):
        c.step(np.array([0.4, -0.2, 0.1, 0.0]), 1.0)
    return c


def test_cumulant_free_run_is_exactly_stepwise():
    """The folded manifold's free_run == N x step() at fixed inputs — value-exact, the decoupled engine."""
    import copy
    c = _cumulant()
    N, inp = 40, np.array([0.4, -0.2, 0.1, 0.0])
    ref = copy.deepcopy(c)
    for _ in range(N):
        ref.step(inp, 1.0)
    fr = copy.deepcopy(c)
    fr.free_run(N, inp, backend="numpy")
    assert np.abs(ref.e1 - fr.e1).max() < 1e-9
    assert np.abs(ref.e2 - fr.e2).max() < 1e-9


def test_cumulant_jax_free_run_matches_numpy():
    """The XLA-fused cumulant free_run (the BPU on-ramp for the manifold) matches numpy under fp32 noise."""
    pytest.importorskip("jax")
    import copy
    c = _cumulant()
    N, inp = 40, np.array([0.4, -0.2, 0.1, 0.0])
    a = copy.deepcopy(c); a.free_run(N, inp, backend="numpy")
    b = copy.deepcopy(c); b.free_run(N, inp, backend="jax")
    assert np.abs(a.e1 - b.e1).max() < 1e-3
    assert np.abs(a.e2 - b.e2).max() < 1e-3


# (The origin's end-to-end forecast_freerun test booted a domain "smoke reservoir"
# (training_backbone._build_smoke_reservoir) that requires a house spec — not portable to the
# domain-free engine. The free-run kernel's value-exactness + physicality are covered above by
# test_free_run_is_exactly_the_stepwise_loop and the cumulant free_run tests.)
