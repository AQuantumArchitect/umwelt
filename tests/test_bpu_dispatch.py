"""Guards the BPU dispatch node (meerkat/brain/bpu_dispatch.py) — the gauge-tracked seam the
hindbrain offloads batch matmul to. The contract (the plan's verification, applied to compute):
every dispatch is (1) parity-proven against the float64 CPU reference and (2) reversible — an
unavailable/failing backend falls back to CPU and the gauge SAYS so, never a silent wrong answer.

The BPU backend is unavailable on x86 (no .bin, no hobot_dnn), so these run CPU + expansion now;
they're the regression that locks the seam before the kernel lands.
"""
from __future__ import annotations

import numpy as np

from umwelt.foresight import bpu_dispatch as bd

# ops-health telemetry isn't a separate umwelt module — bpu_dispatch carries a no-op shim exposing
# the ENRICHED/DECOHERED verdict constants + stream_record. Use it directly.
dh = bd.dh


def _wide_complex(d=24, seed=3):
    """A complex matmul in the high-dynamic-range regime — the one that needs the expansion (tiny
    coherences against O(1) populations), where naive global INT8 would annihilate the small terms."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    b = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)))
    b = b * 10.0 ** (-rng.integers(0, 7, size=(d, d)))
    return a, b


def test_cpu_backend_is_the_reference():
    a, b = _wide_complex()
    r = bd.Dispatcher().matmul(a, b, backend="cpu")
    assert r.backend == "cpu" and not r.fell_back
    assert r.ref_err == 0.0 and r.parity
    np.testing.assert_allclose(r.result, a @ b)


def test_expansion_backend_proves_parity_with_cpu():
    """The expansion (BPU-native format) tracks the CPU float64 reference to well under tolerance —
    the small coherences survive (the whole point of the number system)."""
    a, b = _wide_complex()
    r = bd.Dispatcher(prove=True).matmul(a, b, backend="expansion")
    assert r.backend == "expansion" and not r.fell_back
    assert r.ref_err < bd.PARITY_TOL, f"expansion drifted {r.ref_err:.2e}"
    assert r.parity


def test_bpu_unavailable_falls_back_to_cpu_and_records_it():
    """No compiled .bin on x86 → request bpu, get cpu, gauge flags the fallback (reversible, honest)."""
    disp = bd.Dispatcher()
    assert disp.backends_available()["bpu"] is False
    a, b = _wide_complex()
    r = disp.matmul(a, b, backend="bpu")
    assert r.requested == "bpu" and r.backend == "cpu" and r.fell_back
    assert r.parity                                   # CPU vs CPU
    np.testing.assert_allclose(r.result, a @ b)


def test_no_fallback_raises_when_backend_unavailable():
    disp = bd.Dispatcher(fallback=False)
    a, b = _wide_complex()
    try:
        disp.matmul(a, b, backend="bpu")
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError with fallback disabled")


def test_batched_matmul_rolls_up_worst_case_parity():
    disp = bd.Dispatcher(prove=True)
    pairs = [_wide_complex(seed=s) for s in range(5)]
    results, roll = disp.batched_matmul(pairs, backend="expansion")
    assert len(results) == 5
    assert roll.n_ops == 5 and roll.backend == "expansion"
    # rollup worst-case error is the max over the batch, and each result matches its own CPU ref
    for (a, b), res in zip(pairs, results):
        assert np.max(np.abs(res - (a @ b))) <= roll.ref_err + 1e-12
    assert roll.parity == (roll.ref_err <= bd.PARITY_TOL)


def test_gauge_record_is_canonical_enriched_stream():
    disp = bd.Dispatcher(prove=True)
    a, b = _wide_complex()
    rec = disp.gauge_record(disp.matmul(a, b, backend="expansion"))
    assert rec["id"] == bd.STREAM_ID
    assert rec["verdict"] == dh.ENRICHED          # brain output, fenced from inputs
    assert rec["source"] == "bpu_node"
    assert "parity" in rec["gauge"] and "ms" in rec["gauge"] and "backend" in rec["gauge"]


def test_parity_break_surfaces_as_decohered():
    """If a backend silently drifts past tolerance, the gauge degrades the verdict — a drifting
    compute stream is surfaced like any unhealthy datastream, not hidden."""
    res = bd.DispatchResult(backend="bpu", requested="bpu", result=np.zeros((2, 2)),
                            ms=1.0, ref_err=0.5, parity=False, fell_back=False)
    rec = bd.Dispatcher().gauge_record(res)
    assert rec["verdict"] == dh.DECOHERED
    assert "drifted" in rec["action"]


def test_expansion_quantize_preserves_small_against_large():
    """The number-system unit test: a 1e-6 coherence beside an O(1) population survives quantization
    (global INT8 would zero it). This is why the matmul keeps the Berry clock."""
    M = np.array([[1.0 + 0j, 1e-6 + 1e-6j], [1e-6 - 1e-6j, 0.5 + 0j]])
    q = bd.expansion_quantize(M, levels=2, bits=8, base=2.0)
    assert abs(q[0, 1]) > 1e-7                      # the small coherence is NOT crushed to zero
    assert abs(q[0, 1] - M[0, 1]) < 1e-7            # and it's accurate


# ── the banded scheme (the BPU-faithful number system, regime_split) ──────────

def test_banded_backend_proves_parity_and_is_bpu_faithful():
    """The banded backend (shared-scale bands + Σ Aᵢ@Bⱼ cross-product) tracks the CPU reference to well
    under tolerance at d=32 — and unlike the per-element expansion, every operation is one shift+scale
    INT8 matmul, so this parity is the kernel's parity."""
    a, b = _wide_complex(d=32)
    r = bd.Dispatcher(prove=True).matmul(a, b, backend="banded")
    assert r.backend == "banded" and not r.fell_back
    assert r.ref_err < bd.PARITY_TOL, f"banded drifted {r.ref_err:.2e}"
    assert r.parity


def test_banded_clears_the_floor_that_one_band_misses():
    """One shared-scale band (= block-float, the d=32 wall) loses the small coherences; the band ladder
    recovers them — each extra band ~+2 orders. This is the floor-break the per-element fractal got but
    a BPU-native single block-float can't."""
    a, b = _wide_complex(d=32)
    ref = a @ b
    small = np.abs(ref) < 0.01 * np.max(np.abs(ref))
    err = lambda C: float(np.max(np.abs((C - ref)[small])) / (np.max(np.abs(ref[small])) + 1e-30))
    e1 = err(bd.banded_matmul(a, b, levels=1))      # one band = block-float = the wall
    e3 = err(bd.banded_matmul(a, b, levels=3, keep_order=4))
    assert e1 > 0.1                                  # the floor: block-float kills the small coherences
    assert e3 < 1e-3                                 # the ladder clears it
    assert e3 < e1 / 100.0


def test_banded_keep_order_drops_the_deep_corner_for_free():
    """'Only the batches that matter': keeping i+j ≤ levels-1 (the anti-diagonal triangle) matches the
    full K² grid to the floor — the dropped corner is below precision, so it costs fewer matmuls, not
    accuracy."""
    a, b = _wide_complex(d=32)
    full = bd.banded_matmul(a, b, levels=4, keep_order=99)      # the whole 4×4 grid
    triangle = bd.banded_matmul(a, b, levels=4, keep_order=3)   # i+j ≤ 3 only
    assert np.max(np.abs(full - triangle)) < 1e-6               # the corner was below the floor


def test_bands_complex_reconstructs_and_is_int8():
    """Each band is an INT8 mantissa (|q| ≤ 127) at a shared scale; their sum reconstructs the array.
    The shared scale (one per band) is exactly the BPU's shift+scale register."""
    a, _ = _wide_complex(d=16)
    bands = bd._bands_complex(a, levels=4, bits=8)
    recon = sum(q * s for q, s in bands)
    assert np.max(np.abs(recon - a)) < 1e-4                     # 4 bands ≈ 32 bits of range
    for q, _s in bands:
        assert np.max(np.abs(q.real)) <= 127 and np.max(np.abs(q.imag)) <= 127


def test_auto_self_schedules_to_a_parity_holding_backend():
    """The compute-self-scheduling seed: in 'auto' mode the node explores each available non-cpu backend
    (>= MIN_SAMPLES probes) then runs the fastest backend that holds parity, with cpu as the floor. On
    this x86 box numpy cpu is fastest, so after exploration auto converges to cpu — and every explored
    backend recorded a high parity rate (banded/expansion match the float64 reference)."""
    a, b = _wide_complex(d=24)
    disp = bd.Dispatcher(backend="auto", prove=True)
    seen = set()
    for _ in range(40):
        r = disp.matmul(a, b)
        seen.add(r.backend)
    stats = disp.stats()
    # explored the available non-cpu backends (banded + expansion are always available on CPU)
    assert stats["banded"]["n"] >= disp._AUTO_MIN_SAMPLES
    assert stats["expansion"]["n"] >= disp._AUTO_MIN_SAMPLES
    assert stats["banded"]["parity_rate"] >= 0.9       # the band system matches the reference
    assert "cpu" in stats and stats["cpu"]["n"] > 0     # cpu measured via the reference run
    # after enough samples it settles on the fastest parity-holding backend — cpu on this hardware
    assert disp.choose() == "cpu"
