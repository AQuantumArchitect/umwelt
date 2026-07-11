"""BPU dispatch — the gauge-tracked enrichment node the hindbrain offloads batch compute to.

The CPU stays MASTER (the field, the forebrain, the live loop). This is the seam where the
compute-heavy, batchy, latency-tolerant work — the hindbrain's consolidation matmuls, the field's
expansion-ρ, later YAMNet/qwen — gets *dispatched* to the accelerator and folded back into the
enrichment flow as just another gauge-tracked datastream node (Luke's framing: "keep the CPU as
the master, dispatch the BPU for the compute-heavy hindbrain/learning tasks, batch at an API level
with the gauge-tracked datastreams — it becomes another node in the enrichment flow").

Today the BPU kernel isn't compiled (the offline HBDK3 wall — see ops/bpu/README.md), so every
dispatch is CPU-backed. That's fine: the SEAM is the deliverable. Mirrors how ask_gateway isolates
the qwen swap behind a URL — here the backend is chosen at the dispatch edge, every call is metered
and PARITY-PROVEN against the CPU reference (the gauge), and any backend failure falls back to CPU.
When the `.bin` lands it's a one-line backend swap; nothing upstream changes.

One primitive: the complex matmul. H@ρ, the expansion-ρ Lindblad step, a CNN conv, a transformer
layer — they all reduce to it, so the node speaks matmul and batches of matmul.

Backends:
  cpu        float64 numpy — the master / the reference everything is judged against. Always up.
  expansion  the 2-term base-2 INT8 floating-point EXPANSION (the fib_fractal finding,
             experiments/fib_fractal.py) — accurate, but PER-ELEMENT scale (one float matmul of
             quantized operands), which the BPU can't natively do. The accuracy upper bound.
  banded     the BPU-FAITHFUL number system (Luke's regime-split, experiments/regime_split.py):
             shared-scale INT8 bands + the Σᵢⱼ Aᵢ@Bⱼ cross-product. Each band is one shift+scale
             tensor and each cross-term one INT8 matmul, summed — bit-for-bit the on-chip op sequence,
             so its CPU parity IS the kernel's parity. Clears the d=32 coherence floor a single
             block-float band can't (levels=3 → small-coh ~1e-5). The interference computed in its own
             bin. Tune via UMWELT_BAND_LEVELS / UMWELT_BAND_KEEP.
  bpu        BAKED-operator `.bin` via hrt_model_exec (the accelerator runtime) — the on-silicon-
             validated path (banded H·ρ rel 4.9e-3 on the RDK). NOT a generic matmul backend: a dynamic×
             dynamic matmul mis-maps on-device, so the generic entry point falls back to cpu/banded. The
             real consumer is the forecast/dream rollout (gen_rollout_banded), which calls hrt_runner.

The gauge: each dispatch carries {backend, ms, ref_err, parity} — the divergence the gauged-ensemble
contract wants. The BPU/expansion path is "proven" only when it matches the float64 CPU reference to
within tolerance (well under actuation hysteresis). gauge_record() emits the canonical
datastream_health ENRICHED record (brain output, fenced from inputs) so the node shows up on the
same health surface as every other stream.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np

# ops-health telemetry ships only with the origin deployment; degrade to a no-op record shim in the
# extracted library (the gauge record is optional enrichment on the health surface).
try:
    from . import datastream_health as dh   # type: ignore
except ImportError:
    class _DHShim:
        HEALTHY, STUCK, DARK = "HEALTHY", "STUCK", "DARK"
        ENRICHED, DECOHERED = "ENRICHED", "DECOHERED"

        @staticmethod
        def stream_record(stream_id, verdict, *, gauge=None, members=None, evidence=None,
                          action="", owner="", source=""):
            ev = evidence if isinstance(evidence, list) else ([evidence] if evidence else [])
            return {"id": stream_id, "verdict": verdict, "gauge": gauge or {},
                    "members": sorted(members) if members else [stream_id], "evidence": ev,
                    "action": action, "owner": owner, "source": source}
    dh = _DHShim()

log = logging.getLogger("bpu_dispatch")

# Parity tolerance: a dispatched matmul is "in parity" with the CPU reference when the worst-case
# element error is below this. 1e-3 is ~100x under the 0.1 actuation hysteresis and ~10x above the
# expansion's measured ~1e-5 wide-range error — so real numerical drift trips it, the expansion doesn't.
PARITY_TOL = float(os.environ.get("BPU_PARITY_TOL", "1e-3"))
STREAM_ID = "bpu.dispatch"


# ── the number systems (one matmul primitive each) ───────────────────────────

def _expansion_quantize_real(x: np.ndarray, levels: int, bits: int, base: float) -> np.ndarray:
    """The validated multi-level INT8 expansion of a real array (fib_fractal `_fractal_real`): greedy,
    each layer an INT8 mantissa at the base-power tranche of the residual's own magnitude (smallest
    power that contains it → mantissa ∈ [1/base, 1), never clips). 2 base-2 layers = Luke's
    value+computed-difference; storing the residual in the 2nd register is the catastrophic-cancellation
    cure that preserves the small coherences (the Berry clock). base-2 = hardware-native free shifts."""
    maxq = 2 ** (bits - 1) - 1
    lb = np.log(base)
    recon = np.zeros_like(x)
    for _ in range(levels):
        residual = x - recon
        mag = np.abs(residual)
        k = np.where(mag > 1e-30, np.floor(np.log(mag + 1e-30) / lb) + 1.0, 0.0)
        scale = base ** k
        mant = np.round(residual / scale * maxq).clip(-maxq, maxq) / maxq
        recon = recon + mant * scale
    return recon


def _block_float_real(x: np.ndarray, levels: int, bits: int) -> np.ndarray:
    """Multi-level BLOCK float — the BPU's ACTUAL native format. Unlike the per-element expansion above,
    every level shares ONE base-2 scale across the whole block (the block's max magnitude), exactly what
    the BPU's shift+scale registers do (hbDNNQuantiShift / hbDNNQuantiScale = one shift per tensor). Each
    successive level re-quantizes the residual at a NEW shared scale — so K levels = K block-float matmuls
    on-chip, summed. This is the honest question for the kernel: how many shared-scale levels does the
    field evolution need? (Per-element is more accurate but the BPU can't do it natively.)"""
    maxq = 2 ** (bits - 1) - 1
    recon = np.zeros_like(x)
    for _ in range(levels):
        residual = x - recon
        m = float(np.max(np.abs(residual)))
        if m < 1e-30:
            break
        k = np.floor(np.log2(m)) + 1.0          # one shared exponent for the whole block (free shift)
        scale = 2.0 ** k
        mant = np.round(residual / scale * maxq).clip(-maxq, maxq) / maxq
        recon = recon + mant * scale
    return recon


def block_float_quantize(M: np.ndarray, *, levels: int = 2, bits: int = 8) -> np.ndarray:
    """The BPU-native K-level block float (shared scale per level), real+imag independently. This is what
    a compiled .bin will actually compute; `expansion_quantize` (per-element) is the more-accurate upper
    bound the BPU can't natively reach. The theory experiment characterizes the gap + the needed depth."""
    M = np.asarray(M)
    if np.iscomplexobj(M):
        return _block_float_real(M.real, levels, bits) + 1j * _block_float_real(M.imag, levels, bits)
    return _block_float_real(M, levels, bits)


def expansion_quantize(M: np.ndarray, *, levels: int = 2, bits: int = 8, base: float = 2.0) -> np.ndarray:
    """Quantize a (complex) array into the per-element INT8 expansion — real + imag independently
    (fib_fractal `q_fractal`). Each element finds its OWN base-power tranche (more accurate than the BPU's
    shared-scale block float; see block_float_quantize for the hardware-faithful version)."""
    M = np.asarray(M)
    if np.iscomplexobj(M):
        return (_expansion_quantize_real(M.real, levels, bits, base)
                + 1j * _expansion_quantize_real(M.imag, levels, bits, base))
    return _expansion_quantize_real(M, levels, bits, base)


# ── the band scheme: interference in its own INT8 bin (Luke's regime-split, experiments/regime_split.py) ─

def _bands_complex(M: np.ndarray, levels: int, bits: int) -> list[tuple[np.ndarray, float]]:
    """Peel `levels` SHARED-scale INT8 bands of a complex array — real + imag share each band's single
    shift+scale (exactly the BPU's one-shift-per-tensor). Band 0 captures the largest magnitudes; each next
    band re-scales to the residual it left behind, carrying progressively smaller values at full INT8
    precision. So the small coherences (the interference) live in their OWN band/bin instead of underflowing
    against the O(1) diagonal — the d=32 floor cure. Returns [(int8_complex_mantissa, scale)], M ≈ Σ q·s."""
    maxq = 2 ** (bits - 1) - 1
    recon = np.zeros_like(M, dtype=complex)
    bands: list[tuple[np.ndarray, float]] = []
    for _ in range(levels):
        res = M - recon
        amax = max(float(np.max(np.abs(res.real))), float(np.max(np.abs(res.imag))))
        if amax < 1e-300:
            break
        scale = amax / maxq
        q = (np.round(res.real / scale).clip(-maxq, maxq)
             + 1j * np.round(res.imag / scale).clip(-maxq, maxq))
        bands.append((q, scale))
        recon = recon + q * scale
    return bands


def banded_matmul(a: np.ndarray, b: np.ndarray, *, levels: int = 3, bits: int = 8,
                  keep_order: int | None = None) -> np.ndarray:
    """C = A@B as Σᵢⱼ Aᵢ@Bⱼ over shared-scale INT8 bands — the BPU-FAITHFUL matmul. Each (band i of A) @
    (band j of B) is one INT8×INT8 → INT32 on-chip matmul at scale sᵢ·sⱼ, summed at the CPU edge. A term's
    scale is sᵢ·sⱼ ∝ 256^-(i+j), so `keep_order` (max i+j) drops the deep low-scale corner that sits below
    the precision floor — 'only the batches that matter'. Default keep_order = levels-1 (the anti-diagonal
    triangle that clears the d=32 floor at ~½·levels² matmuls). Unlike the per-element expansion this is
    EXACTLY what a compiled .bin will compute, so proving its parity on CPU proves the kernel math."""
    a = np.asarray(a)
    b = np.asarray(b)
    cmplx = np.iscomplexobj(a) or np.iscomplexobj(b)
    Ab = _bands_complex(a.astype(complex), levels, bits)
    Bb = _bands_complex(b.astype(complex), levels, bits)
    if keep_order is None:
        keep_order = max(0, levels - 1)
    C = np.zeros((a.shape[0], b.shape[-1]), dtype=complex)
    for i, (qa, sa) in enumerate(Ab):
        for j, (qb, sb) in enumerate(Bb):
            if (i + j) > keep_order:
                continue
            C = C + (qa * sa) @ (qb * sb)
    return C if cmplx else C.real


class Backend:
    """A compute backend: name + availability + the matmul primitive."""
    name = "base"

    def available(self) -> bool:
        return True

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CpuBackend(Backend):
    """float64 numpy — the master and the reference. The CPU stays in charge."""
    name = "cpu"

    def matmul(self, a, b):
        return np.asarray(a) @ np.asarray(b)


class ExpansionBackend(Backend):
    """The 2-term base-2 INT8 expansion — the BPU's native number format, run on CPU so parity is
    provable today. `q(q(A) @ q(B))` (fib_fractal `make_mm`): quantize operands → multiply →
    re-quantize the output, i.e. the on-chip number system end to end."""
    name = "expansion"

    def __init__(self, *, levels: int = 2, bits: int = 8, base: float = 2.0):
        self.levels, self.bits, self.base = int(levels), int(bits), float(base)

    def _q(self, M):
        return expansion_quantize(M, levels=self.levels, bits=self.bits, base=self.base)

    def matmul(self, a, b):
        return self._q(self._q(np.asarray(a)) @ self._q(np.asarray(b)))


class BandedBackend(Backend):
    """The BPU-FAITHFUL number system: shared-scale INT8 bands + the Σᵢⱼ Aᵢ@Bⱼ cross-product (Luke's
    regime-split — interference computed in its own band/bin). Unlike ExpansionBackend (per-element scale,
    one float matmul — accurate but NOT what the BPU can do), every band here is one shift+scale tensor and
    every cross-term one INT8 matmul, summed — bit-for-bit the on-chip operation sequence a .bin runs. This
    is the backend whose CPU parity actually PROVES the kernel. levels=3 clears the d=32 coherence floor
    (small-coh ~1e-5) at ~6 matmuls; the deep corner is dropped via keep_order (default levels-1)."""
    name = "banded"

    def __init__(self, *, levels: int | None = None, bits: int = 8, keep_order: int | None = None):
        self.levels = int(levels if levels is not None else os.environ.get("UMWELT_BAND_LEVELS", "3"))
        self.bits = int(bits)
        _ko = os.environ.get("UMWELT_BAND_KEEP", "")
        self.keep_order = int(keep_order if keep_order is not None else (_ko if _ko != "" else self.levels - 1))

    def matmul(self, a, b):
        return banded_matmul(a, b, levels=self.levels, bits=self.bits, keep_order=self.keep_order)


class BpuBackend(Backend):
    """The real accelerator path — via `hrt_model_exec` (the accelerator runtime), NOT `pyeasy_dnn`.

    Two on-silicon findings reshaped this (project_bpu_kernel, Wall 2): (1) a GENERIC dynamic×dynamic matmul
    mis-maps to the BPU's conv engine and returns garbage on-device — so the BPU is NOT a drop-in generic
    matmul backend; (2) the working path is a BAKED-OPERATOR kernel (H baked as a const weight, only ρ
    streams) run through `hrt_model_exec` — there the banded H·ρ matches float64 (rel 4.9e-3). So this backend
    runs a PRECOMPILED baked-operator `.bin` whose CPU-side reference is supplied by the caller; the
    generic `matmul(a, b)` path stays unavailable on purpose (the dispatcher falls back to `cpu`/`banded`,
    both correct on the A55). The real BPU consumer is the forecast/dream rollout (gen_rollout_banded), which
    calls hrt_runner directly. This class keeps the dispatch seam honest + gauge-metered for that kernel."""
    name = "bpu"

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.environ.get("BPU_NODE_MODEL", "")

    def available(self) -> bool:
        # Honest: only when a compiled baked-operator `.bin` is configured AND the hrt runtime exists (RDK-only).
        if not self.model_path or not os.path.exists(self.model_path):
            return False
        try:
            from . import hrt_runner
        except Exception:
            return False
        return hrt_runner.available()

    def matmul(self, a, b):  # pragma: no cover - exercised only on the RDK with a configured baked .bin
        # Generic dynamic matmul on the BPU is the proven-broken case (cosine 0 on-device). Raising here
        # routes the dispatcher to the CPU reference — the truthful behaviour. The baked-operator kernel
        # is driven through hrt_runner by the forecast path, not this generic entry point.
        raise RuntimeError("BPU generic dynamic matmul is not on-device-correct; use the baked-H rollout "
                           "kernel via hrt_runner (the forecast/dream path)")


# ── the dispatcher: pick a backend, time it, prove parity, fold into the gauge ──

@dataclass
class DispatchResult:
    backend: str                       # the backend that actually ran (post-fallback)
    requested: str                     # the backend that was asked for
    result: np.ndarray
    ms: float                          # wall time of the dispatched op
    ref_err: float                     # worst-case element error vs the CPU float64 reference
    parity: bool                       # ref_err <= PARITY_TOL (proven equivalent to CPU)
    fell_back: bool                    # requested backend was unavailable/failed → CPU
    n_ops: int = 1                     # matmuls in this dispatch (a batch rolls up)
    extra: dict = field(default_factory=dict)

    def gauge(self) -> dict:
        return {"backend": self.backend, "requested": self.requested, "ms": round(self.ms, 3),
                "ref_err": float(self.ref_err), "parity": bool(self.parity),
                "fell_back": bool(self.fell_back), "n_ops": int(self.n_ops), **self.extra}


class Dispatcher:
    """The gauge node. CPU is always the reference; the requested backend runs alongside (when it's
    not CPU) and is judged against it. Reversible: an unavailable or throwing backend falls back to
    CPU and the gauge records the fallback — never a silent wrong answer."""

    # compute-self-scheduling (the metacognition seed): in 'auto' mode the node picks its OWN backend
    # from the live gauge — explore each available backend a few times, then run the FASTEST one that
    # keeps parity (cpu is the always-correct floor). On the A55 today numpy wins (the measured BPU-loses
    # finding, project_bpu_kernel) so auto converges to cpu; when a parity-holding .bin is faster it picks
    # bpu — the brain scheduling its own compute by measured cost, not a hardcoded choice.
    _AUTO_MIN_SAMPLES = int(os.environ.get("BPU_AUTO_MIN_SAMPLES", "5"))   # probes before trusting a backend
    _AUTO_PARITY_RATE = float(os.environ.get("BPU_AUTO_PARITY_RATE", "0.9"))

    def __init__(self, backend: str | None = None, *, fallback: bool = True, prove: bool = True):
        self._backends = {b.name: b for b in
                          (CpuBackend(), ExpansionBackend(), BandedBackend(), BpuBackend())}
        self.primary = backend or os.environ.get("BPU_NODE_BACKEND", "cpu")
        self.fallback = bool(fallback)
        self.prove = bool(prove)   # also run CPU to measure parity (off → trust the backend, no ref)
        # per-backend running stats for 'auto': {name: {"n", "parity_ok", "ema_ms"}}
        self._stats: dict[str, dict] = {}

    def backends_available(self) -> dict:
        return {name: b.available() for name, b in self._backends.items()}

    def _record_stat(self, name: str, ms: float, parity: bool) -> None:
        s = self._stats.setdefault(name, {"n": 0, "parity_ok": 0, "ema_ms": ms})
        s["n"] += 1
        s["parity_ok"] += 1 if parity else 0
        s["ema_ms"] = 0.2 * ms + 0.8 * s["ema_ms"]   # latency EMA

    def stats(self) -> dict:
        """Per-backend {n, parity_rate, ema_ms} — what 'auto' schedules on (and the gauge can surface)."""
        return {n: {"n": s["n"], "parity_rate": (s["parity_ok"] / s["n"]) if s["n"] else 0.0,
                    "ema_ms": round(s["ema_ms"], 4)} for n, s in self._stats.items()}

    def choose(self) -> str:
        """Pick the backend for the next op from the live gauge (the self-scheduling decision). Probe any
        available non-cpu backend that's under-sampled; else run the fastest backend that holds parity,
        with cpu as the always-correct floor."""
        avail = [n for n, b in self._backends.items() if n != "cpu" and b.available()]
        for n in avail:                                   # explore: gather stats before trusting
            if self._stats.get(n, {}).get("n", 0) < self._AUTO_MIN_SAMPLES:
                return n
        # exploit: cpu's measured cost (or +inf until measured) vs each parity-holding backend
        candidates = {"cpu": self._stats.get("cpu", {}).get("ema_ms", float("inf"))}
        for n in avail:
            s = self._stats.get(n)
            if s and s["n"] and (s["parity_ok"] / s["n"]) >= self._AUTO_PARITY_RATE:
                candidates[n] = s["ema_ms"]
        return min(candidates, key=candidates.get) if candidates else "cpu"

    def _pick(self, requested: str) -> tuple[Backend, bool]:
        b = self._backends.get(requested)
        if b is not None and b.available():
            return b, False
        if not self.fallback:
            raise RuntimeError(f"backend {requested!r} unavailable and fallback disabled")
        return self._backends["cpu"], (requested != "cpu")

    def matmul(self, a, b, *, backend: str | None = None) -> DispatchResult:
        a = np.asarray(a)
        b = np.asarray(b)
        requested = backend or self.primary
        if requested == "auto":                 # self-schedule: pick the backend from the live gauge
            requested = self.choose()
        chosen, fell_back = self._pick(requested)

        t0 = time.perf_counter()
        try:
            result = chosen.matmul(a, b)
        except Exception as e:                       # a backend that throws (BPU shape/dtype, …) → CPU
            if not self.fallback or chosen.name == "cpu":
                raise
            log.warning("backend %s failed (%s) — falling back to CPU", chosen.name, type(e).__name__)
            chosen, fell_back = self._backends["cpu"], True
            result = chosen.matmul(a, b)
        ms = (time.perf_counter() - t0) * 1000.0

        # parity vs the float64 CPU reference (skip the redundant ref when CPU already ran)
        if self.prove and chosen.name != "cpu":
            tref = time.perf_counter()
            ref = self._backends["cpu"].matmul(a, b)
            cpu_ms = (time.perf_counter() - tref) * 1000.0
            ref_err = float(np.max(np.abs(np.asarray(result) - ref))) if ref.size else 0.0
            self._record_stat("cpu", cpu_ms, True)   # the reference run measures cpu's own cost (for 'auto')
        else:
            ref_err = 0.0
        parity = ref_err <= PARITY_TOL
        self._record_stat(chosen.name, ms, parity)   # feed the self-scheduling gauge
        return DispatchResult(backend=chosen.name, requested=requested, result=np.asarray(result),
                              ms=ms, ref_err=ref_err, parity=parity,
                              fell_back=fell_back, n_ops=1)

    def batched_matmul(self, pairs, *, backend: str | None = None) -> tuple[list, DispatchResult]:
        """Dispatch a BATCH of (a, b) matmuls as one node call — the throughput unit (amortizing
        dispatch is where the accelerator wins). Returns (results, rolled-up gauge): worst-case
        parity across the batch, total wall time, op count. This is the 'batch at an API level'
        the enrichment flow hands the node."""
        requested = backend or self.primary
        results, worst_err, total_ms, any_fallback, ran = [], 0.0, 0.0, False, requested
        for a, b in pairs:
            r = self.matmul(a, b, backend=requested)
            results.append(r.result)
            worst_err = max(worst_err, r.ref_err)
            total_ms += r.ms
            any_fallback = any_fallback or r.fell_back
            ran = r.backend
        rollup = DispatchResult(backend=ran, requested=requested, result=np.asarray([]),
                                ms=total_ms, ref_err=worst_err, parity=(worst_err <= PARITY_TOL),
                                fell_back=any_fallback, n_ops=len(results))
        return results, rollup

    def gauge_record(self, result: DispatchResult) -> dict:
        """The canonical datastream_health record for a dispatch — ENRICHED (brain output, fenced
        from inputs). The verdict degrades when parity breaks: a backend silently drifting from the
        CPU reference is a DECOHERED compute stream, surfaced like any other unhealthy datastream."""
        verdict = dh.ENRICHED if result.parity else dh.DECOHERED
        return dh.stream_record(
            STREAM_ID, verdict, gauge=result.gauge(), source="bpu_node",
            owner="brain",
            action="" if result.parity else f"backend {result.backend} drifted {result.ref_err:.2e} from CPU",
            evidence=f"{result.n_ops} matmul(s) on {result.backend}"
                     f"{' (fell back from ' + result.requested + ')' if result.fell_back else ''}",
        )


_DISPATCHER: Dispatcher | None = None


def dispatcher() -> Dispatcher:
    """The process-wide dispatch node (lazy singleton). The hindbrain / consolidation call this to
    offload batch matmul; the standalone service (ops/collection/bpu_node.py) wraps the same one."""
    global _DISPATCHER
    if _DISPATCHER is None:
        _DISPATCHER = Dispatcher()
    return _DISPATCHER


def _demo():  # pragma: no cover - the comparative, gauge-tracked bench (run on the laptop now)
    rng = np.random.default_rng(7)
    d = 32
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    B = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)))
    B = B * 10.0 ** (-rng.integers(0, 7, size=(d, d)))        # wide range — the regime that needs it
    disp = Dispatcher(prove=True)
    print(f"backends available: {disp.backends_available()}")
    print(f"{'backend':>10s} {'ms':>9s} {'ref_err':>11s}  parity")
    print("-" * 44)
    for name in ("cpu", "expansion", "banded", "bpu"):
        r = disp.matmul(A, B, backend=name)
        tag = "" if not r.fell_back else f"  (fell back from {r.requested})"
        print(f"{r.backend:>10s} {r.ms:9.3f} {r.ref_err:11.2e}  {r.parity}{tag}")
    print("\ngauge record (last):", disp.gauge_record(r))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
