"""bpu_forecast — frozen-H rollouts on the BPU for the forecast/dream brain (Luke's architecture, realized).

The field's H·ρ runs value-correct on the BPU silicon once H is baked as a fixed on-chip weight and run through
hrt_model_exec (project_bpu_kernel, Wall 2 cracked). This is the in-process consumer: a forecaster that spins
ρ forward N steps under a FROZEN operator on the accelerator, freeing the GIL-bound A55. The operator is baked
OFFLINE (x86 HBDK3) into a `.bin` shipped with the release; the live brain only RUNS it. H is re-baked at the
release/siesta cadence — never per-tick — exactly the frozen-H design (the CPU learns + re-bakes; the BPU
spins). Spin different N for multi-horizon surprise (short N = fast dynamics, long N = slow patterns).

The dynamics are the UNITARY commutator (ρ_{t+1} = ρ_t + c·[H,ρ_t]) — the matmul-heavy coherent core that the
kernel implements; the CPU fallback runs the SAME dynamics, so the parity gauge (BPU vs CPU) is meaningful (it
proves the silicon is faithful, not that the proxy equals the full Lindblad). This is an ADDITIONAL fast/coarse
forecast channel, not a replacement for the value-exact CPU free-run; it feeds surprise like any forecast.

Availability is honest: needs the `.bin` + the hrt runtime (RDK-only), else `available()` is False and the
caller uses the CPU path. Opt-in via UMWELT_BPU_FORECAST + a baked operator at BPU_FORECAST_MODEL.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np

# ops-health telemetry + the accelerator runtime ship only with the origin deployment; in the
# extracted library they degrade to no-op shims. Health emission is optional; the runtime shim's
# available()==False routes every forecast to the CPU reference.
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

try:
    from . import hrt_runner   # type: ignore
except ImportError:
    class hrt_runner:          # noqa: N801 — module-substitute shim
        @staticmethod
        def available() -> bool:
            return False

        @staticmethod
        def infer(*a, **k):
            raise RuntimeError("accelerator runtime unavailable in the extracted library")

log = logging.getLogger("bpu_forecast")

STREAM_ID = "bpu.forecast"
# A forecast is judged on DIRECTION, not magnitude: a coarse N-step unitary rollout drifts a few % in
# magnitude (the on-chip intermediate-ρ precision, project_bpu_kernel) but holds direction (cos≈0.998). A
# BROKEN accelerator gives cos≈0 (the pyeasy_dnn garbage). So parity = cosine ≥ COS_TOL cleanly separates a
# healthy frozen-H forecast from a drifted one; rel_err stays in the gauge as magnitude info.
COS_TOL = float(os.environ.get("BPU_FORECAST_COS", "0.95"))


def enabled() -> bool:
    return os.environ.get("UMWELT_BPU_FORECAST", "0") not in ("", "0", "false", "False")


def commutator_rollout(rho0: np.ndarray, H: np.ndarray, n_steps: int, c: float) -> np.ndarray:
    """The CPU reference: the same unitary-commutator rollout the .bin computes, in float64. ρ_{t+1} =
    ρ_t + c·(H·ρ − ρ·H). This is what the BPU result is judged against (same dynamics → meaningful parity)."""
    r = np.asarray(rho0, dtype=np.float64)
    Hd = np.asarray(H, dtype=np.float64)
    for _ in range(int(n_steps)):
        r = r + c * (Hd @ r - r @ Hd)
    return r


class BpuForecaster:
    """Spin ρ forward N steps under a frozen operator H. `bin_path` is the baked-in fixed-H rollout `.bin`
    (gen_rollout --h-file → build_rollout.sh); `H` is the same snapshot, kept for the CPU fallback + parity.
    `forecast()` runs on the BPU when available, else CPU — always returns a result. `with_parity()` runs both
    and emits the gauge record (the BPU-vs-CPU divergence — proves the silicon is faithful)."""

    def __init__(self, bin_path: str, H: np.ndarray, *, n_steps: int, c: float = 0.01,
                 dim: int | None = None, out_name: str | None = None):
        self.bin_path = bin_path
        self.H = np.asarray(H, dtype=np.float32)
        self.n_steps = int(n_steps)
        self.c = float(c)
        self.dim = int(dim or self.H.shape[-1])
        self.out_name = out_name or f"rho{self.n_steps}"

    @classmethod
    def from_build(cls, build_dir: str) -> "BpuForecaster":
        """Construct from a build_rollout_* dir: reads shapes.json (steps/dim/output) + ref.npz (the baked H)."""
        s = json.load(open(os.path.join(build_dir, "shapes.json")))
        r = dict(np.load(os.path.join(build_dir, "ref.npz")))
        bins = [f for f in os.listdir(os.path.join(build_dir, "model_output")) if f.endswith(".bin")]
        return cls(os.path.join(build_dir, "model_output", bins[0]), r["H"],
                   n_steps=s["steps"], dim=s["dim"], out_name=s["output"])

    def available(self) -> bool:
        return bool(self.bin_path) and os.path.exists(self.bin_path) and hrt_runner.available()

    def rollout_bpu(self, rho0: np.ndarray) -> np.ndarray:
        out = hrt_runner.infer(self.bin_path, {"rho0": np.asarray(rho0, np.float32)},
                               output_names=[self.out_name], dim=self.dim)
        return out[self.out_name]

    def rollout_cpu(self, rho0: np.ndarray) -> np.ndarray:
        return commutator_rollout(rho0, self.H, self.n_steps, self.c)

    def forecast(self, rho0: np.ndarray) -> np.ndarray:
        """ρ_N under the frozen operator — BPU if available (offloads the A55), else the CPU reference."""
        if self.available():
            try:
                return self.rollout_bpu(rho0)
            except Exception as e:                              # any runtime hiccup → CPU, never a wrong answer
                log.warning("BPU forecast failed (%s) — CPU fallback", type(e).__name__)
        return self.rollout_cpu(rho0)

    def with_parity(self, rho0: np.ndarray) -> tuple[np.ndarray, dict]:
        """Roll on the BPU AND the CPU; return (BPU result, gauge dict). The gauge carries the cosine + rel
        error — a drifting accelerator surfaces as a DECOHERED compute stream, like any unhealthy datastream."""
        cpu = self.rollout_cpu(rho0)
        if not self.available():
            return cpu, {"backend": "cpu", "n_steps": self.n_steps, "cos": 1.0, "rel_err": 0.0, "parity": True}
        bpu = self.rollout_bpu(rho0)
        rel = float(np.linalg.norm(bpu - cpu) / (np.linalg.norm(cpu) + 1e-30))
        cos = float(bpu.ravel() @ cpu.ravel() / ((np.linalg.norm(bpu) * np.linalg.norm(cpu)) + 1e-30))
        return bpu, {"backend": "bpu", "n_steps": self.n_steps, "cos": cos, "rel_err": rel,
                     "parity": cos >= COS_TOL}

    def gauge_record(self, gauge: dict) -> dict:
        verdict = dh.ENRICHED if gauge.get("parity") else dh.DECOHERED
        return dh.stream_record(
            STREAM_ID, verdict, gauge=gauge, source="bpu_node", owner="brain",
            action="" if gauge.get("parity") else f"BPU forecast drifted rel={gauge.get('rel_err'):.2e} from CPU",
            evidence=f"{gauge['n_steps']}-step frozen-H rollout on {gauge['backend']}")


class BpuForecastProbe:
    """The live-loop consumer of BpuForecaster — the additive BPU forecast CHANNEL (Luke's frozen-H
    architecture, realized in the running brain). Each cycle it takes ONE dense cluster's live ρ,
    spins it forward N steps under the FROZEN baked operator on the BPU, and compares to the same
    rollout on the CPU — the parity gauge proves the silicon is faithful (cos≈0.998 healthy). The
    value-exact CPU free-run stays the primary forecast; this is an extra fast/coarse channel that
    offloads the GIL-bound A55 and feeds surprise like any forecast. Side-effect-free: it reads ρ, it
    never writes the live belief.

    Inert-by-default + honest availability: constructed only when UMWELT_BPU_FORECAST is set AND a
    baked build dir is given (UMWELT_BPU_FORECAST_MODEL); `available()` further needs the hrt runtime
    + the .bin (RDK-only) and the target cluster's dim to match the baked dim. Off the RDK / without a
    bake it does nothing — the flag-off path is byte-identical to today. The operator is re-baked from
    the live pickle at the release cadence (ops/bpu/bake_from_pickle.py); never per-tick."""

    def __init__(self, forecaster: "BpuForecaster", node: str | None = None):
        self.fc = forecaster
        self.node = node or os.environ.get("UMWELT_BPU_FORECAST_NODE", "")
        self.last_record: dict | None = None

    def _pick_rho(self, field):
        """The target dense ρ: the configured node if it's a dim-matched dense cluster, else the
        largest dense cluster whose dim matches the baked operator (the folded-manifold root)."""
        clusters = getattr(field, "clusters", {})
        if self.node and self.node in clusters:
            rho = getattr(clusters[self.node], "rho", None)
            if rho is not None and rho.shape[0] == self.fc.dim:
                return self.node, rho
        best = None
        from umwelt.substrate.backend import is_dense
        for name, c in clusters.items():
            if not is_dense(c):        # only the dense backend owns a materializable rho
                continue
            rho = getattr(c, "rho", None)
            if rho is not None and rho.shape[0] == self.fc.dim:
                if best is None or rho.shape[0] > best[1].shape[0]:
                    best = (name, rho)
        return best if best else (None, None)

    def available(self, field) -> bool:
        if not self.fc.available():
            return False
        _, rho = self._pick_rho(field)
        return rho is not None

    def run(self, field) -> dict | None:
        """One frozen-H BPU rollout of the target cluster + the parity gauge. Stores + returns the
        datastream-health record (None if unavailable). Blocking (hrt subprocess) — call via a thread."""
        if not self.fc.available():
            return None
        node, rho = self._pick_rho(field)
        if rho is None:
            return None
        rho_real = np.ascontiguousarray(np.real(rho), dtype=np.float32)
        try:
            _bpu, gauge = self.fc.with_parity(rho_real)
        except Exception as e:                                  # never let the probe perturb the loop
            log.warning("BPU forecast probe failed (%s)", type(e).__name__)
            return None
        gauge["node"] = node
        rec = self.gauge_record_for(gauge)
        self.last_record = rec
        return rec

    def gauge_record_for(self, gauge: dict) -> dict:
        return self.fc.gauge_record(gauge)


def make_bpu_forecast_probe() -> "BpuForecastProbe | None":
    """Gated factory (reservoir wiring). Returns a probe only when UMWELT_BPU_FORECAST is set AND a
    readable baked build dir is given (UMWELT_BPU_FORECAST_MODEL = a build_rollout_* dir). Otherwise
    None — the live loop skips the BPU channel entirely (flag-off = byte-identical). The probe may
    still be UNAVAILABLE at runtime (no hrt runtime / no .bin / dim mismatch) → it stays inert."""
    if not enabled():
        return None
    model = os.environ.get("UMWELT_BPU_FORECAST_MODEL", "")
    if not model or not os.path.isdir(model):
        log.warning("UMWELT_BPU_FORECAST set but UMWELT_BPU_FORECAST_MODEL=%r is not a build dir "
                    "— BPU forecast channel inert", model)
        return None
    try:
        fc = BpuForecaster.from_build(model)
    except Exception as e:
        log.warning("BPU forecast model load failed (%s) — channel inert", type(e).__name__)
        return None
    return BpuForecastProbe(fc)
