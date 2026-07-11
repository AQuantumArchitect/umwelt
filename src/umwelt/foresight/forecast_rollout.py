"""FORECAST ROLLOUT — the forecast dissolved into the field's own forward dynamics (no learned W).

The Total Manifold campaign's Stage 2: a forecast is a gradient-trained weight matrix living off the
gauge, with no exact-parity qubit move. So we take the maximal cut — the forecast IS the field evolved
forward dt. There is no W to be off-gauge because there is no W. The forecast skill lives entirely in
gauge coordinates (the fractal H-tower + Lindblad rates); a forecast becomes a pure read of the
manifold's intrinsic forward flow.

Two pieces, both gated behind UMWELT_FORECAST_DISSOLVE (default OFF — the trained regressors stay
until this is validated on real forecast tapes on the RDK):

  • FieldRolloutForecaster — snapshot the field, dream it forward H steps under its OWN dynamics
    (feeding only the deterministic future ephemeris), read the predicted leaf values, RESTORE the
    field exactly. A forecast must never perturb the live belief, so this is side-effect-free.

  • DissolvedDriverForecast — a periodic-driver forecaster with its W pulled. A trained driver
    forecaster mapped features → the driver's Bloch position at +horizon, but that label IS a
    deterministic cycle: the W was only ever approximating a known function. The dissolved form
    serves the deterministic label directly (exact), trains nothing, and keeps no W — a drop-in for
    the trained driver-forecast interface.

The lab prototype + synthetic validation live in the lineage harness (experiments/forecast_dissolve.py).
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------------------------------
# the rollout forecaster — the field dreaming forward under its own learned dynamics
# ---------------------------------------------------------------------------------------------------
class FieldRolloutForecaster:
    """Forecast = the field evolved forward dt under its OWN dynamics. No learned W.

    Snapshot/restore makes it a pure observation: dreaming forward never perturbs the live belief.
    Handles both dense QubitClusters (joint .rho) and the ProductQubitClusters of the param fiber
    (per-qubit matrices)."""

    def __init__(self, field, dt_seconds: float = 1.0, sky_anchor=None):
        self.field = field
        self.dt = float(dt_seconds)
        self.sky_anchor = sky_anchor   # sky_anchor(field, future_dt): inject deterministic ephemeris

    def _snapshot(self) -> dict:
        snap = {}
        for name, c in self.field.clusters.items():
            if getattr(c, "is_product", False):
                snap[name] = ("product", {r: m.copy() for r, m in c.state_matrices().items()})
            elif getattr(c, "is_cumulant", False):
                # cumulant clusters (wide nodes — a merged entity carries several coupled
                # roles) evolve e1/e2 only; the Hamiltonian (_h/_zz) is fixed. Save the cumulants.
                snap[name] = ("cumulant", (c.e1.copy(), c.e2.copy()))
            else:
                snap[name] = ("dense", c.rho.copy())
        return snap

    def _restore(self, snap: dict) -> None:
        for name, (kind, state) in snap.items():
            c = self.field.clusters[name]
            if kind == "product":
                c.load_matrices({r: m.copy() for r, m in state.items()})
            elif kind == "cumulant":
                c.e1, c.e2 = state[0].copy(), state[1].copy()
            else:
                c.rho = state.copy()

    def role_value(self, cluster_name: str, role: str) -> float | None:
        c = self.field.clusters.get(cluster_name)
        if c is None or role not in getattr(c, "role_index", {}):
            return None
        return float(c.role_bloch(role)[2])

    def forecast(self, horizon_steps: int, leaves, now=None) -> dict:
        """Dream the field forward `horizon_steps`; read each (cluster, role) in `leaves`. The field
        is restored afterwards, so this is side-effect-free on the live belief.

        The rollout is the canonical OFFLOAD target — latency-tolerant, repeated, never touches the live
        belief — so under UMWELT_FORECAST_BACKEND (default numpy) it runs the batched evolution through a
        number-system backend (the BPU-native expansion, eventually the .bin). The live field's own ticks
        keep NumpyBackend; only this snapshot→roll→restore loop swaps. Restored in `finally` either way."""
        import os
        from datetime import timedelta
        from umwelt.substrate.batched_evolve import make_backend
        snap = self._snapshot()
        be = os.environ.get("UMWELT_FORECAST_BACKEND", "numpy")
        prev_backend = getattr(self.field, "_evolve_backend", None)
        if be != "numpy":
            self.field._evolve_backend = make_backend(be)
        try:
            for k in range(horizon_steps):
                self.field.step({})
                if self.sky_anchor is not None and now is not None:
                    self.sky_anchor(self.field, now + timedelta(seconds=self.dt * (k + 1)))
            return {(cn, r): self.role_value(cn, r) for (cn, r) in leaves}
        finally:
            self.field._evolve_backend = prev_backend
            self._restore(snap)

    def forecast_freerun(self, horizon_steps: int, leaves, now=None, backend_name=None) -> dict:
        """The DECOUPLED ENGINE forecast — the single whole-field forecast brain.

        Where `forecast()` loops `field.step({})` H times (H dispatches/cluster, the per-step backend
        swap measured 0.86× — the numpy↔kernel conversion eats the fusion), this gathers each same-dim
        dense group's (ρ, H, rates) ONCE and FREE-RUNS all H steps inside ONE fused kernel per dim-group
        (the proven jax.lax.fori_loop: 1.6–4.1× + a single GIL acquire for the whole run — and exactly the
        shape the BPU churns). No per-scope partition: one rollout reads every leaf. Side-effect-free
        (snapshot→roll→restore in `finally`).

        backend_name defaults to UMWELT_FORECAST_BACKEND: 'numpy' = the reference; 'jax' = the XLA-fused
        free-run; 'expansion'/'bpu' = the BPU-native number system (and, when the .bin lands, the BPU
        itself). Holding H/rates fixed across the run is the decoupling thesis — the field evolves under
        its own field between data samples (interpolation = the physics). Forecast leaves are dense
        beliefs; the few product/cumulant clusters that host a leaf (e.g. a merged person) step per-tick.
        """
        import os
        from umwelt.substrate.batched_evolve import make_backend, free_run_groups
        be = backend_name or os.environ.get("UMWELT_FORECAST_BACKEND", "numpy")
        backend = make_backend(be)
        snap = self._snapshot()
        try:
            leaf_nodes = {cn for (cn, _r) in leaves}
            by_dim: dict[int, list] = {}
            other_leaf_clusters = []
            for name, c in self.field.clusters.items():
                if getattr(c, "is_product", False) or getattr(c, "is_cumulant", False):
                    if name in leaf_nodes:
                        other_leaf_clusters.append(c)        # a leaf on a non-dense cluster — step it
                    continue
                rho = getattr(c, "rho", None)
                if rho is None:
                    continue
                by_dim.setdefault(rho.shape[0], []).append(c)
            # the whole dense field forward in one fused kernel per dim-group (the engine)
            free_run_groups(by_dim, {}, horizon_steps, backend=backend)
            # non-dense leaf hosts: the FOLDED MANIFOLD's root cumulant rolls forward in ONE fused
            # free_run (numpy ref / jax = the XLA-fused BPU on-ramp); other product fibers step per-tick.
            cum_backend = "jax" if be == "jax" else "numpy"
            for c in other_leaf_clusters:
                if getattr(c, "is_cumulant", False) and hasattr(c, "free_run"):
                    c.free_run(horizon_steps, None, backend=cum_backend)
                else:
                    for _ in range(horizon_steps):
                        c.step(None, 1.0)
            return {(cn, r): self.role_value(cn, r) for (cn, r) in leaves}
        finally:
            self._restore(snap)

    def forecast_sensitivity(self, horizon_steps: int, leaves, *, n_probes: int = 3,
                             eps: float = 1e-9, conf_scale: float = 0.02, seed: int = 0) -> dict:
        """The forecast PLUS its own confidence, MEASURED from the dynamics — not assumed.

        A forecast near a saturation boundary (a belief pinned against 'certain' with no fresh evidence,
        run through the nonlinear trace-renormalization) is a coin balanced on edge: a machine-noise nudge
        to the initial state sends the rollout one way or the other. That sensitive dependence IS the
        forecast's intrinsic uncertainty. We measure it: roll once unperturbed, then `n_probes` times with
        an `eps` perturbation to the live ρ; each leaf's mean divergence = its sensitivity. Stable leaves
        (a deterministic driver) barely move → high confidence; tipping-point leaves (a saturating belief)
        swing → low confidence. Returns {leaf: {value, sensitivity, confidence}}, confidence =
        exp(-sensitivity/conf_scale) ∈ (0,1] — exactly the down-weight the trust web should apply.

        Costs (n_probes+1) rollouts, so it's an opt-in confidence probe, not every-tick. Uses whatever
        UMWELT_FORECAST_BACKEND is set (numpy = the true dynamics; the BPU adds no extra sensitivity since
        the drift is bifurcation, not arithmetic — see ops/bpu/README expansion theory)."""
        base = self.forecast(horizon_steps, leaves)
        rng = np.random.default_rng(seed)
        divs = {k: [] for k in leaves}
        for _ in range(max(1, n_probes)):
            snap = self._snapshot()
            for c in self.field.clusters.values():
                if getattr(c, "is_product", False):
                    continue
                try:
                    rho = c.rho
                except Exception:
                    continue
                e = (rng.standard_normal(rho.shape) + 1j * rng.standard_normal(rho.shape)) * eps
                rho = rho + e
                rho = 0.5 * (rho + rho.conj().T)
                tr = np.trace(rho)
                c.rho = rho / tr if abs(tr) > 1e-15 else rho
            try:
                pert = self.forecast(horizon_steps, leaves)
            finally:
                self._restore(snap)
            for k in leaves:
                if base.get(k) is not None and pert.get(k) is not None:
                    divs[k].append(abs(base[k] - pert[k]))
        out = {}
        for k in leaves:
            s = float(np.mean(divs[k])) if divs[k] else 0.0
            out[k] = {"value": base.get(k), "sensitivity": round(s, 6),
                      "confidence": round(float(np.exp(-s / max(conf_scale, 1e-9))), 4)}
        return out


# ---------------------------------------------------------------------------------------------------
# the periodic-driver dissolve — the W was approximating a deterministic function
# ---------------------------------------------------------------------------------------------------
class DissolvedDriverForecast:
    """A trained driver-forecaster with the W pulled. Its training label (the driver's Bloch at
    +horizon) is an exact deterministic cycle, so the prediction IS that label — no regression, no
    weights, perfect skill. Drop-in for the trained driver-forecast interface used by the engine +
    calibration (update/predict/skill/error_ema/W).
    """

    def __init__(self):
        self.W = None                  # pulled — there is no weight matrix
        self.error_ema = 0.0           # the deterministic cycle is exact → zero error
        self._last_pred: NDArray[np.floating] | None = None
        self.n_updates = 0

    def update(self, features, label, lr=None, l2=None, ema=None) -> None:
        """No W to train. The label is the deterministic +horizon cycle value — serve it directly."""
        self._last_pred = np.asarray(label, dtype=float)
        self.n_updates += 1

    def predict(self, features) -> NDArray[np.floating] | None:
        """The last driver label (the exact future driver position), or None before the first tick."""
        return self._last_pred

    @property
    def skill(self) -> float:
        return 1.0 if self.n_updates > 0 else 0.0   # exact: it serves the deterministic truth

    @property
    def weight_norm(self) -> float:
        return 0.0                                   # no W

    def snapshot(self) -> dict:
        """Match the trained driver-forecast snapshot() contract so a console renders either backend.
        There is no regression here (no lr/l2/ema, no W), so those slots serve the dissolved
        constants; `dissolved` flags the W-pulled deterministic mode in the console."""
        return {
            "trained": self.n_updates > 0,           # "trained" == serving the exact cycle
            "n_updates": self.n_updates,
            "error_ema": round(float(self.error_ema), 4),   # 0.0 — the ephemeris is exact
            "skill": round(self.skill, 4),           # 1.0 once it has a label to serve
            "lr": 0.0,
            "l2": 0.0,
            "ema": 0.0,
            "weight_norm": round(self.weight_norm, 4),      # 0.0 — no W
            "prediction": [round(float(v), 4) for v in self._last_pred]
            if self._last_pred is not None else None,
            "dissolved": True,
        }


def freerun_enabled() -> bool:
    """The single free-run forecast brain — gather each dim-group's dynamics ONCE and roll H steps in one
    fused kernel (the decoupled engine + the BPU on-ramp), instead of looping the full field tick H times.
    Opt-in (default OFF = the legacy per-tick rollout, unchanged). Pairs with UMWELT_FORECAST_BACKEND
    (numpy / jax-fused / expansion = BPU number system). See FieldRolloutForecaster.forecast_freerun."""
    from umwelt._util import env_flag
    return env_flag("UMWELT_FORECAST_FREERUN")


def _roll(forecaster, steps: int, leaves, now=None) -> dict:
    """Run one whole-field rollout — the fused free-run engine when enabled, else the legacy per-tick path."""
    if freerun_enabled():
        return forecaster.forecast_freerun(steps, leaves, now=now)
    return forecaster.forecast(steps, leaves, now=now)


def _purity_of(cluster, role: str) -> float:
    """Belief purity for a role — the dense joint purity, or |Bloch| for cumulant/product clusters."""
    try:
        return float(cluster.purity)
    except (AttributeError, RuntimeError):
        try:
            b = cluster.role_bloch(role)
            return float((float(b[0]) ** 2 + float(b[1]) ** 2 + float(b[2]) ** 2) ** 0.5)
        except Exception:
            return 1.0


class DissolvedForecastSurface:
    """ForecastSurface with the ~80 per-leaf Ws PULLED: every leaf's +horizon prediction is the field
    DREAMED FORWARD under its own learned dynamics (FieldRolloutForecaster), read off the evolved qubit
    — one rollout per horizon reads ALL leaves at once. No W is trained or stored; the forecast skill
    lives entirely in gauge coordinates (the fractal H-tower + Lindblad rates). Drop-in for the
    ForecastSurface interface (step/predictions/consume_targets/pre_train/snapshot). Gated by
    UMWELT_FORECAST_DISSOLVE; effectiveness vs the trained surface is RDK/real-tape work."""

    def __init__(self, leaves=None, horizons_min=None, lr=0.03, l2=0.005, dt_seconds: float = 1.0):
        from umwelt.foresight.forecast_surface import DEFAULT_HORIZONS_MIN, DEFAULT_FORECAST_LEAVES
        self.leaves = tuple(leaves) if leaves is not None else DEFAULT_FORECAST_LEAVES
        self.horizons = tuple(float(h) for h in (horizons_min or DEFAULT_HORIZONS_MIN))
        self.dt = float(dt_seconds)
        self._field = None
        self._now = None
        self._purity: dict[tuple[str, str], float] = {}

    def step(self, now, field) -> None:
        """No W to train — just hold the field + now for the rollout, and read each leaf's purity (the
        output-side confidence)."""
        self._field, self._now = field, now
        for node, role in self.leaves:
            c = field.clusters.get(node)
            if c is not None and role in getattr(c, "role_index", {}):
                self._purity[(node, role)] = _purity_of(c, role)

    def _rollout(self, horizon_min: float) -> dict:
        if self._field is None:
            return {}
        steps = max(1, int(round(horizon_min * 60.0 / self.dt)))
        return _roll(FieldRolloutForecaster(self._field, dt_seconds=self.dt),
                     steps, list(self.leaves), now=self._now)

    def _row(self, node, role, z, h):
        import math as _m
        conf = max(0.0, min(1.0, self._purity.get((node, role), 1.0)))
        pf = None
        if self._now is not None:
            from datetime import timedelta
            pf = (self._now + timedelta(minutes=h)).isoformat()
        return {"z_pred": round(float(z), 5), "confidence": round(conf, 5),
                "skill": 1.0, "horizon_min": round(h), "prediction_for": pf}

    def predictions(self) -> dict:
        out = {}
        for h in self.horizons:
            preds = self._rollout(h)
            for node, role in self.leaves:
                z = preds.get((node, role))
                if z is not None:
                    out[(node, role, h)] = self._row(node, role, z, h)
        return out

    def consume_targets(self) -> dict:
        if not self.horizons:
            return {}
        h0 = min(self.horizons)
        preds = self._rollout(h0)
        out = {}
        for node, role in self.leaves:
            z = preds.get((node, role))
            if z is not None:
                r = self._row(node, role, z, h0)
                out[(node, role)] = {"z_pred": r["z_pred"], "confidence": r["confidence"],
                                     "horizon_min": r["horizon_min"]}
        return out

    def pre_train(self, *a, **k) -> int:
        return 0                                       # no W to warm-start

    def snapshot(self) -> dict:
        return {"leaves": len(self.leaves), "horizons_min": [round(h) for h in self.horizons],
                "dissolved": True,
                "predictions": {f"{n}.{r}@{int(h)}m": p
                                for (n, r, h), p in self.predictions().items()}}


def dissolve_enabled() -> bool:
    from umwelt._util import env_flag
    return env_flag("UMWELT_FORECAST_DISSOLVE")


def make_forecast_surface(leaves=None, horizons_min=None, lr: float = 0.03, l2: float = 0.005):
    """Gated factory: the dissolved (rollout, no-W) forecast surface when UMWELT_FORECAST_DISSOLVE is
    set, else the trained ForecastSurface (default)."""
    if dissolve_enabled():
        return DissolvedForecastSurface(leaves=leaves, horizons_min=horizons_min, lr=lr, l2=l2)
    from umwelt.foresight.forecast_surface import ForecastSurface, DEFAULT_HORIZONS_MIN
    return ForecastSurface(leaves=leaves, horizons_min=horizons_min or DEFAULT_HORIZONS_MIN, lr=lr, l2=l2)
