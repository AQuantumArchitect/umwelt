"""OnlineRegressor — THE one online linear forecaster (dissolution M3, b9.33).

Every forecaster in the system — driver, environment, leaf, location — is the same
learner pointed at different targets: an online, L2-regularized linear map from a
feature vector to a target vector, with a rolling error EMA for skill, a raw-error
SNR proxy for the calibrator, and lr/l2/ema threaded live from the parameter fiber.
It used to be written twice (`CelestialForecast` ~150 lines / `WeatherForecast`
~240 lines, near-verbatim copies) — and the copies had DRIFTED: one carried a
feature-geometry reset guard another copy lacked, so a live feature-layout change
(sparse-feature migration, field rebuild) would crash that path mid-matmul.
The dedup delivers that guard to both.

The two survivors are thin subclasses (kept at their historical module paths so
imports and pickles are undisturbed):

    WeatherForecast  = OnlineRegressor(error="mean_abs")             sensor targets
    CelestialForecast= OnlineRegressor(error="half_norm", ball=True)  Bloch target

`project_ball` is the sphere chart's post-map (keep the prediction on/in the
Bloch ball — one atlas application, see bloch.py); `error_mode` preserves each
class's historical skill metric exactly: the scalar family tracks mean |err| per target,
the sphere family tracks the halved norm of the on-ball error.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from umwelt._util import round_or_none


class OnlineRegressor:
    """Online linear regressor W: features → targets, trained one gradient step at a
    time (ΔW = lr(err ⊗ feat − l2·W)), geometry-guarded, with rolling error EMAs."""

    def __init__(
        self,
        n_targets: int,
        target_ids: list[str] | None = None,
        lr: float = 0.02,
        l2: float = 1e-4,
        ema: float = 0.02,
        project_ball: bool = False,
        error_mode: str = "mean_abs",          # "mean_abs" | "half_norm"
    ):
        self.n_targets = int(n_targets)
        self.target_ids = list(target_ids) if target_ids else [f"t{i}" for i in range(n_targets)]
        self.W: NDArray[np.floating] | None = None    # (n_targets, feat_dim), lazy-init
        self.lr = lr
        self.l2 = l2
        self.ema = ema                                # error EMA smoothing
        self.project_ball = bool(project_ball)
        self.error_mode = str(error_mode)
        self.error_ema: float | None = None           # rolling normalized error
        self.per_target_err: NDArray[np.floating] | None = None
        self.n_updates = 0
        self._last_pred: NDArray[np.floating] | None = None
        # Raw (un-smoothed) error history — lets the calibrator estimate the
        # signal's SNR and adapt the EMA bandwidth (non-circular: it reads the
        # RAW series, not the smoothed one).
        self._raw_errors: deque[float] = deque(maxlen=64)

    def _reset_weights(self) -> None:
        """Drop the learned mapping when the feature geometry changes. The
        regressor re-learns from live data; cheap relative to a feature-layout
        change, which is rare (migration / field rebuild)."""
        self.W = None
        self.error_ema = None
        self.per_target_err = None
        self._last_pred = None
        self.n_updates = 0
        self._raw_errors.clear()

    def predict(self, features: NDArray[np.floating]) -> NDArray[np.floating] | None:
        """Forecast the target vector, or None before any training. Feature geometry
        can change under us (sparse-feature migration, a live field rebuild) — W is
        lazily shaped to features, so guard the in-process case: a stale W never
        matmuls against a mismatched feature vector."""
        if self.W is None:
            return None
        features = np.asarray(features, dtype=float)
        if self.W.shape[1] != features.shape[0]:
            self._reset_weights()
            return None
        pred = self.W @ features
        if self.project_ball:
            n = float(np.linalg.norm(pred))
            if n > 1.0:                               # keep it on/in the Bloch ball
                pred = pred / n
        self._last_pred = pred
        return pred

    def update(
        self,
        features: NDArray[np.floating],
        label: NDArray[np.floating],
        lr: float | None = None,
        l2: float | None = None,
        ema: float | None = None,
    ) -> None:
        """One online step: train features → label. lr/l2/ema, when given, come
        live from the parameter fiber (calibrated, not buried constants)."""
        if lr is not None:
            self.lr = lr
        if l2 is not None:
            self.l2 = l2
        if ema is not None:
            self.ema = ema
        features = np.asarray(features, dtype=float)
        label = np.asarray(label, dtype=float)
        if self.W is None or self.W.shape[1] != features.shape[0]:
            # (Re)initialize to the live feature width. The shape-mismatch arm
            # handles a feature-geometry change mid-process (see predict()).
            if self.W is not None and self.W.shape[1] != features.shape[0]:
                self._reset_weights()
            self.W = np.zeros((self.n_targets, features.shape[0]))
        pred = self.W @ features
        err_vec = label - pred
        # ΔW = lr(err ⊗ feat − l2·W) — gradient of ½|err|² + ½·l2·|W|²
        self.W += self.lr * (np.outer(err_vec, features) - self.l2 * self.W)
        self.n_updates += 1
        if self.error_mode == "half_norm":
            # measured against the (possibly ball-projected) post-step prediction,
            # halved to [0,1] — the on-sphere skill metric, verbatim.
            e = 0.5 * float(np.linalg.norm(label - self.predict(features)))
        else:
            ae = np.abs(label - (self.W @ features))  # post-step abs error per target
            self.per_target_err = ae if self.per_target_err is None else (
                self.ema * ae + (1.0 - self.ema) * self.per_target_err
            )
            e = float(np.mean(ae))
        self._raw_errors.append(e)
        self.error_ema = e if self.error_ema is None else (
            self.ema * e + (1.0 - self.ema) * self.error_ema
        )

    @property
    def skill(self) -> float:
        """1 − rolling normalized error, clamped to [0, 1]; 0 until it learns."""
        if self.error_ema is None:
            return 0.0
        return max(0.0, 1.0 - self.error_ema)

    @property
    def weight_norm(self) -> float:
        """‖W‖ — drives the L2 calibration (regularization ↔ weight magnitude)."""
        return float(np.linalg.norm(self.W)) if self.W is not None else 0.0

    def raw_error_snr(self) -> float | None:
        """SNR proxy of the RAW error series: var(series) / var(first-difference).

        High → slowly-drifting signal (track it: widen bandwidth / larger α).
        Low  → tick-to-tick noise (smooth it: smaller α). Non-circular — it
        reads the un-smoothed series. None until enough samples.
        """
        if len(self._raw_errors) < 16:
            return None
        x = np.asarray(self._raw_errors, dtype=float)
        var_sig = float(np.var(x))
        var_diff = float(np.var(np.diff(x)))
        if var_diff < 1e-12:
            return None
        return var_sig / var_diff

    def _base_snapshot(self) -> dict:
        return {
            "trained": self.W is not None and self.n_updates > 0,
            "n_updates": self.n_updates,
            "error_ema": round_or_none(self.error_ema, 4),
            "skill": round(self.skill, 4),
            "lr": round(self.lr, 5), "l2": round(self.l2, 7), "ema": round(self.ema, 4),
            "weight_norm": round(self.weight_norm, 4),
        }

    def snapshot(self) -> dict:
        return self._base_snapshot()
