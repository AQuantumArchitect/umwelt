"""
Qubit-backed parameter — pilot of the one-species fractal vision.

A `QubitBackedParam` exposes the same surface as `ScalarParam` (`value`, `sigma`,
`lo`, `hi`, `get/sample`, `kalman_update`, `set_prior`, `snapshot`) but its state
lives on a single qubit inside a `QubitCluster`. The scalar value is just a
rescaled `⟨σ_z⟩`; the posterior width is just `(1 - |r|)·(hi-lo)/2` (purity is
the same physics as Gaussian sigma, one level up). The "update" call is just
`cluster.observe_qubit(idx, target_bloch, α)` — `α` is the Kalman gain, `target`
is the observation rescaled to the Bloch z axis.

No new physics primitive is introduced — `observe_qubit` already does
analog-native partial collapse (`ρ_new = (1-α)ρ + α·ρ_target`). The conversion
is purely value↔z rescaling and obs_sigma → α arithmetic.

Confidence that SETTLES (kalman_update): the observation target is the PURE state on the Bloch sphere
at the value's polar angle — `(√(1−z²), 0, z)` — NOT the z-axis interior point. So repeated consistent
evidence drives purity |r|→1 (σ→0, the wriggle narrows as the brain grows confident), and a surprising
observation mixes it back down (exploration re-opens). The value `⟨σ_z⟩` is unaffected; the accumulated
confidence rides the x-axis. (The earlier pilot used the interior target `(0,0,z)`, which pinned a
mid-range value's purity at |z| — confidence tracked value-POSITION, not evidence. Fixed.)

Residual nuance: an EXTREME-range value (z≈±1) is geometrically near-pure already, so a param seeded at
its boundary starts more confident than a mid-range one. Minor; the common (mid-range) case settles.

See `plans/noble-sleeping-yao.md` for the full pilot context. If this proves out
on `celestial_alpha`, the same `QubitBackedParam` carries the rest of the fiber.
"""
from __future__ import annotations
from umwelt._util import clamp01

import math
import random
from typing import TYPE_CHECKING

# The scalar chart lives in the atlas (bloch.py, M1); re-exported here so the many
# existing call sites (`from .qubit_param import value_to_bloch_z`) stay unchanged.
from umwelt.substrate.bloch import value_to_bloch_z, bloch_z_to_value, bloch_radius  # noqa: F401

if TYPE_CHECKING:
    from umwelt.substrate.cluster import QubitCluster


class QubitBackedParam:
    """Single learnable scalar whose state is a single qubit's Bloch z.

    API matches `ScalarParam` so it slots into `ParameterBundle` without any
    callsite changes. Backed by `(cluster, qubit_idx)`; all reads and writes
    go through the cluster's existing observe/Bloch API — no new substrate.
    """

    def __init__(
        self,
        name: str,
        cluster: "QubitCluster",
        qubit_idx: int,
        lo: float,
        hi: float,
        prior_mean: float | None = None,
        prior_sigma: float | None = None,
        update_count: int = 0,
        frozen: bool = False,
    ):
        if lo >= hi:
            raise ValueError(f"QubitBackedParam '{name}' needs lo < hi (got {lo}, {hi})")
        self.name = name
        self.cluster = cluster
        self.qubit_idx = qubit_idx
        self.lo = lo
        self.hi = hi
        self.frozen = frozen
        # Prior: where the scalar was when bound, so drift_from_prior is meaningful.
        self.prior_mean = prior_mean if prior_mean is not None else self.value
        self.prior_sigma = (
            prior_sigma if prior_sigma is not None else (hi - lo) / 4.0
        )
        # Mirror ScalarParam's bookkeeping so existing snapshot/test code is happy.
        self.update_count = update_count

    # ── value / sigma derived from the qubit's Bloch state ──────────

    @property
    def _bloch(self) -> tuple[float, float, float]:
        b = self.cluster.qubit_bloch(self.qubit_idx)
        return float(b[0]), float(b[1]), float(b[2])

    @property
    def value(self) -> float:
        """Rescaled ⟨σ_z⟩ of the backing qubit."""
        _, _, z = self._bloch
        return bloch_z_to_value(z, self.lo, self.hi)

    @value.setter
    def value(self, v: float) -> None:
        """Hard-set the backing qubit's Bloch z (used by tests / set_prior)."""
        target_z = value_to_bloch_z(v, self.lo, self.hi)
        # alpha=1 with a pole target snaps purity high; interior target keeps it
        # honest. For a hard set we use the interior target — the value lands
        # exactly where asked, purity reflects how confident we are in the value.
        self.cluster.observe_qubit(self.qubit_idx, (0.0, 0.0, target_z), alpha=1.0)

    @property
    def purity_r(self) -> float:
        """Bloch radius |r| ∈ [0, 1]. 1=pure (zero uncertainty), 0=max mixed."""
        return bloch_radius(*self._bloch)

    @property
    def sigma(self) -> float:
        """Posterior width derived from purity: σ = (1-|r|)·(hi-lo)/2.

        Maximally mixed qubit → σ = (hi-lo)/2 (full Bloch ball half-width in
        value units). Pure qubit → σ = 0. Replaces ScalarParam's Kalman σ.
        """
        return (1.0 - self.purity_r) * (self.hi - self.lo) / 2.0

    @property
    def effective_sigma(self) -> float:
        """Mirror ScalarParam — for the qubit-backed param, sigma already
        floors itself naturally at the prior-sigma equivalent when fresh
        (a maximally mixed qubit has the maximum possible σ)."""
        if self.update_count == 0 and self.prior_sigma is not None:
            return max(self.sigma, self.prior_sigma)
        return self.sigma

    # ── Thompson-sample analog ──────────────────────────────────────

    def sample(self) -> float:
        """Thompson-sample equivalent: project onto z-basis and rescale.

        For a single qubit the projective measurement gives +1 with probability
        `(1 + z) / 2` and -1 with probability `(1 - z) / 2` — those are the
        Bloch-ball poles. Rescaling gives hi or lo as the only two outcomes,
        which is too coarse for parameter exploration. Instead we draw a
        Gaussian around the value with σ = effective_sigma — same semantics as
        ScalarParam, just with the qubit's purity-derived width.
        """
        s = random.gauss(self.value, self.effective_sigma)
        return max(self.lo, min(self.hi, s))

    # ── Bayesian update via observe_qubit ───────────────────────────

    def set_prior(self, prior_mean: float, prior_sigma: float) -> None:
        """LLM warm-start — set prior and (if untouched) hard-set the value."""
        self.prior_mean = prior_mean
        self.prior_sigma = prior_sigma
        if self.update_count == 0:
            self.value = prior_mean  # uses the property setter → observe_qubit

    def kalman_update(self, observation: float, obs_sigma: float) -> float:
        """Bayesian update via partial-collapse toward an observation.

        Mapping: σ_qubit (from purity) and obs_sigma combine to a Kalman gain
        α = σ_q² / (σ_q² + obs_σ²). observe_qubit does ρ_new = (1-α)ρ + α·ρ_target,
        whose ⟨σ_z⟩ is exactly the Kalman value update.

        CONFIDENCE that SETTLES (the fix): the value is the Bloch-z coordinate, but the target is the
        PURE state on the sphere at that value's polar angle — (√(1−z²), 0, z) — NOT the z-axis interior
        point (0,0,z). With the interior target the radius could never exceed |z|, so a mid-range value
        (z≈0) was geometrically pinned near the mixed origin (wide wriggle forever, regardless of
        evidence). Aiming at the surface lets repeated CONSISTENT evidence drive |r|→1 (purity→1,
        σ→0, the wriggle settles), while a SURPRISING observation (a different polar angle) mixes the
        state back down (|r| drops → exploration re-opens). The value (⟨σ_z⟩) is unaffected — the extra
        confidence rides the x-axis, which the value getter never reads. Purity now tracks accumulated
        evidence, not where the value sits in its range.

        Returns the innovation (obs - value_before), matching ScalarParam.kalman_update's contract.
        """
        if self.frozen or obs_sigma <= 0:
            return 0.0

        sigma_q = self.sigma
        # Even a fully-certain qubit still accepts some information from a tight observation (match
        # ScalarParam: K shrinks but never freezes). Floor σ_q at a tiny fraction of the range.
        sigma_floor = (self.hi - self.lo) * 1e-6
        sigma_q = max(sigma_q, sigma_floor)

        var_q = sigma_q * sigma_q
        var_o = obs_sigma * obs_sigma
        alpha = var_q / (var_q + var_o)
        alpha = clamp01(alpha)

        innovation = observation - self.value
        target_z = value_to_bloch_z(observation, self.lo, self.hi)
        x_conf = (max(0.0, 1.0 - target_z * target_z)) ** 0.5     # the surface point: |r|→1 as evidence accrues
        self.cluster.observe_qubit(self.qubit_idx, (x_conf, 0.0, target_z), alpha=alpha)

        self.update_count += 1
        return innovation

    def drift_from_prior(self) -> float:
        if self.prior_sigma and self.prior_sigma > 0:
            return abs(self.value - self.prior_mean) / self.prior_sigma
        return 0.0

    def reset_to_prior(self) -> None:
        self.value = self.prior_mean
        self.update_count = 0

    def snapshot(self) -> dict:
        """Match ScalarParam.snapshot, plus a small bloch block for debugging."""
        x, y, z = self._bloch
        return {
            "name": self.name,
            "value": round(self.value, 6),
            "sigma": round(self.sigma, 6),
            "prior_mean": round(self.prior_mean, 6),
            "drift": round(self.drift_from_prior(), 3),
            "updates": self.update_count,
            "frozen": self.frozen,
            "bloch": {
                "x": round(x, 6), "y": round(y, 6), "z": round(z, 6),
                "r": round(self.purity_r, 6),
            },
            "backing": {
                "kind": "qubit",
                "cluster": getattr(self.cluster, "zone_name", "?"),
                "qubit_idx": self.qubit_idx,
            },
        }
