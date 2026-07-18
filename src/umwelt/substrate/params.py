"""
Parameter Fiber — learnable parameters on the world graph.

Each node in the WorldGraph can carry a ParameterBundle — a fiber
of learnable scalar parameters alongside the density matrix (state fiber).

Two fibers over the same base space:
    Fiber 1 (existing): QubitCluster — evolving quantum state (beliefs)
    Fiber 2 (new):      ParameterBundle — learnable parameters (priors)

Parameters update via Bayesian/Kalman filter from observations.

The real geometric (Berry) phase of the system lives with the *state*
fiber, not here: BlochGeometricPhase accumulates it from the qubits' Bloch
trajectories (see reservoir). This module no longer fabricates a phase from
parameter drift.

Layers (fractal hierarchy matching the world graph depth):
    Layer 0: Topology    — bridge coupling weights
    Layer 1: Sensor      — normalization ranges, noise floors
    Layer 2: Field       — gamma, bridge_strength, collapse thresholds
    Layer 3: Cognition   — pattern windows, confidence gates
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class ScalarParam:
    """A single learnable scalar with Gaussian belief (mu, sigma).

    The minimum quantum of the parameter fiber. Updated via Kalman
    filter: observation + noise -> posterior.
    """

    name: str
    value: float
    sigma: float
    prior_mean: float | None = None
    prior_sigma: float | None = None
    lo: float | None = None       # optional hard clamp (lower)
    hi: float | None = None       # optional hard clamp (upper)
    update_count: int = 0
    frozen: bool = False          # if True, skip updates (e.g. physics constants)

    def __post_init__(self):
        if self.prior_mean is None:
            self.prior_mean = self.value
        if self.prior_sigma is None:
            self.prior_sigma = self.sigma

    @property
    def effective_sigma(self) -> float:
        """Sigma with a floor: unexplored parameters stay uncertain.

        If update_count is zero, sigma never drops below prior_sigma.
        This ensures Thompson Sampling explores parameters that haven't
        seen data yet, even if sigma was initialized small.
        """
        if self.update_count == 0 and self.prior_sigma is not None:
            return max(self.sigma, self.prior_sigma)
        return self.sigma

    def sample(self) -> float:
        """Thompson Sampling: draw from the posterior N(value, sigma^2).

        Parameters near zero with large sigma produce wild samples
        (aggressive exploration). Well-learned parameters with small
        sigma barely deviate from their mean (exploitation).

        The transition from exploration to exploitation is continuous
        and automatic — sigma shrinks with each Kalman update.
        """
        s = random.gauss(self.value, self.effective_sigma)
        if self.lo is not None:
            s = max(self.lo, s)
        if self.hi is not None:
            s = min(self.hi, s)
        return s

    def set_prior(self, prior_mean: float, prior_sigma: float):
        """LLM warm-start: set a weak prior that data can easily override.

        The prior_sigma should be large relative to expected observation
        sigma. A 5:1 ratio gives the LLM guess a half-life of ~1-2
        observations. The guess is a crutch, not authority.
        """
        self.prior_mean = prior_mean
        self.prior_sigma = prior_sigma
        # If no data yet, move value to the prior
        if self.update_count == 0:
            self.value = prior_mean
            self.sigma = prior_sigma

    def kalman_update(self, observation: float, obs_sigma: float) -> float:
        """Bayesian update: posterior = likelihood * prior.

        Returns the innovation (observation - prediction) for diagnostics
        and Berry phase accumulation.
        """
        if self.frozen or obs_sigma <= 0:
            return 0.0

        innovation = observation - self.value
        K = self.sigma ** 2 / (self.sigma ** 2 + obs_sigma ** 2)
        self.value += K * innovation
        self.sigma = math.sqrt((1 - K) * self.sigma ** 2)

        if self.lo is not None:
            self.value = max(self.lo, self.value)
        if self.hi is not None:
            self.value = min(self.hi, self.value)

        self.update_count += 1
        return innovation

    def drift_from_prior(self) -> float:
        """Distance (in prior-sigma units) from the prior mean."""
        if self.prior_sigma and self.prior_sigma > 0:
            return abs(self.value - self.prior_mean) / self.prior_sigma
        return 0.0

    def reset_to_prior(self):
        """Reset value and sigma to prior (forgetting)."""
        self.value = self.prior_mean
        self.sigma = self.prior_sigma
        self.update_count = 0

    def snapshot(self) -> dict:
        """Serializable state for logging/API — AND the persistence record the
        fractal stack round-trips. The rounded fields are the display surface;
        `value_exact`/`sigma_exact` carry full precision so a restore is
        bit-exact (a 6-decimal round-off in a restored H coefficient is enough
        to fork an otherwise deterministic replay — the 2026-07-18 lease-drill
        lesson). Loaders prefer the exact fields, falling back to the rounded
        ones on legacy checkpoints."""
        return {
            "name": self.name,
            "value": round(self.value, 6),
            "sigma": round(self.sigma, 6),
            "value_exact": float(self.value),
            "sigma_exact": float(self.sigma),
            "prior_mean": round(self.prior_mean, 6),
            "drift": round(self.drift_from_prior(), 3),
            "updates": self.update_count,
            "frozen": self.frozen,
        }


@dataclass
class BlochGeometricPhase:
    """Real geometric (Berry) phase of the qubits' Bloch trajectories.

    For a two-level system the Berry connection is exact and closed-form,
    so we accumulate the genuine geometric phase directly from the path the
    Bloch vector traces under the Lindblad dynamics — no transported
    eigenstate decomposition needed:

        dγ = -½ (1 - cosθ) dφ · |r|

    (θ, φ) are the polar/azimuthal angles of the Bloch *direction*; |r| is
    the qubit purity (|r|=1 pure, →0 maximally mixed), damping the step so a
    decohered qubit carries a fuzzy phase. The global phase is the sum of
    the per-qubit γ.

    This is honest physics, unlike the parameter-drift sum it replaces:
      • γ returns to a prior value ONLY when (θ, φ) genuinely retraces a loop;
      • one equatorial loop (Δφ = 2π) gives -π, so a spinor needs 4π of
        winding to return to +1;
      • 4π does NOT guarantee return — a path enclosing zero net solid angle
        returns γ → 0 regardless of how much φ wound.
    """

    phases: dict[str, float] = field(default_factory=dict)
    _prev: dict[str, tuple[float, float]] = field(default_factory=dict)  # key -> (theta, phi)

    def update(self, key: str, bloch) -> None:
        """Advance one qubit's geometric phase from its current Bloch vector."""
        x, y, z = float(bloch[0]), float(bloch[1]), float(bloch[2])
        r = math.sqrt(x * x + y * y + z * z)
        if r < 1e-9:
            # maximally mixed: no defined direction, no geometric phase
            return
        theta = math.acos(max(-1.0, min(1.0, z / r)))
        phi = math.atan2(y, x)
        prev = self._prev.get(key)
        self._prev[key] = (theta, phi)
        if prev is None:
            return  # need two samples to form dφ
        _, phi_prev = prev
        # shortest-arc azimuth step in (-π, π] keeps the winding honest
        dphi = math.atan2(math.sin(phi - phi_prev), math.cos(phi - phi_prev))
        dgamma = -0.5 * (1.0 - math.cos(theta)) * dphi * r
        self.phases[key] = self.phases.get(key, 0.0) + dgamma

    @property
    def total(self) -> float:
        """Global geometric phase: sum of per-qubit γ."""
        return sum(self.phases.values())

    def reset(self):
        self.phases.clear()
        self._prev.clear()


@dataclass
class ParameterBundle:
    """Parameter fiber over a WorldNode.

    A collection of named ScalarParams attached to one node in
    the world graph. This is the second fiber bundle (alongside
    QubitCluster) living on the WorldGraph base space.

    Usage:
        bundle = ParameterBundle.from_dict({
            "co2_baseline": (420.0, 20.0),
            "co2_per_occupant": (200.0, 50.0),
        })
        bundle.update("co2_baseline", observation=435.0, obs_sigma=10.0)
        current = bundle.get("co2_baseline")  # ~ 427.5
    """

    params: dict[str, ScalarParam] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        specs: dict[str, tuple],
        frozen_keys: set[str] | None = None,
    ) -> ParameterBundle:
        """Create from {name: (value, sigma)} or {name: (value, sigma, lo, hi)}.

        frozen_keys: parameter names that should not be updated.
        """
        frozen_keys = frozen_keys or set()
        params = {}
        for name, spec in specs.items():
            if len(spec) == 2:
                value, sigma = spec
                lo, hi = None, None
            elif len(spec) == 4:
                value, sigma, lo, hi = spec
            else:
                raise ValueError(
                    f"Spec for '{name}' must be (value, sigma) or (value, sigma, lo, hi)"
                )
            params[name] = ScalarParam(
                name=name,
                value=value,
                sigma=sigma,
                lo=lo,
                hi=hi,
                frozen=name in frozen_keys,
            )
        return cls(params=params)

    def get(self, name: str, default: float | None = None, explore: bool = False) -> float:
        """Get parameter value, optionally Thompson-sampled.

        explore=False: point estimate (for diagnostics, H projection)
        explore=True:  sample from posterior (for learning, exploration)

        The explore flag is the only switch between exploitation and
        exploration. Everything else — sigma decay, prior strength,
        floor enforcement — is continuous and automatic.
        """
        param = self.params.get(name)
        if param is None:
            if default is not None:
                return default
            raise KeyError(f"Parameter '{name}' not found in bundle")
        return param.sample() if explore else param.value

    def get_param(self, name: str) -> ScalarParam | None:
        """Get the full ScalarParam object."""
        return self.params.get(name)

    def update(self, name: str, observation: float, obs_sigma: float,
               channel: str | None = None) -> float:
        """Kalman-update a parameter (a partial collapse toward the observation).

        `channel` is the reward neuromodulator making the write (#310). When given, the REWARD GUARD
        checks the param's receptor profile: a channel that the param doesn't subscribe to is DROPPED
        (log-only by default; raises under UMWELT_REWARD_STRICT) — the cross-reward contamination
        boundary. Phase-1 learners don't pass `channel` (the guard is dormant in the 1:1 happy path);
        a channel that DOES identify itself only collapses its own sector. Returns the innovation.
        """
        param = self.params.get(name)
        if param is None:
            raise KeyError(f"Parameter '{name}' not found in bundle")
        if channel is not None:
            receptor = getattr(param, "receptor", None)
            if receptor is not None and not receptor.responds_to(channel):
                import logging
                from umwelt._util import env_flag
                logging.getLogger(__name__).warning(
                    "reward guard: channel '%s' tried to collapse '%s' (receptors=%s) — dropped",
                    channel, name, getattr(receptor, "channels", None))
                if env_flag("UMWELT_REWARD_STRICT"):
                    raise PermissionError(f"channel '{channel}' may not write '{name}'")
                return 0.0
            # Receptor SENSITIVITY × channel TONE scale the collapse strength (wider obs_sigma → smaller
            # Kalman α → gentler collapse). At weight=1 + release=1 (every Phase-1 channel) this is a NO-OP.
            if receptor is not None:
                from umwelt.learning.reward.registry import CHANNELS
                weight = receptor.channels.get(channel, 1.0)
                release = CHANNELS[channel].release_level if channel in CHANNELS else 1.0
                gain = max(1e-6, float(weight) * float(release))
                if gain != 1.0:
                    obs_sigma = obs_sigma / gain
        return param.kalman_update(observation, obs_sigma)

    def batch_update(self, observations: dict[str, tuple[float, float]]):
        """Update multiple parameters.

        observations: {name: (observation, obs_sigma)}
        """
        for name, (obs, obs_sigma) in observations.items():
            param = self.params.get(name)
            if param is not None:
                param.kalman_update(obs, obs_sigma)

    def snapshot(self) -> dict:
        """Full state for logging/API."""
        return {
            "params": {name: p.snapshot() for name, p in self.params.items()},
        }

    def set_prior(self, name: str, prior_mean: float, prior_sigma: float):
        """Set a weak prior on one parameter (e.g. from LLM scaffold).

        If the parameter hasn't been updated yet, the value moves to
        the prior mean. Otherwise only prior_mean/prior_sigma change
        and the next Kalman update will incorporate the new prior.
        """
        param = self.params.get(name)
        if param is not None:
            param.set_prior(prior_mean, prior_sigma)

    def xavier_init_sigma(self):
        """Scale sigma by 1/sqrt(n) so the combined effect of many
        uncertain parameters doesn't blow up or vanish.

        This is the parameter-space analogue of Xavier initialization
        in neural networks. Applied once at construction.
        """
        n = max(1, len(self.params))
        scale = 1.0 / math.sqrt(n)
        for param in self.params.values():
            if not param.frozen:
                param.sigma *= scale
                if param.prior_sigma is not None:
                    param.prior_sigma *= scale

    def merge(self, other: ParameterBundle):
        """Merge another bundle's params into this one (additive, non-destructive)."""
        for name, param in other.params.items():
            if name not in self.params:
                self.params[name] = param

    def bind_qubit(self, name: str, cluster, qubit_idx: int) -> None:
        """Replace a ScalarParam with a `QubitBackedParam` carrying the same
        bounds, prior, and update_count. The backing qubit's initial Bloch z
        is set from the current scalar value, so the swap is value-preserving
        and the rest of the codebase sees no change through the bundle API.

        This is the entry point for the one-species pilot — see qubit_param.py
        and plans/noble-sleeping-yao.md.
        """
        from umwelt.substrate.qubit_param import QubitBackedParam

        old = self.params.get(name)
        if old is None:
            raise KeyError(
                f"Cannot bind_qubit: parameter '{name}' not in bundle"
            )
        if old.lo is None or old.hi is None:
            raise ValueError(
                f"Cannot bind_qubit: '{name}' has no lo/hi bounds (qubit needs a finite range)"
            )
        new = QubitBackedParam(
            name=name,
            cluster=cluster,
            qubit_idx=qubit_idx,
            lo=old.lo,
            hi=old.hi,
            prior_mean=old.prior_mean,
            prior_sigma=old.prior_sigma,
            update_count=old.update_count,
            frozen=old.frozen,
        )
        # Seed the qubit at the scalar's current value (value-preserving swap).
        new.value = old.value
        self.params[name] = new

    def __contains__(self, name: str) -> bool:
        return name in self.params

    def __len__(self) -> int:
        return len(self.params)
