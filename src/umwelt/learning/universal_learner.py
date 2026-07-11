"""The universal learning law — pole-robust eligibility-trace collapse.

ONE law for every interior coordinate, replacing the zoo of hand-wired learners (see
experiments/wiring_manifest.py, project_universal_learning_law). The principle: a coordinate
collapses toward whatever reduces its own local SURPRISE. The mechanism that makes it work
even for dissipation rates — whose error is observed only AFTER the state has saturated, where
every local gradient vanishes (the dwell-dissolution finding) — is an ELIGIBILITY TRACE:

  • each tick, accrue the coordinate's LIVE INFLUENCE on the state into a decaying trace
        e ← λ·e + influence                       (captured mid-trajectory, while it's alive)
  • when the delayed surprise r arrives, collapse the coordinate toward reducing it, weighted
    by the trace:
        target = value − lr · r · ê               (ê = the geometric-trace's normalized sum)
        param.kalman_update(target, obs_sigma)    (observe_qubit; PURITY settles → the stop)

What makes one coordinate behave differently from another is WHICH influence and WHICH surprise
it is wired to — geometry, not a bespoke rule. The same object learns a dwell, a coupling, a
trust weight. Validated against the hand-tuned dwell learner in experiments/dwell_dissolution.py
(pole solved at gap-48 where the analytic gradient stalls; beats the hand rule on tracking).

The learner's own hyperparameters (lam/lr/obs_sigma) are, by the same thesis, NOT constants: every
holder live-reads them off the gauge and passes them per-call (the dwell from presence_dwell_*, the
signal/driver trust from hebbian_lr/hebbian_obs_sigma, the spawn from spawn_obs_sigma; calibration
already hosted its obs_sigmas + meta-tunes them). The defaults below are only the seed a fresh learner
falls back to pre-attach / in tests — like web_min_activity's prior before it was dissolved. So even
the one law's own knobs are diff-witnessed coordinates: the totality theorem now covers the constants
OF the learning, not just what it learns. (Giving them an active tuning signal — the meta-tower reach —
is the next dig; today they sit at their priors, in the gauge.)
"""
from __future__ import annotations


class UniversalLearner:
    """Eligibility-trace reinforcement over QubitBackedParams. Stateless about WHAT each param
    means — the caller supplies the influence (each tick) and the surprise (on the event)."""

    def __init__(self, lam: float = 0.9, lr: float = 0.4, obs_sigma: float = 0.15):
        self.lam = float(lam)
        self.lr = float(lr)
        self.obs_sigma = float(obs_sigma)
        self._e: dict[int, float] = {}     # eligibility trace, keyed by param identity

    def accrue(self, param, influence: float) -> None:
        """Bump `param`'s eligibility by its live influence on the state this tick.
        Influence is the magnitude of how much this param moved the predicted state — for a
        dissipation rate, |∂(predicted z)/∂rate| = (z + 1) (relaxation toward the absent pole)."""
        k = id(param)
        self._e[k] = self.lam * self._e.get(k, 0.0) + float(influence)

    def eligibility(self, param) -> float:
        return self._e.get(id(param), 0.0)

    def reinforce(self, param, surprise: float, *, lr: float | None = None,
                  obs_sigma: float | None = None) -> float:
        """A delayed surprise (signed local residual) arrives → collapse `param` toward reducing
        it, weighted by the eligibility trace (the credit captured while it was influential).
        Returns the target value (param.value if there is no eligibility yet). Consumes the trace."""
        k = id(param)
        e = self._e.get(k, 0.0)
        if e == 0.0 or surprise == 0.0:
            return param.value
        e_hat = e * (1.0 - self.lam)                          # normalize the geometric trace sum
        lr = self.lr if lr is None else lr
        obs_sigma = self.obs_sigma if obs_sigma is None else obs_sigma
        target = param.value - lr * float(surprise) * e_hat
        lo = getattr(param, "lo", None)
        hi = getattr(param, "hi", None)
        if lo is not None:
            target = max(lo, target)
        if hi is not None:
            target = min(hi, target)
        param.kalman_update(target, obs_sigma)               # collapse; the qubit's purity is the stop
        self._e[k] = 0.0                                       # consume the credit
        return target

    def gradient_step(self, param, influence: float, surprise: float, *,
                      lr: float | None = None, obs_sigma: float | None = None) -> float:
        """Immediate (no-trace) gradient collapse — the eligibility law when credit is IMMEDIATE
        (influence and surprise arrive the same tick): target = value − lr·surprise·influence, then
        collapse toward it. Equivalent to accrue+reinforce at lam=0. For hebbian-style learners whose
        signal isn't delayed (signal/driver trust weights: influence = the signal's input, surprise
        = −residual). Returns the post-collapse value; no-op on zero influence/surprise or None param."""
        if param is None or influence == 0.0 or surprise == 0.0:
            return None if param is None else param.value
        lr = self.lr if lr is None else lr
        target = param.value - lr * float(surprise) * float(influence)
        return self.observe(param, target, obs_sigma)

    # ── supervised-target mode — the OTHER realization of 'collapse toward reducing surprise' ──
    # When a learner HAS a measured/label-derived target (a sensor range, a tracked γ, a source's
    # accuracy = 1−|z−label|), the coordinate collapses toward it DIRECTLY: no eligibility trace,
    # because the credit is immediate and the right value is KNOWN. (reinforce above is the DELAYED,
    # unsupervised mode — only a surprise direction is available, as for the dwell.) Both are the
    # same observe_qubit partial collapse with purity as the settling stop; this is the mode the
    # already-gauge-native supervised learners (calibration, trust) route through, so there is ONE
    # learner object.

    def observe(self, param, target: float, obs_sigma: float | None = None) -> float:
        """Supervised collapse of a fiber PARAM (ScalarParam / QubitBackedParam) toward a known
        target via param.kalman_update — purity is the settling stop. Parity-exact with the
        ParameterBundle.update facade in the no-channel (Phase-1) path that calibration uses.
        None/absent param → no-op (returns None); target is clamped to the param's [lo, hi].
        Returns the post-collapse value."""
        if param is None:
            return None
        target = float(target)
        lo, hi = getattr(param, "lo", None), getattr(param, "hi", None)
        if lo is not None:
            target = max(lo, target)
        if hi is not None:
            target = min(hi, target)
        param.kalman_update(target, self.obs_sigma if obs_sigma is None else float(obs_sigma))
        return param.value

    def collapse(self, cluster, qubit_idx: int, target_bloch, alpha: float) -> None:
        """The most general supervised collapse: pull a RAW qubit toward a KNOWN Bloch target at a
        fixed alpha (= cluster.observe_qubit). Used where the target is a point in a MULTI-AXIS
        manifold — e.g. a preference qubit's (brightness=z, color_temp=x) — not a single scalar axis.
        observe_raw is the scalar specialization (target = the surface point at value_to_bloch_z)."""
        cluster.observe_qubit(qubit_idx, target_bloch, alpha)

    def observe_raw(self, cluster, qubit_idx: int, target: float, lo: float, hi: float,
                    alpha: float) -> None:
        """Supervised collapse of a RAW reliability qubit (not wrapped as a param) at a FIXED alpha
        — the classical-EMA-parity path for trust reliability. r ← (1−α)r + α·target, exactly the
        prior _surface_point(value_to_bloch_z(target)) observe_qubit call."""
        from umwelt.substrate.qubit_param import value_to_bloch_z
        tz = value_to_bloch_z(target, lo, hi)
        x = max(0.0, 1.0 - tz * tz) ** 0.5      # surface point: |r|→1 as evidence concentrates (purity)
        self.collapse(cluster, qubit_idx, (x, 0.0, tz), alpha)

    def reset(self, param=None) -> None:
        if param is None:
            self._e.clear()
        else:
            self._e.pop(id(param), None)
