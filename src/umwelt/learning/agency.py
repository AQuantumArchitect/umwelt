"""AgencyQubit — the act↔listen axis as ONE Bloch qubit (axiomatic, not a pseudo-scalar).

The scalar `actuation_silenced` (Phase 1) is a clamped number: it can hold "how much agency right
now" and nothing else. This makes that axis a real qubit — |act⟩ at one pole, |listen⟩ at the other,
z the continuous balance — so the de-confounding stops being a bolted-on rule and becomes geometry:
the world-learning weight is just the |listen⟩ projection (you cannot cleanly observe a system you
are driving — act and listen are antipodes of the same complementarity).

It lives in the META-PARAMETER stack, not the world field: DELIBERATELY driven (the silence button,
later the confidence coupling + φ-cadence) and NOT entangled with the noisy world clusters — or the
field could rotate the system back toward "act" before the operator releases silence. Its relaxation
is WALL-CLOCK and slow (a week-ish): a silence HOLDS through the day and HEALS over weeks, decaying
not back to full auto but to RECOMMEND — the system returns offering, and re-earns acting from
confidence (Phase 3). That weekly time-constant is what places it mid-stack.

Phase 1's scalar silence remains the hard FLOOR (the dispatcher's guarantee); this is the smooth,
axiomatic layer on top — the substrate the three coherent reads (intention/output/learning) consume.
See [[project_tendril_unification]], qubit_param.py (the value↔z idiom this reuses).
"""
from __future__ import annotations

import math
import time

# act/listen live on z ∈ [-1, +1]; |act⟩ projection = (z+1)/2.
LISTEN_Z = -1.0
ACT_Z = +1.0


class AgencyQubit:
    """One qubit on the act↔listen axis. z=+1 act, z=-1 listen; r = purity/confidence (Bloch radius);
    phase = the agency history (winds as the system cycles — the conjugate the scalar never had)."""

    def __init__(self, *, z: float = ACT_Z, recommend_z: float = 0.0,
                 tau_days: float = 10.0, now: float | None = None):
        self.z = _clip(z, -1.0, 1.0)        # +1 act … -1 listen
        self.r = 1.0                        # confidence / purity (1 = certain, 0.5 = relaxed-mixed)
        self.phase = 0.0                    # azimuth — agency Berry phase (winds only on a real loop)
        self.recommend_z = recommend_z      # decay TARGET: 'recommend', not full act/listen
        self.tau_days = tau_days            # wall-clock relaxation constant (~a week → mid-stack)
        self._last = now if now is not None else time.time()

    # ── readouts (what the three reads consume) ─────────────────────
    def actuate(self) -> float:
        """|act⟩ projection ∈ [0,1] — the continuous actuate level. The dispatcher silences below a
        listen threshold; the learning router uses this as d (confound weight); the forward model
        scales its self-effect by it (Phase 3)."""
        return (self.z + 1.0) / 2.0

    def listen(self) -> float:
        """|listen⟩ ∈ [0,1] — the clean-data weight. world-learning ∝ this (de-confounding as geometry)."""
        return 1.0 - self.actuate()

    # ── stance as a MEASUREMENT of the qubit, not hand-set cutoffs ──────────────
    # The act/recommend/listen stance is which of the three ANCHORS the belief is nearest in z —
    # a projection onto the basis, not a 0.7/0.3 magic line. The boundaries are the midpoints
    # between anchors, so they MOVE with the (learnable) recommend_z: the stance is graph geometry,
    # not scaffolding. (Old 0.7/0.3 were just the midpoints when recommend_z=0; now derived.)
    def _anchors(self) -> dict:
        return {"listen": LISTEN_Z, "recommend": self.recommend_z, "act": ACT_Z}

    @property
    def regime(self) -> str:
        return min(self._anchors().items(), key=lambda kv: abs(self.z - kv[1]))[0]

    @property
    def is_acting(self) -> bool:
        """A MEASUREMENT: the brain is in an acting stance (not nearest the listen pole) — the gate
        the dispatch path reads, replacing the hand-set `actuate() >= 0.3`. Boundary = the
        listen↔recommend anchor midpoint, so it tracks the learnable recommend_z, not a magic 0.3."""
        return self.regime != "listen"

    @property
    def is_listening(self) -> bool:
        return self.regime == "listen"

    # ── dynamics (deliberate drives only) ───────────────────────────
    def drive(self, target_z: float, alpha: float) -> None:
        """Partial collapse toward a pole (the observe_qubit idiom): move z by gain α, and a
        consistent drive SHARPENS confidence (r→1). This is how the silence button pulls to |listen⟩."""
        alpha = _clip(alpha, 0.0, 1.0)
        self.z = _clip(self.z + alpha * (_clip(target_z, -1.0, 1.0) - self.z), -1.0, 1.0)
        self.r = _clip(self.r + alpha * (1.0 - self.r), 0.0, 1.0)

    def prefer(self, target_z: float, alpha: float) -> None:
        """The USER expresses a listen↔act wish — EXTERNAL stimulus that collapses the agency belief
        toward `target_z` (−1 = listen, +1 = act, 0 = recommend). Same observe_qubit partial collapse
        as `drive` (it IS drive — a named edge for the operator's preference, the way a declare or an
        override is stimulus). The brain holds the stance in-graph; the user only nudges it. The
        silence button is the special case prefer(LISTEN_Z, …)."""
        self.drive(target_z, alpha)

    def relax(self, dt_days: float, target_z: float | None = None) -> None:
        """Wall-clock decay toward `target_z` (default 'recommend') + slow decoherence (confidence
        fades back to mixed). Exponential with tau_days, so a silence holds through a day and heals
        over weeks."""
        if dt_days <= 0.0 or self.tau_days <= 0.0:
            return
        tgt = self.recommend_z if target_z is None else _clip(target_z, -1.0, 1.0)
        k = 1.0 - math.exp(-dt_days / self.tau_days)
        self.z = self.z + k * (tgt - self.z)
        self.r = self.r + k * (0.5 - self.r)

    def tick(self, now: float, *, silenced: bool = False, confidence: float = 0.0,
             auto_act: bool = False, silence_alpha: float = 0.5,
             prefer_z: float | None = None, prefer_alpha: float = 0.0,
             knee: float = 0.0) -> None:
        """One agency step. A live USER PREFERENCE (prefer_alpha > 0) wins — collapse toward prefer_z,
        re-asserted each tick so it holds; this subsumes the silence button (silence = a listen
        preference). With no active preference, → relax over the wall-clock toward a CONFIDENCE-earned
        target: `auto_act` off (default) heals to RECOMMEND (the safe floor of the Fibonacci hand);
        `auto_act` on folds confidence∈[0,1] up toward ACT (trust earns agency, the same gentle weekly
        rate, never a slam)."""
        dt_days = max(0.0, (now - self._last)) / 86400.0
        self._last = now
        if prefer_alpha > 0.0 and prefer_z is not None:   # the operator's listen↔act wish (stimulus)
            self.prefer(prefer_z, prefer_alpha)
            return
        if silenced:                                       # legacy binary silence = a listen preference
            self.prefer(LISTEN_Z, silence_alpha)
            return
        target = self.recommend_z
        if auto_act:
            c = _clip(confidence, 0.0, 1.0)
            # The soft KNEE (b9.41): below the knee, competence buys NO climb toward act —
            # a 1%-learned brain must not drift toward autonomy just because weeks pass.
            # Above it, the remaining range rescales smoothly to [0,1] (a continuous ramp,
            # not a cliff). knee=0 is the exact old law (parity); the knee itself is a
            # learnable fiber param (agency_competence_knee), not a hand cutoff.
            if knee > 0.0:
                k = _clip(knee, 0.0, 0.9)
                c = max(0.0, (c - k) / (1.0 - k))
            target = self.recommend_z + c * (ACT_Z - self.recommend_z)
        self.relax(dt_days, target_z=target)

    # ── persistence (re-derives from the scalar silence if absent) ──
    def state(self) -> dict:
        return {"z": self.z, "r": self.r, "phase": self.phase, "last": self._last}

    def load(self, s: dict) -> None:
        self.z = _clip(float(s.get("z", self.z)), -1.0, 1.0)
        self.r = _clip(float(s.get("r", self.r)), 0.0, 1.0)
        self.phase = float(s.get("phase", 0.0))
        self._last = float(s.get("last", self._last))

    def snapshot(self) -> dict:
        return {"actuate": round(self.actuate(), 3), "listen": round(self.listen(), 3),
                "regime": self.regime, "z": round(self.z, 3), "confidence": round(self.r, 3)}


# the general range clamp lives in core/util (M6); local name kept for the call sites
from umwelt._util import clamp as _clip  # noqa: E402
