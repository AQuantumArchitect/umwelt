"""Tendril — the ONE learned-readout actuator pattern over the comprehension manifold.

We're heading toward hundreds of "sensors" and dozens of actuators. Each actuator used to
carry its own bespoke loop (DimmerActuator, ACActuator) with its own param soup and its own
block in `reservoir.ingest`. That doesn't scale and it's brittle. A Tendril is the single
mechanism every actuator latches on through:

    command = decode( β · features(field) )         # read the manifold, decode to a device cmd
              ↓ gated (master-enable / rate-limit / deadband / device-clamp / act-when-out)
              → Action(reason="<name>_auto")          # OutputSurface routes APPROVED → dispatcher

The field IS the reservoir ([[feedback_no_parallel_quantum]]); only the readout learns. β (the
calibration) lives in the tendril pkl, NOT the param bundle — that's where the per-actuator
param soup goes to die ([[project_ac_comfort]] tendril rearchitecture; plan noble-sleeping-yao).
The brain stays UNIT-FREE: unit-mapping + the device clamp live HERE at the actuator edge, never
inside the field ([[feedback_brain_unit_free]]).

This module is Phase A: the abstraction + the single reservoir loop. ACActuator and DimmerActuator
become concrete `Tendril`s conforming to this contract; the reservoir holds `self.tendrils` and
calls them in ONE step loop + ONE override loop (replacing the per-actuator blocks). The learned-β
decode (Phase C/D — retire the soup, seed-then-refine) slots into a tendril's `command_now()`
without the reservoir loop changing. The origin's continuous output migrated LAST, behind a <0.05-RMSE parity gate.

The contract every tendril honours:
  * ``name``                     — stable id (log lines, snapshot keys, the ``<name>_auto`` reason).
  * ``step(now_ts) -> Action``   — one tick; an Action when all gates pass, else None (shadow).
  * ``apply_override() -> None``  — detect the operator overriding our last dispatch and pull the
                                    manifold toward the observed reality (+ surprise). Self-contained
                                    (reads its own sensor key off ``reservoir.sensor_bridge``), so the
                                    reservoir just loops. No-op by default.
  * ``snapshot() -> dict``        — status for /api/console. ``{}`` by default.
"""
from __future__ import annotations

import math

from typing import TYPE_CHECKING

from umwelt._util import clamp01

if TYPE_CHECKING:  # annotation-only: egress imports this module (Action lives there)
    from umwelt.membranes.egress import Action


def sticky_collapse(level: float, purity: float, prev: bool, scale: float = 0.5) -> bool:
    """A continuous belief → a sticky binary actuation, the north-star way: flip at the 0.5
    measurement midpoint with a half-width = (1−purity)·scale. A CONFIDENT belief (purity→1) flips
    sharply; an UNCERTAIN one widens the band so the actuation stays sticky (no flap on noise) —
    the hysteresis IS the belief's own uncertainty geometry, not a hand-set band. `prev` holds inside
    the band. Shared by every flappy-belief→binary actuator (attention's transcribe, the light commit
    tendril). The canonical home; attention re-exports it as `transcribe_hysteresis`."""
    margin = 0.5 * (1.0 - float(purity)) * float(scale)
    if level >= 0.5 + margin:
        return True
    if level <= 0.5 - margin:
        return False
    return bool(prev)


class CommittedBelief:
    """A tendril's BASE-CASE coupling: a slow committed actuator belief (one qubit) PUMPED by a fast
    evidence signal and RELAXING toward a rest pole — the rise(coupling) + fall(decay) geometry that
    DECOUPLES a flappy continuous belief from a sticky actuation. The perception stays free to
    oscillate; the actuator integrates and commits. Read via `sticky_collapse` (purity-derived
    hysteresis) for a binary device, or as a continuous `level` for a dimmable one.

    This IS what we mean by a "tendril": a learned qubit→qubit coupling carrying its own two-timescale
    geometry. An occupancy→level coupling is the base case; richer couplings (gear
    logic with spheres) generalize from here. It's a reservoir-side 1-qubit register (zero
    feature-geometry weight, like agency/attention), so it never bloats the world field.

    pump(evidence, coupling, decay): relax toward `rest_z` (the dark/off pole) at `decay`, then pump
    UP toward |on⟩ ∝ evidence at `coupling`. Evidence-only-up so the decay genuinely owns the fall —
    a brief evidence dip only nudges the belief; sustained absence relaxes it ~1/decay ticks (the
    linger/dwell). coupling = the rise rate, decay = the linger rate; both are the tendril's learnable,
    gauge-tracked geometry."""

    def __init__(self, name: str, *, rest_z: float = -1.0):
        from umwelt.substrate.product_cluster import ProductQubitCluster
        self.rest_z = float(rest_z)
        self.cluster = ProductQubitCluster(f"_commit_{name}", ["commit"])
        self.cluster.observe_qubit(0, (0.0, 0.0, self.rest_z), 1.0)   # seed at the rest pole

    def pump(self, evidence: float, coupling: float, decay: float) -> None:
        self.cluster.observe_qubit(0, (0.0, 0.0, self.rest_z), clamp01(float(decay)))
        e = clamp01(float(evidence))
        if e > 1e-6:
            self.cluster.observe_qubit(0, (0.0, 0.0, 1.0), clamp01(float(coupling) * e))

    @property
    def level(self) -> float:
        return (float(self.cluster.qubit_bloch(0)[2]) + 1.0) / 2.0

    @property
    def confidence(self) -> float:
        bx, by, bz = (float(v) for v in self.cluster.qubit_bloch(0))
        return (bx * bx + by * by + bz * bz) ** 0.5

    def commit(self, prev: bool, hysteresis_scale: float = 0.5) -> bool:
        """The committed binary readout: sticky_collapse on this belief's level + confidence."""
        return sticky_collapse(self.level, self.confidence, prev, hysteresis_scale)

    def snapshot(self) -> dict:
        return {"level": round(self.level, 4), "confidence": round(self.confidence, 4)}


class GearedBelief(CommittedBelief):
    """A tendril with MULTIPLE input spheres geared into one committed belief — "gear logic with
    spheres". Several evidence sources (occupancy, arousal, phase-of-cycle, …) drive one commit qubit
    through a LEARNED gear: a per-input weight, each living on its own qubit. The effective evidence
    is the weighted geometric mean ∏ eᵢ^wᵢ — a learned soft-AND. A weight→0 drops that input (e⁰=1,
    the product identity); equal unit weights = the plain conjunction. So the scalar `CommittedBelief`
    is exactly the 1-input, weight-1 special case, and the origin's hand-multiplied `occupancy·arousal`
    is the 2-input, weight-1 case — both fall out for free (parity).

    The "gear ratio" is the exponent wᵢ ∈ [0, weight_scale], read off the i-th weight qubit's Bloch z
    ((z+1)/2·scale), so w=1 sits at the origin (z=0, purity 0 = the unlearned default gear) and the
    weight SHARPENS (Bloch radius grows) as it learns. The geometric mean is the natural form: it
    preserves a hard gate (an input at 0 with w>0 zeroes the product, like sleep killing the light),
    matches attention's geometric-mean conjunction, and lets the brain learn WHICH spheres matter via
    `learn_weight` (the one law, reinforcement/observe). This is step 1 of the richer tendril; step 2
    (a learned rotation/phase between spheres — a true gear with a phase, not just a gain) builds on it.
    """

    def __init__(self, name: str, inputs, *, rest_z: float = -1.0, weight_scale: float = 2.0):
        super().__init__(name, rest_z=rest_z)
        from umwelt.substrate.product_cluster import ProductQubitCluster
        self.inputs = list(inputs)
        self.weight_scale = float(weight_scale)
        # the GEAR: one weight qubit per input. add_role seeds maximally-mixed (Bloch origin → z=0),
        # so every weight starts at scale·(0+1)/2 = 1 when scale=2 (equal-weight parity) and unlearned
        # (purity 0). The exponents learn off this floor.
        self.gear = ProductQubitCluster(f"_gear_{name}", self.inputs)

    def weight(self, inp: str) -> float:
        """The learned gear ratio (exponent) for one input ∈ [0, weight_scale]."""
        z = float(self.gear.role_bloch(inp)[2])
        return self.weight_scale * (z + 1.0) / 2.0

    def _combine(self, evidences: dict) -> float:
        """The geared effective evidence: weighted geometric mean ∏ eᵢ^wᵢ over the input spheres.
        Missing inputs default to 1.0 (the product identity → no effect), so a partial stamp is safe."""
        eff = 1.0
        for inp in self.inputs:
            e = clamp01(float(evidences.get(inp, 1.0)))
            eff *= e ** self.weight(inp)        # 0^0=1 (drop), 0^(w>0)=0 (hard gate), e^0=1 (drop)
        return clamp01(eff)

    def pump_inputs(self, evidences: dict, coupling: float, decay: float) -> None:
        """Gear the input spheres into one effective evidence, then pump the commit belief with it
        (rise=coupling / fall=decay) — the multi-input generalization of `pump`."""
        self.pump(self._combine(evidences), coupling, decay)

    def learn_weight(self, inp: str, target_weight: float, alpha: float) -> None:
        """Tune one gear ratio toward `target_weight` (observe the weight qubit at `alpha` — the one
        law). The override/surprise signal drives this: an input that consistently predicts the right
        commit is weighted UP; a misleading one is weighted toward 0 (dropped)."""
        frac = clamp01(float(target_weight) / self.weight_scale)
        self.gear.observe_qubit(self.gear.role_index[inp], (0.0, 0.0, 2.0 * frac - 1.0), float(alpha))

    def observe_outcome(self, evidences: dict, desired_level: float, lr: float) -> None:
        """Learn the gear from a revealed OUTCOME (the operator's override / surprise): nudge each
        input's weight so the geared effective evidence moves toward `desired_level`. Credit
        assignment is the log-gradient of the geometric mean: ∂eff/∂wᵢ = eff·ln(eᵢ). The input that
        most drags eff away from what the operator wanted gets the biggest correction; an uninformative
        input (eᵢ≈1 → ln≈0) gets none, so a noisy/misleading sphere is weighted toward 0 (dropped)
        while a predictive one is kept. Each weight then observes toward its corrected target — the
        one law. This is how the brain learns WHICH spheres matter, not a hand-set gate."""
        eff = self._combine(evidences)
        err = clamp01(float(desired_level)) - eff          # +: want more evidence; -: want less
        for inp in self.inputs:
            e = clamp01(float(evidences.get(inp, 1.0)))
            if not (0.0 < e < 1.0):
                continue                                    # ln(0)→-∞ (hard gate) / ln(1)=0: no gradient
            grad = eff * math.log(e)                        # ≤ 0 (eff drops as wᵢ rises)
            new_w = clamp01((self.weight(inp) + lr * err * grad) / self.weight_scale) * self.weight_scale
            self.learn_weight(inp, new_w, abs(lr * err) )   # confidence ∝ how wrong we were

    def gear_snapshot(self) -> dict:
        """The learned gear ratios + their confidence (Bloch radius) per input, for /console."""
        out = {}
        for inp in self.inputs:
            bx, by, bz = (float(v) for v in self.gear.role_bloch(inp))
            out[inp] = {"weight": round(self.weight(inp), 4),
                        "confidence": round((bx * bx + by * by + bz * bz) ** 0.5, 4)}
        return out

    def snapshot(self) -> dict:
        snap = super().snapshot()
        snap["gear"] = self.gear_snapshot()
        return snap


class Tendril:
    """Base contract for a learned actuator tendril over the comprehension manifold.

    Subclasses read their relevant qubits from the field, decode to a device command, gate, and
    emit an Action. The reservoir treats them uniformly through this interface — one step loop, one
    override loop. New actuators are a new subclass, not a new bypass block + a new pair of hooks.

    Tendril is the UNIVERSAL actuator interface — not just continuous β-readouts. Two axes let
    one interface cover every actuator without flattening the distinctions that matter:

      * decode shape — CONTINUOUS (β·features → a setpoint/level) vs
        DISCRETE (a thresholded/hysteretic read → on/off: lights, plugs, locks). decode_binary()
        is the discrete primitive; a binary fixture is still a tendril, it just decodes to a bool.
      * route mode — `route_mode` ∈ {"auto","recommend"}. AUTO emits reason "<name>_auto" →
        OutputSurface routes it APPROVED (auto-executed). RECOMMEND emits "<name>_rec" → SUGGESTED →
        the confirmation lifecycle (in-transit hold → confirm/decline → observe_feedback). The
        observe-vs-actuate gauge, per actuator: continuous comforts auto; discrete/high-stakes
        actions (and ALWAYS locks) recommend. This is the Fibonacci-hand coordinate with one home.
    """

    #: stable identifier — overridden by each concrete tendril (e.g. "level", "setpoint").
    name: str = "tendril"

    #: routing for this tendril's Action — "auto" (APPROVED, auto-executed) or "recommend"
    #: (SUGGESTED → human confirmation lifecycle). Locks should pin this to "recommend".
    route_mode: str = "auto"

    #: this actuator's NODE NAME in the world graph (its identity). The learning router resolves
    #: what learned roles a recent dispatch confounds from the graph (confounding.actuator_confounding)
    #: via this name — so the tendril hand-codes NO cluster mapping, only its own identity. None →
    #: not a graph-represented actuator (contributes no confounding signal).
    graph_node: str | None = None

    #: seconds over which a dispatch's confounding ECHO decays to 0 (slow HVAC → long, fast light →
    #: short). Used by dispatch_echo(); overridden per tendril.
    echo_window: float = 300.0

    def step(self, now_ts: float | None = None) -> Action | None:
        """One actuator tick. Return an Action when every gate passes, else None (shadow)."""
        raise NotImplementedError

    def apply_override(self) -> None:
        """Detect the operator overriding this tendril's most-recent dispatch and pull the
        manifold toward the observed reality (+ emit surprise). Self-contained: a tendril reads
        its own readback sensor off the reservoir's sensor_bridge. No-op by default so a tendril
        without an override channel costs the reservoir nothing."""
        return None

    def observed(self) -> float | None:
        """The device's CURRENT observed primary value (cool setpoint °F, brightness 0-1, …), read
        from the sensor readback — for the act-when-out gate. None when unknown → the tendril acts
        (we can't tell it's already satisfied). Override per tendril; base returns None."""
        return None

    def last_dispatch_ts(self) -> float | None:
        """Wall-clock ts of this tendril's most-recent dispatch, or None if it hasn't dispatched.
        Each tendril tracks its own dispatch record (`_last`, `_last_dispatch`, …); this exposes it
        uniformly so dispatch_echo() can age it. Base returns None (no dispatch tracked)."""
        return None

    def dispatch_echo(self, now_ts: float) -> float | None:
        """The confounding ECHO of this tendril's last dispatch ∈ (0,1], decaying linearly over
        echo_window, or None if it hasn't dispatched recently. The learning router maps this to the
        cluster + roles the tendril confounds (via graph_node → confounding.actuator_confounding) and
        discounts that cluster's world-model learning as self-caused (downstream-from-us). Uniform —
        EVERY actuating tendril confounds the learned roles its device state projects onto; there is
        no per-device special-case (every output is a first-class graph node like every other device).
        See learning_router.py + confounding.py + the heritage-braid observe-vs-actuate exploration."""
        ts = self.last_dispatch_ts()
        if ts is None:
            return None
        age = now_ts - ts
        if age < 0 or age >= self.echo_window or self.echo_window <= 0:
            return None
        return 1.0 - age / self.echo_window

    def route_reason(self) -> str:
        """The Action.reason that encodes this tendril's routing. "auto" → "<name>_auto" (the suffix
        OutputSurface reads to APPROVE + auto-execute); "recommend" → "<name>_rec" (no _auto suffix →
        SUGGESTED → the confirmation lifecycle). One rule, no per-actuator routing list."""
        return f"{self.name}_auto" if self.route_mode == "auto" else f"{self.name}_rec"

    @staticmethod
    def decode_binary(
        signal: float, prev_on: bool, *, on_at: float, off_at: float,
    ) -> bool:
        """The DISCRETE decode primitive: a continuous manifold signal → on/off with HYSTERESIS.
        A binary fixture is a tendril whose readout collapses to a bool — but a bare threshold
        chatters on the boundary, so we use a band: turn ON only above `on_at`, turn OFF only below
        `off_at` (on_at ≥ off_at), and HOLD the previous state in the dead-band between them. This is
        the on/off half of what the bespoke person_lights loop hand-codes; folding it onto the base
        lets every binary fixture (lights, plugs) share it. Continuous tendrils ignore this."""
        if signal >= on_at:
            return True
        if signal <= off_at:
            return False
        return prev_on        # dead-band → hold (hysteresis, no chatter)

    @staticmethod
    def out_of_preference(desired: float, observed: float | None, deadband: float) -> bool:
        """Act-when-out: True when reality is OUT of the preference deadband (or observed unknown).
        The dispatch PRINCIPLE — a tendril fires only when the world has drifted out of the
        operator's preference field, not to re-assert a state reality already holds. A pure
        SUPPRESSION gate: it can only skip a dispatch the tendril would otherwise make, never add
        one, so it can't fight the operator (the rate-limit + override-learning handle that)."""
        if observed is None:
            return True
        return abs(float(desired) - float(observed)) > float(deadband)

    def snapshot(self) -> dict:
        """Status for /api/console (gate states + last dispatch). Empty by default."""
        return {}
