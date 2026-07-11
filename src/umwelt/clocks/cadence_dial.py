"""The autonomic cadence — one shared control surface for the dual-brain's two clocks.

The flagship runs two brains over a shared CPU (flops/wall-clock) budget:
  • FOREBRAIN (live) samples the world at a WALL-CLOCK cadence and drives the world.
  • HINDBRAIN (learning) replays the event backlog → pickle; it LAGS and must CATCH UP.

This module is the breathing controller that lets them share the substrate:

  FOREBRAIN cadence (`CadenceDial`): a pure r=1 qubit whose eased angle IS the wall-clock interval.
    fastness = clip(w_demand·D − w_stress·S − w_lag·(L·(1−D)), 0, 1);  interval = SLOW·(FAST/SLOW)^fastness
    D = decode-demand (adaptive_clock.transition_intensity: surprise + Berry velocity + event-imminence)
        pulls FAST; S = substrate-stress (rdk cpu_load/cpu_temp) pulls SLOW; L = hindbrain lag.
    The crux is the `L·(1−D)` gate: lag throttles the forebrain ONLY during lulls — a far-behind
    hindbrain never costs us fidelity on an interesting moment (high D → (1−D)→0 → lag loses authority).
    Fast-attack / slow-release θ easing = anti-thrash inertia.

  HINDBRAIN geometry (`hindbrain_dt_factor`): during a real-world lull (low D) and when behind (high L),
    replay COARSENS the φ-ladder (high dt_factor) → fewer learning updates / simulated-sec → faster
    drain. CRITICAL: gated on the REPLAYED segment's own intensity, not live D — else a quiet evening
    coarsens while replaying this afternoon's burst, blurring exactly the dense history we captured.

Stability is structural: the hindbrain is niced + CPU-capped (slack only), so the two controllers never
contend one resource; the lag loop is negative (more lag → slower forebrain → less inflow → lag drains);
the two run on φ-separated timescales (forebrain per-tick, hindbrain per-replay-batch). The coupling
weights live on the root ParameterBundle (cadence_* keys) so the meta-loop tunes them against
skill-per-compute. See plan noble-sleeping-yao, [[project_context_gauge]], [[project_sensor_dials]].

    python -m umwelt.clocks.cadence_dial      # self-test (no engine needed)
"""
from __future__ import annotations

import math

from umwelt.substrate.cluster import QubitCluster
from umwelt.clocks.phi_clock import PHI

# Wall-clock cadence band (seconds). FAST aims well under the human 1–3s transit timescale (Nyquist
# headroom); SLOW is the calm-coast cadence. The dial slides geometrically between them. These are the
# SEED values — the live dial reads them from the bundle (keys cadence_fast_s/cadence_slow_s), learnable.
CADENCE_FAST_S = 0.5
CADENCE_SLOW_S = 20.0

# Every tunable is a learnable param on the root bundle (key `cadence_<name>`), read live via `_g`.
# These are the fallback defaults (pre-attach / sandbox) — the param_bundles.py seeds mirror them.
DEFAULTS: dict[str, float] = {
    "cadence_w_demand": 1.0,    # decode-demand → faster forebrain
    "cadence_w_stress": 0.8,    # CPU/heat → slower forebrain (protect the substrate)
    "cadence_w_lag":    0.6,    # hindbrain lag → slower forebrain, BUT only during lulls (the (1−D) gate)
    "cadence_w_dorm":   0.7,    # real-world dormancy → coarser hindbrain replay (catch up)
    "cadence_w_lagH":   0.7,    # hindbrain lag → coarser replay
    "cadence_k_max":    6.0,    # max φ-rungs the hindbrain replay may climb (φ^6 ≈ 18× compression)
    "cadence_l_ref":    3600.0, # lag horizon (s): backlog this deep ⇒ L=1 ("badly behind"), ~1h
    "cadence_fast_s":   CADENCE_FAST_S,   # wall-clock FAST band edge
    "cadence_slow_s":   CADENCE_SLOW_S,   # wall-clock SLOW band edge
    "cadence_attack":   0.8,    # fast-attack easing (toward fast — catch transitions)
    "cadence_release":  0.12,   # slow-release easing (toward slow — anti-thrash)
    "cadence_throughput_mult":   2.0,     # hindbrain: max extra batches when urgent
    "cadence_busy_ref":          120.0,   # hindbrain: events/window ⇒ busy (dormancy=0)
    "cadence_dormancy_window_s": 120.0,   # hindbrain: recent-window for the dormancy estimate
}


CADENCE_META_KEYS = ("cadence_w_demand", "cadence_w_stress", "cadence_w_lag",
                     "cadence_fast_s", "cadence_slow_s")   # the knobs meta-PBT learns (skill/compute)


def stress_from_field(field, node: str = "_body") -> float:
    """Substrate-stress S ∈ [0,1] from the compute-body cluster's cpu_load + cpu_temp beliefs
    (Bloch z→[0,1]). Whichever is more stressed gates: temp integrates sustained load, load is
    the fast signal. `node` names the domain's hardware node; absent → 0 (no stress signal)."""
    body = getattr(field, "clusters", {}).get(node) if field is not None else None
    if body is None:
        return 0.0
    ri = getattr(body, "role_index", {})
    def b(role):
        return (float(body.role_bloch(role)[2]) + 1.0) / 2.0 if role in ri else 0.0
    return max(b("cpu_load"), b("cpu_temp"))


def lag_to_L(lag_seconds: float, bundle=None) -> float:
    """Normalize a raw backlog lag (seconds, from event_replay.hindbrain_lag_seconds) to L ∈ [0,1]
    against the learnable horizon cadence_l_ref. The L both brains feed into the control law."""
    l_ref = (float(bundle.get("cadence_l_ref"))
             if (bundle is not None and "cadence_l_ref" in getattr(bundle, "params", {}))
             else DEFAULTS["cadence_l_ref"])
    return _clip01(float(lag_seconds) / max(1.0, l_ref))


from umwelt._util import clamp01 as _clip01  # [0,1] clamp — one home (#313)


class CadenceDial:
    """The FOREBRAIN's wall-clock cadence as a pure r=1 dial qubit (the SensorDial "glass bead").

    drive(demand, stress, lag) eases an internal angle toward a target FASTNESS and re-prepares the
    qubit as a pure surface state (|r|≈1). Fast-attack toward fast (catch a transition instantly),
    slow-release back toward slow (anti-thrash). Reads its coupling weights live from an optional
    ParameterBundle (so the meta-loop tunes them), falling back to DEFAULTS.
    """

    def __init__(self, bundle=None, dt: float = 0.1):
        self.c = QubitCluster("cadence", ["_dial_cadence"], gamma=0.0, dt=dt,
                              role_modes={"_dial_cadence": "unitary"})
        self.idx = self.c.role_index["_dial_cadence"]
        self.bundle = bundle
        self.theta = math.pi                 # polar angle; π → z=-1 (slow end) at rest

    def _g(self, key: str) -> float:
        if self.bundle is not None and key in getattr(self.bundle, "params", {}):
            return float(self.bundle.get(key))
        return float(DEFAULTS[key])

    def drive(self, demand: float, stress: float, lag: float = 0.0) -> float:
        """Update the dial from demand D, substrate-stress S, hindbrain-lag L; return the interval (s)."""
        D, S, L = _clip01(demand), _clip01(stress), _clip01(lag)
        # lag only asserts during lulls — never sacrifice fidelity on an interesting moment.
        fastness = _clip01(self._g("cadence_w_demand") * D
                           - self._g("cadence_w_stress") * S
                           - self._g("cadence_w_lag") * (L * (1.0 - D)))
        theta_target = math.pi * (1.0 - fastness)        # fastness=1 → θ=0 (z=+1, fast)
        # fast-attack toward fast / slow-release toward slow (both live-read, learnable)
        e = self._g("cadence_attack") if theta_target < self.theta else self._g("cadence_release")
        self.theta += e * (theta_target - self.theta)
        # faithful pure-surface representation (|r|≈1): z = cos θ
        self.c.observe_qubit(self.idx, (math.sin(self.theta), 0.0, math.cos(self.theta)), 1.0)
        return self.interval

    @property
    def fastness(self) -> float:
        return 1.0 - self.theta / math.pi                # the eased angle IS the rate

    @property
    def interval(self) -> float:
        f = _clip01(self.fastness)
        fast, slow = self._g("cadence_fast_s"), self._g("cadence_slow_s")
        return float(slow * (fast / slow) ** f)


def hindbrain_dt_factor(dormancy: float, lag: float, replayed_intensity: float = 0.0,
                        bundle=None) -> float:
    """The HINDBRAIN's replay geometry: how far up the φ-ladder to compress this batch.

    dormancy = 1−D of the LIVE world (lull → free to compress); lag = how far behind; replayed_intensity
    = the event density of the batch being replayed NOW. catchup = clip(w_dorm·dormancy + w_lagH·lag) sets
    the φ-rungs, but it is CAPPED by (1 − replayed_intensity): a dense recorded burst must NOT be coarsened
    (we captured it densely for a reason). Returns a dt_factor on an exact φ-rung so `fib_strides_at`
    doesn't chatter. dt_factor=1 (no compression) when there's nothing to catch up or the segment is busy.
    """
    g = (lambda k: float(bundle.get(k)) if (bundle is not None and k in getattr(bundle, "params", {}))
         else DEFAULTS[k])
    catchup = _clip01(g("cadence_w_dorm") * _clip01(dormancy) + g("cadence_w_lagH") * _clip01(lag))
    catchup *= (1.0 - _clip01(replayed_intensity))       # a busy replayed segment vetoes coarsening
    k_rungs = round(g("cadence_k_max") * catchup)
    return float(PHI ** k_rungs)


def _selftest() -> None:
    print("╔══ CADENCE DIAL self-test")
    d = CadenceDial()
    for label, D, S, L in [
        ("calm",                 0.02, 0.05, 0.0),
        ("occupant moving",      0.85, 0.05, 0.0),
        ("moving + CPU spike",   0.85, 0.90, 0.0),
        ("LULL + hindbrain far behind", 0.03, 0.05, 1.0),
        ("BUSY + hindbrain far behind", 0.85, 0.05, 1.0),   # (1−D) gate: lag must NOT slow this
    ]:
        for _ in range(40):
            iv = d.drive(D, S, L)
        print(f"   {label:>30}  D={D:.2f} S={S:.2f} L={L:.1f} → interval {iv:6.2f}s  fastness {d.fastness:.2f}")
    print("\n   hindbrain dt_factor (φ-rungs):")
    for label, dorm, lag, seg in [
        ("lull, behind, quiet history",  0.97, 1.0, 0.0),
        ("lull, behind, BUSY history",   0.97, 1.0, 0.9),    # must NOT coarsen
        ("active world (caught up)",     0.10, 0.0, 0.0),
    ]:
        dtf = hindbrain_dt_factor(dorm, lag, seg)
        print(f"   {label:>30}  dt_factor {dtf:6.2f}  (φ^{round(math.log(dtf)/math.log(PHI))})")


if __name__ == "__main__":
    _selftest()
