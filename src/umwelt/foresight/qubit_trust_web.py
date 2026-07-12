"""QubitTrustWeb — the per-leaf trust fuser with reliability as a learned QUBIT (gated upgrade).

A drop-in subclass of TrustWeb (same fuse / _eff_weight / compensation / facade, so every
bridge callsite is unchanged) that moves the per-source reliability r_s from a plain scalar dict
to one independent reliability QUBIT per source on a ProductQubitCluster (O(N), project_qubit_fiber),
learned via observe_qubit partial collapse — the same pattern that made celestial_alpha a qubit
(qubit_param.QubitBackedParam).

  • reliability r_s  = (z + 1)/2 of the source's qubit  (z=+1 → r=1, the trusted prior).
  • confidence-in-r  = the qubit's PURITY |r| — a new gauge DOF the scalar never had: how SETTLED
    the reliability estimate is. Consistent rewards concentrate it; contradictory rewards keep it
    mixed ("we don't yet know how reliable this source is").

EXACT EMA PARITY (this is an upgrade, not a rewrite): observe_qubit mixes z as
z ← (1−a)z + a·target_z; with target_z = 2·reward−1, reliability r=(z+1)/2 evolves as
r ← (1−a)r + a·reward — the SAME EMA as TrustWeb.learn. Each new qubit is seeded pure at z=+1,
so day-1 fusion is byte-identical to the classical web; it earns its way off the prior on outcomes.

Compensation c_{s,t} stays the classical sparse term. Off by default — bridge.attach_trust_web
builds this only when UMWELT_TRUST_QUBIT is set. See the lineage harness experiments for the
validation: 14/14 field-health agreement + exact reliability parity vs the classical TrustWeb.
"""
from __future__ import annotations

import math

from umwelt.foresight.trust_web import TrustWeb
from umwelt.substrate.product_cluster import ProductQubitCluster
from umwelt.substrate.qubit_param import value_to_bloch_z, bloch_z_to_value


def _surface_point(tz: float) -> tuple[float, float, float]:
    """A PURE Bloch target at the given z (on the sphere). The x-component injects the coherence
    whose accumulation IS the purity = confidence-in-the-reliability-estimate."""
    tz = max(-1.0, min(1.0, tz))
    return (math.sqrt(max(0.0, 1.0 - tz * tz)), 0.0, tz)


class QubitTrustWeb(TrustWeb):
    """A TrustWeb whose per-source reliability r_s lives on a reliability qubit (drop-in)."""

    def __init__(self, lr: float = 0.05, cluster: ProductQubitCluster | None = None):
        super().__init__(lr=lr)
        self.rel = cluster if cluster is not None else ProductQubitCluster("trust_reliability")
        # self.r (the inherited classical dict) is left unused; reliability reads from the qubits.
        # Compensation c_{s,t} also lives on qubits now (a CouplingBank): its update is an EMA, so it
        # is the same partial-collapse move as r_s — finishing the trust qubit (no classical halo left
        # in this web). Tuple-keyed, so the inherited _eff_weight/compensation read it unchanged.
        from umwelt.substrate.qubit_ema import CouplingBank
        self.c = CouplingBank("trust_compensation", 0.0, 1.0)
        # The reliability learning is routed through the ONE learner object (UniversalLearner), in its
        # SUPERVISED-target mode: r_s collapses toward the source's accuracy reward — already the
        # universal principle (collapse toward reducing the source's own surprise), now on the shared
        # learner. Exact observe_qubit/EMA parity with the classical web. See project_universal_learning_law.
        from umwelt.learning.universal_learner import UniversalLearner
        self._learner = UniversalLearner()

    # ── reliability now lives on a qubit ──────────────────────────────────────
    def _role(self, s: str) -> str:
        return f"_rel_{s}"

    def _ensure(self, s: str) -> int:
        """Index of source s's reliability qubit; seed a fresh one pure at z=+1 (r=1) so day-1
        fusion matches the classical prior exactly."""
        role = self._role(s)
        if role in self.rel.role_index:
            return self.rel.role_index[role]
        idx = self.rel.add_role(role)
        self.rel.observe_qubit(idx, (0.0, 0.0, 1.0), alpha=1.0)   # pure |0> → reliability 1.0
        return idx

    def _reliability(self, s: str) -> float:
        return bloch_z_to_value(float(self.rel.qubit_bloch(self._ensure(s))[2]), 0.0, 1.0)

    def confidence(self, s: str) -> float:
        """The new DOF: purity |r| of the reliability qubit — how settled the estimate is."""
        bx, by, bz = self.rel.qubit_bloch(self._ensure(s))
        return math.sqrt(bx * bx + by * by + bz * bz)

    # ── learn: partial-collapse the reliability qubit toward the reward ────────
    def learn(self, inputs: dict[str, tuple[float, float, bool]],
              label_z: "float | dict[str, float]", lr: float | None = None) -> None:
        a = self.lr if lr is None else float(lr)
        live = {s: z for s, (z, conf, lv) in inputs.items() if lv and conf > 0.0}
        for s in live:
            self.seen.add(s)
        down = {s for s in self.seen if s not in live}
        labels = label_z if isinstance(label_z, dict) else {s: label_z for s in live}
        for s, z in live.items():
            if s not in labels:
                continue
            reward = max(0.0, 1.0 - abs(z - labels[s]))          # sharp reward (== classical) = 1 − surprise
            r = self._reliability(s)
            # r ← (1−a)r + a·reward via the ONE learner's supervised-target collapse (raw-qubit form,
            # parity-exact with the prior direct observe_qubit; the reward IS 1 − the source's surprise).
            self._learner.observe_raw(self.rel, self._ensure(s), reward, 0.0, 1.0, a)
            surplus = reward - r                                 # compensation now lives on qubits too
            for t in down:
                # c ← (1−a)c + a·surplus on the coupling qubit; the qubit's [0,1] bounds clamp the
                # target (the classical web clamped the post-update value — equivalent for surplus∈[0,1],
                # the normal regime, and both floor compensation at 0).
                self.c.observe((s, t), surplus, a, default=0.0)

    # ── persistence: pickle the qubit matrices instead of the r dict ──────────
    def snapshot(self) -> dict:
        return {
            "rel_matrices": {role: m.tolist() for role, m in self.rel.state_matrices().items()},
            # compensation qubit matrices, keyed by the joined role (a coupling cluster)
            "c_matrices": {role: m.tolist() for role, m in self.c.cluster.state_matrices().items()},
            "seen": sorted(self.seen),
            "lr": self.lr,
            "kind": "qubit_trust_web",
        }

    def load(self, state: dict) -> None:
        import numpy as np
        mats = {role: np.asarray(m, dtype=complex) for role, m in state.get("rel_matrices", {}).items()}
        for role in mats:
            if role not in self.rel.role_index:
                self.rel.add_role(role)
        self.rel.load_matrices(mats)
        cmats = {role: np.asarray(m, dtype=complex) for role, m in state.get("c_matrices", {}).items()}
        for role in cmats:
            if role not in self.c.cluster.role_index:
                self.c.cluster.add_role(role)
        if cmats:
            self.c.cluster.load_matrices(cmats)
        self.seen = set(state.get("seen", []))
        self.lr = float(state.get("lr", self.lr))
