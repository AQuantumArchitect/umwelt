"""TrustWeb — the per-leaf conditional-trust fuser (the keystone of FORESIGHT.md).

One small TrustWeb per comprehension leaf fuses EVERY input to that leaf — redundant
sensors AND competing forecast brains, uniformly — into a single confidence-gated
observation. Because a sensor is "a forecast with horizon 0" and a forecast is "a sensor
that predicts the future," the same operator handles three things at once:

  • health / redundancy — when a sensor drops (confidence→0 / not live), reroute trust to
    the correlated inputs that remain ("trust A more when B is out"), instead of merely
    decaying the belief;
  • foresight ensembling — fuse N scoped forecast brains' opinions on the leaf;
  • brain-chaining — the braid generalized to many upstream brains.

It is NOT a hand-coded health table. Each source has a learned baseline reliability r_s
(how well it predicts reality) plus SPARSE pairwise compensation c_{s,t} (extra weight s
earns specifically WHEN t is down). Initialized at the confidence prior (r=1, c=0), so the
day-1 fusion is exactly today's confidence-weighted observation; it earns its way off that
prior on realized outcomes (delayed-label). State is a small dict → pickled as heritage.

See docs/FORESIGHT.md, the confidence contract in docs/COMPREHENSION.md §2,
project_confidence_gauge_braid, project_sparse_features (the single+pairwise sparse shape).
"""
from __future__ import annotations
from umwelt._util import clamp01


class TrustWeb:
    """Conditional-trust fuser for one leaf. Sources are discovered as they appear."""

    def __init__(self, lr: float = 0.05):
        self.lr = float(lr)
        # baseline reliability per source (prior 1.0 = trusted; defaults applied lazily)
        self.r: dict[str, float] = {}
        # sparse compensation: c[(s, t)] = extra weight s earns when t is DOWN
        self.c: dict[tuple[str, str], float] = {}
        # every source ever seen live — a source in `seen` but absent this tick is "down"
        self.seen: set[str] = set()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _reliability(self, s: str) -> float:
        return self.r.get(s, 1.0)

    def _eff_weight(self, s: str, conf: float, down: set[str]) -> float:
        """w_s = conf_s × max(0, r_s + Σ_{t down} c_{s,t}). Compensation only ADDS trust
        (it lifts a survivor when a peer is out); never drives the weight negative."""
        comp = 0.0
        for t in down:
            comp += self.c.get((s, t), 0.0)
        eff = self._reliability(s) + comp
        return float(conf) * (eff if eff > 0.0 else 0.0)

    # ── fuse: {source: (z, conf, live)} → (z_fused, conf_fused) ───────────────
    def fuse(self, inputs: dict[str, tuple[float, float, bool]]) -> tuple[float, float]:
        """Fuse this tick's inputs into one (target_z, confidence). A source with
        live=False or conf≤0 contributes nothing; its absence (if previously seen) makes
        it "down" and triggers its peers' compensation. All-down → (0, 0): a provable
        no-op, the belief free-evolves (the confidence contract, preserved)."""
        for s, (_z, _conf, live) in inputs.items():
            if live:
                self.seen.add(s)
        live_sources = {s: (z, conf) for s, (z, conf, live) in inputs.items()
                        if live and conf > 0.0}
        down = {s for s in self.seen if s not in live_sources}

        weights: dict[str, float] = {}
        total = 0.0
        for s, (z, conf) in live_sources.items():
            w = self._eff_weight(s, conf, down)
            if w > 0.0:
                weights[s] = w
                total += w
        if total <= 0.0:
            return 0.0, 0.0

        z_fused = sum(weights[s] * live_sources[s][0] for s in weights) / total
        # disagreement = weighted variance of z around the fused value (z∈[-1,1] → var≤1).
        # Agreement falls when trusted live sources disagree → honest uncertainty.
        disagree = sum(weights[s] * (live_sources[s][0] - z_fused) ** 2 for s in weights) / total
        agreement = 1.0 - min(1.0, disagree)
        # confidence rises with total trusted weight (corroboration) and with agreement.
        conf_fused = min(1.0, total) * agreement
        return float(z_fused), float(max(0.0, conf_fused))

    # ── learn (delayed-label): ground reliability + compensation on outcomes ──
    def learn(self, inputs: dict[str, tuple[float, float, bool]], label_z: float,
              lr: float | None = None) -> None:
        """Update reliabilities + compensation from a realized leaf value `label_z`
        (a high-confidence direct observation, or the realized future for a forecast
        horizon). Good predictors gain reliability; a source that predicts well
        SPECIFICALLY when a peer is down grows that peer's compensation term."""
        a = self.lr if lr is None else float(lr)
        live = {s: z for s, (z, conf, lv) in inputs.items() if lv and conf > 0.0}
        for s in live:
            self.seen.add(s)
        down = {s for s in self.seen if s not in live}
        for s, z in live.items():
            # reward in [0,1]; 1 = perfect. Sharp (1−|err|, not 1−|err|/2) so a source
            # that is consistently ~0.7 off earns LOW trust, not a soft pass — the web
            # has to actually discriminate skill.
            reward = max(0.0, 1.0 - abs(z - label_z))
            r = self._reliability(s)
            self.r[s] = clamp01(r + a * (reward - r))  # EMA toward reward
            # compensation: did s beat its OWN baseline while peer t was down?
            surplus = reward - r
            for t in down:
                cur = self.c.get((s, t), 0.0)
                self.c[(s, t)] = clamp01(cur + a * (surplus - cur))

    # ── introspection / persistence ──────────────────────────────────────────
    def reliability(self, s: str) -> float:
        return self._reliability(s)

    def compensation(self, s: str, t: str) -> float:
        return self.c.get((s, t), 0.0)

    def snapshot(self) -> dict:
        return {
            "r": dict(self.r),
            "c": [[s, t, v] for (s, t), v in self.c.items()],
            "seen": sorted(self.seen),
            "lr": self.lr,
        }

    def load(self, state: dict) -> None:
        self.r = dict(state.get("r", {}))
        self.c = {(s, t): float(v) for s, t, v in state.get("c", [])}
        self.seen = set(state.get("seen", []))
        self.lr = float(state.get("lr", self.lr))
