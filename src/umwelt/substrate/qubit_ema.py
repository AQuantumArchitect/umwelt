"""QubitEMABank — a dict-of-EMA whose every entry is a learned QUBIT (gated upgrade).

The calibration loop keeps several per-cluster EMAs (`_surprise_ema`, `_tracking_ema`,
`_collapse_rate_ema`) as plain `dict[str, float]`. Each is updated by the textbook recurrence
`ema ← α·signal + (1−α)·prev` — which is EXACTLY a partial-collapse `observe_qubit` at α toward the
signal: `z ← (1−α)z + α·target_z`, so the rescaled value `v ← (1−α)v + α·signal`. So the EMA dict is
a classical halo learner that already has a qubit form — the same move that made `driver_alpha`
(qubit_param.QubitBackedParam) and the trust web (qubit_trust_web.QubitTrustWeb) qubits.

This module supplies that form as a drop-in for the dict:

  • `QubitEMABank` — each key is one independent qubit on a ProductQubitCluster. `observe(key, signal,
    alpha, default)` is the EMA step; the value reads off the qubit's Bloch z; the qubit's PURITY |r|
    is a new DOF the scalar never had (how SETTLED that bandwidth estimate is). It also keeps an
    `update_count` per key — the LEARNING LEDGER the non-training certificate reads (the
    decoherence-invariant witness: it moves only when a learner calls observe, never from field
    evolution). Held OUTSIDE field.clusters on purpose: these banks grow lazily, and a growing cluster
    in the field would shift the readout feature geometry. They are witnessed by the ledger, not by
    field_gauge.
  • `ClassicalEMABank` — a `dict` subclass with the same `observe`/`ledger` API, the exact scalar EMA.
    The default when the flag is off, so day-1 behaviour is byte-identical.
  • `make_ema_bank(name, lo, hi)` — returns the qubit bank iff `UMWELT_CALIB_QUBIT=1`, else classical.

EXACT PARITY (upgrade, not rewrite): seeded at the same `default`, fed the same (signal, α) stream,
the qubit bank's value tracks the classical EMA to ~1e-6 (verified in tests). Off by default.
"""
from __future__ import annotations

import math
import os

from umwelt.substrate.product_cluster import ProductQubitCluster
from umwelt.substrate.qubit_param import bloch_z_to_value, value_to_bloch_z


def _surface_point(tz: float) -> tuple[float, float, float]:
    """Pure Bloch target at z=tz (on the sphere). The x-component injects the coherence whose
    accumulation IS the purity = how settled the estimate is. Value (⟨σ_z⟩) is unaffected."""
    tz = max(-1.0, min(1.0, tz))
    return (math.sqrt(max(0.0, 1.0 - tz * tz)), 0.0, tz)


class ClassicalEMABank(dict):
    """The plain scalar EMA dict, with the shared observe()/ledger() API (the gated-off default)."""

    def observe(self, key: str, signal: float, alpha: float, default: float) -> float:
        prev = self.get(key, default)
        ema = alpha * signal + (1.0 - alpha) * prev
        self[key] = ema
        return ema

    def ledger(self) -> dict[str, int]:
        return {}                                # classical state is not ledger-witnessed


class QubitEMABank:
    """A dict-of-EMA where each entry is one independent qubit (exact-EMA parity + a purity DOF)."""

    def __init__(self, name: str, lo: float, hi: float):
        self.cluster = ProductQubitCluster(name)
        self.lo, self.hi = float(lo), float(hi)
        self._count: dict[str, int] = {}         # the learning ledger (update_count per key)

    # ── allocation ────────────────────────────────────────────────────────────
    def _ensure(self, key: str, default: float) -> int:
        if key in self.cluster.role_index:
            return self.cluster.role_index[key]
        idx = self.cluster.add_role(key)
        # seed pure at the default so the first observe matches classical (prev = default)
        self.cluster.observe_qubit(idx, _surface_point(value_to_bloch_z(default, self.lo, self.hi)),
                                   alpha=1.0)
        self._count[key] = 0
        return idx

    # ── the EMA step = partial collapse toward the signal ───────────────────────
    def observe(self, key: str, signal: float, alpha: float, default: float) -> float:
        idx = self._ensure(key, default)
        self.cluster.observe_qubit(idx, _surface_point(value_to_bloch_z(signal, self.lo, self.hi)),
                                   alpha=alpha)
        self._count[key] += 1
        # read straight off the qubit (NOT self.get — a subclass like CouplingBank overrides get with
        # different key semantics; this internal read must use the already-resolved string key)
        return bloch_z_to_value(float(self.cluster.role_bloch(key)[2]), self.lo, self.hi)

    def ledger(self) -> dict[str, int]:
        return dict(self._count)

    # ── dict-compatible read surface (so stats()/meta-learn callsites are unchanged) ──
    def get(self, key: str, default: float = 0.0) -> float:
        if key not in self.cluster.role_index:
            return default
        return bloch_z_to_value(float(self.cluster.role_bloch(key)[2]), self.lo, self.hi)

    def confidence(self, key: str) -> float:
        """The new DOF: purity |r| of the key's qubit — how settled the bandwidth estimate is."""
        if key not in self.cluster.role_index:
            return 0.0
        b = self.cluster.role_bloch(key)
        return math.sqrt(sum(float(v) * float(v) for v in b))

    def keys(self):
        return list(self.cluster.role_index)

    def values(self):
        return [self.get(k) for k in self.cluster.role_index]

    def items(self):
        return [(k, self.get(k)) for k in self.cluster.role_index]

    def __getitem__(self, key: str) -> float:
        return self.get(key)

    def __iter__(self):
        return iter(self.cluster.role_index)

    def __contains__(self, key: str) -> bool:
        return key in self.cluster.role_index

    def __len__(self) -> int:
        return len(self.cluster.role_index)

    def __bool__(self) -> bool:
        return len(self.cluster.role_index) > 0


def make_ema_bank(name: str, lo: float, hi: float):
    """Qubit-backed EMA bank iff UMWELT_CALIB_QUBIT=1, else the classical scalar dict (default)."""
    if os.environ.get("UMWELT_CALIB_QUBIT") == "1":
        return QubitEMABank(name, lo, hi)
    return ClassicalEMABank()


_SEP = "\x1f"   # unit-separator: joins a (source, target) pair into one qubit role name


class CouplingBank(QubitEMABank):
    """A QubitEMABank keyed by PAIRS (s, t) — for the trust web's pairwise compensation c_{s,t}.

    c is learned by `c ← (1−a)c + a·surplus` (clamped) — an EMA, so it is the same partial-collapse
    move as every other migrated learner: each pair gets one qubit, exact parity, ledger-witnessed.
    This recognises 'compensation' as gauge coordinates (a learned coupling between two reliability
    sources) rather than a classical sparse dict — the gorgeous half of finishing the trust qubit.
    Exposes the tuple-keyed dict surface (`get((s,t), default)`, `observe((s,t), ...)`, `pairs()`)
    so it drops into TrustWeb's `self.c` with no callsite changes."""

    def _k(self, pair) -> str:
        return f"{pair[0]}{_SEP}{pair[1]}"

    def get(self, pair, default: float = 0.0) -> float:
        return super().get(self._k(pair), default)

    def observe(self, pair, signal: float, alpha: float, default: float = 0.0) -> float:
        return super().observe(self._k(pair), signal, alpha, default)

    def pairs(self):
        """[(s, t, value), ...] — for snapshot/inspection (splits the joined role back to a pair)."""
        out = []
        for role in self.cluster.role_index:
            s, _, t = role.partition(_SEP)
            out.append((s, t, super().get(role)))
        return out
