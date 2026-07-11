"""The normalizer registry — declarative {type, **params} → a unit-free callable.

Every observation enters the field as a scalar in [-1, +1]; a normalizer is the edge
function that takes a raw reading (a temperature, a price return, a parse score) and
produces that unit-free value. The engine ships only DOMAIN-NEUTRAL normalizer types;
a domain registers its own idioms via `register_normalizer` (e.g. a °C→°F regime, a
log-return squash) — vocabulary is data, never engine code.

A binding's normalizer in a spec is data — `{"type": "regime", "center": 21, "width": 4}`
— resolved here at build time. Module-level functions (not lambdas) where a normalizer
must stay picklable.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def binary_norm(val: float) -> float:
    """Binary signal (0/1) -> (-1/+1)."""
    return 1.0 if val > 0.5 else -1.0


def forecast_zflip_norm(val: float) -> float:
    """A forecast's published value IS a Bloch-z in [-1, +1] (the same z it read off the
    field and predicted forward). The observe path computes `target_z = -r_obs × direction`,
    so to land the live belief AT the forecast z we flip the sign here (with r_obs=1.0):
    direction = -z_pred → target_z = +z_pred. Module-level so the binding stays picklable."""
    return -max(-1.0, min(1.0, float(val)))


def range_norm(lo: float, hi: float) -> Callable[[float], float]:
    """Linear normalization from [lo, hi] -> [-1, +1]."""
    def _norm(val: float) -> float:
        if hi == lo:
            return 0.0
        clamped = max(lo, min(hi, val))
        return 2.0 * (clamped - lo) / (hi - lo) - 1.0
    return _norm


def threshold_norm(threshold: float, invert: bool = False) -> Callable[[float], float]:
    """Soft threshold: tanh((val - threshold) / scale)."""
    def _norm(val: float) -> float:
        x = (val - threshold) * 2.0
        result = float(np.tanh(x))
        return -result if invert else result
    return _norm


def regime_norm(center: float, width: float, invert: bool = False) -> Callable[[float], float]:
    """Sigmoidal regime classifier: two states with a transition band.

    Maps a scalar to [-1, +1] using tanh: values well below `center` → -1 (clearly in
    state |0⟩), well above → +1 (clearly |1⟩), near center → ≈0 (genuine uncertainty,
    superposition). `width` is the half-width of the transition band — narrow = sharp
    boundary, wide = a gradual band where quantum uncertainty is honest."""
    def _norm(val: float) -> float:
        x = (val - center) / max(width, 1e-6)
        result = float(np.tanh(x))
        return -result if invert else result
    return _norm


def cyclic_norm(period: float, peak: float = 0.0) -> Callable[[float], float]:
    """Position on a cycle -> [-1, +1], +1 at `peak`, -1 half a period away.
    The generic form of a time-of-day / session-phase / seasonal normalizer."""
    def _norm(val: float) -> float:
        return float(np.cos(2.0 * np.pi * (val - peak) / period))
    return _norm


# ── the registry: declarative type name → factory over the callables above ───────────
NORMALIZER_FACTORIES: dict[str, Callable[..., Callable[[float], float]]] = {
    # stateless (no params)
    "binary":         lambda: binary_norm,
    "forecast_zflip": lambda: forecast_zflip_norm,
    # parametric
    "range":          lambda lo, hi: range_norm(lo, hi),
    "threshold":      lambda threshold, invert=False: threshold_norm(threshold, invert),
    "regime":         lambda center, width, invert=False: regime_norm(center, width, invert),
    "cyclic":         lambda period, peak=0.0: cyclic_norm(period, peak),
}


def register_normalizer(name: str, factory: Callable[..., Callable[[float], float]]) -> None:
    """Register a domain normalizer type. `factory(**params)` must return the callable.
    Re-registering an engine-shipped name raises — a domain extends the vocabulary, it
    does not silently redefine what "binary" means."""
    if name in NORMALIZER_FACTORIES:
        raise ValueError(f"normalizer type {name!r} already registered")
    NORMALIZER_FACTORIES[name] = factory


def resolve_normalizer(spec: "str | dict | Callable") -> Callable[[float], float]:
    """Resolve a declarative normalizer → a callable. Accepts a bare type name ("binary"),
    a {"type": ..., **params} dict, or an already-built callable (pass-through). Unknown
    types raise — a spec that names a normalizer we can't build should fail loudly, not
    silently mis-bind a signal."""
    if callable(spec):
        return spec
    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError(f"normalizer spec must be a type name, {{type,...}} dict, or callable: {spec!r}")
    t = spec["type"]
    factory = NORMALIZER_FACTORIES.get(t)
    if factory is None:
        raise ValueError(f"unknown normalizer type {t!r}; known: {sorted(NORMALIZER_FACTORIES)}")
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    return factory(**kwargs)
