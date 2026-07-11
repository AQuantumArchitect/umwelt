"""Derived beliefs — a parent's belief synthesized from its children's shared role.

The generic replacement for per-domain "layered" summary fields (blocker 3 of the
extraction): where the origin deployment had a module that hard-baked "is ANYONE
present, per region" into tensor semantics, here a NodeSpec declares `reduce` and the
engine synthesizes the parent's role from its children with a registered reducer.

Reducers operate on the children's Bloch-z values for the shared role and return the
parent's z. `or_` is the soft-OR (max) with the convention z=+1 ⇒ asserted.
"""
from __future__ import annotations

from typing import Callable

REDUCERS: dict[str, Callable[[list[float]], float]] = {
    "max": lambda zs: max(zs) if zs else 0.0,
    "mean": lambda zs: (sum(zs) / len(zs)) if zs else 0.0,
    "or": lambda zs: max(zs) if zs else -1.0,   # soft-OR: any child asserted ⇒ parent asserted
}


def register_reducer(name: str, fn: Callable[[list[float]], float]) -> None:
    if name in REDUCERS:
        raise ValueError(f"reducer {name!r} already registered")
    REDUCERS[name] = fn


def reduce_children(reduce: str, child_zs: list[float]) -> float:
    fn = REDUCERS.get(reduce)
    if fn is None:
        raise ValueError(f"unknown reducer {reduce!r}; known: {sorted(REDUCERS)}")
    return fn([float(z) for z in child_zs])
