"""umwelt — a belief-field engine.

Describe a world as data (a DomainSpec); feed it observations as weak measurements;
it holds a live, honest, uncertain comprehension of that world — and forecasts from it.

Public facade (grows as extraction phases land):
    Event                       — the neutral event wire format
    (P1) DomainSpec, NodeSpec, BridgeSpec, BindingSpec, OutputSpec, DriverSpec, load_spec
    (P2) build_engine, BeliefEngine
"""
from __future__ import annotations

from .events import Event

__version__ = "0.1.0.dev0"

__all__ = ["Event", "__version__"]
