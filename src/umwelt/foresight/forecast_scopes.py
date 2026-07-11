"""Forecast-brain scopes — the (RETIRED) sparse-graph partition for a multi-brain forecast ensemble.

RETIRED — the scoped forecast-brain services are gone. The forebrain now forecasts ITSELF via the
in-process free-run engine (forecast_rollout.forecast_freerun: one whole-field rollout reads EVERY
leaf, no partition needed — the free-run kernel made the scoping obsolete). This module survives for
reference and for any domain that still wants to partition a sparsely-connected field into scoped
forecast brains.

A scope is a set of cluster (node) names + the forecast leaves that scope owns. The cluster set scopes
the field (build via a cluster_filter); the leaves feed a ForecastSurface(leaves=…). Per-leaf the trust
web fuses whichever brains cover it. A leaf whose cluster isn't built is simply skipped by the surface.

The scope table is empty by default (domain-free) — a domain registers its own scopes.
"""
from __future__ import annotations

# scope name → {clusters: set of node names in the scoped field, leaves: forecast targets}.
# Empty by default; a domain registers its partition via register_scope().
FORECAST_SCOPES: dict[str, dict] = {}


def register_scope(name: str, clusters: set[str], leaves: tuple[tuple[str, str], ...]) -> None:
    """Register a forecast scope: a cluster-name set + the forecast leaves it owns."""
    FORECAST_SCOPES[name] = {"clusters": set(clusters), "leaves": tuple(leaves)}


def scope_cluster_filter(scope_name: str):
    """A `cluster_filter(node) -> bool` that keeps only the scope's clusters (module-level closure,
    not a lambda, so a scoped field stays picklable)."""
    clusters = FORECAST_SCOPES[scope_name]["clusters"]

    def _keep(node) -> bool:
        return getattr(node, "name", None) in clusters

    return _keep


def scope_leaves(scope_name: str) -> tuple[tuple[str, str], ...]:
    return tuple(FORECAST_SCOPES[scope_name]["leaves"])
