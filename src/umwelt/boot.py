"""build_engine — the ONE place a fully-wired BeliefEngine is constructed.

The origin deployment built this inline at boot, and ~8 other call sites (the
hindbrain, the training backbone, half a dozen experiments) each kept a near-copy
that drifted. This is the single source of truth — the same construction
everywhere, parameterized by the learner flags + an optional run-mode role.

The spec is REQUIRED: the engine has no default world (the origin's built-in
domain catalog is gone — a world arrives as a DomainSpec, never as code). Boot:

    resolve spec → materialize driver anchor nodes → build graph → construct
    engine → attach param bundles → apply declarative bindings → build drivers →
    register reward channels → bridge upgrades → qubit-back the param fiber →
    seed profile → optional run-mode role.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    from typing import Callable
    from umwelt.engine import BeliefEngine
    from umwelt.learning.context import ContextState

logger = logging.getLogger(__name__)


def set_role(engine, state) -> None:
    """Write a ContextState run-mode gauge (actuate/dt_factor/learn/persist) onto the
    engine's root bundle — the one seam every brain role goes through (forebrain,
    hindbrain/REPLAY, shadow, test, a picker's per-situation graft)."""
    root = engine.field.graph.root
    if root is not None and root.param_bundle is not None:
        state.write_to_bundle(root.param_bundle)


def _materialize_driver_nodes(spec):
    """Ensure every DriverSpec's anchor node exists in the topology.

    A driver names a (node, role) qubit to anchor; a spec that declares a driver
    but not its synthetic node (the common case — `_clock` is engine furniture,
    not domain topology) gets the node appended automatically: a `clock`-kind
    leaf under the root carrying the driver's role in unitary mode (a phase
    lives on the Bloch equator; amplitude damping would bias it)."""
    from umwelt.spec.schema import NodeSpec
    names = {n.name for n in spec.nodes}
    root_name = next((n.name for n in spec.nodes if n.parent is None), None)
    extras: list[NodeSpec] = []
    for d in spec.drivers:
        if d.node in names or any(e.name == d.node for e in extras):
            continue
        extras.append(NodeSpec(
            d.node, parent=root_name, kind="clock",
            roles=(d.role,), role_modes={d.role: "unitary"},
        ))
    if not extras:
        return spec
    return replace(spec, nodes=spec.nodes + tuple(extras))


def build_engine(
    *,
    population: bool = True,
    calibration: bool = True,
    fractal: bool = True,
    role: "ContextState | None" = None,
    seed_profile: str | None = None,
    cluster_filter: "Callable[[object], bool] | None" = None,
    subdomains: bool | None = None,
    spec=None,
    dispatch=None,
) -> "BeliefEngine":
    """Construct + fully wire a BeliefEngine: build → param fiber → declarative
    bindings → drivers → reward channels → param-norm + weight upgrades → whole-fiber
    qubit-backing → seed profile. Pass `role` to stamp the run-mode gauge in one step
    (e.g. ContextState.forebrain() or .replay()). Pass `cluster_filter` to SCOPE the
    field to a subgraph — a lightweight scoped brain (a forecast brain holds only the
    clusters it forecasts + their context); a live forebrain leaves it None.

    `spec` is the world: a DomainSpec, or a 'module:ATTR' string, else the UMWELT_SPEC
    env. No spec anywhere → ValueError (there is no default domain).

    `dispatch` is the app's transport: a callable(Action) the OutputSurface invokes for
    AUTO, non-shadow tendril decisions. None (the default) → every decision is recorded
    on engine.output_surface.recommendations and nothing leaves the process — combined
    with OutputSpec.shadow=True defaults, a fresh world decides visibly and dispatches
    nothing until the app opts in (every output enters as a signal first)."""
    from umwelt.engine import BeliefEngine
    from umwelt.learning.calibration import CalibrationConfig
    from umwelt.substrate.population import PopulationConfig
    from umwelt.substrate.fractal_stack import FractalStackConfig
    from umwelt.substrate.param_bundles import configure_param_bundles
    from umwelt.spec.schema import load_spec
    from umwelt.spec.build import build_graph_from_spec

    import os

    # The "any world" seam: explicit arg wins; else the env; else FAIL LOUDLY.
    if spec is None:
        _ref = (os.environ.get("UMWELT_SPEC") or "").strip()
        if not _ref:
            raise ValueError(
                "build_engine requires a spec (a DomainSpec, a 'module:ATTR' ref, "
                "or the UMWELT_SPEC env) — the engine has no default world")
        spec = load_spec(_ref)
    elif isinstance(spec, str):
        spec = load_spec(spec)

    # Fractal sub-domains (opt-in): group each region's identical device banks under
    # rich parent clusters. Explicit arg wins; else the spec flag; else the gated env
    # (default OFF). The post-build transform itself is a later phase — the flag rides
    # through to the engine for surface-compat.
    if subdomains is None:
        subdomains = bool(getattr(spec, "enable_subdomains", False)
                          or os.environ.get("UMWELT_SUBDOMAINS"))

    # Topology: materialize the drivers' synthetic anchor nodes, then build.
    spec = _materialize_driver_nodes(spec)
    graph = build_graph_from_spec(spec)

    engine = BeliefEngine(
        graph=graph,
        calibration=CalibrationConfig(hamiltonian_enabled=calibration),
        population=PopulationConfig(enabled=population, generation_interval=30, min_age=15),
        fractal_stack=FractalStackConfig(enabled=fractal),
        cluster_filter=cluster_filter,
        subdomains=bool(subdomains),
    )
    # Param fiber must attach before the bridge upgrades read sensor_*_lo/hi off
    # bundles. The spec rides along so per-node priors (NodeSpec.params) land too.
    configure_param_bundles(engine.graph, spec)
    engine.sensor_bridge.refresh_node_params()

    # The declarative binding seam — the spec's own signal vocabulary, nothing domain-
    # coded. Per-binding membrane-guarded inside (a bad binding never breaks the rest).
    from umwelt.membranes.ingress import apply_spec_bindings
    apply_spec_bindings(engine.sensor_bridge, spec)

    # Periodic drivers — the domain's clocks, resolved through the driver registry.
    from umwelt.clocks.drivers import build_driver
    engine.drivers = [build_driver(d) for d in spec.drivers]

    # Output tendrils — the spec's decisions, alive on the uniform surface (blocker 7):
    # each OutputSpec becomes a SpecTendril (committed belief + decoder + gates), and the
    # OutputSurface routes what they emit (shadow/recommend → recorded; auto → dispatch).
    from umwelt.membranes.egress import OutputSurface, build_tendrils
    engine.tendrils = build_tendrils(engine, spec)
    engine.output_surface = OutputSurface(dispatch=dispatch)

    # Declarative reward-vocabulary extensions: assign spec params to channels
    # (a pattern ending "_" is a prefix, else exact — the registry idiom).
    if getattr(spec, "param_channels", None):
        from umwelt.learning.reward.registry import register_param_channel
        for pat, ch in spec.param_channels:
            try:
                if pat.endswith("_"):
                    register_param_channel(prefix=(pat,), channel=ch)
                else:
                    register_param_channel(exact={pat}, channel=ch)
            except ValueError as exc:
                logger.warning("spec param_channel (%r, %r) skipped: %s", pat, ch, exc)

    engine.sensor_bridge.upgrade_to_param_norms()
    engine.sensor_bridge.upgrade_weights()
    # Whole-fiber qubit-backing: constructor ran against missing bundles; re-init now
    # they're attached.
    engine._bind_param_fiber()
    # Initial-condition gauge: the DEFAULT is the maximally-mixed BLANK floor —
    # preference seeds wiped to max-entropy, nothing assumed. The opted-in profile
    # (UMWELT_SEED_PROFILE, or the seed_profile arg) ADDS a deployment's mined seeds
    # back. See learning/seed_profile.py.
    from umwelt.learning.seed_profile import apply_seed_profile, seed_profile_from_env
    apply_seed_profile(engine, seed_profile or seed_profile_from_env())
    if role is not None:
        set_role(engine, role)
    return engine


# Origin-name alias for ported call sites (see tools/RENAMES.md).
build_reservoir = build_engine
