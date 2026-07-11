"""Role classification as DATA — how each qubit role receives its input.

Three orthogonal properties, all registry-backed so a domain declares its vocabulary
instead of the engine hard-coding it:

INPUT MODE ("unitary" | "dissipative"): a unitary role is event-driven — an observation
kicks it (σx rotation / partial collapse) and it free-evolves between events. A
dissipative role is continuously driven — the belief thermalizes toward the reading
through decoherence, at a rate set by the evolver's gamma_diss. Default for an
unregistered role is DISSIPATIVE: continuous tracking is the safer default (it cannot
spike the density matrix).

OBSERVE roles: discrete qualities held as a *belief* that drifts between observations
and snaps toward the seen state when reality is observed (QubitCluster.observe_qubit).
Driven OUT OF BAND in the engine ingest loop, never through the continuous field.step
input array. As an input channel they behave like unitary qubits.

ANALOG roles: continuous quantities that must NOT be committed as ±1 poles — the
sky→ground collapse stores their real value/position and emits no discrete transition.
A periodic driver's anchor role (a clock reading, an ephemeris position) is the
canonical case and is registered automatically when the driver attaches.

Engine-internal prefixes (`_param_`, `_pref_`, `_clock_`) are always unitary: isolated
memory cells whose only drive is observe_qubit.
"""
from __future__ import annotations

# ── the registries (module-level, mutated only via the register_* functions) ─────────
_UNITARY_ROLES: set[str] = set()
_DISSIPATIVE_ROLES: set[str] = set()
_OBSERVE_ROLES: set[str] = set()
_DRIVER_ROLES: set[str] = set()          # periodic-driver anchor roles (out of band, analog)
_ANALOG_ROLES: set[str] = set()

_ENGINE_UNITARY_PREFIXES = ("_param_", "_pref_", "_clock_")


def register_role_mode(role: str, mode: str, *, analog: bool = False) -> None:
    """Declare a role's input mode ('unitary' | 'dissipative'); optionally mark it analog.
    Domains call this (directly or via their spec) for every role in their vocabulary."""
    if mode == "unitary":
        _UNITARY_ROLES.add(role)
        _DISSIPATIVE_ROLES.discard(role)
    elif mode == "dissipative":
        _DISSIPATIVE_ROLES.add(role)
        _UNITARY_ROLES.discard(role)
    else:
        raise ValueError(f"role mode must be 'unitary' or 'dissipative', got {mode!r}")
    if analog:
        _ANALOG_ROLES.add(role)


def register_observe_role(role: str) -> None:
    """Declare a role driven by observation/partial-collapse, out of band."""
    _OBSERVE_ROLES.add(role)


def register_driver_role(role: str) -> None:
    """Declare a periodic driver's anchor role: out of band, analog, unitary-channel.
    Called automatically when a driver attaches to the engine."""
    _DRIVER_ROLES.add(role)
    _ANALOG_ROLES.add(role)


def register_analog_role(role: str) -> None:
    """Declare a continuous role whose collapse stores a value, never a ±1 pole."""
    _ANALOG_ROLES.add(role)


def is_observe_role(role: str) -> bool:
    """True for roles driven by observation/partial-collapse, not continuous drive."""
    return role in _OBSERVE_ROLES


def is_driver_role(role: str) -> bool:
    """True for periodic-driver anchor roles driven out of band."""
    return role in _DRIVER_ROLES


def is_analog_role(role: str) -> bool:
    """True for continuous roles that collapse commits as a value/position, not ±1."""
    return role in _ANALOG_ROLES


def role_input_mode(role: str) -> str:
    """Classify a qubit role's input channel: 'unitary' or 'dissipative'.

    Observe and driver roles report 'unitary' so the cluster builds no dissipative
    thermal channel for them — between out-of-band updates they free-evolve (drift).
    Their actual drive is the observe-collapse, not field.step."""
    if role in _OBSERVE_ROLES or role in _DRIVER_ROLES:
        return "unitary"
    if role.startswith(_ENGINE_UNITARY_PREFIXES):
        return "unitary"
    if role in _UNITARY_ROLES:
        return "unitary"
    return "dissipative"
