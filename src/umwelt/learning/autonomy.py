"""Shim — renamed to learning/actuator_autonomy.py (R7.6 2026-07-31: the
agency/autonomy/agency_loop trio are three distinct concepts and now carry
distinct names). Kept for pickled references and external callers; retire
once all live snapshots have re-minted under the new name. NOTE: REGISTRY is
the same list object either way — registrations through one name are seen
through the other.
"""
from umwelt.learning.actuator_autonomy import *  # noqa: F401,F403
from umwelt.learning.actuator_autonomy import (  # noqa: F401
    AUTO,
    REGISTRY,
    SHADOW,
    ActuatorAutonomy,
    posture,
    register_actuator_autonomy,
    report,
    set_posture,
)
