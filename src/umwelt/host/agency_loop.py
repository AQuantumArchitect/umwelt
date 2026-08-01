"""Shim — renamed to host/subroutines.py (R7.6 2026-07-31: the
agency/autonomy/agency_loop trio are three distinct concepts and now carry
distinct names; this one is the sub-routine/attention-budget/earned-automation
loop). Kept for pickled references and external callers; retire once all
live snapshots have re-minted under the new name.
"""
from umwelt.host.subroutines import *  # noqa: F401,F403
from umwelt.host.subroutines import (  # noqa: F401
    AgencyLoop,
    AttentionBudget,
    PromotionGate,
    SubRoutine,
    TimeContraction,
)
