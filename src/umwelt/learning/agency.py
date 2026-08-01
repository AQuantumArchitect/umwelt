"""Shim — renamed to learning/agency_qubit.py (R7.6 2026-07-31: the
agency/autonomy/agency_loop trio are three distinct concepts and now carry
distinct names). Kept so engine snapshots pickled before the rename (module
path `umwelt.learning.agency` baked into the pickle) still restore; retire
once all live snapshots have re-minted under the new name.
"""
from umwelt.learning.agency_qubit import *  # noqa: F401,F403
from umwelt.learning.agency_qubit import AgencyQubit  # noqa: F401
