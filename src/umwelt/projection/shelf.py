"""Shim — the provenance shelf moved down to substrate (R7.5 2026-07-31:
its only importer is substrate/field_gauge.py, and substrate must not import
upward). Kept so the projection-facing name and any external readers keep
working; the projection package docstring still lists the shelf reader.
"""
from umwelt.substrate.shelf import *  # noqa: F401,F403
from umwelt.substrate.shelf import Shelf, ARCHIVE_ROOT  # noqa: F401
