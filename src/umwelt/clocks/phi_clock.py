"""Shim — the φ ladder moved down to substrate (R7.5 2026-07-31: substrate
must not import upward, and fractal_stack climbs this ladder too). Both
towers still share the clock via either name; this one survives for the
existing clocks/learning/foresight importers and any pickled references.
"""
from umwelt.substrate.phi_clock import (  # noqa: F401
    PHI,
    effective_stride,
    fib_strides,
    fib_strides_at,
)
