"""Rewards as neuromodulators — the multi-reward learning substrate.

Each REWARD is a neuromodulator CHANNEL (a named field with its own expression manifold + timescale,
broadcast everywhere). Each learnable PARAM is a cell with a RECEPTOR PROFILE — which channels modulate
it, with what sensitivity. A channel acting on a param is an `observe_qubit` partial collapse toward that
channel's target, scaled by the receptor weight: "chemical attachment = classical measurement" (the
update IS the collapse, already true in this substrate).

Phase 1 (separate sectors): every param has exactly ONE receptor (its sector); the fiber splits into one
ProductQubitCluster per channel; the existing learners (calibration=surprise, meta_pbt=skill_compute,
person_model=override) are mapped declaratively, untouched. Future: multi-receptor profiles, the
user-outcome channel (outcome:sleep), and learnable receptors. See plan noble-sleeping-yao + #310.
"""
from .channel import ReceptorProfile, RewardChannel
from .registry import CHANNELS, channel_for, receptor_for

__all__ = ["ReceptorProfile", "RewardChannel", "CHANNELS", "channel_for", "receptor_for"]
