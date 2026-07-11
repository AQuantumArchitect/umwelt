"""The reward-channel registry + the param→channel manifest, as DATA.

The authoritative place a param is assigned to a reward channel. Phase-1 is 1:1 (every
param → one channel, weight 1.0). The boundary between SURPRISE and UNLEARNED is
intentionally fuzzy/low-stakes (both are "not skill, not override"); what must be exact
is the ISOLATION of the genuinely-distinct rewards: SKILL/COMPUTE (the cadence +
compression family, learned generationally against skill − λ·compute) and OVERRIDE
(params learned from operator corrections).

The engine ships the four generic neuromodulators and the engine-DNA classification
rules. A DOMAIN extends the manifest declaratively:
  declare_outcome_channel("purchase")             → an "outcome:purchase" channel
  register_param_channel(exact={"my_param"}, channel="override")
  register_param_channel(prefix=("fee_",), channel="surprise")
DomainSpec.param_channels rows ((exact_or_prefix, channel)) feed the same registry at
boot. Folded-topology re-keyed params ('{node}_{param}') classify through the spec's
param_key_normalizer hook, applied by the caller (engine._root_param path).
"""
from __future__ import annotations

from .channel import ReceptorProfile, RewardChannel

# The neuromodulators. fiber_cluster = the ProductQubitCluster (sector) each channel's
# params live on. Domain outcome channels are added via declare_outcome_channel.
CHANNELS: dict[str, RewardChannel] = {
    "surprise":      RewardChannel("surprise",      "_fiber_surprise",  timescale="per_tick"),
    "skill_compute": RewardChannel("skill_compute", "_fiber_skill",     timescale="generational"),
    "override":      RewardChannel("override",      "_fiber_override",  timescale="event"),
    "unlearned":     RewardChannel("unlearned",     "_fiber_unlearned", timescale="none"),
}


def declare_outcome_channel(name: str) -> RewardChannel:
    """Declare a domain OUTCOME reward: a stated outcome drives ACTUATION up a valence
    gradient. Distinct from surprise (which has no preference over outcomes); its
    receptors are outputs, not signals. Returns the (possibly existing) channel."""
    full = f"outcome:{name}"
    if full not in CHANNELS:
        CHANNELS[full] = RewardChannel(full, f"_fiber_outcome_{name}", timescale="event")
    return CHANNELS[full]


# ── the engine-DNA manifest ──────────────────────────────────────────────────────────
# SKILL/COMPUTE — the cadence + compression family (generational: skill − λ·compute)
_SKILL_EXACT: set[str] = {"dt_factor_max", "coast_eps"}
_SKILL_PREFIX: tuple[str, ...] = ("cadence_",)   # the whole cadence manifold, by intent

# OVERRIDE — operator corrections (engine ships none; domains register theirs)
_OVERRIDE_EXACT: set[str] = set()

# SURPRISE — what the calibration tower + fractal self-tune nudge toward less error
_SURPRISE_EXACT: set[str] = {
    "gamma", "gamma_diss", "hysteresis", "confidence_threshold", "projection_coupling",
    "bridge_strength",
    "driver_alpha", "driver_hebbian_lr", "driver_anticipation_ema",
    "forecast_lr", "forecast_l2", "forecast_ema",
    "wide_nudge_lo", "wide_nudge_hi", "collapse_rate_ema_alpha",
    "transition_floor", "motion_eps",
}
_SURPRISE_PREFIX: tuple[str, ...] = ("sensor_", "gamma_diss_")

# DOMAIN extensions: (exact_or_prefix, channel) — prefix entries end with "_".
_DOMAIN_EXACT: dict[str, str] = {}
_DOMAIN_PREFIX: list[tuple[str, str]] = []


def register_param_channel(*, exact: set[str] | None = None,
                           prefix: tuple[str, ...] | None = None,
                           channel: str) -> None:
    """Declaratively assign params to a channel. Unknown channel names raise — declare
    outcome channels first (declare_outcome_channel)."""
    if channel not in CHANNELS:
        raise ValueError(f"unknown reward channel {channel!r}; known: {sorted(CHANNELS)}")
    for k in (exact or ()):
        _DOMAIN_EXACT[k] = channel
    for p in (prefix or ()):
        _DOMAIN_PREFIX.append((p, channel))


def channel_for(node_name: str, key: str) -> str:
    """The reward channel that owns a param (Phase-1: its single sector). node_name is
    accepted for future per-node rules; Phase-1 classification is key-based. Domain
    registrations take precedence over the engine manifest."""
    if key in _DOMAIN_EXACT:
        return _DOMAIN_EXACT[key]
    for p, ch in _DOMAIN_PREFIX:
        if key.startswith(p):
            return ch
    if key in _SKILL_EXACT or key.startswith(_SKILL_PREFIX):
        return "skill_compute"
    if key in _OVERRIDE_EXACT:
        return "override"
    if key in _SURPRISE_EXACT or key.startswith(_SURPRISE_PREFIX):
        return "surprise"
    return "unlearned"


def receptor_for(node_name: str, key: str) -> ReceptorProfile:
    """The param's receptor profile (Phase-1: one channel, weight 1.0)."""
    return ReceptorProfile({channel_for(node_name, key): 1.0})
