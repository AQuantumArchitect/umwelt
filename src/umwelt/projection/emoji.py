"""Glyph State Renderer.

Maps qubit axial states (Bloch vector components) to emoji glyphs for human-readable
visualization of the probability field. Each qubit role MAY have a semantic glyph set,
registered by the domain (`register_role_emoji`) — the engine ships only the neutral
fallback. The z-axis (population) determines the primary glyph; coherence (|x|,|y|)
adds a modifier. Cluster-level correlations (fractal levels 2+) get relationship glyphs.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ── registries: domain vocabulary is registered, never shipped ──────────────────────
# role → {"pos": glyph, "zero": glyph, "neg": glyph, "coherent": glyph}
ROLE_EMOJI: dict[str, dict[str, str]] = {}

# node name → icon shown beside its line in field_summary
NODE_ICONS: dict[str, str] = {}

# Fallback for unregistered roles
_DEFAULT_EMOJI = {
    "pos": "🔵", "zero": "⚪", "neg": "🔴", "coherent": "💠",
}

_DEFAULT_NODE_ICON = "📍"


def register_role_emoji(role: str, mapping: dict[str, str]) -> None:
    """Register a role's glyph set: keys pos/zero/neg (+ optional coherent)."""
    missing = {"pos", "zero", "neg"} - set(mapping)
    if missing:
        raise ValueError(f"role emoji mapping for {role!r} missing keys: {sorted(missing)}")
    ROLE_EMOJI[role] = dict(mapping)


def register_node_icon(node: str, icon: str) -> None:
    """Register the icon shown beside a node's line in field_summary."""
    NODE_ICONS[node] = icon


# Correlation glyphs (fractal levels 2+)
CORRELATION_EMOJI = {
    "strong":      "🔗",  # |correlation| > 0.5
    "moderate":    "〰️",   # 0.2 < |correlation| < 0.5
    "weak":        "·",   # |correlation| < 0.2
    "entangled":   "🕸️",   # genuine multi-body correlation
}

# Purity glyphs
PURITY_EMOJI = {
    "pure":  "✨",  # purity > 0.95
    "mixed": "🌫️",  # purity < 0.5
    "mid":   "💎",  # in between
}


def qubit_emoji(
    bloch: NDArray[np.floating],
    role: str = "",
    threshold: float = 0.3,
) -> str:
    """Convert a Bloch vector (x, y, z) to a glyph string.

    Returns 1-2 glyphs: primary (z-axis state) + optional coherence modifier."""
    x, y, z = float(bloch[0]), float(bloch[1]), float(bloch[2])
    emap = ROLE_EMOJI.get(role, _DEFAULT_EMOJI)

    # Primary: z-axis
    if z > threshold:
        glyph = emap["pos"]
    elif z < -threshold:
        glyph = emap["neg"]
    else:
        glyph = emap["zero"]

    # Coherence modifier: transverse components
    coherence = np.sqrt(x**2 + y**2)
    if coherence > 0.5:
        glyph += emap.get("coherent", "💠")

    return glyph


def cluster_emoji(
    bloch_dict: dict[str, NDArray[np.floating]],
) -> str:
    """Render a full cluster state as a glyph string.

    Args:
        bloch_dict: role → Bloch vector mapping from QubitCluster.all_bloch()

    Returns:
        Space-separated glyph string."""
    parts = []
    for role, bloch in bloch_dict.items():
        parts.append(qubit_emoji(bloch, role))
    return " ".join(parts)


def correlation_emoji(strength: float) -> str:
    """Map a correlation strength to a glyph."""
    s = abs(strength)
    if s > 0.5:
        return CORRELATION_EMOJI["strong"]
    elif s > 0.2:
        return CORRELATION_EMOJI["moderate"]
    else:
        return CORRELATION_EMOJI["weak"]


def purity_emoji(purity: float) -> str:
    """Map purity Tr(ρ²) to a glyph."""
    if purity > 0.95:
        return PURITY_EMOJI["pure"]
    elif purity < 0.5:
        return PURITY_EMOJI["mixed"]
    else:
        return PURITY_EMOJI["mid"]


def field_summary(
    node_emojis: dict[str, str],
    node_purities: dict[str, float],
    bridge_strengths: dict[tuple[str, str], float] | None = None,
) -> str:
    """Render a full-field summary as a compact text block: one line per node
    (icon, per-role glyphs, purity), plus a bridges line when correlations exist."""
    lines = []
    for node, emojis in node_emojis.items():
        icon = NODE_ICONS.get(node, _DEFAULT_NODE_ICON)
        pur = purity_emoji(node_purities.get(node, 1.0))
        lines.append(f"  {icon} {node:10s} {emojis}  {pur}")

    if bridge_strengths:
        bridge_parts = []
        for (na, nb), strength in bridge_strengths.items():
            bridge_parts.append(f"{na}↔{nb} {correlation_emoji(strength)}")
        lines.append(f"  bridges: {' '.join(bridge_parts)}")

    return "\n".join(lines)
