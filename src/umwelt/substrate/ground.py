"""
Classical World Model — the ground layer.

Persistent commitments about the state of reality. Updated by
selective collapse from the quantum probability field.

The quantum field holds beliefs (continuous probability).
The world model holds commitments (discrete facts).

    Field (sky): "an occupancy-like role is probably +1 with 87% confidence"
    Model (ground): "that region IS occupied" (committed at step 150)

The gap between field and model is where intelligence lives:
the field anticipates, the model commits, actions follow commitments.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Data structures
# ============================================================================

@dataclass(frozen=True)
class Transition:
    """A state change in the world model."""
    from_state: int
    to_state: int
    node: str
    role: str
    step: int
    confidence: float
    reason: str = ""

    def __repr__(self):
        arrow = "\u2191" if self.to_state > 0 else "\u2193"
        return f"{self.node}.{self.role} {arrow} (conf={self.confidence:.2f} @{self.step} [{self.reason}])"


@dataclass
class CommittedState:
    """A single collapsed value — what the system believes is true."""
    value: int          # +1 or -1 (discrete roles); sign(z) only, for display
    confidence: float   # certainty at collapse time
    step: int           # when last updated
    reason: str         # why it collapsed
    # Analog roles (periodic drivers, temperature, ...) are continuous — committing
    # them as ±1 is a lie. When set, `analog` holds the real committed scalar
    # (Bloch z) and `bloch` the full (x, y, z) position; no ±1 transition is
    # emitted. None for discrete roles (back-compat: old checkpoints, existing
    # readers see the unchanged `value`).
    analog: float | None = None
    bloch: tuple[float, float, float] | None = None


@dataclass
class NodeState:
    """Classical state of one node in the world model."""
    node: str
    states: dict[str, CommittedState] = field(default_factory=dict)

    @property
    def bitstring(self) -> str:
        return "".join("+" if s.value > 0 else "-" for s in self.states.values())

    def asserted(self, role: str) -> bool:
        """True when this node's committed state for `role` is in the + pole."""
        s = self.states.get(role)
        return s is not None and s.value > 0


# Glyphs for collapsed states: role -> (positive glyph, negative glyph).
# EMPTY by default — a domain registers its vocabulary; unregistered roles render
# with the neutral up/down fallback at the read sites.
COLLAPSE_EMOJI: dict[str, tuple[str, str]] = {}


def register_collapse_emoji(role: str, pos: str, neg: str) -> None:
    COLLAPSE_EMOJI[role] = (pos, neg)


class ClassicalWorldModel:
    """
    The ground layer — persistent classical commitments.

    Updated incrementally by selective collapse from the quantum field.
    The LLM and action system read from this, not the field directly.
    """

    def __init__(self):
        self._nodes: dict[str, NodeState] = {}
        self._pending_transitions: list[Transition] = []
        self._all_transitions: list[Transition] = []

    def update(
        self,
        node_name: str,
        role: str,
        value: int,
        confidence: float,
        step: int,
        reason: str = "",
        analog: float | None = None,
        bloch: tuple[float, float, float] | None = None,
    ):
        """Update a single committed state. Detects transitions.

        Analog roles (analog is not None) commit a continuous value/position
        and emit NO ±1 transition — a sun crossing the horizon or a drifting
        temperature is not an on/off event, and binarizing it spams the
        transition log with meaningless -1↔+1 flips.
        """
        if node_name not in self._nodes:
            self._nodes[node_name] = NodeState(node=node_name)

        ns = self._nodes[node_name]
        old = ns.states.get(role)

        if analog is None and old is not None and old.value != value:
            t = Transition(
                from_state=old.value,
                to_state=value,
                node=node_name,
                role=role,
                step=step,
                confidence=confidence,
                reason=reason,
            )
            self._pending_transitions.append(t)
            self._all_transitions.append(t)
            if len(self._all_transitions) > 10000:
                self._all_transitions = self._all_transitions[-5000:]
            logger.info("Transition: %s", t)

        ns.states[role] = CommittedState(
            value=value,
            confidence=confidence,
            step=step,
            reason=reason,
            analog=analog,
            bloch=bloch,
        )

    def get(self, node_name: str) -> NodeState | None:
        return self._nodes.get(node_name)

    def get_value(self, node_name: str, role: str) -> int | None:
        ns = self._nodes.get(node_name)
        if ns is None:
            return None
        cs = ns.states.get(role)
        return cs.value if cs else None

    def pop_transitions(self) -> list[Transition]:
        """Return and clear pending transitions."""
        t = list(self._pending_transitions)
        self._pending_transitions.clear()
        return t

    def asserted_nodes(self, role: str) -> list[str]:
        """Nodes whose committed state for `role` is in the + pole."""
        return [n for n, ns in self._nodes.items() if ns.asserted(role)]

    def emoji_summary(self) -> str:
        """Emoji representation of all committed states."""
        lines = []
        for name, ns in self._nodes.items():
            emojis = []
            for role, cs in ns.states.items():
                pos, neg = COLLAPSE_EMOJI.get(role, ("\u2b06\ufe0f", "\u2b07\ufe0f"))
                emojis.append(pos if cs.value > 0 else neg)
            emoji_str = " ".join(emojis)
            lines.append(f"  {name:15s} {emoji_str}  [{ns.bitstring}]")
        return "\n".join(lines)

    def context(self) -> dict[str, Any]:
        """Full context dict for LLM / action system."""
        if not self._nodes:
            return {"status": "no_data"}

        return {
            "nodes": {
                name: {
                    "bitstring": ns.bitstring,
                    "states": {
                        role: {
                            "value": cs.value,
                            "confidence": cs.confidence,
                            "emoji": COLLAPSE_EMOJI.get(
                                role, ("\u2b06\ufe0f", "\u2b07\ufe0f")
                            )[0 if cs.value > 0 else 1],
                            "uncertain": cs.confidence < 0.3,
                            "step": cs.step,
                            "reason": cs.reason,
                            # Present only for analog roles (periodic drivers/temp/...):
                            # the real committed position, so the display can
                            # render it on the sphere instead of as a pole.
                            **({"analog": cs.analog,
                                "bloch": list(cs.bloch) if cs.bloch else None}
                               if cs.analog is not None else {}),
                        }
                        for role, cs in ns.states.items()
                    },
                }
                for name, ns in self._nodes.items()
            },
        }

    def reset(self):
        self._nodes.clear()
        self._pending_transitions.clear()
