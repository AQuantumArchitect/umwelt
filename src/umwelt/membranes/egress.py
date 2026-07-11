"""Egress — how decisions leave the field.

P2 ships the wire format: `Action`, the one command type every tendril emits and every
dispatcher consumes. The full egress surface (OutputSurface routing, `build_tendrils`
from OutputSpec, the decoder registry) lands in P3. The dispatcher is ALWAYS injected
app code — a callable `dispatch(Action) -> None`; the engine never knows a transport.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Action:
    """A command to send to an output device/channel."""
    actuator_id: str
    command: dict            # device-specific command payload (opaque to the engine)
    node: str                # source node in the world model
    role: str                # source role
    value: int               # committed value (+1 or -1)
    confidence: float        # confidence at collapse time
    reason: str              # why (periodic, query, confidence, "<tendril>_auto", ...)

    def __repr__(self):
        return (
            f"Action({self.actuator_id}: {self.command} "
            f"<- {self.node}.{self.role}={'+'if self.value>0 else '-'} "
            f"conf={self.confidence:.2f} [{self.reason}])"
        )
