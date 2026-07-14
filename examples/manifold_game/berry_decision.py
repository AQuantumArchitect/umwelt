#!/usr/bin/env python3
"""Berry-phase decision demo — a choice that flips because of winding.

CLAIMS.md has carried this as EVALUATION OWED: "geometric phase as a process
clock that can GATE a downstream decision (a choice that flips because of
winding)". This demo pays it with REAL tapes from a foreign game's real
geometry: SpaceWheat integrates the solid angle of a qubit's Bloch path
(L'Huilier, per slice) in its native engine; its harvest law fires at
|γ| ≥ 2π. The two tapes under data/ were recorded by the game's
`berry_tape_drive.py` on the same biome, same duration, same tracking —
only the PATH differs:

    berry_loop.json   Hadamard off the pole, then track: the evolution
                      traces a genuine loop; γ winds past 2π.
    berry_still.json  track only: the qubit sits at its pole; the path
                      encloses nothing; γ stays ≈ 0.

A finding the tapes forced (kept, not hidden): the game's world NEVER sits
still — its boot deliberately kicks every stationary state, so even the
pole-hugging control winds slowly (each orbit encloses a small cap). The
honest contrast is therefore RATE: the loop's gate opens ~8× sooner, and at
any fixed time budget the two processes yield OPPOSITE choices. The decision
function is deliberately one line — the point is not the rule but WHAT it
reads: not elapsed time, not tick count, but accumulated geometric phase.
The tapes also drive this repo's BerryStamper so the stamps land on the
umwelt berry-tape machinery.

Run from the repo root:  python3 examples/manifold_game/berry_decision.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from umwelt.clocks.berry_tape import BerryStamper

HERE = Path(__file__).resolve().parent


def load_tape(name: str) -> dict:
    return json.loads((HERE / "data" / name).read_text(encoding="utf-8"))


def harvest_decision(gamma: float, threshold: float) -> bool:
    """The gate: harvest exactly when the process has WOUND far enough."""
    return abs(gamma) >= threshold


def run_tape(tape: dict) -> dict:
    stamper = BerryStamper()
    flipped_at = None
    for t_phrames, gamma in tape["samples"]:
        stamper.stamp(gamma, "berry_sample", tape["kind"],
                      "t=%d" % t_phrames)
        if flipped_at is None and harvest_decision(gamma, tape["ripe_threshold"]):
            flipped_at = t_phrames
            stamper.stamp(gamma, "decision_flip", tape["kind"],
                          "harvest gate opened at t=%d" % t_phrames)
    final_gamma = tape["samples"][-1][1]
    return {"kind": tape["kind"], "flipped_at": flipped_at,
            "final_gamma": final_gamma, "stamps": len(stamper.tape)}


def decision_at_budget(tape: dict, budget: int) -> bool:
    """The choice an agent with `budget` phrames would make, reading only γ."""
    gamma = 0.0
    for t_phrames, g in tape["samples"]:
        if t_phrames > budget:
            break
        gamma = g
    return harvest_decision(gamma, tape["ripe_threshold"])


def main() -> int:
    loop_tape = load_tape("berry_loop.json")
    still_tape = load_tape("berry_still.json")
    loop = run_tape(loop_tape)
    still = run_tape(still_tape)
    budget = loop_tape["samples"][1][0]  # one sampling interval of play
    for r in (loop, still):
        print(f"{r['kind']:5s}: final γ = {r['final_gamma']:+9.4f}, gate opened at "
              f"t={r['flipped_at']} ({r['stamps']} stamps on the berry tape)")

    at_budget = (decision_at_budget(loop_tape, budget),
                 decision_at_budget(still_tape, budget))
    print(f"at the same budget t={budget}: loop → "
          f"{'HARVEST' if at_budget[0] else 'wait'}, control → "
          f"{'HARVEST' if at_budget[1] else 'wait'}")

    assert loop["flipped_at"] is not None, "the loop tape must open the harvest gate"
    assert at_budget[0] and not at_budget[1], (
        "at a fixed budget the choice must flip on winding alone")
    assert still["flipped_at"] is None or still["flipped_at"] >= 4 * loop["flipped_at"], (
        "the pole-hugging control must wind at least 4x slower")
    print("\nsame machinery, same clock — the geometry of the path decided.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
