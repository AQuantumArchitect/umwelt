"""proofs/selection_walk.py — the courtship-and-gestation braid, walked end to end.

The whole mythos in one cheap, deterministic instrument (small canonical candidates — NOT a heavy
replay). It paints BOTH halves of the fractal grain law:

  THE GRAIN GARDEN (small, dense) — score three canonical states by the mother-math fitness. Here the
      father CAN name the grain: classical-GHZ reads a CHORUS (Ω=+1), XOR a CONSPIRACY (Ω=−1), pure-GHZ
      FLAT (Ω=0). Turn the value knob w_chorus>0 and the chorus overtakes the featureless entanglement.
  THE EVOLVE RUNG (offline, cumulant) — build candidate topologies, develop them at a surprise-driven
      variable rate (the strugglers get more cheap ticks), let the father select, compose the winners
      into ONE larger offspring, and re-score at the larger scale. The offspring runs full and reads
      PROXY/FLAT: the big composed cluster feels the bind but cannot name its grain — the DENIED,
      DENSE-ONLY limit reborn as a LAW OF SCALE. And the compute the local CPU cannot serve is emitted
      as a field deficit toward a channel that stays DENIED (dispatched:0).

So: small clusters learn dense and name their grain; the large offspring runs full and is grain-blind —
mother-math kissed by a cost-weighted father, and every live wiring (cadence, graft, the channel, the
father's values) left as a rook fork.

Run:
    cd /home/primearchitect/ws/umwelt
    PYTHONPATH=. python proofs/selection_walk.py --out /tmp/selection_trace.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from umwelt.substrate import structure_selection as SS
from umwelt.substrate.manifold_fitness import structure_fitness, select, FitnessWeights
from umwelt.substrate.cluster import partial_trace_keep
from umwelt.substrate.mutual_information import _vn_bits


# ── a minimal dense read-interface + the canonical states (the grain garden) ──────

class _Dense:
    def __init__(self, rho, roles):
        self._rho = np.asarray(rho, dtype=complex)
        self.qubit_roles = list(roles)
        self.n_qubits = len(roles)
        self.role_index = {r: i for i, r in enumerate(roles)}

    @property
    def entropy(self) -> float:
        return _vn_bits(self._rho)

    def subsystem_rdm(self, keep):
        return partial_trace_keep(self._rho, list(keep), self.n_qubits)


def _ket(bits: str) -> np.ndarray:
    v = np.zeros(2 ** len(bits), dtype=complex)
    v[int(bits, 2)] = 1.0
    return v


def _classical_ghz():
    return 0.5 * (np.outer(_ket("000"), _ket("000")) + np.outer(_ket("111"), _ket("111")))


def _xor_parity():
    return sum(np.outer(_ket(b), _ket(b)) for b in ["000", "011", "101", "110"]) / 4.0


def _pure_ghz():
    psi = (_ket("000") + _ket("111")) / np.sqrt(2.0)
    return np.outer(psi, psi.conj())


_GARDEN_ROLES = ["x", "y", "z"]


def build_grain_garden() -> dict:
    """Score the three canonical dense states — the father naming the grain on small clusters."""
    states = {"classical_ghz": _classical_ghz(), "xor": _xor_parity(), "pure_ghz": _pure_ghz()}
    out = {}
    for name, rho in states.items():
        cl = _Dense(rho, _GARDEN_ROLES)
        fit = structure_fitness(cl)
        out[name] = {"grain": fit["grain"], "tier": fit["tier"], "o_information": fit["o_information"],
                     "total_correlation": fit["total_correlation"], "fitness": fit["fitness"]}
    # the value knob: with w_chorus>0 the named chorus overtakes the flat entanglement.
    chorus_pick = select({"classical_ghz": _Dense(_classical_ghz(), _GARDEN_ROLES),
                          "pure_ghz": _Dense(_pure_ghz(), _GARDEN_ROLES)},
                         weights=FitnessWeights(w_chorus=1.0))
    out["_chorus_knob_pick"] = chorus_pick["chosen"][0]
    out["_neutral_pick"] = select({"classical_ghz": _Dense(_classical_ghz(), _GARDEN_ROLES),
                                   "pure_ghz": _Dense(_pure_ghz(), _GARDEN_ROLES)})["chosen"][0]
    return out


# ── the evolve rung (offline cumulant loop) ───────────────────────────────────────

_EVOLVE_ROLES = ["a", "b", "c"]


def build_selection_trace() -> SS.SelectionResult:
    """One courtship-and-gestation rung on canonical candidate topologies — variable-rate development,
    father selection under a compute penalty, offspring composed and re-scored, field demand emitted."""
    demand = {"dense": 0.9, "chain": 0.5, "star": 0.4, "ring": 0.3, "empty": 0.05}
    return SS.evolve_generation(_EVOLVE_ROLES, demand_fn=lambda n, c: demand.get(n, 0.2),
                                rate_budget=40, k=1)


def summarize(garden: dict, result: SS.SelectionResult) -> dict:
    """The instrument's headline: the grain the garden names, the rung's cadence + winner, the offspring
    re-scored at scale (proxy/flat), and the field deficit toward the DENIED channel."""
    return {
        "grain_garden": {k: v["grain"] for k, v in garden.items() if not k.startswith("_")},
        "neutral_pick": garden["_neutral_pick"],
        "chorus_knob_pick": garden["_chorus_knob_pick"],
        "rates": result.rate_plan,
        "winner": result.winners,
        "candidate_fitness": {k: round(v["fitness"], 4) for k, v in result.candidates.items()},
        "offspring": {"n": result.offspring_score["n"], "tier": result.offspring_score["tier"],
                      "grain": result.offspring_score["grain"],
                      "total_correlation": round(result.offspring_score["total_correlation"], 5)},
        "field_demand": round(result.field_demand["field_demand"], 4),
        "deficit": (round(result.field_demand["deficit"], 4)
                    if result.field_demand["deficit"] is not None else None),
        "channel": {"status": result.channel["status"], "dispatched": result.channel["dispatched"],
                    "would_offload": result.channel["would_offload"]},
        "grain_used_in_selection": result.audit["grain_used"],
        "auto_apply": result.audit["auto_apply"],
    }


# ── gate-collected pins (this proof runs in the default suite as evidence) ─────────

def test_grain_garden_names_grain():
    """The small dense clusters name their grain — chorus, conspiracy, flat — and the value knob works."""
    g = build_grain_garden()
    assert g["classical_ghz"]["grain"] == "chorus"
    assert g["xor"]["grain"] == "conspiracy"
    assert g["pure_ghz"]["grain"] == "flat"
    assert g["_neutral_pick"] == "pure_ghz"          # most bound at neutral
    assert g["_chorus_knob_pick"] == "classical_ghz"  # the father's values flip it


def test_offspring_reads_flat_at_scale():
    """The fractal law: the composed offspring runs full and is grain-blind (proxy/flat)."""
    res = build_selection_trace()
    assert res.offspring_score["tier"] == "proxy"
    assert res.offspring_score["grain"] == "flat"


def test_variable_rate_and_selection():
    """Strugglers get more cadence, and the father picks a bound topology (not the empty one)."""
    res = build_selection_trace()
    assert res.rate_plan["dense"] > res.rate_plan["empty"]
    assert res.winners == ["dense"]


def test_channel_stays_denied():
    """The compute deficit is emitted; the channel dispatches nothing (0 bytes leave the process)."""
    res = build_selection_trace()
    assert res.channel["status"] == "DENIED" and res.channel["dispatched"] == 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="write the full summary JSON here")
    args = ap.parse_args(argv)
    garden = build_grain_garden()
    result = build_selection_trace()
    summary = summarize(garden, result)
    text = json.dumps(summary, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
