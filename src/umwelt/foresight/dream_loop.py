"""Dream loop — the package-side integration that wires generative dreaming into the hindbrain.

The hindbrain (experiments/hindbrain.py, the lineage harness) replays the live event backlog; when it
finds NO new events, the world is at rest — which is exactly when dreaming should run (docs/MIND.md: feed
the learning-stream counterfactual shadow cassettes when there's nothing live to learn from). This module
is the thin seam the hindbrain calls in that branch, so the dreaming machinery lives in the deployed
package and the hindbrain edit stays a single call.

SHADOW BY DEFAULT (b9.1): gated OFF via UMWELT_DREAM (default "0"), so it ships WIRED BUT INERT — zero
behavior change, the gate logic validated without paying the clone cost in production. b9.7 graduates it
(UMWELT_DREAM=1, consolidation opt-in, validated on real cassettes). The dream learns on a deepcopy clone
and never actuates; with consolidation off (the default) the live pickle is untouched — pure shadow.

It is also where the BPU dispatch node gets its first real client: the dream engine is handed
`bpu_dispatch.dispatcher()` so the replay matmul is gauge-tracked + (eventually) offloadable.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable

from umwelt.foresight import bpu_dispatch
from umwelt.foresight.dreaming import DREAM_KINDS, reservoir_dream_engine, should_dream

log = logging.getLogger("dream_loop")

# Shadow floor: wired but inert until explicitly enabled (b9.7). Consistent with the opt-in-not-gated floor.
DREAM_ENABLED = os.environ.get("UMWELT_DREAM", "0") == "1"
# Consolidation (fold the generalized gradient back into the live field) is a SEPARATE, stricter opt-in.
DREAM_CONSOLIDATE = os.environ.get("UMWELT_DREAM_CONSOLIDATE", "0") == "1"
DREAM_N = int(os.environ.get("UMWELT_DREAM_N", "5"))
# Rest threshold: dream when live surprise is at/under this (nothing live to learn). Also fires in siesta.
DREAM_SURPRISE_REST = float(os.environ.get("UMWELT_DREAM_SURPRISE_REST", "0.15"))
# Cooldown (b9.7 graduation): after a session, hold off this long — so a rest window triggers ONE session,
# not one per ~45s hindbrain idle cycle. A dream is consolidation, not a busy-loop. 0 disables.
DREAM_COOLDOWN_S = float(os.environ.get("UMWELT_DREAM_COOLDOWN_S", "3600"))
# Topology mutation (grow the field's couplings from real data) — a SEPARATE opt-in sub-organ of the dream.
# Default OFF; proposes only unless DREAM_CONSOLIDATE is also on (then it writes the surviving J to _xy, which
# persists through the pickle). Reads the live events.db over a lookback window; selection is PURELY held-out
# forecast-surprise reduction (k-fold CV) — no synthetic shadows. See dream_topology / coupling_learn.
DREAM_TOPOLOGY = os.environ.get("UMWELT_DREAM_TOPOLOGY", "0") == "1"
DREAM_TOPOLOGY_LOOKBACK_DAYS = float(os.environ.get("UMWELT_DREAM_TOPOLOGY_LOOKBACK_DAYS", "14"))
_last_dream = {"t": None}      # monotonic seconds of the last session (None = never)


def _cooling_down(now: float) -> bool:
    return (_last_dream["t"] is not None and DREAM_COOLDOWN_S > 0
            and (now - _last_dream["t"]) < DREAM_COOLDOWN_S)


def _events_db_path() -> str:
    return os.environ.get("UMWELT_DB_PATH") or "umwelt_events.db"


def _grow_topology(reservoir) -> dict | None:
    """Run the topology-mutation organ over the live field on the dream cadence (at rest, cooldown-gated).
    Side-effect-free unless DREAM_CONSOLIDATE (then surviving couplings are written to _xy). Returns a report."""
    from datetime import datetime, timedelta, timezone
    from umwelt.foresight import dream_topology
    since = (datetime.now(timezone.utc) - timedelta(days=DREAM_TOPOLOGY_LOOKBACK_DAYS)).isoformat()
    return dream_topology.grow_couplings(reservoir.field, _events_db_path(), consolidate=DREAM_CONSOLIDATE,
                                         since=since, log=lambda m: log.info(m))


def _topology_summary(report: dict) -> dict:
    applied = [e for n in report.values() for e in n.get("applied", [])]
    proposed = [e for n in report.values() for e in n.get("proposals", [])]
    return {"nodes": len(report), "proposed": len(proposed), "applied": len(applied),
            "edges": [f"{e['node']}:{e['leader']}→{e['follower']} J={e['J']:+.2f} "
                      f"(surprise −{e['surprise_reduction']:.2f})" for e in (applied or proposed)]}


def maybe_dream(reservoir, cassette_fn: Callable[[], list], *, solar_phase: float | None = None,
                live_surprise: float | None = None, publish: Callable[[dict], None] | None = None,
                now: float | None = None):
    """Dream IF enabled AND at rest AND not within the cooldown. `cassette_fn` is called only when we
    actually dream (so the — possibly expensive — cassette build is skipped on the common path). Returns
    the DreamSession, or None when disabled / not resting / cooling down / no cassette. Consolidation is
    the stricter UMWELT_DREAM_CONSOLIDATE opt-in; default OFF = pure shadow (the live pickle is untouched)."""
    if not (DREAM_ENABLED or DREAM_TOPOLOGY):
        return None
    t = time.monotonic() if now is None else now
    if _cooling_down(t):
        return None
    if not should_dream(solar_phase=solar_phase, live_surprise=live_surprise,
                        surprise_rest=DREAM_SURPRISE_REST):
        return None

    session = None
    if DREAM_ENABLED:
        try:
            cassette = cassette_fn()
        except Exception as e:
            log.warning("dream cassette build failed: %s", e)
            cassette = None
        if cassette:
            eng = reservoir_dream_engine(reservoir, apply_gradient=DREAM_CONSOLIDATE,
                                         dispatcher=bpu_dispatch.dispatcher())
            session = eng.dream(cassette, n_dreams=DREAM_N, kinds=DREAM_KINDS, consolidate=DREAM_CONSOLIDATE)
            summary = session.summary()
            log.info("dream session: %s", summary)
            if publish is not None:
                try:
                    publish(summary)
                except Exception:
                    pass

    if DREAM_TOPOLOGY:
        try:
            report = _grow_topology(reservoir)
            if report:
                topo = _topology_summary(report)
                log.info("dream topology: %s", topo)
                if publish is not None:
                    try:
                        publish({"topology": topo})
                    except Exception:
                        pass
        except Exception as e:
            log.warning("dream topology failed: %s", e)

    if session is None and not DREAM_TOPOLOGY:
        return None
    _last_dream["t"] = t                                  # start the cooldown (b9.7)
    return session


def status() -> dict:
    """For /api and logs: is dreaming wired, enabled, consolidating?"""
    return {"enabled": DREAM_ENABLED, "consolidate": DREAM_CONSOLIDATE,
            "n_dreams": DREAM_N, "surprise_rest": DREAM_SURPRISE_REST,
            "cooldown_s": DREAM_COOLDOWN_S, "topology": DREAM_TOPOLOGY,
            "topology_lookback_days": DREAM_TOPOLOGY_LOOKBACK_DAYS}
