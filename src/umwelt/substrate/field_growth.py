"""field_growth — the surprise-minimizing coupling-growth service (SHADOW-first).

The offline learner that grows the field's web from real data, at rest, on a nightly timer. One run:
  1. Load ONLY the target cumulant cluster from a COPY of the live pickle (lightweight —
     never the full engine state, so it never OOMs alongside the live process).
  2. Scan candidate within-node role↔role edges against the bounded gauge-pruned STREAM TAPE
     (~ms reads, not the raw events firehose), k-fold-validate each on held-out real
     data, and CONSOLIDATE the robust survivors onto the clone's _xy.
  3. PUBLISH the grown couplings to grown_couplings.json + record a field_gauge node (the provable learning
     event). It does NOT touch the live engine — applying the web to the running field (the at-rest graft)
     is a separate, gated step. Shadow-first: discover + record now, actuate once proven.

Niced + bounded + reads-only against the live state → safe to run alongside the live engine.
Run: `python -m umwelt.substrate.field_growth`  (an at-rest timer's entry point).
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s field_growth: %(message)s")
logger = logging.getLogger(__name__)

DATA = Path(os.environ.get("UMWELT_DATA_DIR", "var/data"))
PICKLE = Path(os.environ.get("UMWELT_STATE_PATH", str(DATA / "engine_state.pkl")))
STREAMS = Path(os.environ.get("UMWELT_STREAMS_DB", str(DATA / "streams.db")))
OUT = Path(os.environ.get("UMWELT_GROWN_COUPLINGS", str(DATA / "grown_couplings.json")))
ARCHIVE = os.environ.get("UMWELT_ARCHIVE_ROOT", str(DATA / "archive"))
NODE = os.environ.get("UMWELT_GROWTH_NODE", "")   # "" → caller must name the merged cluster node
LOOKBACK_DAYS = float(os.environ.get("UMWELT_GROWTH_LOOKBACK_DAYS", "14"))
FOLDS = int(os.environ.get("UMWELT_GROWTH_FOLDS", "3"))
PROBE = int(os.environ.get("UMWELT_GROWTH_PROBE", "4"))


def load_cluster(node: str = NODE):
    """The lightweight target cumulant from a COPY of the live pickle — never the full engine state."""
    from umwelt.substrate.cumulant_cluster import CumulantCluster
    snap = DATA / "_growth_snap.pkl"
    shutil.copy2(PICKLE, snap)
    try:
        hs = pickle.load(open(snap, "rb"))["cumulant_states"][node]
    finally:
        snap.unlink(missing_ok=True)
    c = CumulantCluster(hs["zone_name"], hs["qubit_roles"])
    c.load(hs)
    return c


def run() -> dict:
    from umwelt.foresight import dream_topology as dt
    from umwelt.learning import stream_tape as st
    if not STREAMS.exists():
        logger.warning("no stream tape at %s — is UMWELT_STREAM_TAPE on + soaked? skipping", STREAMS)
        return {"grown": 0, "reason": "no_tape"}
    cluster = load_cluster()
    field = SimpleNamespace(clusters={NODE: cluster}, _step=0)
    since = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).timestamp()
    src = lambda ev, dev, _since, limit: st.pull(STREAMS, st.stream_id(ev, dev), since=since, limit=limit)
    pairs = dt.presence_activity_pairs(field, NODE)
    logger.info("growing %s: %d curated edges, tape=%s, lookback=%.0fd, folds=%d probe=%d",
                NODE, len(pairs), STREAMS.name, LOOKBACK_DAYS, FOLDS, PROBE)
    t0 = time.perf_counter()
    rep = dt.grow_couplings(field, "", nodes=[NODE], pairs=pairs, source=src, since=since,
                            consolidate=True, folds=FOLDS, probe_budget=PROBE,
                            log=lambda m: logger.info(m))
    node_rep = rep.get(NODE, {})
    grown = node_rep.get("applied", [])
    logger.info("done in %.1fs: scanned=%s probed=%s grown=%d",
                time.perf_counter() - t0, node_rep.get("scanned"), node_rep.get("probed"), len(grown))
    if grown:
        OUT.write_text(json.dumps({"ts": datetime.now().isoformat(), "node": NODE, "couplings": grown}, indent=2))
        logger.info("published %d grown coupling(s) → %s (shadow; graft-path is a separate gated step)",
                    len(grown), OUT)
        try:                                          # the provable learning event in the gauge/git ledger
            from umwelt.substrate import field_gauge
            n = field_gauge.record_field_node(field, archive_root=ARCHIVE, note=f"field_growth {NODE}")
            logger.info("field_gauge node: %s", n["label"] if n else "unchanged")
        except Exception as e:
            logger.warning("field_gauge record skipped: %s", e)
    return {"grown": len(grown), "scanned": node_rep.get("scanned")}


if __name__ == "__main__":
    run()
