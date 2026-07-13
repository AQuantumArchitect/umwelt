"""THE DAEMON-PARITY PROOF — umweltd adds nothing and loses nothing.

The founding claim of the service layer: a world driven OVER THE WIRE (worker
subprocess, HTTP, event log, snapshot) ends at the exact field canon hash the same
stream produces replayed library-direct. Plus the event-sourcing contract: kill the
worker, respawn it, and the reloaded world keeps ingesting from its log tail.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from examples.gridworld.world import agent_walk, gridworld_spec, runner_batches, synthesize_rows
from umweltd.client import UmweltClient

FLUSH_SECS = 30.0


def _pin_rngs():
    import random

    import numpy as np
    random.seed(1234)
    np.random.seed(1234)


def _direct_hash(rows) -> str:
    """The library-direct twin: same seeds, same construction, same bucketing."""
    _pin_rngs()
    from umwelt.boot import build_engine
    from umwelt.learning.runner import BrainRunner
    engine = build_engine(spec="world_spec:SPEC", population=False)
    BrainRunner(engine).replay(runner_batches(rows, flush_secs=FLUSH_SECS))
    return engine.field_canon_hash()


@pytest.fixture()
def world(tmp_path):
    """A gridworld world dir + a live worker subprocess; yields (client, dir, proc)."""
    (tmp_path / "world_spec.py").write_text(
        "from examples.gridworld.world import gridworld_spec\nSPEC = gridworld_spec()\n")
    wdir = tmp_path / "worlds" / "grid"
    wdir.mkdir(parents=True)
    (wdir / "world.json").write_text(json.dumps(
        {"name": "grid", "spec": "world_spec:SPEC", "pin_rngs": True,
         "flush_secs": FLUSH_SECS}))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])

    def spawn():
        (wdir / "worker.port").unlink(missing_ok=True)
        proc = subprocess.Popen(
            [sys.executable, "-m", "umweltd.worker", "--dir", str(wdir)], env=env)
        deadline = time.time() + 60
        while not (wdir / "worker.port").exists():
            assert proc.poll() is None, "worker died at boot"
            assert time.time() < deadline, "worker never wrote its port"
            time.sleep(0.1)
        port = int((wdir / "worker.port").read_text())
        return proc, UmweltClient(f"http://127.0.0.1:{port}")

    proc, client = spawn()
    yield client, wdir, proc, spawn, env
    if proc.poll() is None:
        proc.kill()


def test_wire_replay_hash_equals_library_replay(world, tmp_path):
    client, wdir, proc, spawn, env = world
    sys.path.insert(0, str(tmp_path))
    try:
        walk = agent_walk(seed=7, days=0.25)
        rows = synthesize_rows(gridworld_spec(), walk, seed=7)

        assert client.health()["seed_profile"] == "blank"
        result = client.ingest(rows)               # one request = one bucketing pass
        assert result["appended"] == len(rows) and result["batches"] > 10

        wire_hash = client.snapshot()["field_canon_hash"]
        assert wire_hash == _direct_hash(rows), (
            "the daemon changed the physics: wire hash != library hash")

        # the read surface serves: graph_state has nodes, a belief reads finite
        state = client.state()
        assert state and isinstance(state, dict)
        b = client.belief("cell_1_1", "agent_near")
        assert -1.0 <= b["z"] <= 1.0
    finally:
        sys.path.remove(str(tmp_path))


def test_health_reports_resource_sizes(world, tmp_path):
    client, wdir, proc, spawn, env = world
    sys.path.insert(0, str(tmp_path))
    try:
        assert client.health()["events_db_bytes"] == 0
        assert client.health()["snapshot_bytes"] == 0
        walk = agent_walk(seed=7, days=0.1)
        rows = synthesize_rows(gridworld_spec(), walk, seed=7)
        client.ingest(rows)
        client.snapshot()
        health = client.health()
        assert health["events_db_bytes"] > 0
        assert health["snapshot_bytes"] > 0
    finally:
        sys.path.remove(str(tmp_path))


def test_worker_recovers_from_snapshot_plus_log_tail(world, tmp_path):
    client, wdir, proc, spawn, env = world
    sys.path.insert(0, str(tmp_path))
    try:
        walk = agent_walk(seed=7, days=0.25)
        rows = synthesize_rows(gridworld_spec(), walk, seed=7)
        half = len(rows) // 2
        client.ingest(rows[:half])
        client.snapshot()                          # cursor now at the first half
        client.ingest(rows[half:])                 # in the LOG, not in the snapshot
        h_live = client.snapshot()["field_canon_hash"]

        proc.send_signal(signal.SIGTERM)           # graceful: snapshots on the way out
        proc.wait(timeout=30)
        proc2, client2 = spawn()                   # respawn: snapshot + tail replay
        try:
            assert client2.health()["world"] == "grid"
            h_recovered = client2.snapshot()["field_canon_hash"]
            assert h_recovered == h_live, "recovery lost or invented state"
            more = synthesize_rows(
                gridworld_spec(),
                agent_walk(seed=8, days=0.05,
                           start=rows[-1][0].replace("+00:00", "") + "+00:00"),
                seed=8)
            assert client2.ingest(more)["batches"] > 0
        finally:
            proc2.kill()
    finally:
        sys.path.remove(str(tmp_path))