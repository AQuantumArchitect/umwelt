"""The spec_path contract — a world authored OUTSIDE the installed packages boots,
serves, and event-source-recovers through the manifest alone.

This is the daemon-side half of the forge pipeline: `world.json` carries
`spec_path` (the generated module's home), the worker prepends it to sys.path
before the spec ref imports, and because the manifest IS the event-sourced world
identity, a respawn (watchdog, /start, supervisor restart) needs no environment
reconstruction. The worker subprocess here is deliberately started with a
PYTHONPATH that does NOT contain the module's directory — only the manifest knows.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from umweltd.client import UmweltClient

WORLD_MODULE = """\
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec

SPEC = DomainSpec(
    name="forge-tiny",
    nodes=(
        NodeSpec("top", parent=None, kind="root", roles=()),
        NodeSpec("area_a", parent="top", roles=("level",)),
    ),
    bindings=(BindingSpec("sig_a", zone="area_a", role="level",
                          normalizer="binary"),),
)
"""


@pytest.fixture()
def world(tmp_path):
    """A world whose spec module lives OFF the worker's PYTHONPATH — reachable only
    via the manifest's spec_path. Yields (client, wdir, proc, spawn)."""
    module_home = tmp_path / "forge_home"
    module_home.mkdir()
    (module_home / "world_forge_tiny.py").write_text(WORLD_MODULE)

    wdir = tmp_path / "worlds" / "tiny"
    wdir.mkdir(parents=True)
    (wdir / "world.json").write_text(json.dumps(
        {"name": "tiny", "spec": "world_forge_tiny:SPEC",
         "spec_path": str(module_home), "pin_rngs": True}))

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(          # note: module_home NOT included
        [str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])

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
    yield client, wdir, proc, spawn
    if proc.poll() is None:
        proc.kill()


@pytest.mark.slow
def test_spec_path_boot_ingest_and_event_sourced_respawn(world):
    client, wdir, proc, spawn = world

    # Boots blank from the off-path module, through the manifest alone.
    h = client.health()
    assert h["world"] == "tiny" and h["seed_profile"] == "blank"

    # The declared vocabulary is readable over the wire (drives the playground).
    [binding] = client.bindings()
    assert binding["sensor_id"] == "sig_a"
    assert binding["node"] == "area_a" and binding["role"] == "level"

    # Ingests through the production path.
    rows = [(f"2026-01-05T12:{m:02d}:00+00:00", "sig_a", float(m % 2), None)
            for m in range(10)]
    result = client.ingest(rows)
    assert result["appended"] == 10 and result["batches"] > 0
    z0 = client.belief("area_a", "level")["z"]
    step0 = client.health()["step"]
    assert step0 > 0

    # SIGTERM → snapshot; respawn = fresh process, manifest read, spec_path
    # re-inserted, module re-imported, snapshot + log tail replayed.
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=30)
    proc2, client2 = spawn()
    try:
        h2 = client2.health()
        assert h2["world"] == "tiny" and h2["step"] >= step0
        assert client2.belief("area_a", "level")["z"] == pytest.approx(z0)
        # ...and keeps ingesting.
        more = [("2026-01-05T13:00:00+00:00", "sig_a", 1.0, None)]
        assert client2.ingest(more)["appended"] == 1
    finally:
        proc2.kill()


def test_missing_spec_path_dir_fails_loudly(tmp_path):
    wdir = tmp_path / "worlds" / "ghost"
    wdir.mkdir(parents=True)
    (wdir / "world.json").write_text(json.dumps(
        {"name": "ghost", "spec": "no_such_world_module:SPEC",
         "spec_path": str(tmp_path / "nowhere")}))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])
    proc = subprocess.Popen(
        [sys.executable, "-m", "umweltd.worker", "--dir", str(wdir)],
        env=env, stderr=subprocess.DEVNULL)
    assert proc.wait(timeout=60) != 0, "worker must exit nonzero on an unimportable spec"
    assert not (wdir / "worker.port").exists()
