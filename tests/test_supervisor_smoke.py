"""Supervisor smoke: create a world over the API, proxy to it, stop/start it."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from umweltd.client import UmweltClient


def _wait_http(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(0.1)
    raise TimeoutError(url)


@pytest.mark.slow
def test_supervisor_lifecycle(tmp_path):
    (tmp_path / "world_spec.py").write_text(
        "from proofs._gridworld import gridworld_spec\nSPEC = gridworld_spec()\n")
    env = os.environ.copy()
    env["UMWELTD_HOME"] = str(tmp_path / "home")
    env["UMWELTD_API_KEY"] = "test-key-123"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])
    port = 7099
    sup = subprocess.Popen(
        [sys.executable, "-m", "umweltd.supervisor", "--port", str(port)], env=env)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 30                     # wait for 401 (auth is up)
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"{base}/health", timeout=2)
                raise AssertionError("keyless request must be rejected")
            except urllib.error.HTTPError as e:
                assert e.code == 401
                break
            except Exception:
                time.sleep(0.1)
        client = UmweltClient(base, world="grid", api_key="test-key-123")

        created = UmweltClient(base, api_key="test-key-123").create_world(
            "grid", spec="world_spec:SPEC", pin_rngs=True)
        assert created["name"] == "grid" and created["port"] > 0

        # proxied round trip: health + one ingest through the supervisor
        assert client.health()["seed_profile"] == "blank"
        rows = [["2026-01-05T00:00:00+00:00", "sight_cell_0_0", "1.0", None],
                ["2026-01-05T00:01:00+00:00", "resource_cell_0_0", "5.0", None]]
        assert client.ingest(rows)["appended"] == 2

        # stop snapshots + start recovers (keyed requests)
        key_hdr = {"X-API-Key": "test-key-123"}
        req = urllib.request.Request(f"{base}/worlds/grid/stop", data=b"{}",
                                     method="POST", headers=key_hdr)
        assert json.loads(urllib.request.urlopen(req).read())["running"] is False
        req = urllib.request.Request(f"{base}/worlds/grid/start", data=b"{}",
                                     method="POST", headers=key_hdr)
        assert json.loads(urllib.request.urlopen(req).read())["running"] is True
        assert client.health()["world"] == "grid"
    finally:
        sup.terminate()
        sup.wait(timeout=30)