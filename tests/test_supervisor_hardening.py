"""umweltd ops hardening: constant-time auth, the crash watchdog (and its backoff +
stop-means-stop guarantee), the max-worlds guardrail, and the JSON access log.

All in-process against `Supervisor`/`_Handler` directly — `spawn()` is faked so these
run fast and don't need a real worker subprocess (that end-to-end path stays covered
by test_supervisor_smoke.py and test_daemon_parity.py).
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from umweltd.supervisor import Supervisor, _Handler
from umweltd.worldstore import WorldDir


class _FakeProc:
    """Stands in for subprocess.Popen so the watchdog tests never spawn a real worker."""

    def __init__(self, alive: bool = True, returncode: int = 1):
        self._alive = alive
        self.returncode = None if alive else returncode

    def poll(self):
        return None if self._alive else self.returncode

    def terminate(self):
        self._alive = False

    def wait(self, timeout=None):
        pass


def _supervisor(tmp_path, monkeypatch):
    """A Supervisor with spawn() faked to record calls and mark the world alive.
    The front-door spec gate is faked too ('x:Y' resolves nothing real) — its real
    subprocess behavior is pinned by tests/test_front_door.py."""
    monkeypatch.setenv("UMWELTD_HOME", str(tmp_path))
    monkeypatch.delenv("UMWELTD_MAX_WORLDS", raising=False)
    sup = Supervisor()
    spawn_calls: list[str] = []

    def fake_spawn(name):
        spawn_calls.append(name)
        sup.procs[name] = _FakeProc(alive=True)
        return 12345

    monkeypatch.setattr(sup, "spawn", fake_spawn)
    monkeypatch.setattr(sup, "_gate_spec", lambda body: None)
    return sup, spawn_calls


def _declare(sup, name):
    WorldDir(sup.worlds_root() / name).write_manifest({"name": name, "spec": "x:Y"})


# ── auth ──────────────────────────────────────────────────────────────────────────

def test_authorized_rejects_wrong_or_missing_key_and_accepts_the_right_one():
    h = _Handler.__new__(_Handler)          # bypass socket-requiring __init__
    h.api_key = "secret"
    h.headers = {}
    assert not h._authorized()
    h.headers = {"X-API-Key": "wrong"}
    assert not h._authorized()
    h.headers = {"X-API-Key": "secret"}
    assert h._authorized()
    h.api_key = None                        # no key configured = open (localhost trust)
    assert h._authorized()


# ── watchdog ──────────────────────────────────────────────────────────────────────

def test_watchdog_restarts_a_crashed_world(tmp_path, monkeypatch):
    sup, spawn_calls = _supervisor(tmp_path, monkeypatch)
    sup.desired.add("w1")
    sup.procs["w1"] = _FakeProc(alive=False)
    sup.watchdog_tick()
    assert spawn_calls == ["w1"]
    assert sup.procs["w1"].poll() is None   # the fake respawn is alive again


def test_watchdog_ignores_a_healthy_world(tmp_path, monkeypatch):
    sup, spawn_calls = _supervisor(tmp_path, monkeypatch)
    sup.desired.add("w1")
    sup.procs["w1"] = _FakeProc(alive=True)
    sup.watchdog_tick()
    assert spawn_calls == []


def test_watchdog_gives_up_after_repeated_crashes(tmp_path, monkeypatch):
    sup, spawn_calls = _supervisor(tmp_path, monkeypatch)
    sup.desired.add("w1")
    for _ in range(5):
        sup.procs["w1"] = _FakeProc(alive=False)
        sup.watchdog_tick()
    assert "w1" in sup._watchdog_disabled
    n_before = len(spawn_calls)
    sup.procs["w1"] = _FakeProc(alive=False)
    sup.watchdog_tick()
    assert len(spawn_calls) == n_before      # no further restart attempts


def test_stop_removes_desired_so_watchdog_leaves_it_down(tmp_path, monkeypatch):
    sup, spawn_calls = _supervisor(tmp_path, monkeypatch)
    _declare(sup, "w1")
    sup.procs["w1"] = _FakeProc(alive=True)
    sup.desired.add("w1")
    sup.stop("w1")
    assert "w1" not in sup.desired
    sup.procs["w1"] = _FakeProc(alive=False)   # the just-stopped process reads as exited
    sup.watchdog_tick()
    assert spawn_calls == []                   # an intentional stop is never a "crash"


def test_manual_start_clears_watchdog_backoff(tmp_path, monkeypatch):
    sup, spawn_calls = _supervisor(tmp_path, monkeypatch)
    _declare(sup, "w1")
    sup._watchdog_disabled.add("w1")
    sup.procs["w1"] = _FakeProc(alive=False)
    sup.start("w1")
    assert "w1" not in sup._watchdog_disabled
    assert "w1" in sup.desired


# ── max-worlds guardrail ─────────────────────────────────────────────────────────

def test_create_refuses_past_max_worlds(tmp_path, monkeypatch):
    sup, spawn_calls = _supervisor(tmp_path, monkeypatch)
    monkeypatch.setenv("UMWELTD_MAX_WORLDS", "1")
    sup.create({"name": "w1", "spec": "x:Y"})
    with pytest.raises(ValueError, match="world cap reached"):
        sup.create({"name": "w2", "spec": "x:Y"})
    assert spawn_calls == ["w1"]


def test_create_unlimited_when_max_worlds_unset(tmp_path, monkeypatch):
    sup, spawn_calls = _supervisor(tmp_path, monkeypatch)
    sup.create({"name": "w1", "spec": "x:Y"})
    sup.create({"name": "w2", "spec": "x:Y"})
    assert spawn_calls == ["w1", "w2"]


# ── access log ────────────────────────────────────────────────────────────────────

def _wait_for_access_log(caplog, timeout: float = 2.0) -> list[str]:
    """The access log line is emitted by the server's own request-handling thread,
    a beat after the client sees the response — poll briefly rather than assume
    same-instant ordering across threads."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        lines = [r.getMessage() for r in caplog.records
                if r.getMessage().startswith('{"event": "access"')]
        if lines:
            return lines
        time.sleep(0.02)
    return []


def test_access_log_emits_parseable_json(tmp_path, monkeypatch, caplog):
    sup, _spawn_calls = _supervisor(tmp_path, monkeypatch)
    _Handler.sup = sup
    _Handler.api_key = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        with caplog.at_level(logging.INFO, logger="umweltd.supervisor"):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            access_lines = _wait_for_access_log(caplog)
        assert access_lines, caplog.text
        rec = json.loads(access_lines[-1])
        assert rec["method"] == "GET"
        assert rec["path"] == "/health"
        assert rec["status"] == 200
        assert rec["latency_ms"] >= 0.0
    finally:
        server.shutdown()


def test_unauthorized_request_is_401_and_logged(tmp_path, monkeypatch, caplog):
    sup, _spawn_calls = _supervisor(tmp_path, monkeypatch)
    _Handler.sup = sup
    _Handler.api_key = "secret"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        with caplog.at_level(logging.INFO, logger="umweltd.supervisor"):
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            assert exc_info.value.code == 401
            access_lines = _wait_for_access_log(caplog)
        assert access_lines, caplog.text
        assert json.loads(access_lines[-1])["status"] == 401
    finally:
        server.shutdown()
        _Handler.api_key = None
