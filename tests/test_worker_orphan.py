"""A worker whose world directory is deleted must exit.

The leak this pins: on 2026-08-03 two `umweltd.worker` processes were found
still running ten hours after test_supervisor_lifecycle finished, serving
`<pytest tmp>/home/worlds/grid` -- a directory pytest had already removed.
The supervisor had been hard-killed (Popen.terminate() on Windows is
TerminateProcess, which skips its `finally: sup.stop(name)`), so nothing
reaped the children.

_orphan_watch touches its world_dir through exactly one call -- .exists() --
so the strike logic is tested by scripting that answer rather than by racing
real mkdir/rmdir against a sleep. The timing version of this file had a
genuinely flaky test: with every=0.05 and a 0.07s deletion window, two checks
legitimately land inside one window, so "non-consecutive misses" failed at
random. Scripting removes the clock from the test entirely.
"""
from __future__ import annotations

import ast
import inspect
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from umweltd import worker  # noqa: E402


class _FakeServer:
    """Records shutdown() the way ThreadingHTTPServer would receive it."""

    def __init__(self) -> None:
        self.stopped = threading.Event()

    def shutdown(self) -> None:
        self.stopped.set()


class _ScriptExhausted(Exception):
    """Raised to end an otherwise-infinite watch loop inside a test."""


class _ScriptedDir:
    """Answers .exists() from a fixed script, then stops the loop."""

    def __init__(self, script: list[bool]) -> None:
        self.script = script
        self.calls = 0

    def exists(self) -> bool:
        if self.calls >= len(self.script):
            raise _ScriptExhausted
        answer = self.script[self.calls]
        self.calls += 1
        return answer

    def __str__(self) -> str:  # the warning line formats the path
        return "<scripted world dir>"


def _run(script: list[bool], strikes: int = 2) -> _FakeServer:
    """Drive the watcher synchronously over a scripted existence sequence."""
    server = _FakeServer()
    dir_ = _ScriptedDir(script)
    try:
        worker._orphan_watch(dir_, server, "probe", every=0, strikes=strikes)
    except _ScriptExhausted:
        pass
    return server


def test_worker_exits_when_its_world_directory_is_deleted():
    server = _run([True, True, False, False])
    assert server.stopped.is_set(), "world directory deleted but the worker kept serving"


def test_worker_keeps_serving_while_the_directory_is_there():
    server = _run([True] * 20)
    assert not server.stopped.is_set(), "shut down while the world still existed"


def test_a_single_stat_blip_does_not_kill_the_worker():
    """These directories live under Syncthing. One miss is not a delete."""
    server = _run([True, False, True, True, True])
    assert not server.stopped.is_set(), "one transient miss must not shut the worker down"


def test_strikes_must_be_consecutive():
    """A miss, a hit, a miss is not two strikes -- the counter resets."""
    server = _run([False, True, False, True, False, True, False, True])
    assert not server.stopped.is_set(), "non-consecutive misses accumulated into a kill"


def test_the_strike_count_is_exact():
    """Exactly `strikes` consecutive misses, not one fewer."""
    assert not _run([True, False], strikes=3).stopped.is_set()
    assert not _run([True, False, False], strikes=3).stopped.is_set()
    assert _run([True, False, False, False], strikes=3).stopped.is_set()


def test_the_guard_is_actually_installed_by_serve():
    """Pins the wiring, not just the function -- an unwired guard is no guard."""
    src = inspect.getsource(worker.serve)
    assert "_orphan_watch" in src, "serve() no longer starts the orphan watcher"
    assert worker.ORPHAN_STRIKES >= 2, "a single stat blip would kill live workers"
    assert worker.ORPHAN_CHECK_S > 0


def test_orphan_exit_does_not_snapshot():
    """Snapshotting on the way out would recreate the world that was deleted.

    Checks the parsed body for a .snapshot() CALL, so the prose explaining why
    it must not snapshot cannot satisfy -- or break -- the assertion.
    """
    tree = ast.parse(inspect.getsource(worker._orphan_watch).lstrip())
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "snapshot" not in called, "orphan exit must not write a snapshot"
    assert "shutdown" in called, "orphan exit must actually stop the server"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
