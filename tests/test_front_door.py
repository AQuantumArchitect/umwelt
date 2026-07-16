"""First-contact kindness: the supervisor's front-door gate.

The first external hive deployment (2026-07-14) registered three broken worlds and
learned about each one as a worker-exit 500 with the truth only in the daemon log:
a bare-module vocabulary ref (AttributeError deep in the worker), a bare-float
NodeSpec param (TypeError in ParameterBundle.from_dict), and a gamma/hold mismatch
that produced no error at all — just a mute world. These tests pin the fix: the
manifest is resolved and sanity-checked in a fresh subprocess BEFORE anything
spawns (the forge-gate discipline, umweltforge.pipeline.run_validation), and the
refusal is a 400 carrying the subprocess's precise error text.

spawn() is faked (a refused world never reaches it; an accepted one doesn't need a
real worker here — that end-to-end path stays with test_supervisor_smoke.py). The
gate subprocess is REAL in every test below.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from umweltd.supervisor import Supervisor, _Handler

REPO = Path(__file__).resolve().parents[1]


def _supervisor(tmp_path, monkeypatch):
    """A Supervisor with spawn() faked but the front-door gate REAL."""
    monkeypatch.setenv("UMWELTD_HOME", str(tmp_path))
    monkeypatch.delenv("UMWELTD_MAX_WORLDS", raising=False)
    sup = Supervisor()
    monkeypatch.setattr(sup, "spawn", lambda name: 12345)
    return sup


def test_bare_vocabulary_ref_is_refused_with_the_fix_named(tmp_path, monkeypatch):
    # Scar #1: vocabulary="hive_world" (no ':fn') used to become an AttributeError
    # inside the worker. Refused at the door, with the shape AND a suggestion.
    sup = _supervisor(tmp_path, monkeypatch)
    with pytest.raises(ValueError) as exc_info:
        sup.create({"name": "hive", "spec": "x:Y", "vocabulary": "hive_world"})
    msg = str(exc_info.value)
    assert "'hive_world'" in msg
    assert "module:function" in msg
    assert "hive_world:register_vocabulary" in msg
    # a refused world leaves no catalog entry behind
    assert not (sup.worlds_root() / "hive").exists()


def test_unresolvable_spec_ref_is_refused_with_the_subprocess_error(
        tmp_path, monkeypatch):
    sup = _supervisor(tmp_path, monkeypatch)
    with pytest.raises(ValueError) as exc_info:
        sup.create({"name": "ghost", "spec": "no.such.module:SPEC"})
    msg = str(exc_info.value)
    assert "refused before spawn" in msg
    assert "No module named" in msg                 # the subprocess's precise error
    assert not (sup.worlds_root() / "ghost").exists()


def test_bad_param_shape_is_refused_naming_node_and_key(tmp_path, monkeypatch):
    # Scar #2, end to end: a spec authored OUTSIDE the installed packages
    # (spec_path honored, exactly as the worker honors it) whose params value is a
    # bare float. The 400 names the node and the key — not a TypeError traceback.
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "fleet_spec.py").write_text(
        "from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec\n"
        "SPEC = DomainSpec(name='fleet-world', nodes=(\n"
        "    NodeSpec('top', parent=None, kind='root', roles=()),\n"
        "    NodeSpec('fleet', parent='top', roles=('momentum',),\n"
        "             params={'gamma_diss_momentum': 0.01}),),\n"
        "  bindings=(BindingSpec('m', zone='fleet', role='momentum',\n"
        "                        normalizer='binary'),))\n")
    sup = _supervisor(tmp_path, monkeypatch)
    with pytest.raises(ValueError) as exc_info:
        sup.create({"name": "fleet", "spec": "fleet_spec:SPEC",
                    "spec_path": str(ws)})
    msg = str(exc_info.value)
    assert "node 'fleet' param 'gamma_diss_momentum'" in msg
    assert "expected (default, sigma, lo, hi)" in msg
    assert not (sup.worlds_root() / "fleet").exists()


def test_healthy_world_passes_the_front_door(tmp_path, monkeypatch):
    # The gate must refuse broken manifests without taxing healthy ones: the
    # canonical gridworld (via spec_path, since the gate subprocess doesn't share
    # this process's sys.path — a list here, pinning the multi-dir form too).
    ws = tmp_path / "ws_good"
    ws.mkdir()
    (ws / "good_spec.py").write_text(
        "from examples.gridworld.world import gridworld_spec\n"
        "SPEC = gridworld_spec()\n")
    sup = _supervisor(tmp_path, monkeypatch)
    created = sup.create({"name": "grid", "spec": "good_spec:SPEC",
                          "spec_path": [str(ws), str(REPO)]})
    assert created == {"name": "grid", "port": 12345}
    assert (sup.worlds_root() / "grid" / "world.json").exists()


def test_front_door_refusal_is_an_http_400(tmp_path, monkeypatch):
    # The wire contract: the client sees a 400 with the teaching text, never a
    # worker-exit 500 whose truth lives only in the daemon log.
    sup = _supervisor(tmp_path, monkeypatch)
    _Handler.sup = sup
    _Handler.api_key = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        body = json.dumps({"name": "hive", "spec": "x:Y",
                           "vocabulary": "hive_world"}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/worlds", data=body,
                                     method="POST",
                                     headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=10)
        assert exc_info.value.code == 400
        payload = json.loads(exc_info.value.read())
        assert "module:function" in payload["error"]
    finally:
        server.shutdown()
