"""The world worker — one engine, one process, one vocabulary, one HTTP surface.

Boot sequence (the event-sourcing contract):
    1. read world.json; import + call the vocabulary ref (registries are
       process-global — this process speaks exactly one domain's language)
    2. build the engine blank from the spec ref
    3. if snapshot.pkl exists: load it, read cursor.txt
    4. replay events.db rows AFTER the cursor through the production ingest path
    5. bind an ephemeral port, write worker.port, serve

Endpoints (JSON in/out):
    GET  /health            {world, step, last_event_ts, seed_profile}
    POST /events            {"events": [[ts, sid, value, meta|null], ...],
                             "flush_secs": 30.0?} -> append to log, bucket via
                            replay_sensor_batches, ingest each batch. Batch
                            boundaries are per-request: a live pusher sends one
                            request per flush window.
    GET  /state             the canonical graph_state projection
    GET  /beliefs?node=&role=   one belief read (raw bloch; domains apply their own
                            read convention client-side)
    GET  /recommendations   the shadow layer — decisions that would have dispatched
    GET  /bindings          the declared signal vocabulary (what a client may ingest)
    POST /snapshot          save engine + cursor; returns {"field_canon_hash": ...}

SIGTERM snapshots before exit. Every engine touch is serialized by one lock.

world.json knobs: {"name", "spec": "module:ATTR", "vocabulary": "module:function"?,
"flush_secs": 30.0?, "pin_rngs": false?, "spec_path": "/abs/dir" | [dirs]?} —
pin_rngs seeds the process RNGs at boot (the parity proof's determinism switch; live
worlds leave it off). spec_path dirs are prepended to sys.path before the spec ref
imports, so a world authored OUTSIDE the installed packages (a forge workspace)
boots and event-source-recovers identically. Trust model: spec_path is arbitrary
code execution by design — exactly the trust already granted to the spec ref itself
(both import a module into this process).
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import signal
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from umweltd.jsonutil import jsonable
from umweltd.worldstore import WorldDir

logger = logging.getLogger("umweltd.worker")

FLUSH_SECS_DEFAULT = 30.0


def _call_ref(ref: str) -> None:
    module_name, _, attr = ref.partition(":")
    if not module_name or not attr:
        # The supervisor's front-door gate refuses this shape with a 400 before a
        # worker ever spawns; this mirror keeps a directly-booted worker's death
        # note just as precise (the log is all a worker has).
        raise ValueError(
            f"vocabulary ref {ref!r} must be 'module:function' — a bare module "
            f"name gives the worker nothing to call")
    fn = getattr(importlib.import_module(module_name), attr)
    fn()


def _make_webhook_dispatch(url: str):
    """The app-owned transport, as a webhook: every AUTO (non-shadow) Action POSTs to
    `url` as JSON. Shadow stays the law — a spec output dispatches nothing until the
    domain flips it; this is merely where the flipped ones go. Failures are logged,
    never fatal (a dead sink must not kill the world)."""
    import urllib.request

    def _dispatch(action) -> None:
        try:
            req = urllib.request.Request(
                url, data=json.dumps(jsonable(action)).encode(), method="POST",
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
        except Exception as exc:
            logger.warning("webhook dispatch failed: %r", exc)
    return _dispatch


class WorldHost:
    """Owns the engine + the world dir; every public method is lock-serialized."""

    def __init__(self, world_dir: Path):
        self.dir = WorldDir(Path(world_dir))
        self.manifest = self.dir.manifest()
        self.lock = threading.Lock()
        self.last_ts: str = ""
        self.flush_secs = float(self.manifest.get("flush_secs", FLUSH_SECS_DEFAULT))

        # spec_path first: the manifest is the event-sourced world identity, so a
        # generated spec module's home rides in world.json and survives every respawn
        # (watchdog, /start, supervisor restart) with no environment to reconstruct.
        import sys
        raw = self.manifest.get("spec_path")
        for p in ([raw] if isinstance(raw, str) else (raw or [])):
            resolved = str(Path(p).expanduser().resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)

        if self.manifest.get("pin_rngs"):
            import random

            import numpy as np
            random.seed(1234)
            np.random.seed(1234)

        if self.manifest.get("vocabulary"):
            _call_ref(self.manifest["vocabulary"])

        from umwelt.boot import build_engine
        dispatch = (_make_webhook_dispatch(self.manifest["webhook_url"])
                    if self.manifest.get("webhook_url") else None)
        self.engine = build_engine(spec=self.manifest["spec"], population=False,
                                   dispatch=dispatch)

        if self.dir.snapshot_path.exists():
            self.engine.load(str(self.dir.snapshot_path))
        self.last_ts = self.dir.cursor()
        # The gauge discipline (train ≡ deploy): the recovery tail replays under the
        # REPLAY gauge (actuate=0 — a catch-up must never re-dispatch old decisions),
        # then the world is stamped LIVE for serving. A manifest "gauge": "replay"
        # keeps it a pure replay world (offline learners behind the same API).
        from umwelt.boot import set_role
        from umwelt.learning.context import ContextState
        set_role(self.engine, ContextState.replay())
        replayed = self._replay_tail()
        if replayed:
            logger.info("replayed %d tail batches after cursor %r", replayed,
                       self.dir.cursor())
        if self.manifest.get("gauge", "live") != "replay":
            set_role(self.engine, ContextState.live())

    @property
    def name(self) -> str:
        return self.manifest.get("name", self.dir.root.name)

    # ── the ingest paths (both go through umwelt.events bucketing) ──────────────────
    def _ingest_rows(self, rows) -> dict:
        from umwelt.events import replay_sensor_batches
        from umwelt.learning.runner import BrainRunner
        counts = {"batches": 0, "actions": 0}

        def on_batch(n, item, result):
            counts["batches"] = n
            counts["actions"] += len((result or {}).get("actions") or ())

        BrainRunner(self.engine).replay(
            ((readings, bt, conf)
             for bt, readings, conf, _last in replay_sensor_batches(
                 rows, flush_secs=self.flush_secs)),
            on_batch=on_batch)
        if rows:
            self.last_ts = max(self.last_ts, max(r[0] for r in rows))
        return counts

    def _replay_tail(self) -> int:
        from umwelt.events import read_events_since
        rows = read_events_since(self.dir.events_db, self.last_ts) \
            if self.dir.events_db.exists() else []
        return self._ingest_rows(rows)["batches"] if rows else 0

    # ── the API surface ──────────────────────────────────────────────────────────────
    def post_events(self, body: dict) -> dict:
        rows = [tuple(r) + (None,) * (4 - len(r)) for r in body.get("events", ())]
        for r in rows:
            datetime.fromisoformat(r[0])            # fail loudly before logging
        with self.lock:
            self.dir.append_events(rows)            # the log first — always
            counts = self._ingest_rows(rows)
        return {"appended": len(rows), **counts}

    def snapshot(self) -> dict:
        with self.lock:
            tmp = self.dir.snapshot_path.with_suffix(".pkl.tmp")
            self.engine.save(str(tmp))
            tmp.replace(self.dir.snapshot_path)
            self.dir.write_cursor(self.last_ts)
            return {"field_canon_hash": self.engine.field_canon_hash(),
                    "cursor": self.last_ts}

    def state(self) -> dict:
        from umwelt.projection.graph_state import graph_state
        with self.lock:
            return jsonable(graph_state(self.engine))

    def belief(self, node: str, role: str) -> dict:
        with self.lock:
            cluster = self.engine.field.clusters.get(node)
            if cluster is None:
                raise KeyError(f"unknown node {node!r}")
            bloch = cluster.role_bloch(role)
        return {"node": node, "role": role, "bloch": jsonable(bloch),
                "z": float(bloch[2])}

    def recommendations(self) -> list:
        surface = getattr(self.engine, "output_surface", None)
        recs = getattr(surface, "recommendations", None) or ()
        return jsonable(list(recs))

    def bindings(self) -> list:
        """The world's declared signal vocabulary — what a client may ingest.
        Drives the playground's push-readings panel."""
        with self.lock:
            return jsonable(self.engine.sensor_bridge.list_bindings())

    def health(self) -> dict:
        def _size(p: Path) -> int:
            return p.stat().st_size if p.exists() else 0
        return {"world": self.name,
                "step": int(getattr(self.engine, "_step", -1)),
                "last_event_ts": self.last_ts,
                "seed_profile": getattr(self.engine, "seed_profile", None),
                "events_db_bytes": _size(self.dir.events_db),
                "snapshot_bytes": _size(self.dir.snapshot_path)}


class _Handler(BaseHTTPRequestHandler):
    host: WorldHost = None          # injected by serve()

    def log_message(self, fmt, *args):  # replaced by the access log below
        pass

    def _send(self, code: int, payload) -> None:
        self._last_status = code
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _fail(self, exc: Exception) -> None:
        if isinstance(exc, (ValueError, KeyError)):
            self._send(400, {"error": str(exc)})
            return
        # Unexpected — degrade, never crash the world, but don't leak internals
        # (paths, tracebacks) into the response; the real exception goes to the log.
        logger.error("unhandled error on %s: %r", self.path, exc)
        self._send(500, {"error": "internal error"})

    def _access_log(self, method: str, t0: float) -> None:
        logger.info(json.dumps({
            "event": "access", "world": self.host.name, "method": method,
            "path": self.path, "status": getattr(self, "_last_status", None),
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }))

    def do_GET(self):
        t0 = time.time()
        self._last_status = None
        url = urlparse(self.path)
        try:
            if url.path == "/health":
                self._send(200, self.host.health())
            elif url.path == "/state":
                self._send(200, self.host.state())
            elif url.path == "/recommendations":
                self._send(200, self.host.recommendations())
            elif url.path == "/bindings":
                self._send(200, self.host.bindings())
            elif url.path == "/beliefs":
                q = parse_qs(url.query)
                self._send(200, self.host.belief(q["node"][0], q["role"][0]))
            else:
                self._send(404, {"error": f"no route {url.path}"})
        except Exception as exc:                     # degrade, never crash the world
            self._fail(exc)
        finally:
            self._access_log("GET", t0)

    def do_POST(self):
        t0 = time.time()
        self._last_status = None
        url = urlparse(self.path)
        try:
            if url.path == "/events":
                self._send(200, self.host.post_events(self._body()))
            elif url.path == "/snapshot":
                self._send(200, self.host.snapshot())
            else:
                self._send(404, {"error": f"no route {url.path}"})
        except Exception as exc:
            self._fail(exc)
        finally:
            self._access_log("POST", t0)


def serve(world_dir: Path, port: int = 0) -> None:
    host = WorldHost(world_dir)
    _Handler.host = host
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    actual_port = server.server_address[1]
    host.dir.port_path.write_text(str(actual_port))
    logger.info("world %r serving on 127.0.0.1:%d", host.name, actual_port)

    def _shutdown(signum, frame):
        try:
            host.snapshot()
        finally:
            host.dir.port_path.unlink(missing_ok=True)
            raise SystemExit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    try:
        server.serve_forever()
    finally:
        host.dir.port_path.unlink(missing_ok=True)


def main() -> None:
    logging.basicConfig(level=os.environ.get("UMWELTD_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="umweltd world worker")
    ap.add_argument("--dir", required=True, help="the world directory")
    ap.add_argument("--port", type=int, default=0, help="TCP port (0 = ephemeral)")
    args = ap.parse_args()
    serve(Path(args.dir), args.port)


if __name__ == "__main__":
    main()
