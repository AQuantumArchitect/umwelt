"""The umweltd supervisor — world catalog, worker lifecycle, and a thin proxy.

One world = one worker process (vocabulary registries are process-global). The
supervisor owns the catalog under UMWELTD_HOME (default ~/.umweltd), spawns a worker
per world, and proxies /worlds/<name>/<rest> to the right worker so clients need one
base URL.

    GET    /health                      supervisor + per-world worker liveness
    GET    /worlds                      the catalog
    POST   /worlds                      {"name", "spec", "vocabulary"?, ...} ->
                                        create dir + manifest, spawn worker
    POST   /worlds/<name>/stop          SIGTERM the worker (it snapshots on the way out)
    POST   /worlds/<name>/start         respawn a stopped world
    *      /worlds/<name>/<rest>        proxied verbatim to the worker

Run: python -m umweltd.supervisor [--port 7071]. Existing worlds respawn at startup.
"""
from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from umweltd.worldstore import WorldDir

logger = logging.getLogger("umweltd.supervisor")

DEFAULT_PORT = 7071
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
SPAWN_TIMEOUT_S = 60.0
WATCHDOG_INTERVAL_S = float(os.environ.get("UMWELTD_WATCHDOG_INTERVAL_S", "10.0"))
CRASH_WINDOW_S = 300.0
CRASH_GIVEUP_COUNT = 5


def home() -> Path:
    return Path(os.environ.get("UMWELTD_HOME", Path.home() / ".umweltd"))


class Supervisor:
    def __init__(self):
        self.procs: dict[str, subprocess.Popen] = {}
        # Worlds that SHOULD be running — create()/start() add, stop() removes. The
        # watchdog only ever acts on this set, so an operator-requested stop is never
        # mistaken for a crash.
        self.desired: set[str] = set()
        self._crash_times: dict[str, list[float]] = {}
        self._watchdog_disabled: set[str] = set()

    def worlds_root(self) -> Path:
        root = home() / "worlds"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def catalog(self) -> list[dict]:
        out = []
        for d in sorted(self.worlds_root().iterdir()):
            wd = WorldDir(d)
            if not wd.manifest_path.exists():
                continue
            port = int(wd.port_path.read_text()) if wd.port_path.exists() else None
            proc = self.procs.get(d.name)
            out.append({"name": d.name, "port": port,
                        "running": bool(proc and proc.poll() is None)})
        return out

    def spawn(self, name: str) -> int:
        wd = WorldDir(self.worlds_root() / name)
        wd.port_path.unlink(missing_ok=True)
        proc = subprocess.Popen(
            [sys.executable, "-m", "umweltd.worker", "--dir", str(wd.root)],
            env=os.environ.copy())
        self.procs[name] = proc
        deadline = time.time() + SPAWN_TIMEOUT_S
        while time.time() < deadline:
            if wd.port_path.exists():
                return int(wd.port_path.read_text())
            if proc.poll() is not None:
                raise RuntimeError(f"worker for {name!r} exited rc={proc.returncode}")
            time.sleep(0.1)
        raise TimeoutError(f"worker for {name!r} never wrote its port file")

    def create(self, body: dict) -> dict:
        name = body.get("name", "")
        if not NAME_RE.match(name):
            raise ValueError(f"bad world name {name!r}")
        if not body.get("spec"):
            raise ValueError("a world needs a spec ref ('module:ATTR')")
        max_worlds = os.environ.get("UMWELTD_MAX_WORLDS")
        if max_worlds is not None and len(self.catalog()) >= int(max_worlds):
            raise ValueError(
                f"world cap reached ({max_worlds}, UMWELTD_MAX_WORLDS) — refusing to "
                f"create {name!r}")
        wd = WorldDir(self.worlds_root() / name)
        if wd.manifest_path.exists():
            raise ValueError(f"world {name!r} already exists")
        wd.write_manifest({k: v for k, v in body.items() if v is not None})
        port = self.spawn(name)
        self.desired.add(name)
        return {"name": name, "port": port}

    def stop(self, name: str) -> dict:
        self.desired.discard(name)
        self._watchdog_disabled.discard(name)
        self._crash_times.pop(name, None)
        proc = self.procs.get(name)
        if proc and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=30)
        return {"name": name, "running": False}

    def start(self, name: str) -> dict:
        wd = WorldDir(self.worlds_root() / name)
        if not wd.manifest_path.exists():
            raise KeyError(f"unknown world {name!r}")
        self.desired.add(name)
        self._watchdog_disabled.discard(name)
        self._crash_times.pop(name, None)
        proc = self.procs.get(name)
        if proc and proc.poll() is None:
            return {"name": name, "port": int(wd.port_path.read_text()), "running": True}
        return {"name": name, "port": self.spawn(name), "running": True}

    def respawn_all(self) -> None:
        for d in self.worlds_root().iterdir():
            if WorldDir(d).manifest_path.exists():
                try:
                    self.spawn(d.name)
                    self.desired.add(d.name)
                except Exception as exc:
                    logger.warning("world %r failed to spawn: %r", d.name, exc)

    def watchdog_tick(self) -> None:
        """Notice a world that died on its own (not via stop()) and restart it. Backs
        off after repeated crashes in a rolling window rather than tight-looping a
        world that can never come up — a manual start() clears the backoff."""
        now = time.time()
        for name in list(self.desired):
            if name in self._watchdog_disabled:
                continue
            proc = self.procs.get(name)
            if proc is None or proc.poll() is None:
                continue                                    # never spawned yet, or still alive
            history = self._crash_times.setdefault(name, [])
            history.append(now)
            history[:] = [t for t in history if now - t <= CRASH_WINDOW_S]
            if len(history) >= CRASH_GIVEUP_COUNT:
                self._watchdog_disabled.add(name)
                logger.error("world %r crashed %d times in %.0fs — giving up "
                            "auto-restart, needs manual /start", name, len(history),
                            CRASH_WINDOW_S)
                continue
            logger.warning("world %r crashed (rc=%s) — auto-restarting (%d/%d in "
                           "window)", name, proc.returncode, len(history),
                           CRASH_GIVEUP_COUNT)
            try:
                self.spawn(name)
            except Exception as exc:
                logger.error("world %r auto-restart failed: %r", name, exc)

    def proxy(self, name: str, rest: str, method: str, body: bytes | None) -> tuple[int, bytes]:
        wd = WorldDir(self.worlds_root() / name)
        if not wd.port_path.exists():
            return 503, json.dumps({"error": f"world {name!r} not running"}).encode()
        port = int(wd.port_path.read_text())
        req = urllib.request.Request(f"http://127.0.0.1:{port}/{rest}", data=body,
                                     method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()


class _Handler(BaseHTTPRequestHandler):
    sup: Supervisor = None
    api_key: str | None = None      # UMWELTD_API_KEY; None = open (localhost trust)

    def log_message(self, fmt, *args):
        pass                                                # replaced by the access log below

    def _authorized(self) -> bool:
        # Constant-time compare — a plain `==` short-circuits on the first mismatched
        # byte, which leaks key-length/prefix information through response timing.
        return self.api_key is None or hmac.compare_digest(
            self.headers.get("X-API-Key") or "", self.api_key)

    def _send(self, code: int, payload) -> None:
        self._last_status = code
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, code: int, page: str) -> None:
        self._last_status = code
        body = page.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, method: str, parts: list) -> bool:
        """The unauthenticated static surface: the playground page and the rendered
        docs. Static content only — it holds no world data (the page's own API calls
        carry the visitor's X-API-Key); every JSON route stays behind auth. Gate the
        whole surface off with UMWELTD_UI=off."""
        if method != "GET" or os.environ.get("UMWELTD_UI", "on").lower() in (
                "off", "0", "false"):
            return False
        from umweltd import docsite, playground
        if parts == []:                                  # / → the playground
            self._last_status = 302
            self.send_response(302)
            self.send_header("Location", "/ui")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return True
        if parts == ["ui"]:
            self._send_html(200, playground.PLAYGROUND_HTML)
            return True
        if parts == ["docs"]:
            self._send_html(200, docsite.render_index())
            return True
        if len(parts) == 2 and parts[0] == "docs":
            page = docsite.render_doc(parts[1])
            if page is None:
                self._send(404, {"error": f"no doc {parts[1]!r}"})
            else:
                self._send_html(200, page)
            return True
        return False

    def _route(self, method: str) -> None:
        t0 = time.time()
        self._last_status = None
        try:
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            if self._static(method, parts):
                return
            if not self._authorized():
                self._send(401, {"error": "missing or wrong X-API-Key"})
                return
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else None
            if parts == ["health"]:
                self._send(200, {"ok": True, "worlds": self.sup.catalog()})
            elif parts == ["worlds"] and method == "GET":
                self._send(200, self.sup.catalog())
            elif parts == ["worlds"] and method == "POST":
                self._send(201, self.sup.create(json.loads(raw or b"{}")))
            elif len(parts) >= 2 and parts[0] == "worlds":
                name = parts[1]
                rest = "/".join(parts[2:])
                if rest == "stop" and method == "POST":
                    self._send(200, self.sup.stop(name))
                elif rest == "start" and method == "POST":
                    self._send(200, self.sup.start(name))
                else:
                    query = ("?" + self.path.split("?", 1)[1]) if "?" in self.path else ""
                    code, body = self.sup.proxy(name, rest + query, method, raw)
                    self._send(code, body)
            else:
                self._send(404, {"error": f"no route {self.path}"})
        except (ValueError, KeyError) as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:
            # Log the real exception server-side; the client only gets a generic
            # message — a 500 body must never leak internals (paths, tracebacks).
            logger.error("unhandled error on %s: %r", self.path, exc)
            self._send(500, {"error": "internal error"})
        finally:
            logger.info(json.dumps({
                "event": "access", "method": method, "path": self.path,
                "status": self._last_status,
                "latency_ms": round((time.time() - t0) * 1000, 2),
            }))

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")


def _watchdog_loop(sup: Supervisor) -> None:
    while True:
        time.sleep(WATCHDOG_INTERVAL_S)
        try:
            sup.watchdog_tick()
        except Exception:
            logger.exception("watchdog tick failed")


def main() -> None:
    logging.basicConfig(level=os.environ.get("UMWELTD_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="umweltd supervisor")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; leaving localhost REQUIRES UMWELTD_API_KEY")
    ap.add_argument("--no-respawn", action="store_true",
                    help="do not respawn existing worlds at startup")
    args = ap.parse_args()
    api_key = os.environ.get("UMWELTD_API_KEY") or None
    if args.host not in ("127.0.0.1", "localhost") and not api_key:
        raise SystemExit("refusing to bind beyond localhost without UMWELTD_API_KEY")
    sup = Supervisor()
    if not args.no_respawn:
        sup.respawn_all()
    _Handler.sup = sup
    _Handler.api_key = api_key
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    # TLS: point UMWELTD_TLS_CERT/UMWELTD_TLS_KEY at a PEM pair (self-signed is fine
    # for a single-tenant box; a real deployment terminates at its proxy).
    cert, key = os.environ.get("UMWELTD_TLS_CERT"), os.environ.get("UMWELTD_TLS_KEY")
    scheme = "http"
    if cert and key:
        import ssl
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    threading.Thread(target=_watchdog_loop, args=(sup,), daemon=True).start()
    logger.info("supervising %s on %s://%s:%s%s", home(), scheme, args.host, args.port,
               " (api-key required)" if api_key else "")
    try:
        server.serve_forever()
    finally:
        for name in list(sup.procs):
            sup.stop(name)


if __name__ == "__main__":
    main()
