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
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from umweltd.worldstore import WorldDir

DEFAULT_PORT = 7071
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
SPAWN_TIMEOUT_S = 60.0


def home() -> Path:
    return Path(os.environ.get("UMWELTD_HOME", Path.home() / ".umweltd"))


class Supervisor:
    def __init__(self):
        self.procs: dict[str, subprocess.Popen] = {}

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
        wd = WorldDir(self.worlds_root() / name)
        if wd.manifest_path.exists():
            raise ValueError(f"world {name!r} already exists")
        wd.write_manifest({k: v for k, v in body.items() if v is not None})
        port = self.spawn(name)
        return {"name": name, "port": port}

    def stop(self, name: str) -> dict:
        proc = self.procs.get(name)
        if proc and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=30)
        return {"name": name, "running": False}

    def start(self, name: str) -> dict:
        wd = WorldDir(self.worlds_root() / name)
        if not wd.manifest_path.exists():
            raise KeyError(f"unknown world {name!r}")
        proc = self.procs.get(name)
        if proc and proc.poll() is None:
            return {"name": name, "port": int(wd.port_path.read_text()), "running": True}
        return {"name": name, "port": self.spawn(name), "running": True}

    def respawn_all(self) -> None:
        for d in self.worlds_root().iterdir():
            if WorldDir(d).manifest_path.exists():
                try:
                    self.spawn(d.name)
                except Exception as exc:
                    print(f"[umweltd] world {d.name!r} failed to spawn: {exc}",
                          flush=True)

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
        pass

    def _authorized(self) -> bool:
        return self.api_key is None or self.headers.get("X-API-Key") == self.api_key

    def _send(self, code: int, payload) -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _route(self, method: str) -> None:
        if not self._authorized():
            self._send(401, {"error": "missing or wrong X-API-Key"})
            return
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else None
        try:
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
            self._send(500, {"error": str(exc)})

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")


def main() -> None:
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
    print(f"[umweltd] supervising {home()} on {scheme}://{args.host}:{args.port}"
          f"{' (api-key required)' if api_key else ''}", flush=True)
    try:
        server.serve_forever()
    finally:
        for name in list(sup.procs):
            sup.stop(name)


if __name__ == "__main__":
    main()
