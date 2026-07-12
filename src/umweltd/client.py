"""A thin stdlib client for umweltd — what a harness imports to talk to the daemon."""
from __future__ import annotations

import json
import urllib.request


class UmweltClient:
    """Point at a supervisor (`/worlds/<name>/...` routes) or directly at one worker
    (base_url with world=None and worker-local paths)."""

    def __init__(self, base_url: str, world: str | None = None, timeout: float = 120.0,
                 api_key: str | None = None):
        self.base = base_url.rstrip("/")
        self.world = world
        self.timeout = timeout
        self.api_key = api_key

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _url(self, path: str) -> str:
        prefix = f"/worlds/{self.world}" if self.world else ""
        return f"{self.base}{prefix}/{path.lstrip('/')}"

    def _call(self, method: str, path: str, payload: dict | None = None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self._url(path), data=data, method=method,
                                     headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read() or b"null")

    # ── supervisor-level ────────────────────────────────────────────────────────────
    def create_world(self, name: str, spec: str, vocabulary: str | None = None, **knobs):
        payload = {"name": name, "spec": spec, "vocabulary": vocabulary, **knobs}
        data = json.dumps(payload).encode()
        req = urllib.request.Request(f"{self.base}/worlds", data=data, method="POST",
                                     headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    # ── world-level ─────────────────────────────────────────────────────────────────
    def ingest(self, events: list[tuple], flush_secs: float | None = None) -> dict:
        payload: dict = {"events": [list(e) for e in events]}
        if flush_secs is not None:
            payload["flush_secs"] = flush_secs
        return self._call("POST", "events", payload)

    def health(self) -> dict:
        return self._call("GET", "health")

    def state(self) -> dict:
        return self._call("GET", "state")

    def belief(self, node: str, role: str) -> dict:
        return self._call("GET", f"beliefs?node={node}&role={role}")

    def recommendations(self) -> list:
        return self._call("GET", "recommendations")

    def snapshot(self) -> dict:
        return self._call("POST", "snapshot", {})
