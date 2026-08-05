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
GATE_TIMEOUT_S = 60.0
WATCHDOG_INTERVAL_S = float(os.environ.get("UMWELTD_WATCHDOG_INTERVAL_S", "10.0"))
CRASH_WINDOW_S = 300.0
CRASH_GIVEUP_COUNT = 5

# ── the pacemaker ────────────────────────────────────────────────────────────
# The engine has no background tick: `umweltd.worker` advances only on ingest, so
# a dissipative world left alone does not idle — it relaxes to its ground state,
# which under the house sign law is the COLD pole. Unfed worlds do not go quiet,
# they go confidently wrong. Something must drive them.
#
# What that something must NOT be is a fixed wall-clock interval. A world's decay
# rate is its own property, and `gamma_diss` is not even a constant — calibration
# channel 5 nudges `gamma_diss_{role}` from live tracking error, so a world that
# learns its beliefs perish faster than declared should start asking to be fed
# sooner, with nobody editing anything. That is what a scheduled task can never do.
#
# So: each world reports tau from its own measured physics (GET /pace) and the
# pacemaker services it at a FRACTION of tau. Fast-decaying worlds get fed more
# often, fresh worlds are not touched at all, and a newly registered world is
# paced correctly on its first tick with no configuration.
PACE_ENABLED = os.environ.get("UMWELTD_PACEMAKER", "1") != "0"
# How much of a decay constant may elapse before a belief stops being evidence.
# This is the one honest policy number here: not "how often to run", but "how
# stale is too stale", which is a judgement about evidence rather than a clock.
PACE_FRACTION = float(os.environ.get("UMWELTD_PACE_FRACTION", "0.3"))
# Applied to the RESOLUTION deadline instead of PACE_FRACTION. Not a taste knob:
# resolution is the point past which the membrane discards wall time outright, so
# this is scheduling margin (the watchdog looks every WATCHDOG_INTERVAL_S), not a
# decay tolerance. Raising it toward 1.0 trades safety margin for fewer wakeups.
PACE_RESOLUTION_MARGIN = float(os.environ.get("UMWELTD_PACE_RESOLUTION_MARGIN", "0.9"))
# Bounds on the derived interval. The floor stops a pathological tau (a world that
# reports absurdly fast decay) from spinning the hose; the ceiling is the fallback
# for a world that cannot report tau at all — a world with no measurable step rate
# is not therefore immortal.
PACE_FLOOR_S = float(os.environ.get("UMWELTD_PACE_FLOOR_S", "60"))
PACE_CEILING_S = float(os.environ.get("UMWELTD_PACE_CEILING_S", "3600"))
PACE_WEBHOOK = os.environ.get("UMWELTD_PACE_WEBHOOK", "http://127.0.0.1:7099")
PACE_TIMEOUT_S = 5.0
# Scarcity, if any. Unset => the engine has no opinion about cost and the budget
# is unlimited; umwelt stays domain-agnostic and yurt supplies the meaning by
# pointing this at `hive-purse`. Any world can play the role — the engine only
# needs a warmth in [0,1].
PACE_BUDGET_WORLD = os.environ.get("UMWELTD_PACE_BUDGET_WORLD", "").strip()
PACE_BUDGET_NODE = os.environ.get("UMWELTD_PACE_BUDGET_NODE", "purse")
PACE_BUDGET_ROLE = os.environ.get("UMWELTD_PACE_BUDGET_ROLE", "warmth")
PACE_BUDGET_MAX = int(os.environ.get("UMWELTD_PACE_BUDGET", "1000"))
# Never zero. A budget of nothing is a hive that can never observe its way back
# out of scarcity, and the floor is what lets a cold purse still recover.
PACE_BUDGET_MIN = int(os.environ.get("UMWELTD_PACE_BUDGET_MIN", "1"))


def home() -> Path:
    return Path(os.environ.get("UMWELTD_HOME", Path.home() / ".umweltd"))


def _ref_shape_ok(ref: str) -> bool:
    """True iff ref has the 'module:attr' shape every worker seam requires."""
    module_name, _, attr = str(ref).partition(":")
    return bool(module_name and attr)


class Supervisor:
    def __init__(self):
        self.procs: dict[str, subprocess.Popen] = {}
        # Worlds that SHOULD be running — create()/start() add, stop() removes. The
        # watchdog only ever acts on this set, so an operator-requested stop is never
        # mistaken for a crash.
        self.desired: set[str] = set()
        self._crash_times: dict[str, list[float]] = {}
        self._watchdog_disabled: set[str] = set()
        self._pace_last: dict[str, float] = {}
        self._pace_rank: list[dict] = []
        # Per world: did the LAST beat actually land? Transport success is not
        # landing — a pulse at an actuator nobody registered used to return
        # `{"ok": true}` like any other, so the heart reported 280 successful
        # beats a day into the void. Surfaced on /pace so a detached limb is a
        # queryable fact rather than something you find by grepping a log.
        self._pace_outcome: dict[str, dict] = {}

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
        # First-contact kindness: refuse a manifest that would kill the worker HERE,
        # with the fix named, instead of a worker-exit 500 whose truth lives only in
        # the daemon log (the first external hive deployment's three scars).
        vocab = body.get("vocabulary")
        if vocab and not _ref_shape_ok(vocab):
            raise ValueError(
                f"vocabulary ref {vocab!r} must be 'module:function' — a bare module "
                f"name gives the worker nothing to call (did you mean "
                f"{vocab}:register_vocabulary?)")
        if not _ref_shape_ok(body["spec"]):
            raise ValueError(f"spec ref {body['spec']!r} must be 'module:ATTR' "
                             f"(e.g. 'examples.gridworld.world:SPEC')")
        max_worlds = os.environ.get("UMWELTD_MAX_WORLDS")
        if max_worlds is not None and len(self.catalog()) >= int(max_worlds):
            raise ValueError(
                f"world cap reached ({max_worlds}, UMWELTD_MAX_WORLDS) — refusing to "
                f"create {name!r}")
        wd = WorldDir(self.worlds_root() / name)
        if wd.manifest_path.exists():
            raise ValueError(f"world {name!r} already exists")
        self._gate_spec(body)
        wd.write_manifest({k: v for k, v in body.items() if v is not None})
        port = self.spawn(name)
        self.desired.add(name)
        return {"name": name, "port": port}

    def _gate_spec(self, body: dict) -> None:
        """Resolve + sanity-check the manifest's spec (and vocabulary) in a FRESH
        subprocess BEFORE anything spawns — the forge-gate discipline
        (umweltforge.pipeline.run_validation), applied at the daemon's front door.
        Cheap by construction: `--resolve-only` stops the gate after resolve +
        schema_sanity (no engine boot, no worker). spec_path is honored exactly the
        way the worker honors it (dirs prepended to the import path). Raises
        ValueError (→ a 400 with the subprocess's precise error text) on refusal."""
        cmd = [sys.executable, "-m", "umwelt.spec.validate", body["spec"],
               "--json", "--resolve-only"]
        if body.get("vocabulary"):
            cmd += ["--vocabulary", body["vocabulary"]]
        env = os.environ.copy()
        raw = body.get("spec_path")
        dirs = [raw] if isinstance(raw, str) else list(raw or [])
        if dirs:
            resolved = [str(Path(p).expanduser().resolve()) for p in dirs]
            env["PYTHONPATH"] = os.pathsep.join(
                resolved + [env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        try:
            out = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                 timeout=GATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise ValueError(f"spec gate timed out after {GATE_TIMEOUT_S:.0f}s "
                             f"resolving {body['spec']!r}")
        if out.returncode == 0:
            return
        try:
            report = json.loads(out.stdout)
            detail = "; ".join(
                f"{c['name']}: {c['detail']}" for c in report.get("checks", ())
                if not c.get("ok") and not c.get("skipped")) or "spec gate failed"
        except (json.JSONDecodeError, TypeError, KeyError):
            detail = ((out.stderr or out.stdout).strip()[-2000:]
                      or "spec gate emitted nothing")
        raise ValueError(
            f"world {body.get('name')!r} refused before spawn — {detail}")

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

    # ── pacemaker + sinking ─────────────────────────────────────────────────────
    def _worker_get(self, name: str, path: str) -> dict | None:
        """One cheap read from a running worker. None on any failure — a world that
        cannot answer is simply not paced this tick, never a reason to raise."""
        wd = WorldDir(self.worlds_root() / name)
        if not wd.port_path.exists():
            return None
        try:
            port = int(wd.port_path.read_text())
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}{path}", timeout=PACE_TIMEOUT_S) as resp:
                return json.loads(resp.read())
        except Exception:                            # noqa: BLE001 — best-effort probe
            return None

    def _pace_due(self, pace: dict) -> tuple[bool, float, float | None]:
        """(due, interval_s, age_s) for one world, from ITS numbers, not a constant."""
        age = pace.get("age_s")
        # The two deadlines mean different things and must not share a coefficient.
        #
        # DRIFT is a tolerance: "how much decay am I willing to accept", so
        # PACE_FRACTION genuinely belongs on it.
        # RESOLUTION is a hard membrane limit: past hold*(cap+1) the catch-up
        # saturates and the extra wall gap is discarded outright. Taking 30% of a
        # hard limit just feeds three times more often than necessary — the margin
        # below is for scheduling jitter, not for taste.
        drift = pace.get("drift_s")
        resolution = pace.get("resolution_s")
        deadlines = []
        if drift:
            deadlines.append(PACE_FRACTION * float(drift))
        if resolution:
            deadlines.append(PACE_RESOLUTION_MARGIN * float(resolution))
        if deadlines:
            tau = min(deadlines)
        else:
            # A worker that reports only tau_s does not say WHICH bound it is, so
            # apply the drift tolerance: feeding a shade too often is recoverable,
            # under-feeding a fast-decaying world is the failure this exists to stop.
            raw = pace.get("tau_s")
            tau = (PACE_FRACTION * float(raw)) if raw else None
        if tau and tau > 0:
            interval = min(max(float(tau), PACE_FLOOR_S), PACE_CEILING_S)
        else:
            # No measurable step rate yet (a world that has never been fed cannot
            # report one). Fall back to the ceiling rather than treating unknown
            # decay as no decay — the same "silence is absence, not health" rule
            # the rest of the hive runs on.
            interval = PACE_CEILING_S
        if age is None:
            # Never ingested anything. It is maximally stale, not fresh.
            return True, interval, None
        return (float(age) >= interval), interval, float(age)

    def pacemaker_tick(self) -> list[dict]:
        """Feed the worlds that have decayed past usefulness, worst first.

        Two jobs in one pass, because they are the same decision:

        PACING — a world is due when its age exceeds PACE_FRACTION of its own tau.
        Nothing here knows how often any world "should" run.

        SINKING — under scarcity the tick has a service budget, and worlds are
        serviced in rank order. A world at the bottom is simply never reached: it
        is not archived, unregistered or deleted, it just does not get fed, and it
        comes straight back the moment there is room. Dormancy, not extinction —
        which is also why umweltd still needs no DELETE route.

        Rank is urgency (how far past due) divided by cost to service. A world that
        is barely stale and expensive sinks below one that is badly stale and cheap.
        """
        if not PACE_ENABLED:
            return []
        budget = self.service_budget()
        candidates = []
        for entry in self.catalog():
            name = entry["name"]
            if not entry.get("running"):
                continue
            pace = self._worker_get(name, "/pace")
            if pace is None:
                continue
            due, interval, age = self._pace_due(pace)
            # Never re-fire inside one due-window: a pulse already in flight has not
            # had a chance to land, and firing again would stack hoses on a box whose
            # scarcest resource is memory.
            last = self._pace_last.get(name, 0.0)
            if time.time() - last < interval:
                due = False
            overdue = (age / interval) if (age is not None and interval > 0) else 99.0
            candidates.append({
                "world": name, "due": due, "age_s": age, "interval_s": round(interval, 1),
                "tau_s": pace.get("tau_s"), "gamma_max": pace.get("gamma_max"),
                "hottest_axis": pace.get("hottest_axis"),
                "rank": round(overdue / max(1e-6, self.service_cost(name)), 4),
            })

        candidates.sort(key=lambda c: -c["rank"])
        self._pace_rank = candidates
        fired = []
        for cand in candidates:
            if not cand["due"]:
                continue
            if len(fired) >= budget and not self._budget_exempt(cand["world"]):
                cand["skipped"] = "service budget exhausted — sank this tick"
                logger.info("pacemaker: %r sank (budget %d exhausted)",
                            cand["world"], budget)
                continue
            if self._pulse(cand["world"], cand):
                self._pace_last[cand["world"]] = time.time()
                fired.append(cand)
        return fired

    def service_budget(self) -> int:
        """How many worlds may be fed this tick.

        With no PACE_BUDGET_WORLD configured this is effectively unlimited, which
        keeps umwelt domain-agnostic: the engine has no opinion about money. Point
        it at a world (yurt points it at `hive-purse`) and scarcity becomes a felt
        belief rather than a configured cap — a cold scarcity signal shrinks the
        budget, the ranking decides who eats, and the bottom of the stack sinks.

        Nothing is archived, unregistered or deleted by sinking. A sunk world stays
        a first-class catalog entry and comes straight back when there is room,
        which is dormancy rather than extinction — and is why umweltd still needs
        no DELETE route.
        """
        if not PACE_BUDGET_WORLD:
            return PACE_BUDGET_MAX
        belief = self._worker_get(
            PACE_BUDGET_WORLD,
            f"/beliefs?node={PACE_BUDGET_NODE}&role={PACE_BUDGET_ROLE}")
        if not belief or belief.get("value") is None:
            # Unreadable scarcity signal is not permission to spend the whole
            # roster's worth of work — but it must not be a stop either, or a
            # dead signal freezes the hive. Fall back to the floor.
            return PACE_BUDGET_MIN
        try:
            warmth = float(belief["value"])          # (z+1)/2 in [0, 1]
        except (TypeError, ValueError):
            return PACE_BUDGET_MIN
        span = max(0, PACE_BUDGET_MAX - PACE_BUDGET_MIN)
        return PACE_BUDGET_MIN + int(round(max(0.0, min(1.0, warmth)) * span))

    def _budget_exempt(self, name: str) -> bool:
        """The scarcity signal's own world is never sunk.

        Without this the design deadlocks, and quietly: a cold purse shrinks the
        budget, a shrunken budget can stop the purse's own poster from running,
        and an unfed purse decays further cold. It would be a hive that starves
        itself to death by correctly observing that it is starving. Whatever
        measures scarcity has to be exempt from it.
        """
        return bool(PACE_BUDGET_WORLD) and name == PACE_BUDGET_WORLD

    def service_cost(self, name: str) -> float:
        """Relative cost of feeding this world. 1.0 until the burn face can price a
        world individually; kept as a seam so ranking is cost-aware by construction
        rather than retrofitted."""
        return 1.0

    def _pulse(self, name: str, detail: dict) -> bool:
        """Ask the action router to run this world's poster.

        Deliberately the SAME envelope a field Action uses (`actuator_id` +
        `command.on`), so a pulse rides the registry's existing gate, cooldown and
        risk_class machinery instead of a private side channel. `world_pulse_*`
        entries are risk_class "safe": they run a read-only scan and post readings,
        and spend nothing.
        """
        payload = {
            "actuator_id": f"world_pulse_{name.replace('-', '_')}",
            "command": {"on": True},
            "source": "umweltd.pacemaker",
            "why": {k: detail.get(k) for k in
                    ("age_s", "interval_s", "tau_s", "gamma_max", "hottest_axis")},
        }
        try:
            req = urllib.request.Request(
                PACE_WEBHOOK, data=json.dumps(payload).encode(), method="POST",
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=PACE_TIMEOUT_S) as resp:
                raw = resp.read()
            # `accepted` says the actuator EXISTS and the decision was ON. It is
            # not a success claim — the action runs after the response. An older
            # router has no such field; absence reads as accepted so a version
            # skew does not manufacture a fleet of phantom failures.
            try:
                body = json.loads(raw) if raw else {}
            except (ValueError, TypeError):
                body = {}
            accepted = body.get("accepted", True) is not False
            reason = body.get("reason") or ""
            self._pace_outcome[name] = {"at": time.time(), "landed": bool(accepted),
                                        "reason": reason}
            if accepted:
                logger.info("pacemaker: pulsed %r (age %.0fs >= %.0fs, tau %s)",
                            name, detail.get("age_s") or -1,
                            detail.get("interval_s") or -1, detail.get("tau_s"))
            else:
                # Deliberately NOT a return of False. False means "no cooldown,
                # retry next tick", which is right for a dead router and wrong
                # for a permanent registry gap — that would hot-loop the router
                # every watchdog tick forever. A beat that was delivered and
                # refused is still a delivered beat; it just did not land.
                logger.warning("pacemaker: pulse for %r was REFUSED by the router "
                               "(%s) — the world is not being fed", name,
                               reason or "no reason given")
            return True
        except Exception as exc:                     # noqa: BLE001 — a dead router must not kill the loop
            self._pace_outcome[name] = {"at": time.time(), "landed": False,
                                        "reason": f"transport: {exc!r}"}
            logger.warning("pacemaker: pulse for %r failed: %r", name, exc)
            return False

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
            elif parts == ["pace"] and method == "GET":
                # The heart, visible. Shows every world's derived interval and its
                # rank, so "why has nothing fed X" is answerable without reading logs.
                # `last_pulse` is grafted on at read time rather than stored in the
                # rank rows, because the rank is rebuilt every tick from the
                # workers' own numbers and the outcome outlives any single tick.
                _rank = [dict(r, last_pulse=self.sup._pace_outcome.get(r.get("world")))
                         for r in self.sup._pace_rank]
                self._send(200, {
                    "enabled": PACE_ENABLED, "fraction": PACE_FRACTION,
                    "floor_s": PACE_FLOOR_S, "ceiling_s": PACE_CEILING_S,
                    "webhook": PACE_WEBHOOK, "budget": self.sup.service_budget(),
                    "unfed": sorted(w for w, o in self.sup._pace_outcome.items()
                                    if not o.get("landed")),
                    "worlds": _rank})
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
        # The heart shares the watchdog's thread on purpose: it is one process with
        # a lifecycle, not a second scheduled task. The tick rate here is only how
        # often the pacemaker LOOKS; how often it FEEDS is each world's own tau.
        try:
            sup.pacemaker_tick()
        except Exception:
            logger.exception("pacemaker tick failed")


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
