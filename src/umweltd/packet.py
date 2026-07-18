"""evolve-packet — stateless lease-based evolution over a synced world directory.

The yurt charter moved the canonical belief field into the shared folder
(`yurt-sync/hearth-state/worlds/<name>/`). Nodes are peers the yurt hires for
services; THIS module is the "evolve" and "verify" hires. It wraps the existing
worldstore recovery contract + BrainRunner replay — zero engine changes.

Protocol spec: yurt repo `docs/HEARTH_STATE.md`. In one breath:

    lease.d/<node>.json     per-node bid files, ONE WRITER EACH (Syncthing-safe)
    winner                  a pure function every node computes identically:
                            earliest non-expired bid_ts, lexicographic node
                            tie-break; a 90s sync-grace re-read before believing
                            you won; ttl expiry reaps crashed holders; any
                            sync-conflict twin in lease.d/ = back off
    evolve                  bid won -> resolve spec via MANIFEST.toml (PYTHONPATH
                            to a pinned checkout; resolves the per-node
                            spec_path split) -> orphan check -> load snapshot +
                            replay events to --until-ts -> snapshot -> cursor ->
                            sync-confirm (Syncthing REST /rest/db/completion,
                            90s wall fallback) -> write release {cursor,
                            snapshot_sha256, field_canon_hash}
    verify (--verify-only)  the redundant-evolution referee on a second node:
                            rebuild from the LOG alone, recompute
                            field_canon_hash, compare to the release; posts an
                            `evolution_referee` event (high eta) when a hearth
                            URL is given; the exit code is real either way.

Recovery law (unchanged from worldstore): THE LOG IS TRUTH. A snapshot newer
than its cursor, or whose sha256 disagrees with the last release, is an orphan
of a crashed/partially-synced writer — it is set aside and the world replays
from the log.

CLI:
    python -m umweltd.packet bid     --world DIR --node N [--purpose evolve-packet]
                                     [--ttl 600] [--until-ts ISO] [--freeze-learning]
    python -m umweltd.packet winner  --world DIR [--grace 0]
    python -m umweltd.packet evolve  --world DIR --node N [--until-ts ISO]
                                     [--freeze-learning] [--grace 90]
                                     [--no-sync-confirm]
    python -m umweltd.packet verify  --world DIR --node N [--freeze-learning]
                                     [--hearth-url URL --hearth-key KEY
                                      --hearth-world hive-ops]
    python -m umweltd.packet withdraw --world DIR --node N

Syncthing REST config rides env vars (per-device, never synced):
    SYNCTHING_URL     default http://127.0.0.1:8384
    SYNCTHING_APIKEY  the device's REST api key
    SYNCTHING_FOLDER  default yurt-sync
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from umweltd.worldstore import WorldDir

logger = logging.getLogger("umweltd.packet")

PURPOSES = ("serve", "evolve-packet", "migrate")
DEFAULT_TTL_SECS = 600
SYNC_GRACE_SECS = 90.0
SYNC_FALLBACK_SECS = 90.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── the lease directory ──────────────────────────────────────────────────────────────

def lease_dir(world: Path) -> Path:
    return Path(world) / "lease.d"


def lease_path(world: Path, node: str) -> Path:
    return lease_dir(world) / f"{node}.json"


def read_bids(world: Path) -> tuple[list[dict], list[str]]:
    """All parseable bids + the basenames of any sync-conflict twins (twins mean
    two nodes fought over ONE bid file — a one-writer breach; callers back off)."""
    d = lease_dir(world)
    bids, twins = [], []
    if not d.is_dir():
        return bids, twins
    for p in sorted(d.iterdir()):
        if "sync-conflict" in p.name:
            twins.append(p.name)
            continue
        if p.suffix != ".json":
            continue
        try:
            bid = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("unreadable bid %s: %r (skipped)", p.name, exc)
            continue
        bid.setdefault("node", p.stem)
        bids.append(bid)
    return bids, twins


def bid_expiry(bid: dict) -> datetime | None:
    """bid_ts (or renewed_ts, whichever is later) + ttl. None = malformed."""
    try:
        base = datetime.fromisoformat(bid["bid_ts"])
        if bid.get("renewed_ts"):
            base = max(base, datetime.fromisoformat(bid["renewed_ts"]))
        return base + timedelta(seconds=float(bid.get("ttl", DEFAULT_TTL_SECS)))
    except (KeyError, TypeError, ValueError):
        return None


def winner(bids: list[dict], now: datetime | None = None,
           purpose: str | None = None) -> dict | None:
    """The pure function every node computes identically: among non-expired,
    non-released bids (optionally filtered to one purpose), the earliest bid_ts
    wins; ties break lexicographically on node name. Malformed bids never win."""
    now = now or _now()
    live = []
    for b in bids:
        if purpose is not None and b.get("purpose") != purpose:
            continue
        if b.get("release"):
            continue                       # already completed — not a live claim
        exp = bid_expiry(b)
        if exp is None or exp <= now:
            continue                       # malformed or reaped by ttl expiry
        try:
            key = (datetime.fromisoformat(b["bid_ts"]), str(b["node"]))
        except (KeyError, TypeError, ValueError):
            continue
        live.append((key, b))
    if not live:
        return None
    live.sort(key=lambda kb: kb[0])
    return live[0][1]


def write_bid(world: Path, node: str, purpose: str, ttl: int,
              until_ts: str | None, from_cursor: str,
              freeze_learning: bool) -> dict:
    bid = {
        "node": node,
        "purpose": purpose,
        "bid_ts": _iso(_now()),
        "ttl": ttl,
        "renewed_ts": None,
        "packet": {
            "from_cursor": from_cursor,
            "until_ts": until_ts,
            "freeze_learning": bool(freeze_learning),
        },
        "release": None,
    }
    d = lease_dir(world)
    d.mkdir(parents=True, exist_ok=True)
    lease_path(world, node).write_text(json.dumps(bid, indent=1))
    return bid


def _update_own_bid(world: Path, node: str, **fields) -> dict:
    p = lease_path(world, node)
    bid = json.loads(p.read_text())
    bid.update(fields)
    p.write_text(json.dumps(bid, indent=1))
    return bid


def i_won(world: Path, node: str, purpose: str,
          grace_secs: float = SYNC_GRACE_SECS) -> bool:
    """Compute the winner, wait the sync grace, re-read, recompute. Only believe
    a win that survives the grace re-read (a slower peer's earlier bid may still
    be in flight). Any bid twin in lease.d/ = back off unconditionally."""
    for attempt in range(2):
        bids, twins = read_bids(world)
        if twins:
            logger.warning("lease twins present (%s) — backing off", twins)
            return False
        w = winner(bids, purpose=purpose)
        if w is None or w.get("node") != node:
            return False
        if attempt == 0 and grace_secs > 0:
            logger.info("provisional win for %r — %.0fs sync grace", node, grace_secs)
            time.sleep(grace_secs)
    return True


# ── MANIFEST.toml — the shared spec pin ──────────────────────────────────────────────

def read_manifest_toml(world: Path) -> dict:
    """`MANIFEST.toml` beside world.json: the SHARED spec reference. world.json's
    `spec_path` is a per-node absolute path (the Windows/WSL split); MANIFEST's
    `pythonpath` entries are relative to the world dir (or absolute), so every
    node resolves the same pinned checkout from the same synced bytes.

        spec = "hive_world:HIVE_SPEC"      # optional — defaults to world.json spec
        vocabulary = "module:function"     # optional
        pythonpath = ["../../checkouts/spacewheat-hive"]
        [pins]
        umwelt_sha = "..."                 # provenance, recorded not enforced
        yurt_repo_sha = "..."
    """
    p = Path(world) / "MANIFEST.toml"
    if not p.exists():
        return {}
    import tomllib
    return tomllib.loads(p.read_text())


def resolve_pythonpath(world: Path) -> list[str]:
    """MANIFEST pythonpath (world-relative) first, then world.json spec_path
    (per-node absolute, legacy) as fallback — prepended to sys.path."""
    world = Path(world)
    out: list[str] = []
    man = read_manifest_toml(world)
    for entry in man.get("pythonpath", ()):
        q = Path(entry)
        out.append(str(q if q.is_absolute() else (world / q).resolve()))
    raw = {}
    try:
        raw = WorldDir(world).manifest()
    except (OSError, json.JSONDecodeError):
        pass
    sp = raw.get("spec_path")
    for pth in ([sp] if isinstance(sp, str) else (sp or [])):
        out.append(str(Path(pth).expanduser().resolve()))
    return out


# ── boot: worldstore recovery + BrainRunner replay (the engine contract) ─────────────

def orphan_check(world: Path, last_release: dict | None) -> str | None:
    """THE LOG IS TRUTH. A snapshot mtime-newer than its cursor (a writer died
    between the two) or sha-mismatched against the last release (a partial sync)
    is an orphan: set it aside so boot replays from the log. Returns the reason
    string when an orphan was quarantined, else None."""
    wd = WorldDir(Path(world))
    snap, cur = wd.snapshot_path, wd.cursor_path
    if not snap.exists():
        return None
    reason = None
    if not cur.exists():
        reason = "snapshot without cursor"
    elif snap.stat().st_mtime > cur.stat().st_mtime + 1.0:   # 1s fs-timestamp slack
        reason = "snapshot newer than cursor"
    elif last_release and last_release.get("snapshot_sha256"):
        if _sha256(snap) != last_release["snapshot_sha256"]:
            reason = "snapshot sha256 disagrees with last release"
    if reason is None:
        return None
    quarantine = snap.with_name(snap.name + f".orphan-{int(time.time())}")
    snap.rename(quarantine)
    if cur.exists():
        cur.unlink()
    logger.warning("orphan snapshot (%s) -> %s; will replay from log",
                   reason, quarantine.name)
    return reason


def last_release(world: Path) -> dict | None:
    """The most recent release across all bids (by released_ts)."""
    bids, _ = read_bids(world)
    rels = [dict(b["release"], node=b.get("node"))
            for b in bids if isinstance(b.get("release"), dict)]
    rels = [r for r in rels if r.get("released_ts")]
    return max(rels, key=lambda r: r["released_ts"]) if rels else None


def boot_engine(world: Path, *, freeze_learning: bool, until_ts: str | None,
                from_log_only: bool = False):
    """The worker's boot sequence without the HTTP surface: pythonpath ->
    vocabulary -> build_engine blank -> (snapshot + cursor unless from_log_only)
    -> BrainRunner replay of the event tail to until_ts under the REPLAY gauge
    (learn=0 when freeze_learning). Returns (engine, last_ts, batches)."""
    world = Path(world)
    wd = WorldDir(world)
    manifest = wd.manifest()
    man = read_manifest_toml(world)

    # Packet runs ALWAYS pin the process RNGs (the parity proof's determinism
    # switch, worker.py's pin_rngs). Deterministic evolution is what makes a
    # second node's --verify-only referee possible at all; a live worker may
    # leave this off, a packet may not.
    import random

    import numpy as np
    random.seed(1234)
    np.random.seed(1234)

    for p in resolve_pythonpath(world):
        if p not in sys.path:
            sys.path.insert(0, p)

    vocab = man.get("vocabulary") or manifest.get("vocabulary")
    if vocab:
        import importlib
        mod, _, attr = vocab.partition(":")
        getattr(importlib.import_module(mod), attr)()

    from umwelt.boot import build_engine, set_role
    from umwelt.learning.context import ContextState
    spec = man.get("spec") or manifest["spec"]
    engine = build_engine(spec=spec, population=False)

    last_ts = ""
    if not from_log_only and wd.snapshot_path.exists():
        engine.load(str(wd.snapshot_path))
        last_ts = wd.cursor()

    gauge = ContextState.replay()
    if freeze_learning:
        gauge = ContextState(actuate=gauge.actuate, dt_factor=gauge.dt_factor,
                             learn=0.0, persist=gauge.persist)
    set_role(engine, gauge)

    from umwelt.events import read_events_since, replay_sensor_batches
    from umwelt.learning.runner import BrainRunner
    rows = (read_events_since(wd.events_db, last_ts, until=until_ts)
            if wd.events_db.exists() else [])
    flush = float(manifest.get("flush_secs", 30.0))
    n = BrainRunner(engine).replay(
        (readings, bt, conf)
        for bt, readings, conf, _last in replay_sensor_batches(rows, flush_secs=flush))
    if rows:
        last_ts = max(last_ts, max(r[0] for r in rows))
    return engine, last_ts, n


# ── sync-confirm ─────────────────────────────────────────────────────────────────────

def sync_confirm(folder: str | None = None, url: str | None = None,
                 apikey: str | None = None, timeout: float = 300.0,
                 fallback_secs: float = SYNC_FALLBACK_SECS) -> str:
    """Block until the local Syncthing reports the folder 100% accepted by its
    peers (GET /rest/db/completion), or fall back to a flat wall-clock wait when
    the REST surface is missing/unreachable. Returns how it confirmed."""
    folder = folder or os.environ.get("SYNCTHING_FOLDER", "yurt-sync")
    url = (url or os.environ.get("SYNCTHING_URL", "http://127.0.0.1:8384")).rstrip("/")
    apikey = apikey or os.environ.get("SYNCTHING_APIKEY", "")
    if not apikey:
        logger.info("no SYNCTHING_APIKEY — falling back to %.0fs wait", fallback_secs)
        time.sleep(fallback_secs)
        return "fallback-wait"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                f"{url}/rest/db/completion?folder={folder}",
                headers={"X-API-Key": apikey})
            with urllib.request.urlopen(req, timeout=10) as resp:
                comp = json.loads(resp.read()).get("completion", 0.0)
            if comp >= 100.0:
                return "rest-completion-100"
            logger.info("sync completion %.1f%% — waiting", comp)
        except Exception as exc:
            logger.warning("completion poll failed (%r) — %.0fs fallback",
                           exc, fallback_secs)
            time.sleep(fallback_secs)
            return "fallback-wait-after-error"
        time.sleep(5)
    return "timeout"                      # release still written; verify will referee


# ── the verbs ────────────────────────────────────────────────────────────────────────

def cmd_bid(a) -> int:
    wd = WorldDir(Path(a.world))
    bid = write_bid(a.world, a.node, a.purpose, a.ttl, a.until_ts,
                    from_cursor=wd.cursor(), freeze_learning=a.freeze_learning)
    print(json.dumps(bid, indent=1))
    return 0


def cmd_winner(a) -> int:
    if a.grace:
        time.sleep(a.grace)
    bids, twins = read_bids(a.world)
    w = winner(bids, purpose=a.purpose)
    print(json.dumps({"winner": (w or {}).get("node"), "bid": w,
                      "twins": twins}, indent=1))
    return 0 if (w and not twins) else 1


def cmd_evolve(a) -> int:
    world = Path(a.world)
    if not lease_path(world, a.node).exists():
        wd = WorldDir(world)
        write_bid(world, a.node, "evolve-packet", a.ttl, a.until_ts,
                  from_cursor=wd.cursor(), freeze_learning=a.freeze_learning)
        logger.info("bid placed for %r", a.node)
    if not i_won(world, a.node, "evolve-packet", grace_secs=a.grace):
        print(json.dumps({"evolved": False, "reason": "did not win the lease"}))
        return 3

    orphan_check(world, last_release(world))
    engine, last_ts, batches = boot_engine(
        world, freeze_learning=a.freeze_learning, until_ts=a.until_ts)

    # Write-order law: snapshot -> cursor -> sync-confirm -> release. A reader that
    # sees the release can trust the pair; a crash before the release leaves an
    # orphan the next boot quarantines (the log is truth either way).
    wd = WorldDir(world)
    tmp = wd.snapshot_path.with_suffix(".pkl.tmp")
    engine.save(str(tmp))
    tmp.replace(wd.snapshot_path)
    wd.write_cursor(last_ts)

    confirmed = ("skipped" if a.no_sync_confirm
                 else sync_confirm(a.folder, a.syncthing_url, a.syncthing_key))

    release = {
        "released_ts": _iso(_now()),
        "cursor": last_ts,
        "snapshot_sha256": _sha256(wd.snapshot_path),
        "field_canon_hash": engine.field_canon_hash(),
        "batches_replayed": batches,
        "freeze_learning": bool(a.freeze_learning),
        "sync_confirm": confirmed,
    }
    _update_own_bid(world, a.node, release=release)
    print(json.dumps({"evolved": True, "node": a.node, **release}, indent=1))
    return 0


def cmd_verify(a) -> int:
    """--verify-only: the redundant-evolution referee. Rebuild the field from the
    LOG ALONE (blank engine, full replay to the release cursor), recompute the
    canon hash, compare. Learning gauge mirrors the release's freeze flag unless
    overridden. Exit code is the verdict."""
    world = Path(a.world)
    rel = last_release(world)
    if not rel:
        print(json.dumps({"verified": False, "reason": "no release to verify"}))
        return 2
    freeze = a.freeze_learning if a.freeze_learning is not None \
        else bool(rel.get("freeze_learning"))
    snap_sha = _sha256(WorldDir(world).snapshot_path) \
        if WorldDir(world).snapshot_path.exists() else None
    engine, last_ts, batches = boot_engine(
        world, freeze_learning=freeze, until_ts=rel.get("cursor") or None,
        from_log_only=True)
    got = engine.field_canon_hash()
    ok_hash = (got == rel.get("field_canon_hash"))
    ok_sha = (snap_sha == rel.get("snapshot_sha256"))
    verdict = {
        "verified": bool(ok_hash),
        "field_canon_hash": {"release": rel.get("field_canon_hash"), "recomputed": got},
        "snapshot_sha256": {"release": rel.get("snapshot_sha256"), "on_disk": snap_sha,
                            "match": bool(ok_sha)},
        "release_node": rel.get("node"), "referee_node": a.node,
        "batches_replayed": batches, "cursor": last_ts,
    }
    print(json.dumps(verdict, indent=1))
    if a.hearth_url:
        _post_referee(a, verdict)
    return 0 if ok_hash else 1


def _post_referee(a, verdict: dict) -> None:
    """Post `evolution_referee` to the hearth's world (high eta — a recomputation,
    not an opinion). Failures are logged, never change the exit code: the referee's
    authority is the exit code; the post is a courtesy to the hive."""
    try:
        body = json.dumps({"events": [[
            _iso(_now()), "evolution_referee",
            "1.0" if verdict["verified"] else "-1.0",
            json.dumps({"eta": 0.9, **verdict}),   # meta is a JSON STRING
        ]]}).encode()
        req = urllib.request.Request(
            f"{a.hearth_url.rstrip('/')}/worlds/{a.hearth_world}/events",
            data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "X-API-Key": a.hearth_key or ""})
        with urllib.request.urlopen(req, timeout=15) as resp:
            logger.info("evolution_referee posted: %s", resp.read()[:200])
    except Exception as exc:
        logger.warning("evolution_referee post failed: %r", exc)


def cmd_withdraw(a) -> int:
    p = lease_path(Path(a.world), a.node)
    if p.exists():
        p.unlink()
        print(json.dumps({"withdrawn": True}))
    else:
        print(json.dumps({"withdrawn": False, "reason": "no bid on file"}))
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=os.environ.get("UMWELTD_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(prog="umweltd.packet", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, node=True):
        p.add_argument("--world", required=True, help="the synced world directory")
        if node:
            p.add_argument("--node", required=True, help="this node's name (its one lease file)")

    p = sub.add_parser("bid", help="write/refresh this node's lease bid")
    common(p)
    p.add_argument("--purpose", default="evolve-packet", choices=PURPOSES)
    p.add_argument("--ttl", type=int, default=DEFAULT_TTL_SECS)
    p.add_argument("--until-ts", default=None)
    p.add_argument("--freeze-learning", action="store_true")
    p.set_defaults(fn=cmd_bid)

    p = sub.add_parser("winner", help="compute the current winner (pure function)")
    common(p, node=False)
    p.add_argument("--purpose", default="evolve-packet", choices=PURPOSES)
    p.add_argument("--grace", type=float, default=0.0)
    p.set_defaults(fn=cmd_winner)

    p = sub.add_parser("evolve", help="run one evolve-packet under the lease")
    common(p)
    p.add_argument("--until-ts", default=None)
    p.add_argument("--ttl", type=int, default=DEFAULT_TTL_SECS)
    p.add_argument("--freeze-learning", action="store_true")
    p.add_argument("--grace", type=float, default=SYNC_GRACE_SECS)
    p.add_argument("--no-sync-confirm", action="store_true")
    p.add_argument("--syncthing-url", default=None)
    p.add_argument("--syncthing-key", default=None)
    p.add_argument("--folder", default=None)
    p.set_defaults(fn=cmd_evolve)

    p = sub.add_parser("verify", help="--verify-only referee: recompute from the log, compare to the release")
    common(p)
    p.add_argument("--freeze-learning", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--hearth-url", default=None)
    p.add_argument("--hearth-key", default=os.environ.get("UMWELTD_API_KEY"))
    p.add_argument("--hearth-world", default="hive-ops")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("withdraw", help="remove this node's bid file")
    common(p)
    p.set_defaults(fn=cmd_withdraw)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
