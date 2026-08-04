#!/usr/bin/env python
"""Measure the continuation gap: what `load(save(E))` fails to restore, and whether it matters.

The claim this retires is "~800 uncaptured state slots" — a figure that appears in
three places as prose and is derivable from no artifact on this machine. Every
number this tool prints comes with the command that reprints it.

Four verbs:

  prepare  mint a genuine snapshot/tail split from a world's log, in a scratch copy
  capture  boot one side and write its continuation surface
  diff     compare two captures, per owner, ranked by magnitude
  ablate   splice one owner at a time from the from-log twin and see who flips the hash

`capture` boots in a FRESH PROCESS, always. This is not style: a second engine
booted inside a process that already ran a replay inherits polluted module
globals (role registries are process-global — `umwelt/spec/validate.py` says so
in its own docstring, and `tests/test_packet.py:41` documents the consequence).
`diff` and `ablate` orchestrate; they never boot in the parent.

  --stage cursor   both sides at the snapshot cursor. The PURE CAPTURE GAP: what
                   `load` does not restore, before a single tail event replays.
                   This is the enumeration that replaces "~800".
  --stage tail     both sides after the tail. IMPACT: which of those slots
                   actually moved the field_canon_hash, which is the only
                   divergence the protocol's referee can see.

The gap between the two stages is the whole point. A slot can fail to round-trip
and never matter; a slot can round-trip and still fork the chain through a
learner's firing schedule. Only measuring both tells you which repairs to make.

**On lease-drill.** The 2026-07-18 drill world survives at
`hearth-state/worlds/lease-drill/` but carries no `events.db` and no live
snapshot — only `snapshot.pkl.orphan-1784357678`. There is no log to replay, so
it cannot serve as a two-sided fixture. Use `tests/fixtures/packet_mood_stream.json`,
which `tests/test_packet.py` records as that same stream with anonymized organ
names, proven hash-equivalent by replaying its head to the drill's released hash.

Examples
--------
    # a real live world, split at the halfway point of its own log
    python tools/continuation_diff.py prepare --world ~/yurt-sync/hearth-state/worlds/hive-ops \\
        --out /tmp/probe/hive-ops --at-fraction 0.5

    python tools/continuation_diff.py capture --world /tmp/probe/hive-ops \\
        --mode from-log    --stage cursor --out A.json
    python tools/continuation_diff.py capture --world /tmp/probe/hive-ops \\
        --mode incremental --stage cursor --out B.json
    python tools/continuation_diff.py diff A.json B.json --md report.md

    python tools/continuation_diff.py ablate --world /tmp/probe/hive-ops --md ablate.md

Exit codes follow the packet referee's discipline — 0 clean / 1 findings /
2 nothing to measure — so this is usable as a gate, not just as a report.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from umwelt.substrate import continuation as C   # noqa: E402

EXIT_CLEAN, EXIT_FINDINGS, EXIT_NOTHING = 0, 1, 2


# ── shared helpers ────────────────────────────────────────────────────────────────

def _world_dir(world: Path):
    from umweltd.worldstore import WorldDir
    return WorldDir(Path(world))


def _log_stamps(world: Path) -> list[str]:
    """Every event timestamp in the world's log, ascending.

    Read straight from sqlite rather than through `read_events_since`: choosing a
    split point is a question about the log itself, and going through the replay
    path would make the answer depend on the engine we have not booted yet.
    """
    db = _world_dir(world).events_db
    if not db.exists():
        return []
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as cx:
        return [r[0] for r in cx.execute(
            "SELECT timestamp FROM events ORDER BY timestamp")]


def _boot(world: Path, *, from_log: bool, until_ts: str | None, freeze: bool):
    from umweltd.packet import boot_engine
    return boot_engine(Path(world), freeze_learning=freeze, until_ts=until_ts,
                       from_log_only=from_log)


def _self(*argv: str) -> subprocess.CompletedProcess:
    """Re-invoke this script in a fresh interpreter — see the module docstring."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return subprocess.run([sys.executable, str(Path(__file__).resolve()), *argv],
                          capture_output=True, text=True, env=env)


def _run_or_die(*argv: str) -> dict:
    cp = _self(*argv)
    if cp.returncode not in (EXIT_CLEAN, EXIT_FINDINGS):
        sys.stderr.write(cp.stdout + cp.stderr)
        raise SystemExit(f"subprocess failed ({cp.returncode}): {' '.join(argv)}")
    try:
        return json.loads(cp.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        sys.stderr.write(cp.stdout + cp.stderr)
        raise SystemExit(f"subprocess produced no JSON: {' '.join(argv)}")


# ── prepare ───────────────────────────────────────────────────────────────────────

def cmd_prepare(a) -> int:
    """Mint a snapshot/tail split in a SCRATCH COPY of the world.

    Copying is not politeness. Three live worlds carry a log and no snapshot, so
    measuring in place would mean writing a snapshot and cursor into the synced
    hearth — the probe would alter the thing it measures, and every later boot of
    that world would resume from an artifact this tool invented.
    """
    src, out = Path(a.world), Path(a.out)
    stamps = _log_stamps(src)
    if not stamps:
        print(json.dumps({"prepared": False,
                          "reason": f"{src.name} has no events.db — nothing to split"}))
        return EXIT_NOTHING
    if len(stamps) < 4:
        print(json.dumps({"prepared": False, "reason": f"only {len(stamps)} events"}))
        return EXIT_NOTHING

    idx = max(1, min(len(stamps) - 2, int(len(stamps) * float(a.at_fraction))))
    split = stamps[idx]
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns(
        "lease.d", "*.orphan-*", "worker.port", "snapshot.pkl", "cursor.txt"))

    res = _run_or_die("_snap", "--world", str(out), "--until", split)
    print(json.dumps({"prepared": True, "world": str(out), "events": len(stamps),
                      "split_index": idx, "split_ts": split,
                      "tail_events": len(stamps) - idx - 1,
                      "head_hash": res["field_canon_hash"]}, indent=1))
    return EXIT_CLEAN


def cmd_snap(a) -> int:
    """Internal: boot from-log to `--until`, write snapshot + cursor. Fresh process."""
    engine, last_ts, batches = _boot(Path(a.world), from_log=True,
                                     until_ts=a.until, freeze=False)
    wd = _world_dir(Path(a.world))
    engine.save(str(wd.snapshot_path))
    wd.write_cursor(last_ts)
    print(json.dumps({"cursor": last_ts, "batches": batches,
                      "field_canon_hash": engine.field_canon_hash()}))
    return EXIT_CLEAN


# ── capture ───────────────────────────────────────────────────────────────────────

def cmd_capture(a) -> int:
    world = Path(a.world)
    wd = _world_dir(world)
    from_log = (a.mode == "from-log")

    if a.stage == "cursor":
        # Both sides land ON the snapshot cursor: from-log replays the head,
        # incremental loads the snapshot and replays nothing. Any difference is
        # purely what `load` failed to restore.
        until = wd.cursor() or None
        if not until:
            print(json.dumps({"captured": False,
                              "reason": "no cursor — run `prepare` first"}))
            return EXIT_NOTHING
    else:
        until = a.until or None       # tail: run the log out

    if not from_log and not wd.snapshot_path.exists():
        print(json.dumps({"captured": False,
                          "reason": "no snapshot.pkl — run `prepare` first"}))
        return EXIT_NOTHING

    engine, last_ts, batches = _boot(world, from_log=from_log, until_ts=until,
                                     freeze=a.freeze_learning)
    surf = C.walk(engine, raw=bool(a.raw))
    payload = surf.to_json()
    payload["meta"] = {
        "world": str(world), "mode": a.mode, "stage": a.stage,
        "until_ts": until, "cursor": last_ts, "batches_replayed": batches,
        "freeze_learning": bool(a.freeze_learning),
        "field_canon_hash": engine.field_canon_hash(),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.raw:
        # Raw leaves are ndarrays; JSON cannot carry them and the ablate splice
        # needs the values themselves, not their digests.
        with open(out, "wb") as f:
            blob, names = _pickle_top_level(engine)
            pickle.dump({"payload": payload, "raw": surf.raw,
                         "objects_blob": blob, "objects": names}, f)
    else:
        out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps({"captured": True, "out": str(out), "slots": len(surf.slots),
                      "aliases": len(surf.aliases), "digest": surf.digest(),
                      **payload["meta"]}))
    return EXIT_CLEAN


def _pickle_top_level(engine) -> tuple[bytes | None, list[str]]:
    """ALL picklable top-level attributes, in ONE pickle. Returns (blob, names).

    Value-splicing alone cannot close the gap: a slot present in the donor and
    absent in the target has nothing to be stamped into. `berry_tape.stamper.tape`
    is a deque the incremental boot never restored, so writing `tape[5]` raises
    269 times and the ablation silently measures a subtree it never moved.
    Splicing the object handles exactly that.

    One blob, not one per attribute, and this is the whole subtlety. Pickling
    each attribute separately and unpickling them separately gives every holder
    its own private copy of anything they shared — `calibration.field` stops
    being `engine.field`. Splice two such owners and the engine now has two
    fields where it had one, so the greedy closure can splice EVERY owner and
    still reproduce nothing, then report the gap as lying outside the walked
    surface. That conclusion is false and expensive: it is a measurement
    artifact of the measuring tool, which is the one bug this whole exercise
    exists to stop shipping.

    Unpickling the blob once rebuilds the shared structure as shared.
    `CalibrationLoop` holds lambdas and will not pickle at all, so it drops out
    and falls back to the value path — reported per owner, never assumed.
    """
    keep = {k: v for k, v in vars(engine).items()
            if not (v is None or isinstance(v, (bool, int, float, str)))}
    while keep:
        try:
            return pickle.dumps(keep, protocol=pickle.HIGHEST_PROTOCOL), sorted(keep)
        except Exception as exc:                  # noqa: BLE001
            # Drop the attribute the error names and retry, so one unpicklable
            # holder does not cost us the aliasing of all the others.
            bad = next((k for k in keep if k in str(exc)), None) or _probe_unpicklable(keep)
            if bad is None:
                return None, []
            keep.pop(bad, None)
    return None, []


def _probe_unpicklable(d: dict) -> str | None:
    for k, v in d.items():
        try:
            pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:                          # noqa: BLE001
            return k
    return None


def _load_capture(p: Path) -> tuple[C.Surface, dict, dict]:
    if p.suffix == ".pkl":
        with open(p, "rb") as f:
            d = pickle.load(f)
        return (C.Surface.from_json(d["payload"]), d["payload"].get("meta", {}),
                {"raw": d["raw"], "objects": list(d.get("objects") or []),
                 "objects_blob": d.get("objects_blob")})
    d = json.loads(p.read_text(encoding="utf-8"))
    return (C.Surface.from_json(d), d.get("meta", {}),
            {"raw": {}, "objects": [], "objects_blob": None})


# ── diff ──────────────────────────────────────────────────────────────────────────

def cmd_diff(a) -> int:
    left, lmeta, _ = _load_capture(Path(a.left))
    right, rmeta, _ = _load_capture(Path(a.right))
    d = C.diff(left, right)
    owners = d.by_owner(a.depth)

    lh, rh = lmeta.get("field_canon_hash"), rmeta.get("field_canon_hash")
    report = {
        "left": {"file": str(a.left), **{k: lmeta.get(k) for k in
                                         ("mode", "stage", "cursor", "field_canon_hash")}},
        "right": {"file": str(a.right), **{k: rmeta.get(k) for k in
                                           ("mode", "stage", "cursor", "field_canon_hash")}},
        "slots": {"left": len(left.slots), "right": len(right.slots)},
        "hash": {"left": lh, "right": rh,
                 "match": (lh == rh) if (lh and rh) else None},
        "differing_slots": d.total(),
        "changed": len(d.changed), "only_left": len(d.only_left),
        "only_right": len(d.only_right), "alias_changed": len(d.alias_changed),
        "owners": dict(sorted(owners.items(),
                              key=lambda kv: -(kv[1]["changed"] + kv[1]["only_left"]
                                               + kv[1]["only_right"]))),
    }
    if a.md:
        Path(a.md).write_text(_md_report(report, d, a.top), encoding="utf-8")
        report["md"] = str(a.md)
    print(json.dumps(report, indent=1))
    return EXIT_CLEAN if d.clean else EXIT_FINDINGS


def _md_report(r: dict, d: "C.SurfaceDiff", top: int) -> str:
    L: list[str] = []
    L.append("# Continuation gap\n")
    L.append(f"- left  (reference): `{r['left']['mode']}` / `{r['left']['stage']}` "
             f"— {r['slots']['left']} slots, canon hash `{r['left']['field_canon_hash']}`")
    L.append(f"- right (subject):   `{r['right']['mode']}` / `{r['right']['stage']}` "
             f"— {r['slots']['right']} slots, canon hash `{r['right']['field_canon_hash']}`")
    L.append(f"- **{r['differing_slots']} differing slots** "
             f"({r['changed']} changed, {r['only_left']} only-left, "
             f"{r['only_right']} only-right, {r['alias_changed']} alias moves)")
    match = r["hash"]["match"]
    L.append(f"- field_canon_hash: **{'MATCH' if match else 'FORKED' if match is False else 'n/a'}**\n")
    L.append("## By owner\n")
    L.append("| owner | changed | only-left | only-right | alias | max magnitude | example |")
    L.append("|---|---:|---:|---:|---:|---:|---|")
    for owner, v in r["owners"].items():
        mag = v["max_magnitude"]
        L.append(f"| `{owner}` | {v['changed']} | {v['only_left']} | {v['only_right']} "
                 f"| {v['alias_changed']} | {'—' if mag < 0 else f'{mag:.3g}'} "
                 f"| `{(v['examples'] or ['—'])[0]}` |")
    L.append(f"\n## Largest movers (top {top})\n")
    L.append("| slot | left | right | magnitude |")
    L.append("|---|---|---|---:|")
    for sd in sorted(d.changed, key=lambda s: -s.magnitude)[:top]:
        f = lambda v: str(v)[:44]
        L.append(f"| `{sd.path}` | {f(sd.left)} | {f(sd.right)} "
                 f"| {'—' if sd.magnitude < 0 else f'{sd.magnitude:.3g}'} |")
    if d.only_left:
        L.append(f"\n## Present only in the reference ({len(d.only_left)})\n")
        for p in d.only_left[:top]:
            L.append(f"- `{p}`")
    return "\n".join(L) + "\n"


# ── ablate ────────────────────────────────────────────────────────────────────────

def cmd_ablate(a) -> int:
    """Which owner, spliced ALONE from the from-log twin, flips the hash?

    The minimal sufficient set is the closure spec — that is the whole reason
    this verb exists rather than a ranked list of "things that look wrong".
    """
    world = Path(a.world)
    scratch = Path(a.scratch or (world.parent / f"_ablate_{world.name}"))
    scratch.mkdir(parents=True, exist_ok=True)
    donor = scratch / "twin_cursor.pkl"

    # The reference: what the from-log twin reaches after the whole log.
    ref = _run_or_die("capture", "--world", str(world), "--mode", "from-log",
                      "--stage", "tail", "--out", str(scratch / "ref_tail.json"))
    target_hash = ref["field_canon_hash"]

    # The donor: the from-log twin AT THE CURSOR, with raw values to splice.
    _run_or_die("capture", "--world", str(world), "--mode", "from-log",
                "--stage", "cursor", "--raw", "--out", str(donor))

    base = _run_or_die("_ablate_run", "--world", str(world), "--donor", str(donor),
                       "--owners", "")
    if base["field_canon_hash"] == target_hash:
        print(json.dumps({"ablated": False, "reason":
                          "incremental already matches from-log — nothing to ablate",
                          "hash": target_hash}, indent=1))
        return EXIT_CLEAN

    surf, _, donor_d = _load_capture(donor)
    owners = sorted(surf.owners(a.depth))
    singles = []
    for owner in owners:
        r = _run_or_die("_ablate_run", "--world", str(world), "--donor", str(donor),
                        "--depth", str(a.depth), "--owners", owner)
        r["owner"] = owner
        r["flips"] = (r["field_canon_hash"] == target_hash)
        # INCOMPLETE is MEASURED, not inferred from picklability. An owner absent
        # from the donor's pickle blob is not automatically a lower bound — a
        # scalar splices perfectly by value. What makes a splice incomplete is a
        # slot it could not land: one the target has no place for.
        sp = r["spliced"]
        r["incomplete"] = bool(sp["missing"] or sp["failed"])
        singles.append(r)
        if a.verbose:
            sp = r["spliced"]
            print(f"  {'FLIPS ' if r['flips'] else '      '} {owner:28s} "
                  f"{r['field_canon_hash']}  moved={r['moved_slots']:<5d} "
                  f"({sp['in_place']}ip/{sp['rebound']}rb/{sp['missing']}miss/"
                  f"{sp['failed']}fail)" + (f" {sp['errors'][0]}" if sp["errors"] else ""),
                  file=sys.stderr)

    # Greedy closure: add owners by descending single-splice effect until the
    # hash lands. A single owner that flips it alone is the whole answer; when
    # none does, the interaction is the finding.
    chosen: list[str] = []
    closure_hash = base["field_canon_hash"]
    if not any(s["flips"] for s in singles):
        order = [s["owner"] for s in sorted(
            singles, key=lambda s: -(s["moved_slots"] or 0))]
        for owner in order:
            chosen.append(owner)
            r = _run_or_die("_ablate_run", "--world", str(world), "--donor", str(donor),
                            "--depth", str(a.depth), "--owners", ",".join(chosen))
            closure_hash = r["field_canon_hash"]
            if a.verbose:
                print(f"  +{owner:28s} -> {closure_hash}", file=sys.stderr)
            if closure_hash == target_hash:
                break
        else:
            chosen = []          # even everything did not reproduce it

    out = {
        "world": str(world),
        "target_hash": target_hash,
        "incremental_hash": base["field_canon_hash"],
        "sufficient_alone": [s["owner"] for s in singles if s["flips"]],
        "minimal_set": chosen if closure_hash == target_hash else None,
        "incomplete_splices": {s["owner"]: {"missing": s["spliced"]["missing"],
                                            "failed": s["spliced"]["failed"]}
                               for s in singles if s["incomplete"]},
        "still_needed": ([] if (any(s["flips"] for s in singles)
                                or closure_hash == target_hash)
                         else [_unreached_reason(
                             [s["owner"] for s in singles if s["incomplete"]])]),
        "singles": [{"owner": s["owner"], "flips": s["flips"],
                     "hash": s["field_canon_hash"], "moved_slots": s["moved_slots"],
                     "spliced": s["spliced"]} for s in singles],
    }
    if a.md:
        Path(a.md).write_text(_md_ablate(out), encoding="utf-8")
        out["md"] = str(a.md)
    print(json.dumps(out, indent=1))
    return EXIT_CLEAN if out["sufficient_alone"] or out["minimal_set"] else EXIT_FINDINGS


def _unreached_reason(incomplete: list[str]) -> str:
    """Why the closure did not land — and the distinction that decides what to do.

    Blaming the walked surface when the splice simply could not move an owner is
    the most expensive possible wrong answer here: it sends the next reader
    looking for hidden state that is not hidden at all.
    """
    if incomplete:
        return ("UNREACHED, and the result is a LOWER BOUND: the splice could not "
                f"fully move {len(incomplete)} owner(s) ({', '.join(incomplete)}). "
                "Those hold state the incremental engine has no slot for — a value "
                "splice can overwrite a slot, never create one — and they are "
                "unpicklable, so they cannot be spliced whole either. "
                "CalibrationLoop's lambdas are the blocker. Make them picklable "
                "before reading anything stronger into this table.")
    return ("UNREACHED: every owner was spliced completely and the from-log hash "
            "still did not reproduce — the gap is genuinely outside the walked "
            "surface (see `opaque` in the capture).")


def _md_ablate(o: dict) -> str:
    L = ["# Ablation — which owner closes the gap\n",
         f"- from-log reference hash: `{o['target_hash']}`",
         f"- incremental hash:        `{o['incremental_hash']}`",
         f"- **sufficient alone:** {o['sufficient_alone'] or '— none'}",
         f"- **minimal sufficient set:** {o['minimal_set'] or '— not reached'}",
         f"- incompletely spliced (makes this a lower bound): "
         f"{list(o.get('incomplete_splices') or {}) or '— none'}",
         ""]
    if o["still_needed"]:
        L += ["> " + o["still_needed"][0], ""]
    L += ["| owner | flips the hash | slots moved by splice | resulting hash |",
          "|---|:--:|---:|---|"]
    for s in sorted(o["singles"], key=lambda s: (not s["flips"], -(s["moved_slots"] or 0))):
        inc = (o.get("incomplete_splices") or {}).get(s["owner"])
        note = (f" _(incomplete: {inc['missing']} missing, {inc['failed']} failed)_"
                if inc else "")
        L.append(f"| `{s['owner']}` | {'**yes**' if s['flips'] else 'no'} "
                 f"| {s['moved_slots']}{note} | `{s['hash']}` |")
    return "\n".join(L) + "\n"


def cmd_ablate_run(a) -> int:
    """Internal: boot incremental, splice the named owners, replay the tail. Fresh process."""
    world = Path(a.world)
    wd = _world_dir(world)
    surf, meta, donor = _load_capture(Path(a.donor))

    engine, _, _ = _boot(world, from_log=False, until_ts=wd.cursor() or None,
                         freeze=False)
    spliced = {"in_place": 0, "rebound": 0, "missing": 0, "failed": 0, "errors": []}
    method: dict[str, str] = {}
    moved = 0
    owners = [o for o in (a.owners or "").split(",") if o]
    if owners:
        before = C.walk(engine)
        by_value: list[str] = []
        # ONE unpickle for every owner spliced as an object, so anything they
        # shared in the donor is still shared here. See _pickle_top_level.
        obj_owners = ([o for o in owners if o in donor["objects"]]
                      if a.depth == 1 and donor["objects_blob"] else [])
        if obj_owners:
            twin = pickle.loads(donor["objects_blob"])
            for owner in obj_owners:
                setattr(engine, owner, twin[owner])
                method[owner] = "object"
        for owner in owners:
            if owner not in method:
                by_value.append(owner)
                method[owner] = "value"
        if by_value:
            want = {p for p in surf.slots if C.owner_of(p, a.depth) in set(by_value)}
            spliced = C.splice(engine, donor["raw"], want)
        # What the splice ACTUALLY moved, not what it was asked to move. An
        # owner is named by path prefix but objects hold back-references, so
        # "I spliced calibration" is intent; this number is the measurement.
        moved = len(C.diff(before, C.walk(engine)).changed)

    # Now run the tail on the spliced engine, exactly as an incremental evolve would.
    from umwelt.events import read_events_since, replay_sensor_batches
    from umwelt.learning.runner import BrainRunner
    rows = (read_events_since(wd.events_db, wd.cursor() or "", until=None)
            if wd.events_db.exists() else [])
    flush = float(wd.manifest().get("flush_secs", 30.0))
    BrainRunner(engine).replay((r, bt, conf) for bt, r, conf, _l
                               in replay_sensor_batches(rows, flush_secs=flush))
    print(json.dumps({"owners": owners, "spliced": spliced, "method": method,
                      "moved_slots": moved, "tail_rows": len(rows),
                      "field_canon_hash": engine.field_canon_hash()}))
    return EXIT_CLEAN


# ── cli ───────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="continuation_diff",
        description=__doc__.split("\n\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("prepare", help="mint a snapshot/tail split in a scratch copy")
    q.add_argument("--world", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--at-fraction", type=float, default=0.5, dest="at_fraction")
    q.set_defaults(fn=cmd_prepare)

    q = sub.add_parser("capture", help="boot one side and write its surface")
    q.add_argument("--world", required=True)
    q.add_argument("--mode", choices=("from-log", "incremental"), required=True)
    q.add_argument("--stage", choices=("cursor", "tail"), default="cursor")
    q.add_argument("--until", default=None)
    q.add_argument("--out", required=True)
    q.add_argument("--raw", action="store_true",
                   help="also carry leaf VALUES (pickle) so ablate can splice them")
    q.add_argument("--freeze-learning", action="store_true", dest="freeze_learning")
    q.set_defaults(fn=cmd_capture)

    q = sub.add_parser("diff", help="compare two captures")
    q.add_argument("left"); q.add_argument("right")
    q.add_argument("--md", default=None)
    q.add_argument("--top", type=int, default=40)
    q.add_argument("--depth", type=int, default=1)
    q.set_defaults(fn=cmd_diff)

    q = sub.add_parser("ablate", help="find the minimal sufficient owner set")
    q.add_argument("--world", required=True)
    q.add_argument("--depth", type=int, default=1)
    q.add_argument("--md", default=None)
    q.add_argument("--scratch", default=None)
    q.add_argument("-v", "--verbose", action="store_true")
    q.set_defaults(fn=cmd_ablate)

    q = sub.add_parser("_snap"); q.add_argument("--world", required=True)
    q.add_argument("--until", default=None); q.set_defaults(fn=cmd_snap)

    q = sub.add_parser("_ablate_run"); q.add_argument("--world", required=True)
    q.add_argument("--donor", required=True); q.add_argument("--owners", default="")
    q.add_argument("--depth", type=int, default=1)
    q.set_defaults(fn=cmd_ablate_run)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
