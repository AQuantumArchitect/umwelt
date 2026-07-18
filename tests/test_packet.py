"""THE EVOLVE-PACKET PROTOCOL SUITE — the lease ledger tells the truth and the
referee's exit code is real.

Born from the 2026-07-18 lease-drill chain fork: an incremental evolve (snapshot
+ log tail) released a `field_canon_hash` the from-log referee could not
reproduce (exit 1). The drill's root cause: `boot_engine` seeds the process
globals once per boot, but the replay path CONSUMES the global `random` stream
(fractal projection sampling, Thompson draws, surprise-tape reservoir draws) —
an incremental booter replayed its tail with the stream at position 0 while the
full-log referee reached the same tail mid-stream. FIXED: `engine.save` persists
the stream positions (`rng_state`) at the snapshot cursor; `engine.load`
restores them LAST, after every other restore step; a legacy snapshot without
the block loads with a loud warning (seed-once fallback).

Three more determinism repairs fell out of verifying that fix:
  * fractal params round-tripped through 6-decimal DISPLAY rounding
    (`ScalarParam.snapshot`) — now carried exactly (`value_exact`/`sigma_exact`);
  * egress tendrils ticked on WALL clock inside replay (rate-limit gating was
    replay-speed-dependent — the "deterministic replay" contract leaked); they
    now tick in event time (the batch's `now`);
  * tendril continuation state (commit qubit + dispatch memory + learned
    rise/fall) rides the snapshot.

KNOWN OPEN (xfail-pinned below, see CLAIMS.md): those fixes are NECESSARY but
not yet SUFFICIENT. `engine.save` is a curated learned-state cache, not a full
process continuation: at any cursor a snapshot-booted engine differs from its
from-log twin in ~800 live-state slots (calibration EMAs, sensor-bridge
per-sensor memory, berry tapes, collapse touched-roles, driver anticipation,
ingest-hold memory, fractal residuals...), and one learning-hot batch off that
hidden state can fork the canon hash. The daemon-parity suite never sees this
because it pins the hash AT the cursor, not continued evolution. Until the
closure lands, chains that must referee clean re-evolve `--from-log` (the log is
truth; nothing is lost but compute).

The fixture world mirrors the protocol's real deployment shape (the hearth's
mood worlds: dissipative organs + graded observe bindings + a day driver +
ingest hold). `tests/fixtures/packet_mood_stream.json` is the ACTUAL lease-drill
stream (anonymized organ names — proven hash-equivalent to the real artifact:
its head replays to the drill's own released hash). Every packet verb runs as a
fresh subprocess, exactly as deployed — a second engine booted inside a process
that already ran a replay inherits polluted module-global state.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from umweltd import packet
from umweltd.worldstore import WorldDir

FIXTURE = REPO / "tests" / "fixtures" / "packet_mood_stream.json"

SPEC_SHIM = '''\
"""Mood-organ world — the evolve-packet protocol's deployment shape: dissipative
belief axes on organs, graded observe bindings, a slow driver, an ingest hold.
Structurally the lease-drill world (7 organs x 5 axes x 3 tiers + root wall)."""
from umwelt.spec.schema import BindingSpec, DomainSpec, DriverSpec, NodeSpec, OutputSpec

NORM = {"type": "regime", "center": 0.0, "width": 0.5, "invert": True}
ORGANS = tuple(f"organ_{c}" for c in "abcdefg")
AXES = ("activity", "confidence", "blocked", "freshness", "coordination")

_nodes = [NodeSpec("camp", parent=None, kind="root", roles=("coordination",),
                   role_modes={"coordination": "dissipative"},
                   params={"gamma_diss": (0.0002, 0.0001, 0.0, 1.0)})]
_bindings = [BindingSpec("camp_coordination_wall", zone="camp", role="coordination",
                         normalizer=NORM, force_observe=True, collapse_alpha=0.40)]
for organ in ORGANS:
    _nodes.append(NodeSpec(organ, parent="camp", roles=AXES,
                           role_modes={a: "dissipative" for a in AXES},
                           params={"gamma_diss": (0.0003, 0.0001, 0.0, 1.0)}))
    for axis in AXES:
        for tier, eta in (("claim", 0.25), ("wall", 0.40), ("referee", 0.92)):
            _bindings.append(BindingSpec(f"{organ}_{axis}_{tier}", zone=organ, role=axis,
                                         normalizer=NORM, force_observe=True,
                                         collapse_alpha=eta))

SPEC = DomainSpec(
    name="packet-mood",
    nodes=tuple(_nodes),
    bindings=tuple(_bindings),
    outputs=(OutputSpec("mood_hint", node="camp", role="coordination"),),
    drivers=(DriverSpec("day", period_s=86400.0),),
    ingest_hold_s=5.0,
)
'''


# ── the world + the verbs (one fresh process per verb, as deployed) ─────────────────

def make_world(tmp_path: Path, name: str = "mood") -> Path:
    wdir = tmp_path / "worlds" / name
    wdir.mkdir(parents=True)
    (wdir / "world_spec_packet.py").write_text(SPEC_SHIM)
    (wdir / "world.json").write_text(json.dumps(
        {"name": name, "spec": "world_spec_packet:SPEC", "flush_secs": 30.0}))
    (wdir / "MANIFEST.toml").write_text(
        'spec = "world_spec_packet:SPEC"\npythonpath = ["."]\n')
    return wdir


def fixture_stream() -> tuple[list[tuple], list[tuple]]:
    fx = json.loads(FIXTURE.read_text())
    to_rows = lambda rr: [tuple(r) + (None,) for r in rr]
    return to_rows(fx["head"]), to_rows(fx["tail"])


def run(*argv: str) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])
    proc = subprocess.run(
        [sys.executable, "-m", "umweltd.packet", *argv],
        env=env, capture_output=True, text=True, timeout=600)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr[-2000:])
    return proc.returncode


def evolve(wdir: Path, node: str, *extra: str) -> int:
    return run("evolve", "--world", str(wdir), "--node", node,
               "--grace", "0", "--no-sync-confirm", *extra)


def read_lease(wdir: Path, node: str) -> dict:
    return json.loads(packet.lease_path(wdir, node).read_text())


def evolved_world(tmp_path: Path) -> Path:
    """A world evolved over the drill head by wargen-A, tail appended but not yet
    replayed — the incremental-resume brink."""
    wdir = make_world(tmp_path)
    head, tail = fixture_stream()
    WorldDir(wdir).append_events(head)
    assert evolve(wdir, "wargen-A") == 0
    WorldDir(wdir).append_events(tail)
    return wdir


# ── 1. the headline: the RNG stream position rides the snapshot and STEERS the
#      replay (the drill's root cause, pinned) ──────────────────────────────────────

def test_snapshot_rng_state_steers_the_incremental_replay(tmp_path):
    """Two incremental evolves from the SAME field state, differing ONLY in the
    snapshot's rng_state block (present vs stripped-to-legacy), must release
    DIFFERENT field_canon_hashes: the tail's draws come off the resumed stream
    position, not the boot seed. Pre-fix this is impossible — no snapshot
    carries rng_state, both worlds are byte-identical, the hashes collide, and
    this test FAILS (recorded: baseline 6597455 releases 748c... == 748c...).

    Also pins: the snapshot carries both streams, and stripping them still
    evolves (exit 0, the legacy fallback) rather than crashing."""
    import shutil
    wdir = evolved_world(tmp_path)

    with open(WorldDir(wdir).snapshot_path, "rb") as f:
        snap = pickle.load(f)
    assert "rng_state" in snap, "snapshot must persist the RNG stream positions"
    assert "random" in snap["rng_state"] and "numpy" in snap["rng_state"]

    w_intact = tmp_path / "worlds" / "mood-intact"
    w_legacy = tmp_path / "worlds" / "mood-legacy"
    shutil.copytree(wdir, w_intact)
    shutil.copytree(wdir, w_legacy)

    # forge the pre-fix format on the legacy twin; keep its ledger honest
    # (release sha + cursor mtime) so the orphan check doesn't quarantine it
    wd = WorldDir(w_legacy)
    with open(wd.snapshot_path, "rb") as f:
        data = pickle.load(f)
    data.pop("rng_state", None)
    with open(wd.snapshot_path, "wb") as f:
        pickle.dump(data, f)
    lease = read_lease(w_legacy, "wargen-A")
    lease["release"]["snapshot_sha256"] = packet._sha256(wd.snapshot_path)
    packet.lease_path(w_legacy, "wargen-A").write_text(json.dumps(lease, indent=1))
    wd.write_cursor(wd.cursor())                    # cursor mtime >= snapshot mtime

    hashes = {}
    for w in (w_intact, w_legacy):
        assert run("bid", "--world", str(w), "--node", "wargen-A") == 0
        assert evolve(w, "wargen-A") == 0
        hashes[w.name] = read_lease(w, "wargen-A")["release"]["field_canon_hash"]

    assert hashes["mood-intact"] != hashes["mood-legacy"], (
        "stripping rng_state did NOT move the released hash — the replay is not "
        "consuming the resumed stream position, i.e. the snapshot carries no "
        "effective RNG state (the pre-fix defect)")


# ── 2. the north star: incremental evolve == from-log referee (OPEN — xfail) ───────

@pytest.mark.xfail(
    strict=False,
    reason="continuation-state closure OPEN: engine.save does not yet capture the "
           "full live-state surface (calibration/sensor-bridge/berry/driver/hold "
           "memory, ~800 slots enumerated 2026-07-18), so a learning-hot tail "
           "batch can fork off the hidden state even with rng_state resumed. "
           "Chains that must referee clean re-evolve --from-log. See CLAIMS.md.")
def test_incremental_evolve_matches_from_log_referee(tmp_path):
    """The full protocol promise, pinned as the target: evolve from blank to
    cursor A, snapshot; tail arrives; re-bid + evolve INCREMENTALLY; a second
    node's from-log referee reproduces the released hash (exit 0). The stream is
    the real lease-drill artifact's — the exact shape that forked in the drill."""
    wdir = evolved_world(tmp_path)
    assert run("bid", "--world", str(wdir), "--node", "wargen-A") == 0
    assert evolve(wdir, "wargen-A") == 0
    assert run("verify", "--world", str(wdir), "--node", "wargen-B") == 0, (
        "CHAIN FORK: the from-log referee could not reproduce the incremental "
        "evolve's field_canon_hash")


# ── 3. from-log evolve + from-log referee is deterministic TODAY ────────────────────

def test_from_log_evolve_verifies_clean(tmp_path):
    """The lane that must always referee clean: a --from-log evolve (ignores the
    snapshot, replays the whole log, mints a state-carrying snapshot) followed by
    a second node's from-log verify — exit 0. This is the honest remedy for any
    forked or legacy chain, and it is pinned deterministic."""
    wdir = evolved_world(tmp_path)
    assert run("bid", "--world", str(wdir), "--node", "wargen-A") == 0
    assert evolve(wdir, "wargen-A", "--from-log") == 0
    assert run("verify", "--world", str(wdir), "--node", "wargen-B") == 0


# ── 4. verify runs the orphan check ─────────────────────────────────────────────────

def test_verify_quarantines_orphan_snapshot(tmp_path):
    """Spec: EVERY boot runs the orphan check. A snapshot whose sha disagrees
    with the last release (a crashed/partially-synced writer) must be set aside
    by the VERIFY boot too — and the referee still passes from the log alone."""
    wdir = make_world(tmp_path)
    wd = WorldDir(wdir)
    head, _ = fixture_stream()
    wd.append_events(head)
    assert evolve(wdir, "wargen-A") == 0

    # a crashed writer's partial snapshot: bytes disagree with the release sha
    wd.snapshot_path.write_bytes(wd.snapshot_path.read_bytes() + b"\x00garbage")

    assert run("verify", "--world", str(wdir), "--node", "wargen-B") == 0
    assert not wd.snapshot_path.exists(), "orphan snapshot was not quarantined"
    assert list(wdir.glob("snapshot.pkl.orphan-*")), "quarantine file missing"
    assert not wd.cursor_path.exists(), "orphan's cursor was not cleared"


# ── 5. a re-bid preserves the node's completed releases ─────────────────────────────

def test_rebid_preserves_prior_releases(tmp_path):
    """write_bid used to overwrite the lease with release: None, erasing that
    node's completed history — the ledger kept at most one release per node.
    Now the previous release rides forward in prior_releases (newest last,
    capped at MAX_PRIOR_RELEASES)."""
    world = tmp_path / "w"
    world.mkdir()

    def bid():
        packet.write_bid(world, "alpha", "evolve-packet", 600, None, "", False)

    def release(h):
        packet._update_own_bid(world, "alpha", release={
            "released_ts": f"2026-07-18T00:00:{h % 60:02d}+00:00",
            "field_canon_hash": f"hash-{h}"})

    bid()
    release(1)
    bid()                                   # the re-bid that used to erase hash-1
    lease = read_lease(world, "alpha")
    assert lease["release"] is None
    assert [r["field_canon_hash"] for r in lease["prior_releases"]] == ["hash-1"]

    release(2)
    bid()
    lease = read_lease(world, "alpha")
    assert [r["field_canon_hash"] for r in lease["prior_releases"]] == \
        ["hash-1", "hash-2"]

    # the cap: newest MAX_PRIOR_RELEASES survive, oldest fall off
    for h in range(3, 3 + packet.MAX_PRIOR_RELEASES + 5):
        release(h)
        bid()
    kept = [r["field_canon_hash"]
            for r in read_lease(world, "alpha")["prior_releases"]]
    assert len(kept) == packet.MAX_PRIOR_RELEASES
    assert kept[-1] == f"hash-{2 + packet.MAX_PRIOR_RELEASES + 5}"

    # a live (unreleased) re-bid keeps the history untouched
    bid()
    assert [r["field_canon_hash"]
            for r in read_lease(world, "alpha")["prior_releases"]] == kept


# ── 6. a legacy snapshot (no rng_state) still loads, loudly ─────────────────────────

def test_legacy_snapshot_without_rng_state_still_loads(tmp_path, caplog):
    """Real snapshots predate the fix (the lease-drill artifact among them). A
    snapshot missing rng_state must still boot — seed-once fallback — with a
    loud warning that incremental chains off it referee only from the log."""
    wdir = make_world(tmp_path)
    wd = WorldDir(wdir)
    head, _ = fixture_stream()
    wd.append_events(head)
    assert evolve(wdir, "wargen-A") == 0

    with open(wd.snapshot_path, "rb") as f:
        data = pickle.load(f)
    assert "rng_state" in data
    del data["rng_state"]                       # forge the pre-fix format
    with open(wd.snapshot_path, "wb") as f:
        pickle.dump(data, f)

    with caplog.at_level(logging.WARNING, logger="umwelt.engine"):
        engine, last_ts, _ = packet.boot_engine(
            wdir, freeze_learning=False, until_ts=None)
    assert engine.field_canon_hash()
    assert last_ts >= wd.cursor()
    assert any("LEGACY snapshot" in r.message for r in caplog.records), \
        "legacy snapshot loaded silently — the fallback must be loud"
