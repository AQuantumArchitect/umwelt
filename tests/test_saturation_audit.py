"""The saturation detector, exercised against synthesized event logs.

The bug class it exists to catch is invisible to ordinary CI because synthetic
proof worlds always feed both polarities -- whoever writes the fixture writes
the interesting case, and the interesting case is the one that never happens in
production. So these tests synthesize the LOG, not the world: a one-sided
events.db is about twenty lines against the same four-column DDL umweltd's
worldstore writes, and every verdict below is reachable without booting an
engine or running a daemon.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from umwelt.spec.normalizers import one_signed_on_nonnegative
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec
from umwelt.tools import saturation_audit as sat


# ── log synthesis ────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE events (
    timestamp     TEXT,
    source_device TEXT,
    value         TEXT,
    metadata      TEXT
)
"""


def write_log(path, rows):
    """rows: (offset_seconds, sensor_id, raw_value)."""
    base = datetime(2026, 8, 1, tzinfo=timezone.utc)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(DDL)
        conn.executemany(
            "INSERT INTO events VALUES (?, ?, ?, ?)",
            [
                ((base + timedelta(seconds=off)).isoformat(), sid, str(val), json.dumps({}))
                for off, sid, val in rows
            ],
        )
        conn.commit()
    finally:
        conn.close()


def bursty(n_bursts=4, per_burst=10, tight=60, quiet=6000):
    """Readings that arrive in tight bursts separated by long silences.

    The shape that makes SATURATION-RISK distinguishable: most gaps are the
    hose's real cadence and a few are the silence where the other polarity
    would have been. Spacing every reading evenly by `quiet` would put the
    median AT the long gap and the ratio at 1 -- which is the always-on case,
    not this one.
    """
    rows, off = [], 0
    for b in range(n_bursts):
        for i in range(per_burst):
            rows.append((off, "hose", 1.0))
            off += tight
        off += quiet - tight
    return rows


def make_world(tmp_path, spec_module_src: str, rows, name="probe"):
    """A hearth-state world directory: world.json, a spec package, events.db."""
    spec_dir = tmp_path / "spec"
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "world.py").write_text(spec_module_src, encoding="utf-8")

    world_dir = tmp_path / "world"
    world_dir.mkdir(parents=True, exist_ok=True)
    (world_dir / "world.json").write_text(
        json.dumps({"name": name, "spec": "world:SPEC", "spec_path": str(spec_dir)}),
        encoding="utf-8",
    )
    if rows is not None:
        write_log(world_dir / "events.db", rows)
    return world_dir


def spec_src(role: str, mode: str, normalizer: dict | str) -> str:
    return f"""
from umwelt.spec.schema import BindingSpec, DomainSpec, NodeSpec
SPEC = DomainSpec(
    name="probe",
    nodes=(
        NodeSpec("root", parent=None, kind="root", roles=("{role}",),
                 role_modes={{"{role}": "{mode}"}}),
    ),
    bindings=(
        BindingSpec("hose", zone="root", role="{role}", normalizer={normalizer!r}),
    ),
)
"""


def verdict_of(world_dir, sensor="hose"):
    rep = sat.audit(world_dir)
    rows = rep["findings"] + rep["clean"]
    return next(r for r in rows if r["sensor_id"] == sensor)["verdict"]


# ── the three verdicts ───────────────────────────────────────────────────────


def test_starved_when_nothing_ever_arrived(tmp_path):
    """Today's live defect: registered, bound, and never fed once."""
    w = make_world(tmp_path, spec_src("health", "dissipative", "binary"), rows=[])
    assert verdict_of(w) == "STARVED"


def test_bursty_one_sided_unitary_is_saturation_risk(tmp_path):
    """Tight bursts with long silences: the quiet half is a timeout, not a world.

    Most gaps are the hose's 60s cadence and three are 6000s, so p95 clears the
    10x p50 bar and the detector can tell "OFF never posts" from "genuinely
    always on" -- which polarity alone cannot.
    """
    w = make_world(tmp_path, spec_src("attention", "unitary", "binary"), bursty())
    assert verdict_of(w) == "SATURATION-RISK"


def test_evenly_paced_one_sided_unitary_is_only_monopolar(tmp_path):
    """Same 100% polarity, even cadence -- consistent with a world that is on.

    This is the case that keeps the tool from crying wolf, and it is the reason
    the gap ratio is in the rule at all rather than polarity alone.
    """
    rows = [(i * 600, "hose", 1.0) for i in range(40)]
    w = make_world(tmp_path, spec_src("attention", "unitary", "binary"), rows)
    assert verdict_of(w) == "MONOPOLAR"


def test_one_sided_dissipative_is_monopolar_not_a_defect(tmp_path):
    """git-stream's learning axis is monopolar by design; it relaxes on its own."""
    w = make_world(tmp_path, spec_src("learning", "dissipative", "binary"), bursty())
    assert verdict_of(w) == "MONOPOLAR"


def test_two_sided_stream_is_clean(tmp_path):
    rows = [(i * 600, "hose", 1.0 if i % 2 else 0.0) for i in range(40)]
    w = make_world(tmp_path, spec_src("attention", "unitary", "binary"), rows)
    rep = sat.audit(w)
    assert rep["findings"] == []


def test_below_threshold_is_not_judged(tmp_path):
    """Under MIN_SAMPLES the tool says so rather than guessing from noise."""
    rows = [(i * 600, "hose", 1.0) for i in range(5)]
    w = make_world(tmp_path, spec_src("attention", "unitary", "binary"), rows)
    assert verdict_of(w) == "OK"


# ── the normalizer is the thing that decides polarity ────────────────────────


def test_polarity_is_measured_after_the_normalizer(tmp_path):
    """An inverting normalizer flips the sign the field actually saw.

    Every raw value here is positive. Read raw, the stream is 100% positive.
    Read through yurt's WARM_NORM shape, which inverts, the field saw 100%
    NEGATIVE. Grading the poster's sign would report the opposite of the truth,
    so this pins that the fold happens after build_normalizer().
    """
    warm = {"type": "regime", "center": 0.0, "width": 0.5, "invert": True}
    rows = [(i * 600, "hose", 0.9) for i in range(40)]
    w = make_world(tmp_path, spec_src("attention", "unitary", warm), rows)
    rep = sat.audit(w)
    row = (rep["findings"] + rep["clean"])[0]
    assert row["neg"] == 1.0 and row["pos"] == 0.0


# ── the stream and the spec disagreeing ──────────────────────────────────────


def test_unmatched_sensor_is_reported_not_silently_dropped(tmp_path):
    """A poster writing an id no binding declares is dropped by the bridge.

    Live instance: hive-ops takes node_body_claim and project-membrane takes
    yurt_blocked, both unbound, both discarded without a word.
    """
    rows = [(i * 600, "hose", 1.0) for i in range(3)]
    rows += [(i * 600, "ghost_sensor", 1.0) for i in range(7)]
    w = make_world(tmp_path, spec_src("attention", "unitary", "binary"), rows)
    rep = sat.audit(w)
    assert rep["unmatched_sensors"] == [{"sensor_id": "ghost_sensor", "events": 7}]


def test_missing_spec_path_cannot_be_audited(tmp_path):
    """butler-life's live shape: registered against a directory that is gone."""
    world_dir = tmp_path / "world"
    world_dir.mkdir()
    (world_dir / "world.json").write_text(
        json.dumps({"name": "gone", "spec": "world:SPEC",
                    "spec_path": str(tmp_path / "nope")}),
        encoding="utf-8",
    )
    assert sat.main(["--world-dir", str(world_dir)]) == sat.NOTHING


def test_exit_codes_follow_the_referee_discipline(tmp_path):
    """0 clean / 1 findings / 2 nothing to audit -- the primary contract."""
    clean = make_world(
        tmp_path / "a", spec_src("attention", "unitary", "binary"),
        [(i * 600, "hose", 1.0 if i % 2 else 0.0) for i in range(40)],
    )
    starved = make_world(tmp_path / "b", spec_src("attention", "unitary", "binary"), [])
    (tmp_path / "c").mkdir()

    assert sat.main(["--world-dir", str(clean)]) == sat.CLEAN
    assert sat.main(["--world-dir", str(starved)]) == sat.FINDINGS
    assert sat.main(["--world-dir", str(tmp_path / "c")]) == sat.NOTHING


@pytest.fixture(autouse=True)
def _isolate_spec_modules():
    """Each world writes its own spec as `world.py`; drop it between tests.

    Otherwise the second import returns the first test's module from
    sys.modules and every case after the first grades the wrong spec -- the
    same process-global trap that makes proofs/run_all.py use subprocesses.
    """
    import sys
    yield
    sys.modules.pop("world", None)
    sys.path[:] = [p for p in sys.path if "spec" not in p.split("\\")[-1:]]


# ── the static half ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "spec,expect_flagged",
    [
        ({"type": "regime", "center": 0.0, "width": 0.5, "invert": True}, True),
        ({"type": "regime", "center": 21, "width": 4}, False),
        ({"type": "threshold", "threshold": -1.0}, True),
        ({"type": "threshold", "threshold": 5.0}, False),
        ({"type": "range", "lo": -10, "hi": 5}, True),
        ({"type": "range", "lo": -1, "hi": 5}, False),
        ({"type": "range", "lo": 0, "hi": 10}, False),
        ({"type": "range", "lo": 3, "hi": 3}, True),
        ("binary", False),
        ("forecast_zflip", False),
        ({"type": "cyclic", "period": 24}, False),
        ({"type": "domain_registered_thing"}, False),   # unknown is not safe, but not guessed
    ],
)
def test_one_signed_analysis(spec, expect_flagged):
    assert bool(one_signed_on_nonnegative(spec)) is expect_flagged


def test_validate_warns_on_a_one_sided_unitary_role():
    """The warning teaches; it must never gate."""
    from umwelt.spec.validate import validate_spec

    spec = DomainSpec(
        name="one_sided",
        nodes=(
            NodeSpec("root", parent=None, kind="root", roles=("attention",),
                     role_modes={"attention": "unitary"}),
        ),
        bindings=(
            BindingSpec("hose", zone="root", role="attention",
                        normalizer={"type": "regime", "center": 0.0,
                                    "width": 0.5, "invert": True}),
        ),
    )
    report = validate_spec(spec)
    check = next(c for c in report.checks if c.name == "one_sided_risk")
    assert check.warning and check.ok
    assert "root.attention" in check.detail


def test_validate_stays_quiet_when_a_binding_can_push_back():
    """One two-sided binding is enough; the role has a way back."""
    from umwelt.spec.validate import validate_spec

    spec = DomainSpec(
        name="balanced",
        nodes=(
            NodeSpec("root", parent=None, kind="root", roles=("attention",),
                     role_modes={"attention": "unitary"}),
        ),
        bindings=(
            BindingSpec("warm", zone="root", role="attention",
                        normalizer={"type": "regime", "center": 0.0,
                                    "width": 0.5, "invert": True}),
            BindingSpec("cold", zone="root", role="attention", normalizer="binary"),
        ),
    )
    report = validate_spec(spec)
    assert not any(c.name == "one_sided_risk" for c in report.checks)
