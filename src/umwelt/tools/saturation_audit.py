"""Grade a world's real evidence stream for one-sided saturation.

The scar: a role fed evidence that can only ever push one direction pins at
|z|≈1 and stays there. One real incident held z≈+0.95 for thirteen days and
nothing anywhere reported it, because every layer was working exactly as
designed — the poster posted, the binding normalized, the field integrated, and
the belief was simply wrong with full confidence.

CI cannot catch this. Synthetic proof worlds always feed both polarities,
because whoever writes the fixture writes the interesting case. The reframe that
makes it tractable: a detector needs no world at all. Saturation is a pure
function of (spec, event log), so this reads the spec for shape and the log for
what actually arrived, and never boots an engine.

Polarity is measured AFTER the binding's own normalizer, never on the raw posted
value. yurt's WARM_NORM inverts, so the sign a poster writes is frequently not
the sign the field saw — grading raw values would have graded the wrong number
and confidently reported the opposite of the truth.

Three verdicts, because "one-sided" is not one thing:

  SATURATION-RISK  unitary, n>=30, one polarity >=98%, and gap_p95 >= 10*gap_p50.
                   The gap ratio is what separates "one-sided because the world
                   is genuinely always-on" from "one-sided because OFF is a
                   re-arm timeout that never posts". At the hoses' ten-minute
                   cadence n>=30 is about five hours, which beats the thirteen-day
                   horizon by construction.

  MONOPOLAR        same one-sidedness, but the role is dissipative. NOT a defect:
                   git-stream's learning axis is monopolar by design and relaxes
                   toward ground on its own. It is reported anyway because the
                   cold half of that axis is unreachable by evidence, so any
                   reader treating z as "the evidence says cold" is wrong. This
                   tier is what keeps the tool honest instead of alarmist.

  STARVED          registered, bound, and zero events ever arrived.

Exit codes follow the packet referee's discipline: 0 clean / 1 findings /
2 nothing to audit.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sqlite3
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

MIN_SAMPLES = 30
ONE_SIDED_FRACTION = 0.98
GAP_RATIO = 10.0

CLEAN = 0
FINDINGS = 1
NOTHING = 2


# ---------------------------------------------------------------- spec loading


def load_spec(spec_path: Path, spec_ref: str):
    """Import a world's SPEC the way the daemon's worker does.

    spec_ref is "<module>:<attr>". Two registration styles exist in the wild --
    a bare `world:SPEC` with spec_path pointing at the world's own directory,
    and a dotted `worlds.X.world:SPEC` with spec_path at a repo root -- and both
    work here because spec_path goes on sys.path first either way.
    """
    module_name, _, attr = spec_ref.partition(":")
    attr = attr or "SPEC"
    if str(spec_path) not in sys.path:
        sys.path.insert(0, str(spec_path))
    return getattr(importlib.import_module(module_name), attr)


def resolve_world(world_dir: Path) -> tuple[object, dict]:
    manifest = json.loads((world_dir / "world.json").read_text(encoding="utf-8"))
    spec_path = manifest.get("spec_path")
    if not spec_path:
        raise ValueError(f"{world_dir.name}: world.json declares no spec_path")
    if not Path(spec_path).is_dir():
        raise FileNotFoundError(
            f"{world_dir.name}: spec_path does not exist: {spec_path}"
        )
    return load_spec(Path(spec_path), manifest.get("spec", "world:SPEC")), manifest


# ---------------------------------------------------------------- measurement


@dataclass
class Stream:
    sensor_id: str
    node: str
    role: str
    mode: str
    values: list[float] = field(default_factory=list)   # post-normalizer
    stamps: list[datetime] = field(default_factory=list)
    unnormalizable: int = 0

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def pos(self) -> float:
        return sum(1 for v in self.values if v > 0) / self.n if self.n else 0.0

    @property
    def neg(self) -> float:
        return sum(1 for v in self.values if v < 0) / self.n if self.n else 0.0

    @property
    def zero(self) -> float:
        return sum(1 for v in self.values if v == 0) / self.n if self.n else 0.0

    def gaps(self) -> list[float]:
        s = sorted(self.stamps)
        return [(b - a).total_seconds() for a, b in zip(s, s[1:])]

    def gap_stats(self) -> tuple[float | None, float | None]:
        g = self.gaps()
        if len(g) < 2:
            return None, None
        g.sort()
        p50 = statistics.median(g)
        # Nearest-rank p95: with the tens of samples this tool targets,
        # interpolation invents a value no reading ever had.
        p95 = g[min(len(g) - 1, max(0, int(round(0.95 * len(g))) - 1))]
        return p50, p95

    def verdict(self) -> tuple[str, str]:
        if self.n == 0:
            return "STARVED", "bound but no events ever arrived"
        if self.n < MIN_SAMPLES:
            return "OK", f"only {self.n} samples (need {MIN_SAMPLES} to judge)"
        share = max(self.pos, self.neg)
        if share < ONE_SIDED_FRACTION:
            return "OK", f"two-sided ({self.pos:.0%} pos / {self.neg:.0%} neg)"

        side = "positive" if self.pos >= self.neg else "negative"
        p50, p95 = self.gap_stats()
        if self.mode != "unitary":
            return "MONOPOLAR", (
                f"{share:.0%} {side} on a dissipative role -- relaxes on its own, "
                f"but the other half of this axis is unreachable by evidence"
            )
        if p50 and p95 and p95 >= GAP_RATIO * p50:
            return "SATURATION-RISK", (
                f"{share:.0%} {side} on a unitary role, and gaps are bursty "
                f"(p50 {p50:.0f}s, p95 {p95:.0f}s) -- the quiet half looks like a "
                f"re-arm timeout that never posts, not a genuinely always-on world"
            )
        return "MONOPOLAR", (
            f"{share:.0%} {side} on a unitary role, but evenly paced "
            f"(p50 {p50:.0f}s, p95 {p95:.0f}s) -- consistent with a world that "
            f"really is always on"
        )


def parse_stamp(raw: str) -> datetime | None:
    try:
        return datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None


def collect(world_dir: Path, spec) -> tuple[dict[str, Stream], dict[str, int]]:
    """Fold the event log through each binding's own normalizer."""
    modes = {n.name: (n.role_modes or {}) for n in spec.nodes}
    try:
        from umwelt.spec.roles import role_input_mode
    except ImportError:                                   # pragma: no cover
        def role_input_mode(_role: str) -> str:
            return "dissipative"

    streams: dict[str, Stream] = {}
    norms: dict[str, object] = {}
    for b in (spec.bindings or ()):
        mode = modes.get(b.zone, {}).get(b.role) or role_input_mode(b.role)
        streams[b.sensor_id] = Stream(b.sensor_id, b.zone, b.role, mode)
        try:
            norms[b.sensor_id] = b.build_normalizer()
        except Exception:
            norms[b.sensor_id] = None

    unmatched: dict[str, int] = {}
    db = world_dir / "events.db"
    if not db.is_file():
        return streams, unmatched

    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT timestamp, source_device, value FROM events ORDER BY timestamp"
        ).fetchall()
    finally:
        conn.close()

    for ts, sensor_id, raw in rows:
        st = streams.get(sensor_id)
        if st is None:
            unmatched[sensor_id] = unmatched.get(sensor_id, 0) + 1
            continue
        norm = norms.get(sensor_id)
        try:
            value = norm(float(raw)) if norm else float(raw)
        except (TypeError, ValueError, ZeroDivisionError):
            st.unnormalizable += 1
            continue
        st.values.append(float(value))
        stamp = parse_stamp(ts)
        if stamp:
            st.stamps.append(stamp)
    return streams, unmatched


# ---------------------------------------------------------------- reporting


def audit(world_dir: Path) -> dict:
    spec, manifest = resolve_world(world_dir)
    streams, unmatched = collect(world_dir, spec)

    findings, clean = [], []
    for st in sorted(streams.values(), key=lambda s: s.sensor_id):
        verdict, why = st.verdict()
        p50, p95 = st.gap_stats()
        row = {
            "sensor_id": st.sensor_id,
            "node": st.node,
            "role": st.role,
            "mode": st.mode,
            "n": st.n,
            "pos": round(st.pos, 4),
            "neg": round(st.neg, 4),
            "zero": round(st.zero, 4),
            "gap_p50_s": round(p50, 1) if p50 else None,
            "gap_p95_s": round(p95, 1) if p95 else None,
            "unnormalizable": st.unnormalizable,
            "verdict": verdict,
            "why": why,
        }
        (clean if verdict == "OK" else findings).append(row)

    return {
        "world": manifest.get("name", world_dir.name),
        "world_dir": str(world_dir),
        "spec_path": manifest.get("spec_path"),
        "bindings": len(streams),
        "events": sum(s.n for s in streams.values()),
        "findings": findings,
        "clean": clean,
        # A sensor posting into a world that has no binding for it is a poster
        # and a spec that disagree -- silent today, since the bridge drops it.
        "unmatched_sensors": [
            {"sensor_id": k, "events": v} for k, v in sorted(unmatched.items())
        ],
    }


def render(rep: dict) -> str:
    lines = [
        f"world     : {rep['world']}",
        f"bindings  : {rep['bindings']}",
        f"events    : {rep['events']}",
        "",
    ]
    rows = rep["findings"] + rep["clean"]
    if not rows:
        return "\n".join(lines + ["no bindings declared -- nothing to audit"])
    w = max(len(r["sensor_id"]) for r in rows)
    for r in rep["findings"] + rep["clean"]:
        lines.append(
            f"  {r['sensor_id']:<{w}}  {r['verdict']:<16} "
            f"{r['node']}.{r['role']} [{r['mode']}] n={r['n']}"
        )
        lines.append(f"  {'':<{w}}  {r['why']}")
    if rep["unmatched_sensors"]:
        lines.append("")
        lines.append("  unmatched sensors (posted, but no binding declares them):")
        for u in rep["unmatched_sensors"]:
            lines.append(f"    {u['sensor_id']}  ({u['events']} events, all dropped)")
    lines.append("")
    n = len(rep["findings"])
    lines.append(f"{n} finding(s)" if n else "clean")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--world-dir", type=Path, required=True,
                    help="a hearth-state world directory (holds world.json and events.db)")
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args(argv)

    if not (args.world_dir / "world.json").is_file():
        msg = f"{args.world_dir} has no world.json -- nothing to audit"
        print(json.dumps({"ok": None, "why": msg}) if args.json else msg)
        return NOTHING

    try:
        rep = audit(args.world_dir)
    except (FileNotFoundError, ValueError, ImportError, AttributeError) as exc:
        msg = f"{args.world_dir.name}: cannot audit -- {type(exc).__name__}: {exc}"
        print(json.dumps({"ok": None, "why": msg}) if args.json else msg)
        return NOTHING

    print(json.dumps(rep, indent=2) if args.json else render(rep))
    if not rep["bindings"]:
        return NOTHING
    return FINDINGS if rep["findings"] else CLEAN


if __name__ == "__main__":
    sys.exit(main())
