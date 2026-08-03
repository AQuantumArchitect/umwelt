"""The continuation surface — every live slot a snapshot would have to carry.

`engine.save` is a hand-maintained registry of what matters, and drifting is
exactly what it did: the 2026-07-18 lease-drill chain forked because state the
registry had never heard of kept evolving after a load. So this module does NOT
read the registry. It walks the live object graph reflectively and reports what
is *there*, which is the only way to find a slot nobody remembered to declare.

The claim this replaces is "~800 uncaptured state slots", a number that appears
in three places as prose and is derivable from no artifact on this machine. A
number you cannot recompute is not a measurement, so the point of this walker is
that every figure it prints comes with the command that reprints it.

**The ownership boundary.** We recurse into `__dict__` only for types whose
module starts with `umwelt.`. That one rule does a surprising amount of work:
numpy arrays, stdlib containers, loggers, sockets and RNG objects all become
natural leaves without an exclusion list to maintain, and any umwelt type added
later is picked up automatically. What it costs is honest and declared: state
hiding inside a third-party object is invisible here, so `walk()` also returns
the set of opaque types it refused to open (`Surface.opaque`) — you can read
what you are not being shown.

**Precision.** Floats canonicalize at 12 decimals, two more than
`field_unify.canon_hash`'s 10. The probe must be strictly more sensitive than
the observable it exists to explain; a divergence this walker cannot see but the
canon hash can would make the whole tool a liar.

**Aliasing and cycles are load-bearing, not edge cases.**
`collapse_engine.touched_roles` *is* `sensor_bridge.touched_roles` (one set, two
holders — `engine.py:148`), and `CalibrationLoop.graph` back-references its
owner. Memoizing by `id()` handles the cycle; recording the alias as a pointer
to the first path that reached the object means a loader that restores two
copies where there was one shows up as a diff rather than as silence. That is
the exact shape of a well-intentioned loader quietly breaking this.

Nothing here has side effects: we read `__dict__` (and `__slots__`) directly and
never call a property getter. A computed projection is not state — and per the
rule this whole bug class comes from, a display projection is never the
persistence record.
"""

from __future__ import annotations

import collections
import datetime as _dt
import hashlib
import json
import math
from dataclasses import dataclass, field as _dc_field
from typing import Any, Callable, Iterable

import numpy as np

# Two more decimals than field_unify.canon_hash rounds to (10). Strictly more
# sensitive than the observable it explains — see the module docstring.
FLOAT_DECIMALS = 12

# Recursion depth is a backstop against a pathological graph, not a tuning knob.
# The live engine bottoms out around 12; anything approaching this is a bug worth
# seeing, so hitting it is recorded in the surface rather than silently truncated.
MAX_DEPTH = 60

# Which engine attribute owns a shared object. The first path to reach an object
# becomes its home and every other holder becomes an alias, so without a declared
# order the home is decided by alphabetical accident: `calibration` sorts before
# `field`, and the entire belief field ends up filed under the learner that merely
# points at it. That reads fine and ruins the ablation ranking, whose whole
# question is "does THIS owner alone flip the hash".
#
# Order is primary-holder first. It is part of the capture format — two captures
# taken under different orders disagree on alias paths and nothing else, which is
# a diff made entirely of bookkeeping.
ENGINE_ORDER = (
    "field", "graph", "world", "collapse_engine", "sensor_bridge",
    "output_surface", "berry_tape", "bloch_berry", "fractal_stack", "agency",
    "population", "tendrils", "drivers", "calibration", "training",
)


_UNSET = object()

# Slots seeded from the wall clock AT CONSTRUCTION, so two engines built in two
# processes differ here even with every RNG pinned. This is not a defect and not
# an exemption: each one round-trips correctly through save/load, so the digest
# includes them and should. It exists so that comparing two independently BUILT
# engines can subtract the difference it already knows about — and so that a
# newly-added clock-seeded slot shows up as a test failure naming its own path
# rather than as an intermittent "the walker is nondeterministic".
WALL_CLOCK_SEEDED: dict[str, str] = {
    "engine.agency._last": (
        "AgencyQubit.__init__ defaults to time.time() (agency_qubit.py:41). It "
        "rides the snapshot via state()/load() ('last'), so it restores exactly; "
        "only two separate BUILDS disagree."),
}


def _is_owned(obj: Any) -> bool:
    """True for types this project owns — the recursion boundary."""
    mod = getattr(type(obj), "__module__", "") or ""
    return mod == "umwelt" or mod.startswith("umwelt.")


def canon_float(x: float) -> float | str:
    """A float in a form two processes agree on.

    -0.0 and 0.0 hash differently as bytes and mean the same thing; NaN is not
    equal to itself, so it cannot ride as a number at all without making every
    comparison false for a reason that has nothing to do with divergence.
    """
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    r = round(float(x), FLOAT_DECIMALS)
    return 0.0 if r == 0.0 else r          # collapses -0.0


def canon_array(a: np.ndarray) -> dict:
    """An ndarray as a comparable leaf, with a magnitude so a diff can rank.

    A digest alone answers "did it move" but not "by how much", and the ablate
    verb needs the second question — an owner whose subtree moved in the 1e-13
    range is a rounding artifact, one that moved by 0.3 is the reason the hash
    forked. Both travel, so the report can say which.
    """
    arr = np.ascontiguousarray(np.asarray(a))
    if arr.dtype.kind in "fc":
        # Round before hashing: bit-level float noise below the declared
        # precision is not a divergence, and treating it as one buries the
        # real ones in a report nobody finishes reading.
        rounded = np.round(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0),
                           FLOAT_DECIMALS)
        digest = hashlib.sha256(rounded.tobytes()).hexdigest()[:16]
        finite = arr[np.isfinite(arr)] if arr.size else arr
        norm = float(np.linalg.norm(finite.ravel())) if finite.size else 0.0
        has_nan = bool(arr.size and not np.isfinite(arr).all())
    else:
        digest = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
        norm = 0.0
        has_nan = False
    return {"__ndarray__": True, "shape": list(arr.shape), "dtype": str(arr.dtype),
            "digest": digest, "norm": canon_float(norm),
            **({"nonfinite": True} if has_nan else {})}


def _opaque_tag(obj: Any) -> str:
    """A leaf for something we deliberately do not open.

    The TYPE, never the repr: most reprs embed a memory address, so a repr leaf
    would report a divergence on every single boot and the tool would be
    unusable within one run of itself.

    Callables are the exception worth making. A bound callback stored on an
    engine attribute is a real piece of wiring, and `<builtins.method>` for all
    of them means swapping one for another is invisible — so those carry their
    qualname, which is stable and address-free.
    """
    if callable(obj) and hasattr(obj, "__qualname__"):
        return f"<callable {getattr(obj, '__module__', '?')}.{obj.__qualname__}>"
    t = type(obj)
    return f"<{getattr(t, '__module__', '?')}.{getattr(t, '__qualname__', t.__name__)}>"


def _sortable(keys: Iterable) -> bool:
    ks = list(keys)
    return all(isinstance(k, str) for k in ks) or all(
        isinstance(k, int) and not isinstance(k, bool) for k in ks)


def _key_repr(k: Any) -> str:
    return k if isinstance(k, str) else repr(k)


@dataclass
class Surface:
    """One engine's continuation surface: flat path -> canonical leaf."""

    slots: dict[str, Any] = _dc_field(default_factory=dict)
    aliases: dict[str, str] = _dc_field(default_factory=dict)
    opaque: dict[str, int] = _dc_field(default_factory=dict)
    truncated: list[str] = _dc_field(default_factory=list)
    # Actual leaf values, populated only when walk(raw=True). This is what the
    # ablate verb splices: stamping VALUES by path, rather than pickling the live
    # objects, is the difference between a tool that works and one that dies on
    # the first lambda in CalibrationLoop — and it preserves object identity, so
    # splicing cannot silently break the aliases the walk exists to watch.
    raw: dict[str, Any] = _dc_field(default_factory=dict)

    def digest(self) -> str:
        """Content hash of the whole surface. Stable across processes.

        Aliases ride in the digest on purpose: two holders sharing one object is
        part of the state, and a loader that restores them as two objects has
        changed the engine even when every number matches.
        """
        h = hashlib.sha256()
        for path in sorted(self.slots):
            h.update(path.encode())
            h.update(b"\x00")
            h.update(json.dumps(self.slots[path], sort_keys=True,
                                separators=(",", ":")).encode())
            h.update(b"\x01")
        for path in sorted(self.aliases):
            h.update(f"@{path}={self.aliases[path]}".encode())
            h.update(b"\x01")
        return h.hexdigest()[:16]

    def owners(self, depth: int = 1) -> dict[str, list[str]]:
        """Group slot paths by their top `depth` segments — the ablation unit."""
        out: dict[str, list[str]] = {}
        for path in sorted(self.slots):
            out.setdefault(owner_of(path, depth), []).append(path)
        return out

    def to_json(self) -> dict:
        return {"format": 1, "float_decimals": FLOAT_DECIMALS,
                "root_order": list(ENGINE_ORDER),
                "digest": self.digest(), "slots": self.slots,
                "aliases": self.aliases, "opaque": self.opaque,
                "truncated": self.truncated}

    @classmethod
    def from_json(cls, d: dict) -> "Surface":
        # Both guards exist because the failure they prevent is a diff that looks
        # entirely real: mismatched precision produces one made of rounding,
        # mismatched root order one made of alias bookkeeping. Either would be
        # read as divergence and chased.
        got = int(d.get("float_decimals", -1))
        if got != FLOAT_DECIMALS:
            raise ValueError(
                f"capture was taken at {got} decimals, this build canonicalizes at "
                f"{FLOAT_DECIMALS} — recapture both sides")
        order = d.get("root_order")
        if order is not None and tuple(order) != ENGINE_ORDER:
            raise ValueError(
                "capture was taken under a different ENGINE_ORDER — recapture both sides")
        return cls(slots=d.get("slots", {}), aliases=d.get("aliases", {}),
                   opaque=d.get("opaque", {}), truncated=d.get("truncated", []))


def owner_of(path: str, depth: int = 1) -> str:
    """The owning subtree of a slot path — `engine.a.b.c` at depth 1 is `a`.

    Integer subscripts are NOT owner boundaries; string keys are. A list is one
    thing (`_anticipation_raw[0]` and `[1]` are the same owner), whereas a dict
    keyed by world or cluster name is genuinely many. Getting this wrong makes
    ablation useless in the most expensive way available: it turns one owner
    into a hundred owners of one slot each, and then boots an engine per owner.
    """
    steps = parse_path(path)[1:]              # drop the root name
    segs: list[str] = []
    for kind, key in steps:
        if kind == _STEP_ATTR:
            segs.append(str(key))
        elif kind == _STEP_KEY and segs:
            segs[-1] += f"[{key!r}]"
        # _STEP_IDX: a position in a sequence, not a separate owner
        if len(segs) > depth:
            break
    return ".".join(segs[:depth]) or path


def walk(root: Any, *, name: str = "engine",
         skip: Callable[[str], bool] | None = None,
         first: tuple[str, ...] = ENGINE_ORDER,
         raw: bool = False) -> Surface:
    """Walk the live object graph and return its continuation surface.

    `skip` is given each path and may prune a subtree. It exists for the
    declared-cosmetic set, and every caller that uses it should be able to say
    why in one line — an unexplained exclusion here is how the registry this
    module replaces went wrong in the first place.

    `first` names the root attributes to visit before the rest, deciding which
    holder owns a shared object — see ENGINE_ORDER.
    """
    surf = Surface()
    seen: dict[int, str] = {}
    keep_alive: list[Any] = []   # ids are only unique while the object lives

    def emit(path: str, value: Any, source: Any = _UNSET) -> None:
        surf.slots[path] = value
        if raw and source is not _UNSET:
            surf.raw[path] = source

    def visit(obj: Any, path: str, depth: int) -> None:
        if skip is not None and skip(path):
            return
        if depth > MAX_DEPTH:
            surf.truncated.append(path)
            return

        if isinstance(obj, bool) or obj is None:
            return emit(path, obj, obj)
        if isinstance(obj, float):
            return emit(path, canon_float(obj), obj)
        if isinstance(obj, int):
            return emit(path, obj, obj)
        if isinstance(obj, str):
            return emit(path, obj, obj)
        if isinstance(obj, bytes):
            return emit(path, hashlib.sha256(obj).hexdigest()[:16], obj)
        if isinstance(obj, np.ndarray):
            return emit(path, canon_array(obj), obj)
        if isinstance(obj, np.generic):                    # numpy scalar
            v = obj.item()
            return emit(path, canon_float(v) if isinstance(v, float) else v, obj)
        if isinstance(obj, complex):
            return emit(path, [canon_float(obj.real), canon_float(obj.imag)], obj)
        if isinstance(obj, (_dt.datetime, _dt.date, _dt.timedelta)):
            # A timestamp is continuation state — "when did this last fire" is
            # precisely the kind of slot whose absence makes a reloaded engine
            # take a different branch on its first tick.
            return emit(path, str(obj))

        # Identity BEFORE container/object descent: this is what makes cycles
        # terminate and aliasing visible.
        oid = id(obj)
        if oid in seen:
            surf.aliases[path] = seen[oid]
            return
        seen[oid] = path
        keep_alive.append(obj)

        if isinstance(obj, dict):
            items = list(obj.items())
            if _sortable(obj.keys()):
                items.sort(key=lambda kv: kv[0])
            for k, v in items:
                visit(v, f"{path}[{_key_repr(k)!r}]" if not isinstance(k, str)
                      else f"{path}[{k!r}]", depth + 1)
            return
        if isinstance(obj, (list, tuple, collections.deque)):
            # deque belongs here, not in the opaque bucket. A bounded deque of
            # recent values IS continuation state — rolling windows, recent
            # surprise, tick history — and the ownership rule would have filed
            # every one of them as an unopened stdlib leaf, hiding exactly the
            # kind of slot that keeps evolving after a load.
            if isinstance(obj, collections.deque) and obj.maxlen is not None:
                emit(f"{path}.maxlen", obj.maxlen)
            for i, v in enumerate(obj):
                visit(v, f"{path}[{i}]", depth + 1)
            return
        if isinstance(obj, (set, frozenset)):
            # A set has no order, so it must be canonicalized as a VALUE, not
            # walked as a sequence — otherwise the same set produces different
            # paths on two boots and every comparison is noise.
            try:
                elems = sorted(json.dumps(_leaf_of(e), sort_keys=True,
                                          separators=(",", ":")) for e in obj)
            except TypeError:
                elems = sorted(_opaque_tag(e) for e in obj)
            return emit(path, {"__set__": True, "n": len(obj),
                               "digest": hashlib.sha256(
                                   "|".join(elems).encode()).hexdigest()[:16]})

        if _is_owned(obj):
            d = getattr(obj, "__dict__", None)
            if d:
                for k in _attr_order(d, first if depth == 0 else ()):
                    visit(d[k], f"{path}.{k}", depth + 1)
            for sl in _slot_names(type(obj)):
                if hasattr(obj, sl):
                    visit(getattr(obj, sl), f"{path}.{sl}", depth + 1)
            if d is None and not _slot_names(type(obj)):
                emit(path, _opaque_tag(obj))
            return

        tag = _opaque_tag(obj)
        surf.opaque[tag] = surf.opaque.get(tag, 0) + 1
        emit(path, tag)

    def _leaf_of(e: Any) -> Any:
        if isinstance(e, float):
            return canon_float(e)
        if isinstance(e, (bool, int, str, type(None))):
            return e
        return _opaque_tag(e)

    visit(root, name, 0)
    return surf


def _attr_order(d: dict, first: tuple[str, ...]) -> list[str]:
    """Declared names first (in their declared order), then the rest sorted."""
    lead = [k for k in first if k in d]
    return lead + sorted(k for k in d if k not in set(lead))


def _slot_names(t: type) -> tuple[str, ...]:
    out: list[str] = []
    for k in t.__mro__:
        sl = k.__dict__.get("__slots__")
        if isinstance(sl, str):
            out.append(sl)
        elif isinstance(sl, Iterable):
            out.extend(str(s) for s in sl)
    return tuple(dict.fromkeys(out))


# ── comparison ────────────────────────────────────────────────────────────────────

@dataclass
class SlotDiff:
    path: str
    left: Any
    right: Any

    @property
    def magnitude(self) -> float:
        """How far apart, when that is answerable. -1.0 when it is not.

        Ranking owners by "did anything move" puts a 1e-13 rounding artifact
        level with the slot that forked the chain. The report needs to tell
        those apart or the ablation ordering is arbitrary.
        """
        return _magnitude(self.left, self.right)


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict) and v.get("__ndarray__"):
        n = v.get("norm")
        return float(n) if isinstance(n, (int, float)) else None
    return None


def _magnitude(left: Any, right: Any) -> float:
    a, b = _num(left), _num(right)
    if a is None or b is None:
        return -1.0
    return abs(a - b)


@dataclass
class SurfaceDiff:
    changed: list[SlotDiff] = _dc_field(default_factory=list)
    only_left: list[str] = _dc_field(default_factory=list)
    only_right: list[str] = _dc_field(default_factory=list)
    alias_changed: list[str] = _dc_field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.changed or self.only_left or self.only_right
                    or self.alias_changed)

    def by_owner(self, depth: int = 1) -> dict[str, dict]:
        out: dict[str, dict] = {}
        def bucket(p: str) -> dict:
            return out.setdefault(owner_of(p, depth),
                                  {"changed": 0, "only_left": 0, "only_right": 0,
                                   "alias_changed": 0, "max_magnitude": -1.0,
                                   "examples": []})
        for d in self.changed:
            b = bucket(d.path)
            b["changed"] += 1
            b["max_magnitude"] = max(b["max_magnitude"], d.magnitude)
            if len(b["examples"]) < 3:
                b["examples"].append(d.path)
        for p in self.only_left:
            b = bucket(p); b["only_left"] += 1
            if len(b["examples"]) < 3:
                b["examples"].append(p)
        for p in self.only_right:
            b = bucket(p); b["only_right"] += 1
            if len(b["examples"]) < 3:
                b["examples"].append(p)
        for p in self.alias_changed:
            b = bucket(p); b["alias_changed"] += 1
        return out

    def total(self) -> int:
        return (len(self.changed) + len(self.only_left) + len(self.only_right)
                + len(self.alias_changed))


def diff(left: Surface, right: Surface) -> SurfaceDiff:
    """Compare two surfaces. `left` is the reference (the from-log twin)."""
    out = SurfaceDiff()
    lk, rk = set(left.slots), set(right.slots)
    out.only_left = sorted(lk - rk)
    out.only_right = sorted(rk - lk)
    for p in sorted(lk & rk):
        a, b = left.slots[p], right.slots[p]
        if a != b:
            out.changed.append(SlotDiff(p, a, b))
    for p in sorted(set(left.aliases) | set(right.aliases)):
        if left.aliases.get(p) != right.aliases.get(p):
            out.alias_changed.append(p)
    return out


# ── stamping values back by path (what `ablate` splices with) ─────────────────────

_STEP_ATTR, _STEP_KEY, _STEP_IDX = "attr", "key", "idx"


def parse_path(path: str) -> list[tuple[str, Any]]:
    """`engine.field.clusters['camp'].rho` -> the steps to walk it.

    Only the forms `walk()` itself emits are accepted, and anything else raises
    rather than being guessed at — a stamper that silently mis-resolves a path
    writes learned state into the wrong slot, which is a worse outcome than
    every failure this module was built to find.
    """
    steps: list[tuple[str, Any]] = []
    i, n = 0, len(path)
    buf = ""
    while i < n:
        ch = path[i]
        if ch == ".":
            if buf:
                steps.append((_STEP_ATTR, buf))
                buf = ""
            i += 1
        elif ch == "[":
            if buf:
                steps.append((_STEP_ATTR, buf))
                buf = ""
            j = path.index("]", i)
            inner = path[i + 1:j]
            if inner and inner[0] in "'\"":
                steps.append((_STEP_KEY, inner[1:-1]))
            else:
                steps.append((_STEP_IDX, int(inner)))
            i = j + 1
        else:
            buf += ch
            i += 1
    if buf:
        steps.append((_STEP_ATTR, buf))
    return steps


def stamp(root: Any, path: str, value: Any) -> str:
    """Write `value` at `path` under `root`. Returns how it landed.

    Arrays are written THROUGH the existing object (`a[...] = v`) whenever the
    shape allows, never rebound. Rebinding would give one holder of a shared
    array a private copy — silently converting an alias into two objects, which
    is the precise failure `test_aliased_state_survives_restore` exists to catch.
    Doing that inside the tool that measures aliasing would be self-defeating.
    """
    steps = parse_path(path)
    if not steps:
        raise ValueError(f"empty path: {path!r}")
    cur = root
    for kind, key in steps[1:-1]:                       # steps[0] is the root name
        cur = getattr(cur, key) if kind == _STEP_ATTR else cur[key]
    kind, key = steps[-1]
    if kind == _STEP_ATTR:
        old = getattr(cur, key, None)
        if isinstance(old, np.ndarray) and isinstance(value, np.ndarray) \
                and old.shape == value.shape:
            old[...] = value
            return "in-place"
        setattr(cur, key, value)
        return "rebound"
    old = cur[key] if (kind == _STEP_IDX or key in cur) else None
    if isinstance(old, np.ndarray) and isinstance(value, np.ndarray) \
            and old.shape == value.shape:
        old[...] = value
        return "in-place"
    cur[key] = value
    return "rebound"


def splice(target: Any, raw: dict[str, Any], paths: Iterable[str]) -> dict:
    """Stamp the given paths from a donor's raw capture onto `target`.

    Reports what actually landed. An owner's subtree is named by path prefix,
    but objects hold back-references, so "I spliced `calibration`" is a claim
    about intent — the counts here are what makes it a measurement instead.
    """
    landed = {"in_place": 0, "rebound": 0, "missing": 0, "failed": 0,
              "errors": []}
    for p in paths:
        if p not in raw:
            landed["missing"] += 1
            continue
        try:
            how = stamp(target, p, raw[p])
            landed["in_place" if how == "in-place" else "rebound"] += 1
        except Exception as exc:                          # noqa: BLE001
            landed["failed"] += 1
            if len(landed["errors"]) < 5:
                landed["errors"].append(f"{p}: {type(exc).__name__}: {exc}")
    return landed


# ── the round-trip digest (the live-boot guard) ───────────────────────────────────

# Slots excluded from the ROUND-TRIP digest only — never from the probe, which
# must keep showing them. Each entry is a claim that the slot cannot round-trip
# *and does not need to*, and each one is a place this guard is blind, so they
# are listed individually with a reason rather than as a pattern.
DIGEST_EXEMPT: dict[str, str] = {
    "engine._continuation": (
        "The verdict OF this check, stored on the engine. Including it makes the "
        "digest self-referential: a snapshot saved by an engine that had itself "
        "loaded carries '_continuation=exact' inside its digest, while the engine "
        "being graded still reads 'fresh' at check time — so the second generation "
        "of every chain would report a mismatch caused entirely by the reporting."),
}


def _digest_skip(path: str) -> bool:
    return any(path == p or path.startswith(p + ".") or path.startswith(p + "[")
               for p in DIGEST_EXEMPT)


def continuation_digest(engine: Any) -> str | None:
    """The engine's continuation surface as one hash, or None if it cannot be taken.

    Recorded by `engine.save` and recomputed after `engine.load`. A mismatch
    proves `load(save(E)) != E` on the declared surface — at the cursor, before
    a single tail event replays, for the cost of one walk. That is strictly
    weaker than the probe (it cannot see a slot the walker never reaches) and
    strictly cheaper (it runs on every live boot, not in a drill).

    Never raises: a snapshot must be writable even when this cannot be computed,
    exactly as `field_canon_hash` already rides best-effort in `save`.
    """
    try:
        return walk(engine, name="engine", skip=_digest_skip).digest()
    except Exception:            # noqa: BLE001 — a diagnostic may never fail a save
        return None
