"""The continuation walker must not be able to lie about what it found.

This module measures whether `load(save(E))` restores the engine. That makes it
the last place a false negative can hide: a walker that quietly skips a subtree
reports a clean round-trip for an engine that did not round-trip, and the
resulting confidence is worse than the ignorance it replaced.

So these tests grade the instrument, not the engine. Two properties carry most
of the weight:

  * determinism — the same object graph must hash identically in two processes,
    or every comparison is noise
  * completeness — state the walker cannot see is state the digest cannot guard,
    so anything it declines to open must be REPORTED as declined

The engine-level tests live at the bottom and are marked slow; they boot a real
spec, which is the only way to grade the ownership boundary against the object
graph it was written for.
"""

from __future__ import annotations

import collections
import math
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from umwelt.substrate import continuation as C   # noqa: E402


# A stand-in for an umwelt-owned type: the walker's recursion boundary is the
# module name, so a fake must live in a module that starts with "umwelt." to be
# treated as owned. Defining it here would make it `tests.test_continuation`.
class _Owned:
    """Patched into an umwelt module below so `_is_owned` accepts it."""
    __module__ = "umwelt.substrate.continuation"

    def __init__(self, **kw):
        self.__dict__.update(kw)


def W(obj, **kw):
    return C.walk(obj, name="root", first=(), **kw)


# ── canonicalization ──────────────────────────────────────────────────────────────

def test_negative_zero_and_zero_are_the_same_number():
    """They hash differently as bytes and mean the same thing. Left alone, every
    sign flip through zero reads as a divergence."""
    assert C.canon_float(-0.0) == C.canon_float(0.0) == 0.0


def test_nan_and_inf_do_not_ride_as_numbers():
    """NaN != NaN, so a NaN slot would make every comparison false forever, for
    a reason that has nothing to do with the engine."""
    assert C.canon_float(float("nan")) == "nan"
    assert C.canon_float(float("inf")) == "inf"
    assert C.canon_float(float("-inf")) == "-inf"


def test_precision_is_strictly_finer_than_the_canon_hash():
    """The probe must see everything the observable it explains can see.

    `field_unify.canon_hash` rounds to 10 decimals. A probe rounding coarser
    could report a clean surface for a field whose hash moved — the tool would
    contradict the thing it exists to diagnose.
    """
    from umwelt.substrate import field_unify
    src = Path(field_unify.__file__).read_text(encoding="utf-8")
    assert ".round(10)" in src, "canon_hash's precision changed; re-check this"
    assert C.FLOAT_DECIMALS > 10


def test_floats_below_the_declared_precision_do_not_register():
    a = _Owned(x=1.0)
    b = _Owned(x=1.0 + 1e-15)
    assert C.diff(W(a), W(b)).clean


def test_floats_at_the_declared_precision_do_register():
    a = _Owned(x=1.0)
    b = _Owned(x=1.0 + 1e-11)
    assert not C.diff(W(a), W(b)).clean


# ── containers the ownership rule would otherwise hide ────────────────────────────

def test_a_deque_is_walked_not_treated_as_an_opaque_leaf():
    """A bounded deque of recent values IS continuation state - rolling windows,
    recent surprise, tick history. The ownership boundary makes stdlib types
    leaves, which would have filed every one of them as unopened."""
    a = _Owned(hist=collections.deque([1.0, 2.0], maxlen=8))
    b = _Owned(hist=collections.deque([1.0, 9.0], maxlen=8))
    d = C.diff(W(a), W(b))
    assert not d.clean
    assert any("hist[1]" in s.path for s in d.changed)


def test_a_deques_maxlen_is_part_of_its_state():
    a = _Owned(hist=collections.deque([1.0], maxlen=8))
    b = _Owned(hist=collections.deque([1.0], maxlen=16))
    assert not C.diff(W(a), W(b)).clean


def test_a_set_is_canonicalized_by_value_not_walked_in_order():
    """A set has no order. Walking it as a sequence gives the same set different
    paths on two boots, and every comparison becomes noise."""
    a = _Owned(s={"x", "y", "z"})
    b = _Owned(s={"z", "y", "x"})
    assert C.diff(W(a), W(b)).clean
    c = _Owned(s={"x", "y"})
    assert not C.diff(W(a), W(c)).clean


def test_a_datetime_is_state_not_an_opaque_type_name():
    import datetime as dt
    a = _Owned(t=dt.datetime(2026, 8, 3, 12, 0))
    b = _Owned(t=dt.datetime(2026, 8, 3, 12, 5))
    assert not C.diff(W(a), W(b)).clean


def test_a_swapped_callback_is_visible():
    """`<builtins.method>` for every bound callback makes rewiring invisible."""
    def one(): pass
    def two(): pass
    assert not C.diff(W(_Owned(cb=one)), W(_Owned(cb=two))).clean


# ── identity: cycles and aliasing ─────────────────────────────────────────────────

def test_a_cycle_terminates():
    a = _Owned(); a.self = a
    a.child = _Owned(parent=a)
    surf = W(a)                                      # must not recurse forever
    assert surf.aliases


def test_two_holders_of_one_object_are_recorded_as_an_alias():
    """`collapse_engine.touched_roles` IS `sensor_bridge.touched_roles` - one set,
    two holders. That sharing is part of the state."""
    shared = _Owned(v=1.0)
    root = _Owned(a=_Owned(t=shared), b=_Owned(t=shared))
    surf = W(root)
    assert any(p.endswith("b.t") for p in surf.aliases)


def test_splitting_an_alias_into_two_copies_is_a_diff():
    """The exact way a well-intentioned loader breaks this silently: it restores
    two equal objects where the engine had one shared object. Every NUMBER
    matches, so only the alias record can catch it."""
    shared = _Owned(v=1.0)
    joined = _Owned(a=_Owned(t=shared), b=_Owned(t=shared))
    split = _Owned(a=_Owned(t=_Owned(v=1.0)), b=_Owned(t=_Owned(v=1.0)))
    d = C.diff(W(joined), W(split))
    assert d.alias_changed, "an alias that became two objects must not read as clean"
    assert not d.changed, "and it must be the ALIAS that catches it, not a value"


# ── determinism ───────────────────────────────────────────────────────────────────

def test_the_digest_is_stable_across_processes():
    """Two captures are taken in two interpreters. A digest that folds in a
    memory address or a dict's insertion order would pass in-process and fail
    exactly where it is used."""
    prog = textwrap.dedent(f"""
        import sys, collections
        sys.path.insert(0, {str(REPO / 'src')!r}); sys.path.insert(0, {str(REPO)!r})
        from umwelt.substrate import continuation as C
        class O:
            __module__ = "umwelt.substrate.continuation"
            def __init__(self, **kw): self.__dict__.update(kw)
        shared = O(v=2.5)
        root = O(z=O(t=shared), a=O(t=shared), d={{"b": 1.0, "a": 2.0}},
                 q=collections.deque([1.0, 2.0], maxlen=4), s={{"y", "x"}})
        print(C.walk(root, name="root", first=()).digest())
    """)
    outs = {subprocess.run([sys.executable, "-c", prog], capture_output=True,
                           text=True, check=True).stdout.strip() for _ in range(2)}
    assert len(outs) == 1 and outs != {""}, f"digest not reproducible: {outs}"


def test_dict_key_order_does_not_change_the_digest():
    a = _Owned(d={"a": 1.0, "b": 2.0})
    b = _Owned(d={"b": 2.0, "a": 1.0})
    assert W(a).digest() == W(b).digest()


# ── what the walker refuses to open, it must report ───────────────────────────────

def test_opaque_leaves_are_counted_not_silently_dropped():
    """State inside a third-party object is invisible here. That is a declared
    cost, and a cost you are not told about is just a bug."""
    a = _Owned(sock=object())
    surf = W(a)
    assert surf.opaque, "an unopened object must appear in Surface.opaque"


def test_a_capture_taken_at_another_precision_is_refused():
    """Comparing across precisions produces a diff made entirely of rounding,
    which reads exactly like a real one."""
    d = W(_Owned(x=1.0)).to_json()
    d["float_decimals"] = 6
    with pytest.raises(ValueError, match="decimals"):
        C.Surface.from_json(d)


def test_a_capture_taken_under_another_root_order_is_refused():
    d = W(_Owned(x=1.0)).to_json()
    d["root_order"] = ["something", "else"]
    with pytest.raises(ValueError, match="ENGINE_ORDER"):
        C.Surface.from_json(d)


# ── owner attribution: the ablation unit ──────────────────────────────────────────

def test_list_indices_are_not_separate_owners():
    """Otherwise one owner becomes a hundred owners of one slot each, and ablate
    boots an engine per index."""
    assert C.owner_of("engine._anticipation_raw[0]") == "_anticipation_raw"
    assert C.owner_of("engine._anticipation_raw[7]") == "_anticipation_raw"


def test_string_keys_are_separate_owners():
    """A dict keyed by world or cluster name genuinely holds many things."""
    assert C.owner_of("engine.clusters['camp'].rho", 1) == "clusters['camp']"


def test_owner_depth_walks_down_the_path():
    p = "engine.berry_tape.stamper.tape[0].source"
    assert C.owner_of(p, 1) == "berry_tape"
    assert C.owner_of(p, 2) == "berry_tape.stamper"


# ── stamping (what ablate splices with) ───────────────────────────────────────────

def test_stamp_writes_an_array_in_place_so_aliases_survive():
    """Rebinding would give one holder of a shared array a private copy -
    converting an alias into two objects inside the tool that measures aliasing."""
    shared = np.zeros(3)
    root = _Owned(a=_Owned(arr=shared), b=_Owned(arr=shared))
    assert C.stamp(root, "root.a.arr", np.array([1.0, 2.0, 3.0])) == "in-place"
    assert root.b.arr is shared and root.b.arr[0] == 1.0


def test_stamp_rebinds_when_the_shape_cannot_take_it():
    root = _Owned(a=_Owned(arr=np.zeros(3)))
    assert C.stamp(root, "root.a.arr", np.zeros(5)) == "rebound"
    assert root.a.arr.shape == (5,)


def test_stamp_resolves_dict_and_index_steps():
    root = _Owned(d={"k": 1.0}, lst=[1.0, 2.0])
    C.stamp(root, "root.d['k']", 9.0)
    C.stamp(root, "root.lst[1]", 8.0)
    assert root.d["k"] == 9.0 and root.lst[1] == 8.0


def test_parse_path_refuses_what_it_cannot_resolve():
    """A stamper that guesses writes learned state into the wrong slot - worse
    than any failure this module was built to find."""
    with pytest.raises(ValueError):
        C.parse_path("root.a[")


def test_splice_reports_what_it_could_not_land():
    """A splice that silently did nothing reads as evidence the owner does not
    matter, which is the most expensive possible wrong answer from `ablate`."""
    root = _Owned(a=_Owned(x=1.0))
    got = C.splice(root, {"root.a.x": 5.0}, ["root.a.x", "root.a.missing"])
    assert got["rebound"] == 1 and got["missing"] == 1
    assert root.a.x == 5.0


# ── the round-trip digest ─────────────────────────────────────────────────────────

def test_the_digest_never_raises_on_a_hostile_object():
    """It rides in `engine.save`. A diagnostic may never be the reason a snapshot
    fails to write."""
    class Hostile:
        __module__ = "umwelt.substrate.continuation"
        @property
        def boom(self):                       # never read: we use __dict__
            raise RuntimeError("no")
        def __init__(self):
            self.__dict__["ok"] = 1.0
    assert C.continuation_digest(Hostile()) is not None


def test_digest_exempt_entries_all_carry_a_reason():
    """Each exemption is a place this guard is blind. An unexplained one is how
    the hand-maintained registry this module replaces went wrong."""
    for path, why in C.DIGEST_EXEMPT.items():
        assert why and len(why) > 40, f"{path} is exempt with no real reason"


def test_the_verdict_slot_is_exempt_from_its_own_digest():
    """Otherwise the second generation of every chain reports a mismatch caused
    entirely by the reporting: a snapshot saved by an engine that had loaded
    carries '_continuation=exact' inside the digest, while the engine being
    graded still reads 'fresh' at check time."""
    assert "engine._continuation" in C.DIGEST_EXEMPT


# ── against a real engine ─────────────────────────────────────────────────────────

SPEC_SHIM = '''\
from umwelt.spec.schema import BindingSpec, DomainSpec, DriverSpec, NodeSpec, OutputSpec
NORM = {"type": "regime", "center": 0.0, "width": 0.5, "invert": True}
SPEC = DomainSpec(
    name="cont-probe",
    nodes=(NodeSpec("camp", parent=None, kind="root", roles=("coordination",),
                    role_modes={"coordination": "dissipative"}),
           NodeSpec("organ", parent="camp", roles=("activity", "blocked"),
                    role_modes={"activity": "dissipative", "blocked": "dissipative"})),
    bindings=(BindingSpec("camp_wall", zone="camp", role="coordination",
                          normalizer=NORM, force_observe=True, collapse_alpha=0.4),
              BindingSpec("organ_claim", zone="organ", role="activity",
                          normalizer=NORM, force_observe=True, collapse_alpha=0.25)),
    outputs=(OutputSpec("hint", node="camp", role="coordination"),),
    drivers=(DriverSpec("day", period_s=86400.0),),
    ingest_hold_s=5.0,
)
'''


def _boot_and_walk(tmp_path: Path, script: str) -> str:
    (tmp_path / "cont_spec.py").write_text(SPEC_SHIM, encoding="utf-8")
    prog = textwrap.dedent(f"""
        import sys, random
        import numpy as np
        sys.path.insert(0, {str(REPO / 'src')!r}); sys.path.insert(0, {str(REPO)!r})
        sys.path.insert(0, {str(tmp_path)!r})
        random.seed(1234); np.random.seed(1234)
        from umwelt.boot import build_engine
        from umwelt.substrate import continuation as C
        engine = build_engine(spec="cont_spec:SPEC", population=False)
    """) + textwrap.dedent(script)
    cp = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr[-3000:]
    return cp.stdout.strip()


@pytest.mark.slow
def test_a_real_engine_walks_deterministically_apart_from_declared_clock_seeds(tmp_path):
    """Two same-seed boots must differ ONLY where the engine reads the wall clock.

    Written as a subset check rather than an exclusion, so a newly-added
    clock-seeded slot fails here with its own path printed instead of turning
    the whole probe intermittently nondeterministic — which is how a tool like
    this stops being trusted and then stops being run.
    """
    caps = [_boot_and_walk(tmp_path, f"""
        import json
        json.dump(C.walk(engine).to_json(), open({str(tmp_path)!r} + "/c{i}.json", "w"))
        print("ok")
    """) for i in (0, 1)]
    assert caps == ["ok", "ok"]
    import json
    a = C.Surface.from_json(json.loads((tmp_path / "c0.json").read_text()))
    b = C.Surface.from_json(json.loads((tmp_path / "c1.json").read_text()))
    d = C.diff(a, b)
    moved = {s.path for s in d.changed}
    undeclared = moved - set(C.WALL_CLOCK_SEEDED)
    assert not undeclared, (
        "these slots differ between two identically-seeded boots and are not "
        f"declared in WALL_CLOCK_SEEDED: {sorted(undeclared)}")
    assert not (d.only_left or d.only_right or d.alias_changed), (
        "the SHAPE of the surface must be identical between two boots")


@pytest.mark.slow
def test_every_declared_clock_seed_is_real_and_still_moves(tmp_path):
    """A declaration that has stopped being true is just a hole in the check."""
    for path, why in C.WALL_CLOCK_SEEDED.items():
        assert why and len(why) > 40, f"{path} declared with no real reason"
    out = _boot_and_walk(tmp_path, """
        s = C.walk(engine)
        print(" ".join(p for p in C.WALL_CLOCK_SEEDED if p in s.slots))
    """)
    assert set(out.split()) == set(C.WALL_CLOCK_SEEDED), (
        "a declared clock-seeded slot no longer exists on the engine - remove it")


@pytest.mark.slow
def test_a_real_engine_reaches_thousands_of_slots_and_opens_nearly_everything(tmp_path):
    """The ownership boundary is only useful if it actually bottoms out. If the
    opaque count were large, the digest would be guarding a fraction of the
    engine while reading like it guards all of it."""
    out = _boot_and_walk(tmp_path, """
        s = C.walk(engine)
        non_callable = {k: v for k, v in s.opaque.items() if not k.startswith("<callable")}
        print(len(s.slots), len(s.aliases), sum(non_callable.values()), s.truncated)
    """)
    slots, aliases, opaque_state, trunc = out.split(None, 3)
    assert int(slots) > 1000, "a real engine should present a substantial surface"
    assert int(aliases) > 0, "the engine genuinely shares objects; zero means missed"
    assert int(opaque_state) == 0, (
        f"{opaque_state} non-callable objects went unopened - state the digest "
        "cannot see but claims to guard")
    assert trunc == "[]", f"recursion hit MAX_DEPTH: {trunc}"


@pytest.mark.slow
def test_a_real_engine_round_trips_its_digest_through_save_and_load(tmp_path):
    """Not a claim that the engine round-trips - it does not, and that is the
    finding. This pins that the MECHANISM works: the digest is recorded, it is
    recomputed on load, and the verdict is one of the declared words."""
    out = _boot_and_walk(tmp_path, f"""
        p = {str(tmp_path / 'snap.pkl')!r}
        engine.save(p)
        import pickle
        d = pickle.load(open(p, "rb"))
        fresh = build_engine(spec="cont_spec:SPEC", population=False)
        fresh.load(p)
        print(bool(d.get("continuation_digest")), fresh._continuation)
    """)
    recorded, verdict = out.split()
    assert recorded == "True", "engine.save did not record a continuation_digest"
    assert verdict in ("exact", "approximate"), verdict


@pytest.mark.slow
def test_a_legacy_snapshot_without_a_digest_says_legacy_not_exact(tmp_path):
    """Silence must never read as agreement."""
    out = _boot_and_walk(tmp_path, f"""
        import pickle
        p = {str(tmp_path / 'legacy.pkl')!r}
        engine.save(p)
        d = pickle.load(open(p, "rb")); d.pop("continuation_digest", None)
        pickle.dump(d, open(p, "wb"))
        fresh = build_engine(spec="cont_spec:SPEC", population=False)
        fresh.load(p)
        print(fresh._continuation)
    """)
    assert out == "legacy"
