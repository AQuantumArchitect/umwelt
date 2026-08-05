"""The pacemaker: cadence derived from each world's own decay, not a clock.

Deterministic by construction — no sleeps, no wall-clock dependence, no live
worker. `_pace_due` and `pacemaker_tick` are pure enough to drive from scripted
`/pace` payloads, which is the whole reason the HTTP read was split into
`_worker_get`: the decision logic can be tested without a socket.

Follows tests/test_worker_orphan.py's idiom (script the world's answers, assert
the decision) rather than standing up a real engine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from umweltd import supervisor as sup_mod  # noqa: E402


@pytest.fixture
def sup(monkeypatch, tmp_path):
    """A Supervisor with no disk, no workers, and no network."""
    monkeypatch.setattr(sup_mod, "PACE_ENABLED", True)
    monkeypatch.setattr(sup_mod, "PACE_FRACTION", 0.3)
    monkeypatch.setattr(sup_mod, "PACE_FLOOR_S", 60.0)
    monkeypatch.setattr(sup_mod, "PACE_CEILING_S", 3600.0)
    s = sup_mod.Supervisor()
    monkeypatch.setattr(s, "worlds_root", lambda: tmp_path)
    return s


def _wire(monkeypatch, sup, paces: dict, running=None):
    """Script each world's /pace answer; capture pulses instead of sending them."""
    running = paces.keys() if running is None else running
    monkeypatch.setattr(sup, "catalog", lambda: [
        {"name": n, "port": 1, "running": n in running} for n in paces])
    monkeypatch.setattr(sup, "_worker_get",
                        lambda name, path: paces.get(name))
    fired = []
    monkeypatch.setattr(sup, "_pulse",
                        lambda name, detail: (fired.append((name, detail)), True)[1])
    return fired


# ── the derived interval ─────────────────────────────────────────────────────

def test_faster_decay_is_serviced_sooner(sup):
    """The whole point: tau comes from the world, so cadence differs per world."""
    slow = {"tau_s": 3300.0, "age_s": 0.0}       # gamma 0.0003-ish
    fast = {"tau_s": 600.0, "age_s": 0.0}        # decays 5x faster
    _, slow_interval, _ = sup._pace_due(slow)
    _, fast_interval, _ = sup._pace_due(fast)
    assert fast_interval < slow_interval
    assert slow_interval == pytest.approx(0.3 * 3300.0)
    assert fast_interval == pytest.approx(0.3 * 600.0)


def test_fresh_world_is_not_serviced(sup):
    due, interval, age = sup._pace_due({"tau_s": 3300.0, "age_s": 10.0})
    assert not due and age == 10.0 and interval > 10.0


def test_stale_world_is_serviced(sup):
    due, _, _ = sup._pace_due({"tau_s": 3300.0, "age_s": 2000.0})
    assert due


def test_never_ingested_reads_as_maximally_stale_not_fresh(sup):
    """age_s None means no event has ever arrived. That is the STALEST a world can
    be; treating unknown as fresh is how a world starves unnoticed."""
    due, _, age = sup._pace_due({"tau_s": 3300.0, "age_s": None})
    assert due and age is None


def test_unknown_tau_falls_back_to_ceiling_not_immortality(sup):
    """A world that cannot report tau is not therefore exempt from feeding."""
    due, interval, _ = sup._pace_due({"tau_s": None, "age_s": 4000.0})
    assert interval == sup_mod.PACE_CEILING_S
    assert due


def test_interval_is_clamped_both_ways(sup):
    _, tiny, _ = sup._pace_due({"tau_s": 1.0, "age_s": 0.0})
    _, huge, _ = sup._pace_due({"tau_s": 10_000_000.0, "age_s": 0.0})
    assert tiny == sup_mod.PACE_FLOOR_S
    assert huge == sup_mod.PACE_CEILING_S


# ── staleness is age, never confidence ───────────────────────────────────────

def test_pace_decision_ignores_confidence(sup):
    """Dissipation drives the qubit toward a PURE ground state, so |r| rises as a
    belief goes stale. A pacemaker that read confidence would treat a fully-decayed
    world as maximally trustworthy — yurt-mood's exact bug. Pin that the decision
    depends only on age and tau, whatever confidence claims."""
    hot = {"tau_s": 600.0, "age_s": 5000.0, "confidence": 1.0}
    assert sup._pace_due(hot)[0] is True


# ── the tick: ordering, budget, sinking ──────────────────────────────────────

def test_tick_fires_only_due_worlds(monkeypatch, sup):
    fired = _wire(monkeypatch, sup, {
        "fresh": {"tau_s": 3300.0, "age_s": 5.0},
        "stale": {"tau_s": 3300.0, "age_s": 5000.0},
    })
    sup.pacemaker_tick()
    assert [n for n, _ in fired] == ["stale"]


def test_tick_skips_worlds_that_are_not_running(monkeypatch, sup):
    fired = _wire(monkeypatch, sup,
                  {"down": {"tau_s": 600.0, "age_s": 9999.0}}, running=set())
    sup.pacemaker_tick()
    assert fired == []


def test_tick_survives_a_world_that_cannot_answer(monkeypatch, sup):
    monkeypatch.setattr(sup, "catalog", lambda: [
        {"name": "mute", "port": 1, "running": True},
        {"name": "ok", "port": 2, "running": True}])
    monkeypatch.setattr(sup, "_worker_get", lambda name, path:
                        None if name == "mute" else {"tau_s": 600.0, "age_s": 9999.0})
    fired = []
    monkeypatch.setattr(sup, "_pulse", lambda n, d: (fired.append(n), True)[1])
    sup.pacemaker_tick()
    assert fired == ["ok"]


def test_cooldown_prevents_refiring_inside_one_window(monkeypatch, sup):
    fired = _wire(monkeypatch, sup, {"w": {"tau_s": 3300.0, "age_s": 5000.0}})
    sup.pacemaker_tick()
    sup.pacemaker_tick()          # immediately again — the pulse has not landed yet
    assert len(fired) == 1


# ── sinking (Phase D): the bottom of the stack is simply not reached ─────────

def test_budget_sinks_the_lowest_ranked_and_destroys_nothing(monkeypatch, sup):
    """Under scarcity only the top of the stack is serviced. The sunk world is not
    archived, unregistered or deleted — it is just not fed, and it stays a
    first-class catalog entry ready to come back."""
    paces = {
        "barely": {"tau_s": 3300.0, "age_s": 1000.0},    # ~1.0x overdue
        "badly": {"tau_s": 3300.0, "age_s": 90000.0},    # ~90x overdue
    }
    fired = _wire(monkeypatch, sup, paces)
    monkeypatch.setattr(sup, "service_budget", lambda: 1)
    sup.pacemaker_tick()
    assert [n for n, _ in fired] == ["badly"]
    sank = [c for c in sup._pace_rank if c.get("skipped")]
    assert [c["world"] for c in sank] == ["barely"]
    # nothing was removed from the catalog by sinking
    assert {c["world"] for c in sup._pace_rank} == {"barely", "badly"}


def test_rank_prefers_more_overdue(monkeypatch, sup):
    _wire(monkeypatch, sup, {
        "a": {"tau_s": 600.0, "age_s": 300.0},
        "b": {"tau_s": 600.0, "age_s": 30000.0},
    })
    sup.pacemaker_tick()
    assert [c["world"] for c in sup._pace_rank] == ["b", "a"]


def test_rank_is_cost_aware(monkeypatch, sup):
    """Equal urgency, unequal cost — the cheap world outranks the expensive one."""
    _wire(monkeypatch, sup, {
        "cheap": {"tau_s": 600.0, "age_s": 6000.0},
        "dear": {"tau_s": 600.0, "age_s": 6000.0},
    })
    monkeypatch.setattr(sup, "service_cost", lambda n: 1.0 if n == "cheap" else 10.0)
    sup.pacemaker_tick()
    assert [c["world"] for c in sup._pace_rank] == ["cheap", "dear"]


def test_disabled_pacemaker_does_nothing(monkeypatch, sup):
    monkeypatch.setattr(sup_mod, "PACE_ENABLED", False)
    fired = _wire(monkeypatch, sup, {"w": {"tau_s": 600.0, "age_s": 9999.0}})
    assert sup.pacemaker_tick() == []
    assert fired == []


# ── the envelope ─────────────────────────────────────────────────────────────

def test_pulse_uses_the_same_envelope_as_a_field_action(monkeypatch, sup):
    """A pulse must ride the registry's gate/cooldown/risk_class machinery, which
    means the router has to recognise it — so it carries actuator_id + command.on
    exactly as `_make_webhook_dispatch` does, not a private shape."""
    sent = {}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"{}"

    def _fake_urlopen(req, timeout=None):
        sent["url"] = req.full_url
        sent["body"] = __import__("json").loads(req.data.decode())
        return _Resp()

    monkeypatch.setattr(sup_mod.urllib.request, "urlopen", _fake_urlopen)
    assert sup._pulse("wargen-self", {"age_s": 9999.0, "interval_s": 990.0}) is True
    assert sent["body"]["actuator_id"] == "world_pulse_wargen_self"
    assert sent["body"]["command"] == {"on": True}
    assert sent["body"]["source"] == "umweltd.pacemaker"
    assert sent["body"]["why"]["age_s"] == 9999.0


def test_pulse_failure_is_survivable(monkeypatch, sup):
    def _boom(req, timeout=None):
        raise OSError("router down")
    monkeypatch.setattr(sup_mod.urllib.request, "urlopen", _boom)
    assert sup._pulse("w", {"age_s": 1.0}) is False


def test_failed_pulse_does_not_start_a_cooldown(monkeypatch, sup):
    """A pulse that never reached the router must be retried next tick, not treated
    as delivered — otherwise a router restart silently starves every world for a
    full interval."""
    _wire(monkeypatch, sup, {"w": {"tau_s": 600.0, "age_s": 9999.0}})
    monkeypatch.setattr(sup, "_pulse", lambda n, d: False)
    sup.pacemaker_tick()
    assert "w" not in sup._pace_last


# ── did the beat actually LAND? (transport success is not landing) ───────────

def _reply(monkeypatch, body: bytes):
    """Point _pulse's webhook at a scripted router reply."""
    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return body

    monkeypatch.setattr(sup_mod.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp())


def test_refused_pulse_is_recorded_as_not_landed(monkeypatch, sup):
    """The measured 2026-08-04 failure: seven worlds were pulsed ~40x/day at an
    actuator nobody had registered. The router answered `{"ok": true}` to every
    one, so the heart logged success while the worlds aged past a thousand tau."""
    _reply(monkeypatch, b'{"ok": false, "accepted": false, "reason": "unknown_actuator"}')
    sup._pulse("lease-drill", {"age_s": 1_500_000.0, "interval_s": 165.0})
    out = sup._pace_outcome["lease-drill"]
    assert out["landed"] is False
    assert out["reason"] == "unknown_actuator"


def test_refused_pulse_still_starts_a_cooldown(monkeypatch, sup):
    """A registry gap is permanent. Retrying it every watchdog tick would hot-loop
    the router forever, so a DELIVERED-and-refused beat keeps its cooldown — unlike
    a transport failure, which must be retried."""
    _reply(monkeypatch, b'{"ok": false, "accepted": false, "reason": "unknown_actuator"}')
    assert sup._pulse("lease-drill", {"age_s": 9999.0, "interval_s": 165.0}) is True


def test_accepted_pulse_is_recorded_as_landed(monkeypatch, sup):
    _reply(monkeypatch, b'{"ok": true, "accepted": true}')
    sup._pulse("hive-ops", {"age_s": 9999.0, "interval_s": 100.0})
    assert sup._pace_outcome["hive-ops"]["landed"] is True


def test_router_without_accepted_field_reads_as_landed(monkeypatch, sup):
    """Version skew must not manufacture a fleet of phantom failures: a router that
    predates the `accepted` field is not evidence that nothing is landing."""
    _reply(monkeypatch, b'{"ok": true}')
    sup._pulse("hive-ops", {"age_s": 1.0, "interval_s": 100.0})
    assert sup._pace_outcome["hive-ops"]["landed"] is True


def test_unparseable_router_reply_reads_as_landed(monkeypatch, sup):
    _reply(monkeypatch, b'not json at all')
    sup._pulse("hive-ops", {"age_s": 1.0, "interval_s": 100.0})
    assert sup._pace_outcome["hive-ops"]["landed"] is True


def test_transport_failure_records_not_landed(monkeypatch, sup):
    """Distinct from refusal: the beat never arrived, so it is both not-landed AND
    retried (returns False -> no cooldown)."""
    def _boom(req, timeout=None):
        raise OSError("router down")
    monkeypatch.setattr(sup_mod.urllib.request, "urlopen", _boom)
    assert sup._pulse("w", {"age_s": 1.0}) is False
    assert sup._pace_outcome["w"]["landed"] is False
    assert "transport" in sup._pace_outcome["w"]["reason"]


# ── the worker-side derivation: where wall time actually comes from ──────────

class _FakeBundle:
    def __init__(self, vals): self.vals = vals
    def get_param(self, k): return object() if k in self.vals else None
    def get(self, k): return self.vals[k]


class _FakeNode:
    def __init__(self, name, roles, gamma):
        self.name, self.roles = name, roles
        self.param_bundle = _FakeBundle({"gamma_diss": gamma})


class _FakeGraph:
    def __init__(self, root, kids):
        self.root, self._kids = root, kids
    def nodes_with_roles(self): return [self.root] + self._kids


class _FakeEngine:
    def __init__(self, gammas, hold, cap=32, step=100):
        root = _FakeNode("world", ("whole",), 5.0)       # engine DNA, must be ignored
        kids = [_FakeNode(n, (r,), g) for n, r, g in gammas]
        self.graph = _FakeGraph(root, kids)
        self.ingest_hold_s = hold
        self._wall_catchup_max = cap
        self._step = step


def _host(engine, last_ts=""):
    """A WorldHost with only what pace() touches — __init__ builds a real engine."""
    from umweltd.worker import WorldHost
    import threading
    h = object.__new__(WorldHost)
    h.engine, h.lock, h.last_ts = engine, threading.Lock(), last_ts
    h._pace_wall0, h._pace_step0 = 0.0, 0
    h.manifest = {"name": "fake"}
    # `name` is manifest.get("name", self.dir.root.name) and Python evaluates that
    # default eagerly, so `dir` is touched even when the manifest carries a name.
    h.dir = type("_D", (), {"root": Path("fake")})()
    return h


def test_root_engine_dna_is_excluded_from_gamma():
    """param_bundles seeds the ROOT gamma_diss at 5.0 (bounds 0.5..50) as a shared
    field-dynamics constant. A spec's per-node gamma_diss is a decay rate on
    (0,1) — wargen-self's organs sit at 0.0003. Four orders apart, so letting the
    root in would hand every world the same heartbeat."""
    h = _host(_FakeEngine([("body", "headroom", 0.0003)], hold=5.0))
    p = h.pace()
    assert p["gamma_max"] == pytest.approx(0.0003)
    assert p["hottest_axis"] == "body.headroom"
    assert "world.whole" not in p["gammas"]


def test_tau_ignores_measured_step_rate():
    """The circularity pin. Steps only happen on ingest, so steps-per-second
    measures how often WE FEED IT. Pacing on that starves a rarely-fed world
    further, forever. tau must come from ingest_hold_s, never from observed rate."""
    h = _host(_FakeEngine([("a", "x", 0.0003)], hold=5.0))
    h._pace_step0 = 99            # 1 step observed over the whole process life
    p = h.pace()
    assert p["observed_steps"] == 1
    assert p["tau_s"] is not None
    assert p["bound"] in ("resolution", "drift")
    # the resolution bound is hold*(cap+1) and owes nothing to steps_per_s
    assert p["resolution_s"] == pytest.approx(5.0 * 33)


def test_resolution_bound_is_hold_times_catchup_window():
    """Past hold*(cap+1) the catch-up saturates and the excess wall gap is dropped
    on the floor — the world cannot represent it at all."""
    h = _host(_FakeEngine([("a", "x", 1e-9)], hold=2.0, cap=9))
    p = h.pace()
    assert p["resolution_s"] == pytest.approx(20.0)
    assert p["bound"] == "resolution"      # tiny gamma => drift bound is enormous


def test_drift_bound_wins_when_decay_is_fast():
    """A fast-decaying axis is due long before the membrane runs out of resolution."""
    h = _host(_FakeEngine([("a", "x", 0.5)], hold=1.0, cap=1000))
    p = h.pace()
    assert p["drift_s"] == pytest.approx(2.0)     # tau_steps=2, hold=1
    assert p["bound"] == "drift"


def test_no_hold_means_no_derivable_tau():
    """ingest_hold_s is the ONLY place wall time enters. Without it there is no
    honest seconds conversion, so tau is None and the supervisor uses its ceiling."""
    h = _host(_FakeEngine([("a", "x", 0.0003)], hold=0))
    p = h.pace()
    assert p["tau_s"] is None and p["resolution_s"] is None


# ── the two deadlines are weighted differently, on purpose ──────────────────

def test_drift_tolerance_and_resolution_margin_are_separate(sup, monkeypatch):
    """PACE_FRACTION is a decay TOLERANCE and belongs on drift. Resolution is a hard
    membrane limit — taking 30% of it just triples the feed rate for nothing."""
    monkeypatch.setattr(sup_mod, "PACE_RESOLUTION_MARGIN", 0.9)
    monkeypatch.setattr(sup_mod, "PACE_FLOOR_S", 1.0)
    # resolution binds
    _, interval, _ = sup._pace_due({"age_s": 0.0, "drift_s": 16666.0, "resolution_s": 165.0})
    assert interval == pytest.approx(0.9 * 165.0)
    # drift binds
    _, interval, _ = sup._pace_due({"age_s": 0.0, "drift_s": 100.0, "resolution_s": 3300.0})
    assert interval == pytest.approx(0.3 * 100.0)


def test_pace_due_falls_back_to_tau_when_split_absent(sup):
    """Older workers report only tau_s and do not say which bound it is, so the
    drift tolerance applies — over-feeding is recoverable, under-feeding is not."""
    _, interval, _ = sup._pace_due({"age_s": 0.0, "tau_s": 3300.0})
    assert interval == pytest.approx(0.3 * 3300.0)


# ── scarcity: the budget comes from a belief, and never starves its own source ─

def test_budget_is_unlimited_without_a_scarcity_world(sup, monkeypatch):
    """umwelt has no opinion about cost. yurt supplies the meaning."""
    monkeypatch.setattr(sup_mod, "PACE_BUDGET_WORLD", "")
    assert sup.service_budget() == sup_mod.PACE_BUDGET_MAX


def test_budget_scales_with_scarcity_warmth(sup, monkeypatch):
    monkeypatch.setattr(sup_mod, "PACE_BUDGET_WORLD", "hive-purse")
    monkeypatch.setattr(sup_mod, "PACE_BUDGET_MIN", 1)
    monkeypatch.setattr(sup_mod, "PACE_BUDGET_MAX", 11)
    monkeypatch.setattr(sup, "_worker_get", lambda n, p: {"value": 1.0})
    assert sup.service_budget() == 11
    monkeypatch.setattr(sup, "_worker_get", lambda n, p: {"value": 0.0})
    assert sup.service_budget() == 1


def test_unreadable_scarcity_falls_to_the_floor_not_to_zero(sup, monkeypatch):
    """A dead signal must not freeze the hive, and must not authorise the whole
    roster either."""
    monkeypatch.setattr(sup_mod, "PACE_BUDGET_WORLD", "hive-purse")
    monkeypatch.setattr(sup_mod, "PACE_BUDGET_MIN", 2)
    monkeypatch.setattr(sup, "_worker_get", lambda n, p: None)
    assert sup.service_budget() == 2


def test_scarcity_world_is_never_sunk(monkeypatch, sup):
    """The deadlock this prevents: a cold purse shrinks the budget, the shrunken
    budget stops the purse's own poster, and the unfed purse decays colder. A hive
    that starves itself to death by correctly observing that it is starving."""
    monkeypatch.setattr(sup_mod, "PACE_BUDGET_WORLD", "hive-purse")
    fired = _wire(monkeypatch, sup, {
        "hungry": {"tau_s": 600.0, "age_s": 999999.0},   # ranks far above
        "hive-purse": {"tau_s": 600.0, "age_s": 6000.0},
    })
    monkeypatch.setattr(sup, "service_budget", lambda: 1)
    sup.pacemaker_tick()
    names = [n for n, _ in fired]
    assert "hungry" in names and "hive-purse" in names
    assert not [c for c in sup._pace_rank
                if c["world"] == "hive-purse" and c.get("skipped")]
