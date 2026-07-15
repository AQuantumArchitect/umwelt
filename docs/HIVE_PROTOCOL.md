# The hive protocol — a constitution for LLM fleets sharing one world

**Status: field-derived pattern, second deployment artifact.** Written after
the first multi-LLM fleet ran a real project through a shared umwelt world
(the coordination tape and its lessons: `examples/hive_relay/`). The
protocol answers the two failure modes that deployment measured — sensors
that confabulate, and fixers that patch the problem instead of the design —
and closes the loop the obvious way: **the coordinator goes under the same
contract it enforces.**

## Roles, and what a word is worth

| role | writes | η (ingest confidence) |
|---|---|---|
| **sensor** (small explorer/tester agents) | reports ONLY — never artifacts, never fixes | low (~0.25) |
| **referee** (manifests, directories, test suites, CI) | ground truth | high (~0.9–0.95) |
| **builder** (implementation agents) | artifacts, under plan + invariants + ratchets | its work is refereed, never believed |
| **coordinator** (the supervising model) | audit verdicts, escalations | mid (~0.7) — **a claim, not a ruling** |

## The laws

1. **The sensor never holds the hammer.** A tester empowered to fix will
   patch the magic out of whatever it tests. Sensors report; builders build;
   the separation is structural, not stylistic.
2. **A precise early surrender is a success.** A sensor that stalls on its
   current objective stops early and files a wall report (*tried / saw /
   expected*). Forcing through is the failure mode — it hides exactly the
   information the fleet exists to find.
3. **Walls fix paths, not testers.** Every wall escalates to a design-level
   repair of the world being tested (is the affordance visible? does the
   refusal explain? is the ritual taught?) before another sensor flies that
   section. The world gets more legible; the test never gets "passed harder."
4. **Builders are audited at the level of abstraction.** Landings are
   reviewed for: invariants untouched unless sanctioned, thresholds measured
   rather than slashed, authored voice intact, nothing invented to make a
   bar pass. The audit verdict is itself only a claim (law 5).
5. **The coordinator is under the protocol.** Its verdicts ingest as sensor
   readings at η<1; referees check them; a dedicated belief (the fleet's
   *stewardship*) prices the coordinator's word exactly as the fleet's
   *truthfulness* belief prices the sensors'. A blessed failure is
   remembered by the field.
6. **Claims are never writes.** Everything above lands as observations; the
   trust web assigns each reporter the reliability it earns. With three or
   more heterogeneous reporters, no privileged oracle is needed
   (the leave-one-out isolation pin).

## The loop

```
sensor flies → progress OR early wall report
   walls  → coordinator escalates → builder repairs the PATH → audit lands
   audit  → coordinator claim (low η) + CI referee (high η) → stewardship
   all of it → the shared world (beliefs now) + the chronicle (learned shape)
```

## Implementation sketch (all existing machinery)

A world spec with a `fleet` node carrying `truthful` / `momentum` /
`stewardship` dissipative-analog roles; report-class bindings
(`force_observe=True`, `collapse_alpha` = the role's η — see "Worlds of
reports" in NEW_DOMAIN.md); a wall ledger beside the world; the trust web
for per-reporter pricing. First live instance: the SpaceWheat fleet's
`hive.py` (`protocol` / `wall` / `audit` commands), stewardship reading
+0.93 after its first two confirmed audits.
