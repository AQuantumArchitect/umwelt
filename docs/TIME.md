# TIME — the three times, kept untangled

Many domains want to model time; the engine must never confuse its own machinery
with theirs. There are exactly three time-like concepts, each living in one place,
none leaking into the others:

## 1. Compute time — `dt`, `dt_scale`, the φ-clock, the Berry tape

How big a numerical step is and how often learners run. `dt` is the RK4/Lindblad
step size; `dt_scale` is the smooth adaptive clock's catch-up multiplier (compute
compression on calm ticks); the φ-clock staggers learning-tower cadences; the Berry
tape stamps the field's learning journey by geometric phase, not by any clock. None
of this represents anything in the modeled world. It is solver machinery.

## 2. Membrane cadence — `spec.ingest_hold_s` (zero-order hold)

An adapter for SPARSE FEEDS at the ingest membrane, nothing more. The origin world
was dense-polled (a reading every few seconds), so "one compute step per batch" was
implicitly correct. A daily-bar world starves under that assumption. Declaring
`ingest_hold_s` makes each delivered batch's inputs hold across the gap they close,
advanced in bounded unit compute substeps (one per `ingest_hold_s` of gap). A hold
spans one gap only; a channel absent from the previous batch relaxes honestly.

This knob translates *feed cadence* into *compute steps*. It does not model time,
does not create any node or qubit, and is provably inert when unset
(`tests/test_wall_pacing.py`).

## 3. In-universe time — drivers and their clock qubits (opt-in, and only if the world cares)

The only *modeled* time. A domain that cares about a cycle declares a
`DriverSpec` — a day, a session calendar, a game tick, an ephemeris — and the
engine anchors a clock QUBIT toward the driver's phase each ingest. The phase is
fixed physics; its **comprehension** is learned (anticipation skill calibrates the
anchor down). A world that declares no drivers has no concept of time at all, and
nothing in the engine assumes one.

This is where "wall time as an in-universe entity" lives when a domain wants it:
bind it as a driver (or even as ordinary bindings feeding a clock-like node) and it
becomes part of the world model — beliefs, couplings, forecasts and all. Deploy it
only if you care about it.

## The rule

- Solver knobs (`dt`, `dt_scale`, φ, Berry) never appear in a spec's world model.
- `ingest_hold_s` may read wall timestamps, but only to count compute substeps —
  it can never mint a belief, a node, or a phase.
- Anything a domain wants to *know* about time must enter as declared data
  (drivers/bindings) and earn comprehension like any other signal.
