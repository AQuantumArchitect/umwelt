# Papers

The research artifacts live WITH their data in the origin deployment's repository
(meerkat) — a paper belongs next to the experiments and tapes that produced its numbers.
This page is the index.

## Paper 1 — Causal self-tagging ("the system that knows what it caused")

**Status: drafted, with a measured A/B on real data.** An anticipatory actuator poisons
its own world model: acting on a prediction manufactures the evidence that confirms it.
The paper derives the confounding surface entirely from the world graph (an actuator
confounds exactly the learned roles its state projects onto — no per-device code) and
gates world-model learning by `1 − echo·surface`.

Measured on 24 days of real home data: a naive learner credits the system's own lights
at **10.8×** their true strength and is the only arm to degrade when silenced; the
tagged learner cuts the bias **79%**. The mechanism is ordinary graph + echo
accounting (no Belavkin path required) and ships in this library as
`umwelt.learning.confounding` + `umwelt.learning.learning_router`
(mechanism smoke: `proofs/`; the effect-size numbers belong to the origin's real-tape
run and are not re-claimed from synthetic data — see CLAIMS.md).

Draft: the origin repository, `docs/papers/paper1-causal-self-tagging.md`.

## Papers 2–4 — owed

The origin's `docs/WHITEPAPERS.md` tracks three further candidates (the estimator-ladder
methodology with its negative result; gauge-tracked fibers / provable non-training; the
geometric-phase process-time decision demo). Each is listed there with its evidence
status; none is claimed here until its gate is paid. A natural synthetic home for the
path-topology demo is the gridworld fog example — honestly labeled as such.
