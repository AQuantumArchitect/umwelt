# Butler — email + reminders + coordination for busy people

**Status: IMPLEMENTED + measured (2026-07-28).** The world lives in the yurt repo at
`worlds/butler_life/` (spec factory, η tiers as per-binding `collapse_alpha`
explicit 0.98 / lived 0.90 / shield 0.25 / garbage 0.0, the gamma_diss cadence
lesson) with the owed synthetic family week in `fixtures/synthetic_week.jsonl`
and 5 green proofs in `test_butler_life.py`: ease-with-evidence + decay-in-
silence, η=0 bit-identical no-op, nudge→effect causal self-tag, tier
monotonicity, live-spec build. It is registered on the yurt hearth and fed by
Butler (`~/ws/butler`, the front-desk kernel) whose march marks post
explicit-tier observations. Still owed: the Watch⇄Run dial EARNING autonomy on
live data (see CLAIMS.md, "PARTIALLY PAID").

## The umwelt

- **Nodes**: a person root → life domains (household, kids/school, health, finance,
  work, social) → open threads/commitments as device-like leaves (a permission slip, a
  bill, an RSVP).
- **Roles**: `urgency`, `attention_needed`, `load`, `state_of_play`.
- **Observations**: parsed emails and calendar items where **η is the parser's extraction
  confidence** — an LLM parser is a noisy sensor, and the confidence contract means a
  garbage parse provably cannot slam a belief (η=0 ⇒ the innovation vanishes; see
  docs/THEORY.md). Explicit user statements ("done", "not now") land at η≈1. Reminders
  acknowledged or ignored are observations too.
- **Outputs**: drafted reminders, nudges, proposed schedule moves — and the
  Watch⇄Run earned-autonomy dial IS the product: the butler earns the right to auto-send
  per thread-type as its competence rises, and the person can flip it back at any time.
- **Time**: the human calendar (workday, school term, weekly rhythm) as the periodic
  driver; deadline horizons drive foresight.

## Why this domain wants THIS engine

Three mechanisms carry the product:
1. **The confidence contract** — LLM noise cannot corrupt the belief state faster than
   its own admitted confidence allows.
2. **Causal self-tagging** — the butler's own nudges cause completions; a naive learner
   concludes "she handles this herself" and goes quiet. The tagged learner knows what it
   caused.
3. **Provable non-training as the privacy pitch** — "this subsystem did not train on
   your mail" as an empty diff of its gauge snapshot, not a policy page
   (docs/THEORY.md, the gauge discipline).

## Smallest viable demo (owed)

A synthetic week of inbox+calendar for a fictional family; belief per thread easing as
evidence arrives; one nudge fired and its own effect correctly tagged in the confounding
ledger.
