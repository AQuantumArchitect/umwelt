# Butler — email + reminders + coordination for busy people

**Status: sketch + spec stub.** The domain adapter is designed; the synthetic demo is owed
(see CLAIMS.md — nothing here claims a measured result).

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
