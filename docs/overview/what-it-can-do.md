# What umwelt can actually do

Concrete capabilities, explained without the internals. For the technical picture see
[docs/SPEC.md](../SPEC.md) (how you describe a world) and
[CLAIMS.md](../../CLAIMS.md) (the evidence behind every claim below, tiered by how
proven it is).

## 1. Build understanding from a structured description, not a training pipeline

A world is a small structured file: the things you care about (a room, a stock
ticker, a device), how they relate (a room belongs to a house; a ticker belongs to a
sector), and what data feeds them (a motion sensor, a price feed, a news sentiment
score). umwelt turns that description into a live model — no labeled dataset, no
training run, no MLOps pipeline required to get started.

*Example:* to model "is anyone in the den, and is it warm enough," someone would
declare a `den` with two things to track — occupancy and warmth — point a motion
sensor and a thermostat reading at it, and umwelt starts holding a live,
uncertainty-aware belief about both, easing gracefully between readings instead of
freezing on the last number it happened to see.

**Worth being precise about:** that structured file is not free-form English. Someone
— or something — still authors it, following a documented checklist
([docs/SPEC.md](../SPEC.md)) and a working template
([examples/gridworld/](../../examples/gridworld)). What's new: an **experimental v1
of exactly that "paste in a description" pipeline now exists** (`umwelt-forge`,
[docs/FORGE.md](../FORGE.md)) — an embedded AI coding agent turns a plain-English
description into the structured file, and a deterministic, automated gate decides
whether the result is actually wired correctly before anything runs. Honest caveats:
it's a command-line tool for technical users today, it needs an AI-provider API key,
and while the *safety discipline* is machine-checked on every code change (a wrong or
even dishonest authoring attempt provably cannot register a broken world), the
*authoring quality* on real-world descriptions is not yet measured — that evaluation
is explicitly owed in the claims ledger. See
[how you'd work with it](working-with-it.md) for both workflows.

## 2. Fuse multiple, disagreeing sources — and learn which ones to trust

If two sensors, feeds, or forecasts report on the same thing, umwelt doesn't just
average them naively. It learns each source's reliability over time and weighs
accordingly, and if one source goes bad — drifts, breaks, gets corrupted — it can
isolate that specific source rather than let it drag down the whole picture, as long
as there's at least one other independent source to check it against.

## 3. Forecast forward

Because the model runs on continuous dynamics rather than snapshotted rules, it can be
asked to run forward in time: "what does the model expect if nothing new comes in."
This is the same machinery used for everyday comprehension, just run without new
observations — a forecast and a sensor reading are the same kind of thing, at
different time horizons.

## 4. Recommend safely — decide, but don't act, until you say so

Every decision umwelt can make starts in **shadow mode**: it decides, it logs the
decision, it shows you exactly what it *would have done* — and it dispatches nothing
to the real world until you explicitly promote that specific decision to "live." This
is the default behavior, not an opt-in safety switch you have to remember to flip.

## 5. Avoid a classic trap: mistaking its own footprint for reality

A system that both watches a domain and acts on it can quietly poison itself: it acts,
the world changes, and it "learns" that its action caused the change — even when it
didn't. umwelt tracks exactly what its own actions could plausibly have influenced and
discounts learning accordingly, rather than trusting every before/after comparison at
face value.

## 6. Run lean

No GPU, no heavyweight infrastructure. The reference deployment this was extracted
from ran for a year and a half on a $100 single-board computer. The current service
layer is a small local process you can run directly or in a single Docker container.

## 7. Host a game-shaped belief loop (library-first)

Beyond home-style sensors, the library ships a thin **host API** and a public
**fog corridor** example: places on a graph, scouting as partial observation,
shadow decisions, multi-agent private minds, and earned sub-routine automation —
all gated on synthetic data in CI. This is the “Fledgeling core” path: belief and
agency contracts for a host game, not the game itself. Technical roadmap:
[docs/FLEDGELING_CORE.md](../FLEDGELING_CORE.md).

## 8. Show its work, honestly

Every capability above is backed by a public, tiered evidence ledger: proven and
automatically re-checked on every code change, proven once on independent real data,
designed but not yet evidenced, or tried and explicitly rejected. That last category
exists on purpose — a project willing to publish what didn't work is a project you can
trust about what it says did.
