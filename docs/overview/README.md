# umwelt, in plain terms

*A short explainer for anyone evaluating umwelt as a potential customer, partner, or
investor — no engineering background required. For the technical documentation, start
at [the main README](../../README.md); for the full evidence ledger behind every claim
made here, see [CLAIMS.md](../../CLAIMS.md).*

## What is umwelt?

Most software that tries to "understand" a changing situation — a house, a market, a
fleet of devices, a customer relationship — does one of two things badly: it hard-codes
rules that break the moment the real world doesn't match them, or it bolts on a
statistical model that can't tell you when it's guessing.

umwelt is a general-purpose engine for the middle ground. A world you care about gets
described once, as a short structured file — not a traditional program, but not
free-form English either — and from that point on umwelt builds and continuously
maintains a live, honest model of it. It ingests messy, partial, sometimes-conflicting
data; it always knows, and can tell you, how confident it is in any given belief; and
it can recommend or take action without ever mistaking "I decided X" for "the world
told me X."

Writing that first description is real technical work today, not a plain-English
input box — see [how you'd work with it](working-with-it.md) for exactly who does
that, and how.

It is not a chatbot, not a general-purpose AI, and not a drop-in dashboard. It's
closer to an always-on nervous system for one specific domain: plug it into a house, a
portfolio, a fleet, a workflow, and it holds a running, uncertainty-aware picture of
that one thing, all the time.

**In one sentence:** umwelt turns a structured description of "the things I care about
and how they relate" — authored today with technical or AI-assisted help, not typed
as a paragraph of English — into a live, self-honest model that keeps itself current
and always knows the difference between what it has actually observed and what it
merely assumes.

## Where it comes from

umwelt is the extracted, generalized core of a system that ran for **eighteen months
as the resident "brain" of a real home**, on a $100 single-board computer, making real
decisions from real sensors, for a real resident. That deployment is where every hard
lesson in this project came from. umwelt is what's left once the house-specific pieces
were stripped out and replaced with a plug-in system any domain can use the same way.

## Why this matters

- **It's honest about uncertainty by construction**, not as a bolted-on feature. A
  low-confidence reading provably cannot move a belief as far as a high-confidence
  one — that's a property of the underlying math, not a rule someone remembered to
  add. If a data source contributes nothing (zero confidence), it is guaranteed to
  change nothing.
- **It doesn't confuse its own actions with facts about the world.** Any system that
  both watches a domain and acts on it risks a subtle, common failure: taking an
  action, seeing the world change, and quietly "learning" that its own action caused
  the change. Left unfixed, this kind of self-confusion inflated a real system's
  confidence in its own actions by **10.8×**; the mechanism umwelt uses by default
  cut that bias by **79%**, measured on 24 days of real deployment data.
- **It says what it doesn't know yet, out loud.** Every capability claim in this
  project is tracked in a public ledger, sorted into what's been measured and
  proven, what's designed but not yet evidenced, and what was tried and explicitly
  rejected because it didn't hold up under its own test. Nothing here is oversold —
  and the project ships the ideas that *didn't* work, too.

## Read next

- **[What it can actually do](what-it-can-do.md)** — capabilities, in plain terms,
  with concrete examples.
- **[How you'd work with it](working-with-it.md)** — the two ways to use it, and
  where the project honestly stands today.
