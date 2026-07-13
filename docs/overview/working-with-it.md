# How you'd work with it

## Two ways to use umwelt

**1. As the engine behind a product.** If you're using something *built on* umwelt —
a home advisor, a market-sentiment tracker, whatever a customer-facing app looks like
— you wouldn't touch umwelt directly at all. You'd use that product's own interface;
umwelt is the part quietly holding the live model underneath it.

**2. As a technical integrator.** If you or your team want umwelt to power something
of your own, the shape of the work is:

1. **Describe your world.** The end result is a short structured file listing what
   you're tracking, how it's related, and what data feeds it — closer to a small,
   declarative configuration than a paragraph of prose, following a documented
   schema and hard-won domain-modeling idioms ([docs/SPEC.md](../SPEC.md)). Two ways
   to get there today:
   - **The embedded compiler (experimental).** `umwelt-forge`
     ([docs/FORGE.md](../FORGE.md)) takes a plain-English description, has an
     embedded AI coding agent author the structured file, and — the important part —
     runs a deterministic, automated gate that checks every declared input actually
     drives the model before the world is allowed to exist. The agent's own claim of
     success is never trusted; only a passing gate registers a world. It's a
     command-line tool needing an AI-provider API key, and its authoring quality on
     real descriptions is not yet measured (that evaluation is owed in the ledger).
   - **The demonstrated manual workflow.** A domain expert describes what they want
     in conversation and an AI coding assistant (or a person who knows the schema)
     writes the file, validated by the same automated gate. This is how this
     project's own domains were built, and it remains the recommended path for
     anything you'd put in production.
2. **Point umwelt at it.** Either as a library inside your own application, or as a
   small standalone service (`umweltd`) your application talks to over a plain HTTP
   API — post readings in, read back live beliefs, forecasts, and recommendations.
3. **Start in shadow mode.** Watch it decide before it's ever allowed to act on
   anything; promote individual decisions to "live" once you trust them.

There's also a small command-line tool (`umweltctl`) for the basic operational
loop — create a world, check its health, push data, read results — once the
structured description above exists, useful for a first hands-on evaluation without
writing any application-integration code.

## What running it looks like, concretely

Once a world is described, it's one command away from running:

```
docker run -p 7071:7071 -e UMWELTD_API_KEY=... umweltd
```

A small local service comes up, ready to accept data and answer questions about
whatever domain you described — no cloud dependency, no data leaving your own
infrastructure unless you choose to send it somewhere.

## Where the project stands today

Being upfront, in the same spirit as the evidence ledger:

- **Proven core.** The underlying engine ran for eighteen months as a real home's
  resident decision system, and has since been validated a second time on an
  independent real-world dataset it was never built around — both documented with
  numbers, not just claimed.
- **Freshly productized.** The pieces that make umwelt usable *outside* its origin
  deployment — the service layer, the operator tooling, the container packaging, the
  onboarding docs — were built and hardened very recently. They're tested (130+
  automated tests, all green on every change) but young.
- **Early stage.** This is a private, pre-revenue project. There is no multi-tenant
  hosted product yet — today it's a single-tenant service you run yourself, locally
  or in your own infrastructure. Multi-tenant hosting is on the roadmap, not yet
  built.
- **One technology, several potential products.** The same engine is the intended
  core for several different applications currently in development — a sign the
  underlying abstraction generalizes, not a promise that any specific one ships on a
  particular timeline.

## Questions worth asking — and where the honest answers live

- *"Does this actually work, or is it a nice idea?"* → [CLAIMS.md](../../CLAIMS.md) —
  every claim tiered by evidence, including the ones that were tried and failed.
- *"Can I just describe my domain in plain English and get a working model?"* →
  **Yes, experimentally — with honest caveats.** `umwelt-forge`
  ([docs/FORGE.md](../FORGE.md)) is exactly that pipeline: describe the domain, an
  embedded AI coding agent authors the structured file, an automated deterministic
  gate verifies it's wired correctly, and only a passing gate lets the world run.
  The safety discipline is machine-checked on every code change (a wrong — or even
  dishonest — authoring attempt provably cannot register a broken world). The
  caveats: it's a command-line tool for technical users, it needs an AI-provider
  API key, and how *often* it authors a correct world from a real description is
  not yet measured — that evaluation is owed in the ledger before this is claimed
  as more than an experimental capability. For production work, the demonstrated
  AI-assisted manual workflow above remains the recommended path.
- *"What exactly would I need to describe to model my domain?"* →
  [docs/SPEC.md](../SPEC.md).
- *"What does a finished example look like?"* →
  [examples/gridworld/](../../examples/gridworld) — a complete, working reference
  domain you can run yourself.
