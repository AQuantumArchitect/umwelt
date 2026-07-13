# How you'd work with it

## Two ways to use umwelt

**1. As the engine behind a product.** If you're using something *built on* umwelt —
a home advisor, a market-sentiment tracker, whatever a customer-facing app looks like
— you wouldn't touch umwelt directly at all. You'd use that product's own interface;
umwelt is the part quietly holding the live model underneath it.

**2. As a technical integrator.** If you or your team want umwelt to power something
of your own, the shape of the work is:

1. **Describe your world.** A short file listing what you're tracking, how it's
   structured, and what data feeds it — a domain expert can review this file; it
   isn't code in the traditional sense.
2. **Point umwelt at it.** Either as a library inside your own application, or as a
   small standalone service (`umweltd`) your application talks to over a plain HTTP
   API — post readings in, read back live beliefs, forecasts, and recommendations.
3. **Start in shadow mode.** Watch it decide before it's ever allowed to act on
   anything; promote individual decisions to "live" once you trust them.

There's also a small command-line tool (`umweltctl`) for the basic operational
loop — create a world, check its health, push data, read results — without writing
any integration code at all, useful for a first hands-on evaluation.

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
- *"What exactly would I need to describe to model my domain?"* →
  [docs/SPEC.md](../SPEC.md).
- *"What does a finished example look like?"* →
  [examples/gridworld/](../../examples/gridworld) — a complete, working reference
  domain you can run yourself.
