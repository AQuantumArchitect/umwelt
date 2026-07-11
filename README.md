# umwelt

**A belief-field engine.** Describe a world as data; feed it observations; it holds a live,
honest, uncertain comprehension of that world — and forecasts from it.

> **Status: under extraction (0.1.0.dev).** The engine is being lifted out of
> [meerkat], a home-comprehension system that has run for 18 months on a $100 ARM
> board in a real house with a real resident. This README becomes the full story when
> the extraction's proof gate is green; until then, the phase plan lives in the
> commit history.

The name: an [umwelt](https://en.wikipedia.org/wiki/Umwelt) is a world as modeled by an
organism — which is exactly what this library builds. Give it a domain's umwelt as a
declarative spec (nodes, roles, bridges, bindings); it boots blank, treats every
observation as a weak measurement whose strength is the observer's confidence, and grows
a live world-model it can forecast from and act through.

License: Apache-2.0.
