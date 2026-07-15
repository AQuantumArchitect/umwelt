# Forecasting — the future as first-class beliefs

## A sensor is a forecast with horizon 0

The engine's central foresight identity: a physical signal, a forecasting model, and an
upstream engine's published belief are the same KIND of thing — an estimate of a leaf's
value with a confidence attached. So there is ONE fusion operator for all three problems:

- **sensor health** — a flaky signal's learned reliability reroutes weight to its peers;
- **forecast ensembling** — model predictions fuse with live readings at their own η;
- **engine chaining** — an upstream engine's beliefs enter downstream as one more leaf.

That operator is the **trust web** (`umwelt.foresight.trust_web`; optional denser
variant in `qubit_trust_web`): one learned per-leaf fuser, prior-initialized so day-1
behavior is
provably the confidence-weighted average — turning it on changes nothing until it has
evidence to learn from. Opt-in at the ingress membrane (`UMWELT_TRUST_WEB`); webs-off is
byte-equivalent to last-wins.

## Confidence is learned, not asserted

`umwelt.learning.observation_trust` estimates each source's η online from its innovation
EMA — a source whose surprises systematically exceed its admitted confidence prunes
itself out of the fusion. Downstream, the confidence contract holds everywhere: η=0 is a
provable no-op (docs/THEORY.md).

## Free-run rollouts

`umwelt.foresight.forecast_rollout` runs the field's own dynamics forward — snapshot,
evolve without input, read the future beliefs, restore. The forecast is the model
imagining, not a separate regressor bolted on; where a deterministic future is KNOWN (a
periodic driver's phase), the dissolved form serves the exact label instead of an
approximation of it. Forecasts the engine consumes re-enter through ordinary bindings
(`forecast_zflip` normalizer) at their recorded confidence — and REPLAY preserves that
confidence, so training equals deployment gauge-for-gauge (`umwelt.events`).

## Dreaming

At rest, `umwelt.foresight.dreaming` mutates shadow copies against replayed evidence and
`dream_topology` proposes coupling growth — discover-and-record first, graft only when
proven. Region/stream vocabularies are registries; the engine ships them empty.

## What's owed

Trust-web A/B effect sizes and any domain's forecast-skill claims are OWED until measured
(see CLAIMS.md). The one measured verdict shipped with the library is a negative one, and
it stays: the fanciest estimator rung lost to an α-blend on the origin deployment's real
data (docs/THEORY.md).
