# Resilience recommender — e-commerce for neighborhood preparedness

**Status: sketch + spec stub.** The domain adapter is designed; the synthetic demo is owed
(see CLAIMS.md — nothing here claims a measured result).

## The umwelt

- **Nodes**: a household root → need domains (power, water, food, comms, medical) →
  inventory items as device-like leaves; a neighborhood node bridged to the household
  (shared `risk_exposure`).
- **Roles**: `preparedness`, `stock_level`, `usage_rate`, `risk_exposure`.
- **Observations**: purchases, inventory declarations, consumption inferences — and
  hazard forecasts bound as SENSORS (a forecast is a sensor with horizon > 0: severe-
  weather outlooks and grid-stress alerts enter through the same BindingSpecs as
  everything else, at their published confidence).
- **Outputs**: recommendation tendrils, entering under the wishlist law — a
  recommendation is proposed as a *signal first* (accept/dismiss is an observation of
  preference) before it earns dispatch rights. `shadow=True` until then.
- **Time**: a civil/seasonal calendar driver replaces the origin's solar clock; hazard
  horizons drive the foresight ladder.

## Why this domain wants THIS engine

**The recommender feedback loop is exactly the confounding trap the engine was built to
escape.** "We recommended it, they bought it, therefore demand is high" is the
downstream-from-us error: a naive learner credits its own recommendations as world
signal. On the origin deployment's 24 days of real data, a naive learner credited the
system's own actions at 10.8× their true strength; the graph-derived self-tagging cut
that bias 79% (see CLAIMS.md for provenance). Every recommendation tendril here carries
a `graph_node`, so its dispatch echo discounts exactly the roles its own action projects
onto — no per-product special-casing.

## Smallest viable demo (owed)

One synthetic household, five need domains, a simulated year with three hazard events;
run naive vs tagged recommendation learners and report whichever split the data gives.
