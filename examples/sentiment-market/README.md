# Sentiment ↔ market — belief fusion over price and sentiment streams

**Status: sketch + spec stub.** The domain adapter is designed; the synthetic demo is owed
(see CLAIMS.md — nothing here claims a measured result, and this README will publish
whatever verdict the data gives, including "persistence is hard to beat").

## The umwelt

- **Nodes**: a market root → sectors → tickers; a macro node bridged to sectors.
- **Roles**: `drift`, `sentiment`, `volatility_regime`, `momentum`.
- **Observations**: normalized returns per bar; sentiment scores per source, where **η
  per feed is learned online** by the innovation-EMA estimator — a feed whose surprises
  systematically exceed its admitted confidence prunes itself out of the fusion. Macro
  releases enter as scheduled high-η events.
- **Outputs**: forecast surfaces only in the demo (no actuation); a paper-trade tendril
  later. If ever trading at size, market impact makes causal self-tagging relevant —
  noted, not built.
- **Time**: the exchange calendar (sessions, gaps, weekends) as the periodic driver.
  Process-time vs wall-time is a native fit here — volatility time is a known quant
  concept, and the engine's φ-clock measures elapsed *change*, not elapsed seconds.

## Why this domain wants THIS engine

The **trust web** is the star: sentiment feeds, model forecasts, and upstream estimators
fused by one learned operator with per-source reliability — "a sensor is a forecast with
horizon 0" means a data vendor and a forecasting model are the same kind of thing to the
fuser. And the **ladder-walk methodology is the deliverable**: the demo ships persistence
and α-blend baselines built in, and reports the table. On the origin deployment's data
the fanciest rung LOST to the α-blend and was turned off (docs/THEORY.md) — in this
domain, an engine whose own harness demotes its own machinery is a credibility feature.

## Smallest viable demo (owed)

One year of daily bars + a public sentiment dataset for ~5 tickers; run the ladder;
report the table, whatever it says.
