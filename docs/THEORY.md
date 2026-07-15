# THEORY — the estimator ladder, and what each rung measurably buys

*Internal technical note (not a product brief).* Adapted from the origin deployment's
estimator-ladder document (meerkat `docs/QUANTUM_KALMAN.md`). The origin question was
engineering, not branding: if we already have a classical Kalman and a phasor Kalman,
is there a principled next rung on the Bloch ball (the Belavkin filter of quantum
filtering theory)? That filter is a named object in the literature;
`observe_qubit` was already an ad-hoc approximation of it. It was built
(`umwelt/substrate/belavkin.py`), walked rung-by-rung on the origin's real data —
verdict in §5 — and is a **negative product result that ships as such** (default OFF).
Classically simulated; no quantum hardware is involved or claimed.

If this document and [CLAIMS.md](../CLAIMS.md) ever disagree, the ledger wins.

---

## 1. The reframe: it's a ladder, not a fork

The load-bearing-substrate question is usually framed as a fork between two engines.
The stronger frame is a **strict containment ladder**, where each rung adds one
capability and reduces to the rung below in a limit:

```
L0  α-blend            what ships by default: ρ ← (1−α)ρ + α·ρ_target, α from the binding
L1  scalar Kalman      per-leaf mean+variance, principled gain
L2  phasor Kalman      + rotation: phase/rhythm as 2-vector state
L3  Wonham filter      + boundedness: exact nonlinear filter for a
                         two-state jump process observed in noise
L4  qubit Belavkin     + coherence: L3 and L2 unified on the Bloch
    (cumulant order 2)   ball, with measurement back-action          ← Belavkin / SME filter
L5  full-ρ Belavkin    + multipartite coherence                     (2ⁿ — do not build)
```

**L4 is not a product rebrand of Kalman as "quantum."** It is a containment rung: L4
*contains* L2 and L3 as limits. Set coherences to zero and the filter's z-equation *is*
the Wonham filter; linearize away the boundedness and it's the phasor Kalman; freeze
the gain and it's the α-blend. Building L4 does not abandon the classical upside — it
makes every classical special case reachable by turning dials (measurement strength,
efficiency, Hamiltonian) to zero. The ablation is: *walk the ladder on replayed data
and measure what each rung buys.* That walk has been taken (§5); the machinery lives
in this repo (`proofs/ladder_walk.py`). **What ships by default is L0**, not L4.

## 2. What the Belavkin / SME filter concretely is

Quantum filtering theory (Belavkin 1992; Wiseman & Milburn 2009) gives the exact
belief-update for a system under continuous weak measurement in that formalism. For a
sensor weakly measuring observable L (say σ_z of a binary-state qubit) with strength k
and **efficiency η ∈ [0,1]**, the measurement record is

    dy = ⟨L + L†⟩ dt + dW / √(4kη)

and the conditioned state evolves by the stochastic master equation (SME):

    dρ = −i[H, ρ] dt  +  D[L]ρ dt  +  √η · H[L]ρ · dW      (innovation term)

where `H[L]ρ = Lρ + ρL† − Tr[(L+L†)ρ]·ρ` and `dW = √(4kη)·(dy − ⟨L+L†⟩dt)` is the
**innovation** — the surprise, exactly the quantity the surprise tape records
phenomenologically.

For one qubit measured in σ_z, the Bloch components obey (schematically):

    dz = [Lindblad drift] dt + √(4kη) · (1 − z²) · dW
    dx = [drift − 2k·x] dt − √(4kη) · x·z · dW        (same for y)

Three things to notice, because they are the entire payoff:

1. **The gain is state-dependent and bounded: ∝ (1 − z²).** That is the variance of a
   ±1 variable with mean z. Near the poles (certain), the filter automatically stops
   listening; at the equator (maximally uncertain), it listens hardest. This is the
   Wonham/logistic gain `p(1−p)` — the *optimal nonlinear filter for a two-state jump
   process*, which is what a binary world-state (occupied/vacant, on/off) literally
   is. A linear Kalman never gives you this; you clamp and hand-tune instead. A
   hand-set collapse alpha is a crude constant standing in for exactly this term.
2. **Confidence as efficiency η.** η = 0 → the innovation term vanishes → the belief
   free-evolves. In this formalism the no-op is a theorem, not a bolted-on guard: a
   confidence-0 observation is a detector that detected nothing. `0 < η < 1`
   half-commits. The library's everyday confidence contract is *compatible with* that
   reading; the default α-blend path also enforces η=0 as a no-op without requiring
   the full SME update.
3. **Back-action is explicit and tunable.** The deterministic `−2k·x` term says:
   measuring z at strength k destroys phase coherence at rate 2k — a single knob for
   the trade-off between pinning a coordinate and retaining rhythm/coherence.

**At cumulant order 2** — the closure this library ships (`substrate/
cumulant_cluster.py`) — the SME becomes coupled equations on (e₁, e₂): means, phases,
and covariance blocks, with the innovation feeding all of them. That is the Belavkin
filter closed at second cumulants — sometimes nicknamed a "quantum Kalman" in the
filtering literature. Same state the CumulantCluster already carries; what changes is
the *update rule* — principled state-dependent gain + innovation instead of the
hand-set α-blend.

## 3. How it lands in this library

> **Status: BUILT, EVALUATED AT THE ORIGIN, SHIPS OFF.** `umwelt/substrate/belavkin.py`
> implements the conditioned Kraus update in closed form (log-odds shift
> `z′ = tanh(atanh(z) + s·y)` + coherence factor), the cumulant cross-update (measuring
> one qubit moves correlated peers via the regression gain `cov/v`), and the exact
> full-ρ Kraus reference (the test oracle). The substrates expose
> `measure_qubit(idx, record_z, strength, confidence=None)`; `engine.ingest`'s
> observe path branches on `UMWELT_BELAVKIN=1` (default OFF → the α-blend is
> byte-unchanged). `BindingSpec` carries optional `(strength k, efficiency η)` with
> `measurement_alpha() = k·η` winning over the collapse alpha when set, and
> `learning/observation_trust.py`'s innovation-EMA (`UMWELT_LEARN_COLLAPSE`) is the
> online η-estimator.

Design decisions that held at the origin and carried over:

1. **Not a fifth backend — the upgrade of the observe seam.** State layout (e₁, e₂)
   unchanged; the measurement update is a discrete **Kraus map** (positivity-preserving
   by construction, closed-form e₁/e₂ updates), not Euler–Maruyama on the SME.
2. **The caller folds confidence; the substrate records it.** Fixing this uniformly at
   the origin found a live bug — a path had been double-applying confidence (~conf²),
   trusting readings *less* than declared. One contract, one test, all substrates.
3. **Stay at cumulant order 2.** L5 reintroduces the 2ⁿ wall for coherences no consumer
   reads. The closure *is* the tractability contribution; the Belavkin update composed
   with it is the novel object. What the closure loses is measured, not assumed:
   `proofs/fidelity_harness.py` drives the exact 2ⁿ state and the O(n²) closure side by
   side on the same stream and reports z/purity divergence + decision parity.

## 4. Pathologies, found and pinned

- **The pole pathology (found live at the origin).** Repeated one-sided evidence
  saturates z → the Wonham gain (1−z²) goes deaf *and* the cross-update gain
  `cov/v = cov/(1−z²)` explodes — on the origin's real tape this produced an overflow
  through e₂ feeding the coupled evolution. The purity floor (`Z_CAP = 0.98` in
  `proofs/ladder_walk.py`) is **load-bearing for any L4 use**, not a nicety.
  Mitigations: clip z before each measurement, clip e₂ to [−1,1] per bin,
  `clamp_physical` after evolution.
- **The measurement-model burden** resolved as designed: defaults derive from the
  existing alphas (α ↔ kη·dt to first order); observation_trust adapts η online.
- **The null result** was always a named risk — and is, in fact, the result (§5). It
  decides which estimator ships; that decision was the point.

## 5. The ladder-walk verdict (measured at the origin, 2026-07-04)

This is the signature honesty artifact of the lineage, kept prominent and verbatim in
spirit. The numbers below were measured **on the origin deployment's data** — a real
24-day cassette, 6908 five-minute bins of sparse presence reports from a lived-in home
— by the origin's `experiments/ladder_walk.py` (this repo's `proofs/ladder_walk.py` is
that harness, re-pointed at a synthetic gridworld stream; it does NOT re-measure these
numbers and cannot). Next-bin prequential scoring, per-rule strength dial calibrated on
the train split, held-out last 30%:

| contender | couplings | ALL | GAP | TRANSITION |
|---|---|---|---|---|
| persistence | independent | **0.1346** | **0.1210** | 0.8359 |
| L0 α-blend | independent | 0.1349 | 0.1215 | 0.8292 |
| L4 Belavkin | independent | 0.3034 | 0.2997 | 0.9000 |
| L4η (learned η) | independent | 0.7729 | 0.7781 | 1.1390 |
| L0 α-blend | exchange-coupled | 0.5226 | 0.5613 | **0.7399** |
| L4 Belavkin | exchange-coupled | 0.6139 | 0.6529 | 0.7763 |

1. **The origin's world was persistence-dominated** — persistence 0.1346 ≈ α-blend
   0.1349 ≪ Belavkin 0.3034 overall. Nothing beat hold-the-last-report, the α-blend
   tied it, and **the full Belavkin filter was DENIED by its own experiment — it ships
   default-OFF** (`UMWELT_BELAVKIN=0`). The formalism still clarifies the η contract
   and keeps a reference implementation; the estimator that ships is the one the data
   picked (L0).
2. **Cross-node structure earns at transitions** (−11.5% for the coupled blend) at the
   cost of steady-state cross-talk when J is hand-set → couplings should stay
   **learned-from-zero**, growing only where they pay. The regime, not the law,
   decides.
3. On a transition-dominated synthetic walk, the origin measured the rung's upside:
   L4η-coupled beat persistence at transitions (1.365 vs 1.987). The rung is real;
   the origin's data didn't live where it wins.

## 6. Cross-references

- The walking machinery, alive in this repo on synthetic streams:
  `proofs/ladder_walk.py` (the ladder), `proofs/fidelity_harness.py` (closure vs
  full-ρ), `proofs/deconfound_smoke.py` (causal self-tagging mechanism).
- The claim ledger, tier by tier: [CLAIMS.md](../CLAIMS.md) — if it and this document
  disagree, the ledger wins.
- The flagship theorem that gates the repo: `proofs/blank_slate.py`.
