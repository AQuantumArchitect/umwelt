# THEORY — the estimator ladder, and what each rung measurably buys

*Adapted from the origin deployment's estimator-ladder document (meerkat
`docs/QUANTUM_KALMAN.md`). The origin question: "if we have a classical Kalman and a
phasor Kalman, why not take the next step and build a quantum (Bloch-sphere) Kalman?"
Answer: yes — it's a named object in filtering theory (the Belavkin filter),
`observe_qubit` was already its ad-hoc approximation, and the dropped terms were
confessable. It was built (`umwelt/substrate/belavkin.py`), then walked rung-by-rung on
the origin deployment's real data — the verdict is §5, and it is a **negative result
that ships as such**.*

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
    (cumulant order 2)   ball, with measurement back-action          ← "the quantum Kalman"
L5  full-ρ Belavkin    + multipartite coherence                     (2ⁿ — do not build)
```

The claim worth internalizing: **L4 is not "quantum instead of Kalman." L4 *contains*
L2 and L3 as limits.** Set coherences to zero and the qubit filter's z-equation *is*
the Wonham filter; linearize away the boundedness and it's the phasor Kalman; freeze
the gain and it's the α-blend. Building L4 doesn't abandon the classical upside — it
makes every classical special case reachable by turning physical dials (measurement
strength, efficiency, Hamiltonian) to zero. The ablation becomes: *walk the ladder on
replayed data and measure what each rung buys.* That walk has been taken (§5), and the
walking machinery lives on in this repo (`proofs/ladder_walk.py`).

## 2. What the "full quantum Kalman" concretely is

Quantum filtering (Belavkin 1992; Wiseman & Milburn 2009) gives the exact belief-update
for a quantum system under continuous weak measurement. For a sensor weakly measuring
observable L (say σ_z of a binary-state qubit) with strength k and **efficiency
η ∈ [0,1]**, the measurement record is

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
2. **Confidence = measurement efficiency η, natively.** η = 0 → the innovation term
   vanishes → the belief free-evolves. **The confidence contract's provable no-op is a
   theorem of the formalism**, not a convention imposed on it: a confidence-0
   observation cannot move the belief, because it is a detector that detected nothing.
   `0 < η < 1` half-commits. The engine's confidence contract turns out to be the
   standard quantum-filtering efficiency parameter — the strongest theoretical
   grounding the library's central rule could ask for.
3. **Back-action is explicit and tunable.** The deterministic `−2k·x` term says:
   measuring z at strength k destroys phase coherence at rate 2k. "The harder you pin
   *where the state is*, the faster you forget *where it is in its rhythm*" becomes a
   physical trade-off with one knob, rather than an emergent accident.

**At cumulant order 2** — the closure this library ships (`substrate/
cumulant_cluster.py`) — the SME becomes coupled equations on (e₁, e₂): means, phases,
and covariance blocks, with the innovation feeding all of them. That is "the quantum
Kalman": **the Belavkin filter closed at second cumulants.** Same state the
CumulantCluster already carries; what changes is the *update rule* — principled
state-dependent gain + innovation term instead of the hand-set α-blend.

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
   default-OFF** (`UMWELT_BELAVKIN=0`). The theory value stands — the confidence
   contract is a theorem with a reference implementation — but the estimator that
   ships is the one the data picked.
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
