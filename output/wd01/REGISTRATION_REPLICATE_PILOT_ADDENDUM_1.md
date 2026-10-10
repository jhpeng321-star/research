# WD-01-REP-P — Addendum 1 (analysis only; runs and budget unchanged)

2026-09-28. Main discussion asked for this after the planner's chunk-2 finding.
With K = 1, the relative-safety veto removes most wins: 83.7 % → 46.9 %.

This addendum adds an analysis item to §6 of REGISTRATION_REPLICATE_PILOT.md
(sha b90ef72e…). It changes nothing about the runs, the worlds, the seeds or
the budget.

## Added item: the false-veto rate of a single-realisation relative-safety veto

**Metrics.** For each (world, arm ∈ {pi, fixed1}, H ∈ {30, 60, 120, 600}) and
each metric m ∈ {TTC<3 exposure, hard-brake seconds}:

- The replicate paired deltas are Δ_r = m(arm, r) − m(native, r), for r = 0..4.
  r = 0 is the original chunk-2 branch.
- The expected-value reference is the replicate mean Δ̄ over r = 1..4, with its
  SE = σ_rep / √4.

**Veto definition.** A single-realisation veto fires when Δ_r > tol, as
applied by the planner. The tolerances are:

- tol = 0, the owner default;
- the planner's chunk-2 tolerance, taken from its report and recorded here
  with its source.

**Classification.** Each (pair, r) is scored on two questions: did the
single-realisation veto fire, and is the expected delta harmful?

| | expected delta harmful | expected delta NOT harmful |
| --- | --- | --- |
| veto fired | true veto | **false veto** |
| veto not fired | missed veto | correct pass |

- Expected delta harmful means Δ̄ > tol + 2 SE.
- Expected delta NOT harmful means Δ̄ ≤ tol.
- Pairs with Δ̄ between tol and tol + 2 SE are **indeterminate** and are
  counted separately.

**Reported:**

- the false-veto rate, P(veto fired | expected delta not harmful), with a
  bootstrap CI over worlds;
- the missed-veto rate;
- the indeterminate share;
- all three per H and per metric.

**Model-based estimate.** Under a normal noise model with the measured
σ_rep(H), the probability that a zero-effect pair crosses tol by noise alone
is P(false veto) = 1 − Φ(tol / (√2 · σ_rep)). The √2 applies if the arm and
native noise are independent within r; if CRN correlation is present, the
measured within-r σ of Δ is used directly. The model estimate is shown next
to the empirical rate.

**Caveat:** with W = 4 worlds and 2 arms this is an order-of-magnitude
estimate. The main stage would tighten it.
