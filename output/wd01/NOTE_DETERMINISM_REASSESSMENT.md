# Reassessment after the REP-P finding: the campaign SUMO is deterministic after t0

World model session, 2026-09-29. This note is for main discussion and the owner.
It dispatches no tasks.

## The finding

At the REP-P gate (w8488), the reconstructor ran the scenario again with the
RNG state stripped and a new `--seed`. The trajectory reproduced the saved one
tick-exactly. The campaign configuration has no stochastic element after t0:

- IDM (sigma is not used);
- speedDev 0;
- LC2013 with lcSigma 0;
- departures scheduled from the plan seed.

**Open check.** Our p3e found a dawdle signature in the chunk-1-era *native
Part A roots* (the residual matched −σ·a·dt/2 per vType). Either those roots
used a different configuration, or the seed does not reach the RNG on the
restore path. A cheap fresh-start two-seed test settles which (proposed to the
reconstructor). The rest of this note assumes determinism holds for the
campaign data.

## 1. Part A

**What the truth is.** With the full state (every vehicle's lane, s, v, a and
its vType parameters) plus the arrival schedule, the next 30 s are a
deterministic function. Our emulator's "fitted one-sided dawdle" and its LC
hazard model randomness that is not in the campaign data. The spread they
produce is partly spurious. The genuine uncertainty for a real planner is:

- the hidden per-vehicle vType (car-following and lane-change parameters);
- the reconstruction error of the observed state;
- future arrivals, if they are not known.

**What becomes possible.** An exact re-implementation of SUMO IDM + LC2013
(with its lane-change state machine, the merge logic, and the insertion
logic) should reproduce SUMO to near float precision *when given the true
vTypes*. That gives a hard, new test: an **oracle-vType exactness test**, where
the emulator with true vTypes must match SUMO trajectories on TRAIN roots to
within a tight tolerance.

- The structural redesign (v2, DESIGN_PARTA_STRUCTURAL.md) moves directly
  toward this. Its target changes from "LC2013-like" to "LC2013-exact". Dawdle
  becomes 0 by default (kept only behind a flag for non-campaign data).
- **SUMO itself as Part A** (the planner-owned perfect simulator, contract v2
  §6) is exact given the state. For online use it still needs the hidden
  vTypes, which would be sampled K times from the posterior, and it is slow.
  It is the gold reference for the emulator. The fast emulator remains the
  planner's engine.

**Gate v2 and C2L1.**

- The criteria stay valid. Coverage (c) now measures whether the spread
  represents *hidden-state* uncertainty. The p3o spread factors were tuned
  partly to cover dawdle-like variation; with v2 they are re-fitted on TRAIN
  (vType-posterior spread only).
- C2L1 (amendment 1: v1 and v2 side by side, v2 as the candidate) stays
  unchanged. v2 must still be frozen before the look.
- I propose adding a TRAIN-only oracle-vType exactness check to v2's
  acceptance *before* the freeze.

## 2. A2 re-read (the "30 s action effects are below noise" finding)

That conclusion is **withdrawn**.

- Each arm-vs-native pair from a restored t0 state is the exact effect for
  that root.
- The σ ≈ 15 veh·s is the spread of *true* effects across roots, not noise.
- The naive zero-Δ predictor won because Part A's paired-Δ error was larger
  than the true effects. That is model error, and it is fixable (exact
  dynamics, the right controller emulation, the execution rule).

The A2 report and DEV-2 notes need an erratum. The existing chunk-1–3 branch
data are the validation set for paired Δ: 7 arms × about 140 worlds, with no
replicates needed.

## 3. Part B

**What the labels are.** They are deterministic given the full state and
future arrivals.

**Leak check: the V0 demand vector is the exact future schedule.** It holds
per-minute scheduled arrivals per entrance over (t, t + 600], taken from the
plan. A real planner knows demand rates at best, not the realised arrival
schedule.

Ridge ablation (CAL MAE, v0.2 data; script at
tmp/wd01_v02_diag/ablate_demand.py):

| τ | with exact schedule | without schedule (state + demand level) |
|---|---|---|
| 30 | 84 | 90 |
| 60 | 188 | 238 |
| 120 | 418 | 666 |
| 300 | 1361 | 2694 |
| 570 | 2989 | 6450 |

The exact schedule roughly halves the long-τ error. So the v0.2 net, the
ridge and the v0.3 hybrid all lean on **oracle future arrivals**.

**Recommendation: keep two variants and label them.**

- **V_oracle-arrivals**, the current one:
  - valid for *offline, CRN-paired* planner evaluation, where both plans share
    the same arrivals;
  - it is an upper bound on skill;
  - as a planner-ranking signal, much of the arrival information cancels in
    ΔQ.
- **V_deployable**:
  - conditioned only on what an RSU could know: demand level (the configured
    or estimated rate), observed upstream and entrance counts over the recent
    past, and time of day;
  - this is the number to quote for deployment claims.
  - Build it in v0.3.1 or v0.4 (a feature switch in the featurizer and the
    hybrid). Report both in every eval.

The v0.3 hybrid report will carry this caveat explicitly.

## 4. G-2 (relative safety)

G-2 is now feasible without replicates:

- paired Δ(TTC<3) and Δ(hard-brake) between arms at the same restored state
  are exact per root;
- the validation compares Part A's predicted paired Δ with the truth pairs
  over many roots (sign agreement, rank correlation, MAE vs zero-Δ);
- develop on TRAIN, then do a CAL/fresh-world look under a registration.

The planner's K = 1 relative-safety veto has no noise-driven false vetoes.
Its effect on win rates (83.7% → 46.9%) reflects real predicted safety
differences or model error. G-2 validation tells which.

## 5. What changes, in priority order (for the owner's decision; nothing dispatched)

1. Run the fresh-start two-seed test (reconstructor, about 1,200 sim-s) to
   confirm determinism for the campaign configuration.
2. Part A v2 (the approved structural redesign): retarget it to
   LC2013-exact + IDM-exact with dawdle 0, and add the oracle-vType exactness
   test on TRAIN before the freeze. The C2L1 look stays as amended.
3. Part B: label v0.3 as oracle-arrivals. Add the deployable-feature variant
   next (v0.3.1).
4. A2 erratum, and a G-2 validation plan on the existing branch pairs.
5. REP-P: close it as "σ_rep = 0 by construction" if (1) confirms
   determinism.
