# S-RB2-BREAK registered development diagnosis — 2026-11-01

Base: FULL-v4 `22cc32f65a80af482a1a277c5f22377fa727d0a3`; checkout
`03c81f91` adds reporting only. Owner authorizes inference, no neural training.
Read shared rules, morning report, INT handoff and noon evidence inventory/source
audit before implementation. The reported +14,648 veh-s is prior final-panel
evidence, NOT a development target or an attributable causal fraction.

## Exposure / scope
- Forecast and diagnose ONLY 969420–969431 (subset of FIT 969400–969431).
- Never load an observable start, future outcome or score for 969432–969439.
- Existing aggregate final report is prior owner-supplied context; its outcomes
  must not select changes. Final evaluation belongs to S-RB2-INT.
- All settings from frozen motion-only support, own causal pre-t0 histories and
  counters. All learned acceleration/lane/merge/admission/noise corrections OFF.
- New immutable output roots, identity/hash receipts, exact conservation.
- WM inference only; no SUMO generation; no black-box training.

## One factor at a time, paired latent keys
Development screen initially uses future 0, all 12 worlds, all five arms, H600
(120/300 prefixes retained). Expand promising *source-supported* repairs to K8;
K1 screens are descriptive, not distributional qualification. Reuse saved frozen
FULL-v4 development futures as the baseline after identity checking.

Predeclared diagnostics (not deployable recipes by themselves):
1. `no_mainline_lc`: suppress mainline lateral changes only, retain ramp merges.
2. `no_merge_advice`: remove ramp/mainline merge speed negotiation only, retaining
   gap acceptance and lane-end stopping. This identifies negotiation, not all merge physics.
3. `depart_last`: SUMO documented departSpeed=last, same departure lane policy.
4. `depart_first`: SUMO documented departLane=first, same departSpeed=max.
5. `population`: causal own-driver posterior replaced by same fitted population.
6. `point_v0`: replace v0 support coordinate by its population mean, other coordinates unchanged.
7. `point_tau`: replace tau support coordinate by its population mean, others unchanged.
8. `fixed_rate`: conditional request-rate estimate count/exposure, Poisson timing retained.

Do not infer additive causal shares from nonlinear, one-factor interventions.
Compare road/source/total delay bias at120/300/600, upstream/merge/ramp speed-field
CRPS, unchanged BIC event rates per demand×penetration×arm, onset CDF/IBS and
conservation. Report coverage descriptively only at actual K. Sparse-cell CIs are
NA; no future rows treated as independent worlds. Source parity and causal
motion likelihood, not arbitrary outcome scalars, determine deployable fixes.

## Resources / coordination
Local NVIDIA driver unavailable. gpuA checked at11:59 host time: idle GPU,
49139MiB free, load6.4. Maximum two BREAK workers initially,5GiB each,
one CPU thread/nice19; check actual GPU/load before launch. Do not signal peer PIDs.
S-RB2-LANE owns LC2013 corrections; BREAK owns diagnostics and longitudinal,
insertion/arrival fixes. Publish requests/results under rb-shared with own names.

##12:18 source-supported extension (registered before candidate inference)
`lane_limits`: enable `physics.sumo_cf_lane_speed_limits=true` alone. Audit found
that SUMO CF calls current lane getVehicleMaxSpeed, whereas the deployed CF
adapter uses reference v0 on the22.22m/s ramp approach. Insertion and merge
negotiation already map the lane limit; longitudinal CF does not. Reuse the
existing network mapping (no fitted constant), preserve sampled reference v0
and mainline/acceleration-lane outputs exactly. Source-backed parity correction,
not a forecast improvement claim until matched development screen completes.

##12:27 shrinkage endpoint and resource extension
`raw_posterior` removes ONLY the contamination prior (prior_count=0) when
conditioning the same finite support on the same causal own-start motion. Pair
with `population` (infinite-prior-count endpoint) and baseline19.35568. No scalar
sweep or outcome-fitted parameter. Correction canary CPU import failed before
inference due to local filename shadowing; preserved. Repaired module dispatch,
then launched SAME-device CPU baseline/correction pair after load audit9.36.
Two nice19 single-thread CPU workers, no extra GPU worker; GPU cap remains2.

##12:30 CAV source/admission consistency (before candidate inference)
`cav_settings`: bind CAV admission/initial/lane-geometry settings to the SAME
registered motion-fitted CAV vector already forced by `execution.inferred_follow_cap`
through `C4_EXECUTION_DRIVER`. The HDV encoder's CAV pool otherwise controls
insertion while the external CAV executor ignores that pool. This is a
cross-interface physics consistency correction, not a new execution plant or
driver-truth lookup. Same initial CAV mask and source CAV-ordinal identities;
HDV particles, arrival streams, controller/executor unchanged. Explicit opt-in.
Inherited execution artifact TRAIN fit-world manifest is unchanged; no additional
TRAIN tape opened or physical fit performed by BREAK.
