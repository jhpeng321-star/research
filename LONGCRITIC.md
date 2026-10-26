# S-LONGCRITIC-1/2 (2026-10-26)

Branch `c4/longcritic`, base `d6a426c6` (committed mainstart tip).
Registered in the shared REGISTRATION.md before offline execution. **Not activated.**

## Implementation

Default-OFF features:

1. `[longcritic].enabled`: retain authoritative real-return regression, gamma=1.
   Real share `m` multiplies real regression; `(1-m)` multiplies difference
   assistance. Enabled mode never regresses WM levels. This fixes the original
   opt-in's incompatible mix=1 guard versus inherited0.3. A native critic is warmed exclusively
   on real native returns, then frozen. Auxiliary regression fits
   `V_policy(s) - stop(V_native_real(s))` to a shrunk WM policy-minus-native
   **reward difference**, never a WM level. Its terminal state derivatives carry
   delayed benefit into the actor. One TRAIN start/update; paired all-policy/native
   histories, drivers, arrivals, lane/action noise. No-grad rollouts rotate among
   measured60/120/180/240/300s horizons; entrant capacity scales with horizon.
   Native/real continuation horizons must match; changed native data triggers
   rewarming. Native critic/optimizer/warm status are checkpointed. After the selected
   horizon, residual policy-native advantage is assumed zero.
2. `[exploration].long_persistence`: existing group AR(1) e-folding time ramps
   30→60 s at curriculum positions 0→150, training imagination only. Each
   rollout resets noise; deployed actor/executor and MAPPO are unchanged.
3. `collect.long_group.enabled` plus `collect.long_single_hold.enabled`:
   one GROUP and one SINGLE-CAV hold arm at 300 s per world;
   point index `(world_index + cycle) % branch_points`, stable across retries.
   Reuse existing 300 s B/ALL prefixes; save a separate long credit record with
   full source-plus-road ledger. Single selection is the nearest upstream
   mainline CAV, vehicle-ID tie-break, relative to existing merge-start geometry.
   Only that ID receives acceleration0; others native. This credit probe adds no
   deployed action/reward/behavioral threshold. Missing eligible CAV costs zero
   actual seconds but retains its reservation; incidents enter stop accounting.

## Registered real-share rule

`[real_share].enabled`: fixed candidates0.3/0.5/0.7/0.9/1.0, each20 critic-only
updates from identical critic/Adam state, frozen actor/WM and cached CRN targets.
Reserve TRAIN-development worlds by SHA256(`965140:seed`) modulo5=0; exclude
their labels from regression/native warmup and their starts from fitting.
Select the largest share whose upper95% paired-world-bootstrap real MSE change
versus0.3 is <=0 (2,000 draws, seed965140); otherwise0.3. Require current-policy
SHA pairing; record critic/data/WM fingerprints and adopt the selected critic.
Measured on20 fitting examples/10 worlds and six development examples/worlds
968135/968137/968138: **share1.0 selected**, MSE194.13M→123.19M (−36.5%);
upper95% difference−16.98M. Evidence: `REAL_SHARE_MEASURED.json`. At1.0 the
difference loss has zero gradient mass; targets remain computed for audit.

## Measured reliability and accounting

Weight = `max(0,2*sign_lower95-1) * max(0,corr_lower95)^2`, using existing
group-bootstrap intervals, further multiplied per start by
`mean_delta²/(mean_delta² + variance_delta/n_futures)`. Both target and auxiliary
loss are shrunk. Missing/stale horizons yield zero; no extrapolation or manual
weight. The panel measures t0-HDV differences: a proxy, not full-system validation.
The repaired round-16
panel now supplies 60 s lower bounds 0.920213/0.678008, weight **0.386339**;
panel SHA256 `9ddb32143a16bea2df5dafd157bcbd86a806fc316824717db9f3d6c12e1c55aa`.
Its 120–300 s weights remain **zero** without evidence. S-DELAYDIAG's population
mismatch means the original level ratios cannot calibrate these targets.

13 worlds reserve **7,800 extra sim-s**, primary cycle **36,933** versus29,133.
Combined append-only reservations, including failed attempts and old GROUP
charges, cannot exceed8,000/cycle; after a full primary there is insufficient
reserve to retry a600s world. Total210,000 and all historical charges/seeds remain.

## Measurements / verification

**346 CPU tests pass, two existing skips**: delayed-payoff actor reversal,
uncertainty/level cancellation, CRNs, real-WM/checkpoint integration, share rule,
single-CAV commands and failed-attempt accounting.
Cross-parent identity against d6a426c6 passes **348 exact tensors**, two updates
each on plain/cooperative/real-WM trainers; RNG and metrics also match. Ruff passes.
Main's transfer receipt is accepted. gpuA4090D, allocator0.3, four threads,
48 fixed TRAIN worlds: initial median24.43s OFF/42.20s ON (full audits).
Final SAFE SPEED2 **8.55s OFF/24.45s ON**, peak allocated1.215/1.217GB;
selection418.6s. Timings cover10s actor/60s targets; longer actor stages unmeasured.
Matched300s POLICY/B ledgers were used. A600s-native/300s-policy attempt is INVALID,
preserved; exact owned PID7565 was stopped, no adoption. No new SUMO or live activation.

## Exact proposed registration amendment (NOT installed)

“At the next unstarted cycle boundary, install c4/longcritic and set policy.config
to configs/c4_policy14_longcritic.toml; retain frozen gate-panel/horizon arguments.
Set policy.longcritic=true and initialize state.native_real_returns from the
latest paired B300s export, not the old native600s file; conversion automatically
exports/advances subsequent native targets. Retain current-policy real returns;
add --longcritic-reliability pointing to the accepted-WM-SHA-bound TRAIN panel.
Run the registered development-world share selection each policy phase; no
unmeasured share increase. Enable collect.long_group.enabled and
collect.long_single_hold.enabled. Reserve36,933 primary, total210,000, combined
long-attempt cap8,000/cycle; no automatic full retry reserve. Preserve prior
charges/seeds/failures/gates; record source/config/registration hashes before
dispatch. S-LONGCRITIC does not install this amendment.”
