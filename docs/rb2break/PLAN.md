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

##12:47 incoming-population likelihood investigation (before fitting)
Source admission deficit is essentially invariant to LC/merge advice ablations.
Investigate whether a population fitted to road-resident drivers transports to
incoming drivers. Use ONLY development12 observable pre-t0 histories and observed
birth times/CAV ownership, not arrival assignments/types or future trajectories.
If feasible, fit weights on the SAME frozen joint IDM support using stable-leader
motion likelihood for a birth-defined cohort (no residence-time weighting or
hand-chosen physical parameters). Freeze independent birth/time-prefix validation
inside pre-t0 and report physical acceleration CRPS before outcome inference.
No outcome objective selects mixture weights. Proposed change, if it passes that
motion check, affects entrant HDV population only; present-driver posteriors,
CAV executor, arrival process and ordinal keys remain unchanged. This is an
additional one-factor candidate; not an assumption that it will repair congestion.

##13:45 source-physical acceleration-envelope extension (before v2 fit)
The frozen Gaussian motion-discrepancy likelihood can assign weight to a driver's
accel parameter BELOW an acceleration already observed for that HDV. SUMO's
maxNextSpeed/finalizeSpeed enforces v_next ≤ v + accel*dt, independently of LC
braking, so those profiles are physically inadmissible, not residual noise.
V2 birth-cohort fit adds ONLY this support constraint to each FIT driver's
likelihood, using one-ULP outward bounds on recorded float32 speeds and candidate
float32 accel (representation tolerance, no behavioral threshold). Same support,
same Gaussian scale, same birth-defined FIT/validation cohorts. No support for a
driver => fail closed, never clip an observed acceleration or invent a profile.
Unconditioned later-birth acceleration CRPS must improve before any traffic run.
No throughput/delay/breakdown target enters either fit. Entrant-only ablation
would preserve current-driver posteriors and all arrival/controller streams.

##13:55 current-driver envelope validation (before scoring)
The v2 population gate failed; do NOT infer traffic with either birth-population
fit. Separately validate a current-driver physical support correction, without
refitting ANY population or noise scale: after the existing19.35568 shrinkage,
zero/renormalize profiles that violate that driver's observed positive-acceleration
envelope. This prevents the population-contamination component from resurrecting
physically impossible settings. No evidence leaves the population unchanged;
no compatible positive-weight profile fails closed. Same outward IEEE bounds.
Use only240..270s own motion to condition, then270..300s stable-leader acceleration
CRPS on the same registered surviving HDV cohort. This is an internal conditional
motion check (survivor cohort, not prospective independent-world qualification).
Only a lower mean-world physical acceleration CRPS permits a traffic screen.

##14:15 native-uncommanded CAV parity (before inference)
The existing sumo_physics native callback uses inferred_follow_cap, explicitly
the externally commanded CAV path, not native SUMO finalizeSpeed/LC patching.
New default-OFF native_model=sumo_uncommanded delegates native CAV motion to the
same SUMO CF/merge/LC physical stream as HDVs, while still calling the controller
dispatcher once per tick. The native CAV role, birth ownership and latents remain
unchanged; only motion ownership changes. Require the CAV-setting consistency
switch so physical parameters match the already-deployed fixed execution fit.
Compare against CAV-settings-only parent, not against an unmatched baseline.
All four commanded-controller arms MUST remain exactly unchanged. Unit gate:
with identical physical settings, an uncontrolled CAV and an HDV clone produce
identical motion, including emergency finalization and carried LC speed advice.
No externally commanded plant, controller or black box changes.

## Afternoon source correction: active mode2 actuator vs retired mirror
alwayscmd.py explicitly states that mode2's executed law ignores EXEC_CAP_KEY
and the learned mirror, and calls shared sumo_idm.executed_accel instead. The
v1 CAV binder targeted the retired fit incorrectly. Preserve v1 outputs as old-
vector diagnostics; withdraw the claim of mode2 plant consistency. Corrected
binder reads the ACTUAL registered CAV control-law parameters and mainline
allowed speed from sumo_idm (known actuator design, not hidden-driver truth).
Fresh corrected CAV-parent/native-child dev12/K1 screens, with native-child's
four commanded arms required identical to its corrected parent. No external
execution equation or controller weights change. No mixing v1/v2 receipts.

## Full development K8 confirmation, frozen before corrected K1 score collection
Evaluate the source-backed corrected-mode2/native-uncommanded candidate on ALL
dev12×8futures×5arms/H600, retaining120/300prefixes, regardless of K1 efficacy.
New immutable latest source includes pre-planning CAV consistency and known-
actuator fallback when no CAV donor exists. Two disjoint6world shards, max2BREAK
CUDA workers, each starting only after its owned corrected-K1 predecessor exits.
No K1 data are relabelled as K8; all96tasks generated under one new common model
contract. Score only complete K8 worlds; all12required for final dev readout.
The source-corrected candidate is fixed by actuator/ownership parity, not a
J-offset or post-hoc threshold. Original FULL-v4 development K8 remains comparator.
All CIs requiring replicated cells remain NA; no qualification from dev12.

## Startup-only incoming-motion check (registered before v3 fit)
V1/V2 used up to60s after each birth; later motion can include native LC speed
patching despite stable same-lane leaders, which the isolated CF likelihood
does not model. Test the SAME frozen-support mixture fit/validation cohorts
using only the FIRST10s after birth, emphasizing source startup dynamics.
This is a fitting observation window, not a runtime behavior threshold; no
forward equation, arrival process or parameter value is hand-set by this choice.
All pre-t0, native, dev12; same fixed Gaussian scale. Keep original priors unless
birth-disjoint startup acceleration CRPS improves. Report its restricted scope;
do not claim longer-horizon validation or adopt without autonomous dev evidence.
