# First full-campaign development checkpoint: optimizer 71, episode 20

The 48,000-step checkpoint is **ineligible under the registered development
criteria**. All eight cases and all three arms completed. This is the first
of nine prespecified checkpoint assessments, not the end of training or a
prospective acceptance result. No damping checkpoint is accepted.

## Evidence and verification

The completed comparison is
`runs/rllib/native_damping_campaign_v1/development/optimizer_71_episode_020/summary.json`.
Checkpoint SHA-256 is
`9aef1dae1c2fc0b627476d1287693f86cbebb3568fddee6f01098df9d3f48b60`.
The separate audit is
`runs/probe/native_damping_o71_e020_completed_audit_2026-09-05.json`.

The read-only audit verified the campaign and checkpoint identities, current
and snapshotted source hashes, reconstructed development request/scenario
identities, exact case labels and populations, and all 24 full 2,400-interval
records. It reconciled interval cost sums with recorded totals, recomputed the
registered assessment from raw summaries, and reproduced all **291,784** saved
deterministic CAV actions exactly using the saved checkpoint and original
per-tick batch shapes. This audit ran no SUMO and opened no acceptance data.
It verifies saved-data and inference consistency; it is not an independent
validation of SUMO's physical realism or a proof of controller safety.

## What the comparison establishes

Improvements below are relative to each case's native IDM outcome. The
composite is the equal mean of fractional severe-braking and squared-
acceleration-effort improvements. Effort is not a fuel-consumption measure.
System time includes all requested trips and source waiting. Positive time
differences are worse.

- **2000, seed 911001:** composite improved 5.17%; system time improved
  0.78 s/request. No case-level constraint failed.
- **2000, seed 911002:** composite improved 8.19%; system time improved
  0.15 s/request. TTC exposure rose 7.5 vehicle-seconds, within this case's
  chosen 17.25 vehicle-second margin. No case-level constraint failed.
- **2200, seed 911003:** composite worsened 24.79%; system time increased
  22.40 s/request; TTC exposure increased 517.5 vehicle-seconds against a
  57.5 margin. System-time, TTC and harmful-tail constraints failed.
- **2200, seed 911004:** composite worsened 8.49%; system time increased
  2.90 s/request; TTC exposure increased 144.5 vehicle-seconds against a
  28.725 margin. System-time and TTC constraints failed.
- **2600, seed 911005:** composite worsened 2.90%; system time increased
  8.75 s/request. The system-time constraint failed.
- **2600, seed 911006:** composite improved 7.10%; system time improved
  19.69 s/request, but source waiting increased 5.15 s/request against the
  2 s/request limit. This fails despite its aggregate improvement.
- **2700, seed 911007:** composite worsened 5.76%; system time increased
  15.55 s/request. The system-time constraint failed.
- **2700, seed 911008:** composite worsened 0.70%; system time increased
  11.78 s/request. The system-time constraint failed.

Across the four lower-loading cases, mean braking worsened **4.43%**, mean
effort worsened **5.53%**, and the composite worsened **4.98%**, versus the
required 5% improvement. Mean system time increased **6.09 s/request**.
Across the four overload cases, mean braking improved 0.95%, effort worsened
2.08% (slightly exceeding its 2% bound), and mean system time increased
4.10 s/request. These are arithmetic means of paired case differences;
they are not pooled-vehicle estimates or confidence bounds.

Every arm completed its requested population, with no added collision or
teleport events relative to native. That does not erase the measured delay,
source-wait and TTC failures. Only two learned cases passed every individual
case constraint. The result does not establish a general 2000 veh/h operating
boundary: there are only two exposed development cases at that loading.

The analytical velocity-matching comparator is also ineligible when described
under the same physical criteria. Its lower-group composite improved 1.48%,
below the 5% target; its overload composite improved 2.86%. It had TTC,
system-time, source-wait and harmful-tail failures across individual cases.
Its better group means are not evidence of an accepted fallback, a calibrated
optimal controller, or proof that learning is unnecessary.

## Action-interface finding and its limits

A subsequent two-case diagnosis is complete; see
`NATIVE_DAMPING_MAGNITUDE_ABLATION_RESULTS_2026-09-05.md`. It supports both
substantial effects under near-zero engagement and consequential residual
magnitude, without isolating one exclusive cause. The checkpoint remains failed.

The learned deterministic residuals ranged from **-0.00855 to +0.00936 m/s²**,
with a sample-weighted mean of -0.00158 m/s². None of the raw actor outputs
fell within the native-zero tolerance. Nevertheless, 236,798 of 291,784
decisions (**81.16%**) issued `slowDown` commands. Safety projection and
fallbacks can still return control to native IDM.

This confirms that tiny neural residuals are not equivalent to tiny changes
from SUMO native behavior. Exact zero hands back to native IDM, while nonzero
output engages the separately computed nominal IDM and command/projection
path. That known interface distinction is active in this checkpoint. These
records do not identify the causal share of its benefit or harm attributable
to the nominal model, command discretization, learned feedback, or safety
projection. Measured command tracking is not a counterfactual action effect.

There is therefore no basis to conclude that CAV-local observations are
insufficient. The evidence establishes an unsuccessful early checkpoint,
small deterministic learned residuals, substantial command exposure, and
loading-dependent outcomes. It does not establish learning convergence,
feature necessity, or a failure of every local learned controller. It also
does not establish that continuing training will fix these problems.

## Execution decision

The original stage exited successfully after writing the complete assessment.
The existing coordinator advanced to optimizer 71 through episode 62
(148,800 optimized steps). At verification, coordinator PID was 32444,
trainer PID 30172, and the trainer had completed episode 21 / 50,400 steps.
Use `continuation/state.json` for later PIDs; these numbers are a snapshot.
Tool session 71265 is terminal. Observe the coordinator through session 41027.

Continue the existing registered nine-checkpoint campaign without changing
features, reward, action limits, controller code, selection criteria or seeds.
Retain this failed checkpoint and its diagnostics for the learning curve.
Later assessments should compare both physical outcomes and action behavior
using their already saved records. Do not reinterpret near-zero outputs as
native passthrough or improve the score by removing harmful development cases.

If no checkpoint is eligible at the registered cap, the near-zero/nominal-path
distinction is a concrete issue for a versioned successor design and causal
diagnosis, rather than grounds for an unchanged rerun or an observation-
impossibility claim. Any successor must establish usable control and consistent
learning/evaluation semantics before another campaign. No extra simulation is
launched by this report. Prospective acceptance, wall generation and RSU
predictor generation remain closed until their prerequisites are met.
