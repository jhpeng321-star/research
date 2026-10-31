# S-COOPBEHAV — observable cooperation behaviours

Authority: owner approval 2026-10-31 15:45 in the active C4 registration;
analysis-only worktree `c4/coopbehav`, starting at `bf8c1e20`. No simulator is
imported or run. Observational patterns do not establish positive interaction
credit: `C_G = D_G - sum(D_i)` remains a separate, paired-branch measurement.

## Prospective analysis plan

Read only observable S-E3DATA arrays: time, vehicle identity, presence, CAV
flag, position, lane, speed, length, controller and issued acceleration.
`veh_kin` columns are position 0, lane one-hot 1:4, speed 4, acceleration 5,
length 6. Lane 2 is the ramp, lane 1 its receiving lane. Merge geometry is
`s=0..75.44 m` (inherited `tools/wm_eval_common.py`), not a fitted cutoff.
Analyse each episode's recorded branch window, excluding its history prefix.

All signs are zero crossings, with floating-point roundoff tolerance derived
from the input dtype; no tuned speed, braking, gap, duration or synchrony
thresholds. Episodes are maximal contiguous runs on the recorded time grid.
Gaps use bumper geometry and identity continuity, not slot continuity.
Every slowing interval also requires a resolved displacement deficit against
the physical constant-speed extrapolation `x(t-dt)+v(t-dt)*dt`. This excludes
near-stationary speed-sign jitter that has no resolved braking displacement;
the only tolerance is input-dtype epsilon times the compared coordinate scale.
An open slot stops being a braking-created opportunity when the CAV has
recovered that displacement deficit, even if its leader has continued pulling
away. This prevents attributing leader-created space to an earlier CAV brake.

1. **Yield-to-merge:** a receiving-lane CAV decelerates, remains behind the
   same leader, and increases their bumper gap; a ramp vehicle subsequently
   inserts into that exact slot while it is open. Measure opening in metres,
   headway change in seconds, and interval-censored merge time. With a valid
   paired member-release branch, require additional opening and earlier
   acceptance (or no acceptance before matched observation ends).
2. **Gap creation without accepted merge:** the same opening event has no
   observed insertion before that slot closes, changes identity or is censored.
   Report whether any ramp vehicle geometrically fits; censored events are
   labelled explicitly and never interpreted as a refusal to merge.
3. **Cross-lane hold / wall:** simultaneous upstream deceleration of CAVs in
   distinct mainline lanes whose longitudinal body intervals overlap.
   Record member identities and simultaneous exposure, not merely a group
   mean speed. Metering is supported only by measured paired crossing-time
   delay of observed upstream followers, and is otherwise a candidate pattern.
4. **Follower wave damping:** follow the fixed chain of observed same-lane
   followers during each CAV slowing episode. Measure each follower's temporal
   speed variance on the same clocks and compare with the paired branch.
   Without that branch, report only downstream attenuation relative to the
   CAV's variance. Neither version implies beneficial fleet interaction.
5. **Discovery:** rank complete, positive `C_G` records, extract member and
   immediate-neighbour state/action signatures during their registered window,
   standardize features and deterministically cluster them. Cluster count
   is selected by an explicit documented rule; label physical centroids and
   identify clusters containing no events (1)-(4). Outcome ranking is
   exploratory, never confirmatory evidence or an exclusion rule.

## Sources and evidence status at design

- WMDATA: cpuQ `/home/ubuntu/wmdata20261031/views`.
- WMDATA2: gpuR `/root/wmdata2/views`; manifest inspected, 50 TRAIN worlds,
  six arms, MAPPO/CTRL0 share saved initial-state hashes. These are different
  whole-controller trajectories, **not** member-release counterfactuals.
- S-SUMOCF: `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`
  currently contains the original design, without published trace paths.
- S-COOPMEASURE: `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md` not yet present.
- S-COOPMEAS2: `COOPMEAS2.md` documents the 28-world, lane-on MAPPO/ours
  audit. Its interaction uncertainty is substantial; do not conflate that
  controller with WMDATA's speed-only MAPPO.

The runnable analysis will retain source hashes, missing-field coverage,
per-controller counts and exposure rates, member/group events and example
time-space PNGs. Sources are polled through explicit permitted paths only;
restricted names, seeds and truth/assignment files fail closed before reads.
