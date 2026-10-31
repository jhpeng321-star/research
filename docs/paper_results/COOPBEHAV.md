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

## Exact quantities and counterfactual interpretation

For rear CAV i and its unchanged leader L, bumper gap is
`g_iL = x_L - length_L - x_i`; slot headway is `h_iL = g_iL/v_i` when
`v_i>0`, otherwise missing. There is no artificial minimum speed. A slowing
interval needs `v_i(t)<v_i(t-dt)`, a growing gap and a positive, numerically
resolved constant-speed displacement deficit. The opening episode ends on
gap closure, recovery of that deficit, leader/identity/lane change, acceptance,
or observation censoring. New braking-created opportunities can follow a
closure; counts are episodes, not unique physical slots.

Record maximum opening and its braking displacement contribution, slot
headway change, accepted slot clearance and interval-censored acceptance time.
Paired support additionally requires greater opening, greater rear-CAV
displacement deficit and greater speed loss than the matched reference;
for an accepted merge the comparison is at the acceptance frame. The reference
ramp vehicle must still be observed ramp-side through that frame. Later merge
times and right censoring are retained. These conservative conditions identify
mechanistic support, not intention, fleet benefit or interaction significance.

A wall is a connected group of simultaneous slowing CAVs with overlapping
longitudinal body intervals in different mainline lanes, all upstream of `s=0`.
No fitted upstream-distance, speed or synchrony window is used. Paired metering
support measures delayed `s=0` crossings of matched, initially immediate followers;
it is conditional on followers whose crossings are observed in both branches.
Wave damping follows fixed initial follower identities, requiring the same
leader chain throughout the slowing episode. It compares same-clock temporal
variance, not changing-population cross-sectional variance. Without a reference
it means attenuation relative to the CAV's variance; with a reference it means
reduction of the same follower's variance versus that branch.

Native release is validated as an explicit native controller with no external
issued command. SUMOCF's `policy_mean_reference` / `policy_mean_intervention`
is a separate supported contract: reference issued commands must equal recorded
policy-mean commands. Group interventions support group-level contrasts;
they do not isolate one member's causal contribution. Whole-controller MAPPO
versus CTRL0 is never relabelled as either release experiment. Issued commands
are never treated as realised physical acceleration.

## Completed descriptive MAPPO versus CTRL0 analysis

Final outputs are in `tools/coopbehav/evidence/WMDATA_final/` and
`WMDATA2_final/`; `FINAL_COMPARISON.json` independently reconciles event and
member counts, exposure and output hashes. All 180 arm-episodes cover 300–600 s,
at 0.5 s state resolution. Rates below are **events per CAV-hour**, with wall
events counted once per group; member-event and episode-hour rates are also
exported. Denominators include continuously identified observed CAV exposure.
The two corpora have different cell allocations and are reported separately.

- **WMDATA, 40 worlds/controller:** MAPPO: yield candidates **11 (0.214)**,
  gap creation **1493 (29.104)**, cross-lane holds **174 (3.392)**,
  follower attenuation **3827 (74.602)**. CTRL0: **4 (0.079)**,
  **1008 (19.921)**, **169 (3.340)**, **2124 (41.977)**, respectively.
  Exposure: MAPPO184676.5, CTRL0182159 CAV-s. Only112/1493 MAPPO and118/1008
  CTRL0 unaccepted opening episodes contained a geometrically fitting ramp car.
- **WMDATA2, 50 worlds/controller:** MAPPO: yield candidates **10 (0.108)**,
  gap creation **2965 (32.150)**, cross-lane holds **390 (4.229)**,
  follower attenuation **6665 (72.270)**. CTRL0: **1 (0.0109)**,
  **2259 (24.614)**, **501 (5.459)**, **4045 (44.073)**, respectively.
  Exposure: MAPPO332004, CTRL0330404 CAV-s. Geometrically fitting ramp-car
  opportunities occurred in180/2965 versus131/2259 unaccepted openings.

These are observed patterns, including small physical changes and naturally
occurring slowing. No effect-size or duration cutoff was fitted. Censoring,
opening magnitudes and headway changes remain in the event records and
comparison distributions. A gap with no fitting ramp vehicle is generic gap
creation, not evidence of merge assistance. An unaccepted, censored opportunity
does not establish that a ramp vehicle refused it. All paired-support counts in
these two corpora are zero because no member-release reference was supplied.
Their behaviour rates therefore cannot establish which controller cooperates
more. Example PNGs illustrate extrema, not representative or causal examples.

## Available SUMOCF probe discovery

The completed **81 eligible group/reference pairs** are preserved in
`tools/coopbehav/evidence/live/paired_a0b825aa936da6d27d01e07fc6b1bbd210150272eb95decf5389a0596f6895da/`.
These reuse **four distinct TRAIN worlds** across probe designs, not81
independent worlds. Credit is reconstructed from the saved empty/group/solo
branch costs and checked against supplied `C_G`; only closed results are read.
The reference is the same-checkpoint policy mean. Traces are 1 Hz observed
states through intervention end+30 s; the600 s credit extends beyond that
behaviour observation window. Missing/undersized groups remain missing.

Across the observed fleet there are0 accepted-yield candidates,1293 opening
episodes,123 cross-lane holds and3477 attenuation episodes. Reference-supported
subsets are0,41,0,702, respectively, and apply only to intervened members.
These are repeated-design mechanism counts, not independent successful groups.
`FINAL_COMPARISON.json` also separates registered-group event counts from
whole-observed-fleet counts.

Discovery ranks all36 strictly positive-credit records, from only **two distinct
worlds**. Tiny positives are retained. Features use each intervention's own
window: member issued and realised acceleration, speed/change/variance, fixed-
leader gap change, lane changes, upstream fraction, nearest ramp distance and
initial immediate-neighbour speed signatures. Missing features are mean-imputed
after finite-observation mean/population-SD standardization; numerically constant
columns are removed. Average-linkage Euclidean clustering is deterministic.
Choose k in2..floor(sqrt(N)) by maximum silhouette, ties toward smaller k;
nonpositive numerical silhouettes or insufficient data use one cluster.
No gain value enters the feature distance. Here the rule selects **six clusters**:

1. **Short-window release from slow traffic:** mean member speed4.20 m/s,
   speed gain6.35 m/s, realised acceleration+0.635 m/s², same-leader gap
   increase29.78 m; neighbours gain3.14 m/s. Overlaps follower attenuation.
2. **Longer acceleration with neighbours recovering:** mean8.37 m/s, member
   gain12.64 m/s, neighbours+9.51 m/s; same-leader gap changes−0.18 m.
   Overlaps gap creation and attenuation episodes.
3. **Extended recovery with lane changes and departures:** mean5.77 m/s,
   member gain12.32 m/s and neighbour gain10.42 m/s; member visibility74.6%.
   Overlaps gap creation and attenuation.
4. **Nearly steady mainline slowing:** mean13.23 m/s, member change−0.69 m/s,
   neighbours−0.83 m/s, same-leader gaps−3.81 m. Overlaps attenuation.
5. **Near-steady motion with lane changes:** mean11.46 m/s, member and neighbour
   gains about0.55 m/s, gap increase2.56 m, member visibility88.7%.
   Overlaps attenuation.
6. **Later acceleration, mostly downstream:** mean13.41 m/s, member
   gain13.71 m/s and neighbours+12.88 m/s; only2.4% of member frames upstream,
   member visibility68.9%. Overlaps attenuation.

Each cluster has six repeated-design records. Clusters1–3 share the same world
and C_G=1555.7834 veh-s; clusters4–6 share another and C_G=0.0017309 veh-s.
The clusters align strongly with world and window duration/visibility, and
are not six independent discovered mechanisms. **No wholly unexplained cluster
was found under the declared rule.** Detector overlap does not prove it caused
positive C_G; the accelerating recovery signatures are descriptive hypotheses.

## Runtime, checks and continuing handoff

- CPU suite: **54 synthetic tests** covering all four detectors, identity/slot
  recycling, native/policy-mean contracts, credit reconstruction, discovery,
  source precision/hashes, complete reports and advancing-index publication.
- Final descriptive runs use the physical closure implementation committed as
  `05a2b210`; executed source and input hashes are embedded in every report.
  Earlier pilot/v2 outputs are preserved as QA history and are not the final
  counts above. QA caught recovered braking deficits being attributed to later
  leader-created gaps; the final closure rule and regression test fix this.
- cpuQ lacked Matplotlib. Plotting dependencies were installed only under our
  `/home/ubuntu/coopbehav20261031_analysis/deps` and used only by our reports.
  Existing interpreters were retained; no environment or simulator was created.
- Polling is active every120 s through the own Windows watcher; receipt in
  `tools/coopbehav/evidence/WATCHER.json`, live state/logs in
  `runs/coopbehav_poll/`. It watches both design documents, explicit ready indices
  and the published gpuR roots. Immutable index snapshots, one own paired report
  at a time, verified local publication and task-output-only step commits are
  implemented. All completed snapshots are inventoried even if the live index
  advances. Source snapshots overlap and must never be pooled as independent data.
- COOPMEAS2 has costs/hashes but no replayable frames. The current COOPAUDIT
  crop lacks observable lengths, mainline neighbours and non-wall executed
  command status; its geometric analysis is explicitly blocked. A complete
  normalized ready-index handoff can be consumed automatically. The producer
  contract and commands are in `tools/coopbehav/README.md`.
- This task incurred **zero SUMO sim-s**. No peer process was signalled, and
  restricted source paths, truth/assignment files and prohibited seeds were not read.
