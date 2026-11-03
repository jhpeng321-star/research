# S-COOPBEHAV — observable cooperation behaviours

Authority: owner approval 2026-10-31 15:45 in the active C4 registration;
analysis-only worktree `c4/coopbehav`, starting at `bf8c1e20`. No simulator is
imported or run. Observational patterns do not establish positive interaction
credit: `C_G = D_G - sum(D_i)` remains a separate, paired-branch measurement.

**Recovery correction (2026-10-31):** producer `ARTIFACT_CHECK.md` and
`tools/sumocf/TRACE_HANDOFF.md` invalidate the original saved-state probe's
credits and discovery interpretation. Current evidence is explicitly selected in
`tools/coopbehav/evidence/SELECTION.json` and independently reconciled in
`CURRENT_COMPARISON.json`. WMDATA/WMDATA2 descriptions are unaffected.

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
`WMDATA2_final/`; `CURRENT_COMPARISON.json` independently reconciles event and
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

## Invalidated original SUMOCF probe (provenance only)

The original81-pair report beginning `a0b825aa936d` and its six discovery clusters
are retained on disk and in git, but are **withdrawn as cooperation evidence**.
The producer's zero-noise checks found identical issued actions but different
first physical steps after state restoration at later branch times. The resulting
false credits reproduce the dominant original values, including positive
`C_G=1555.7834 veh-s`. Arithmetic reconciliation cannot validate a defective
branch engine. `FINAL_COMPARISON.json` and broad historical `source_sumocf`
reports must not be used to infer positive interaction or novel cooperation.
Corrected samples use full same-seed prefixes for every branch, with a passing
zero-offset bitwise trajectory/cost gate. See the producer's artifact receipts.

## Corrected SUMOCF behaviour evidence

### Deployed-policy DEV (u03)

Selected report: `tools/coopbehav/evidence/live/paired_bc782dd4b08fb81b8c2775732fb97b4148acb92707716600f10f20ccc934452d/`.
The producer's immutable `COOPBEHAV_dev_u03_fullfleet_INDEX.json` compares
deployed learned means with MAPPO means during the60s intervention, followed by
learned continuation. Of three prospective worlds, two have an eligible group;
one is missing, not zero-filled. Two90s observable-fleet crops cover4498 CAV-s;
the credit horizon is600s, substantially beyond the behaviour crop.

The observed fleet has **0yield candidates,32openings,4cross-lane holds and
60attenuation episodes** (rates0/25.611/3.201/48.021 per CAV-hour). Only two
unaccepted openings contain a geometrically fitting ramp vehicle. **All four
paired-supported counts are zero** for intervened groups. No positive-credit
record enters discovery. These data do not establish a learned cooperation
mechanism. This is the latest deployed full-fleet ready index found at recovery;
TRAIN update numbers below do not imply later DEV evidence is available.

### Sampled TRAIN exploration (u15 snapshot)

Selected report: `tools/coopbehav/evidence/live/paired_1c94ebe359532d3d00e6fa2d72af557791e71709691a78b26196cd4973f76c2e/`.
This is the producer's per-update `COOPBEHAV_u15_INDEX.json`: **31eligible
windows from six TRAIN worlds**, of36attempted windows (five missing groups).
Each uses a60s sampled-offset intervention versus the same checkpoint's current
policy mean, with full600s credit and a90s behaviour crop. The report contains
71120 CAV-s of observed-fleet exposure. It is exploratory and does not measure
the deployed mean's cooperation relative to MAPPO.

Whole-fleet counts are **4yield candidates,569openings,61cross-lane holds and
1507attenuation episodes** (rates0.202/28.802/3.088/76.282 per CAV-hour).
Reference-supported intervened-group subsets are **0/1/0/5**, respectively;
registered-member overlap counts are0/33/0/5. Only32unaccepted openings contain
a fitting ramp vehicle. Whole-fleet patterns and group-intervention support are
distinct quantities; neither is a count of independently beneficial groups.

### Corrected discovery and unanticipated signatures

The unchanged declared algorithm ranks strictly positive `C_G` and uses only
the intervention window for signatures: member issued/realised acceleration,
speed/change/variance, fixed-leader gap change, lane changes, upstream fraction,
nearest ramp distance and initial immediate-neighbour speed signatures. Finite
feature means/population SDs standardize the data; missing values are mean-imputed
and numerically constant columns removed. Deterministic average-linkage Euclidean
clustering chooses k in2..floor(sqrt(N)) by maximum positive silhouette (ties
toward smaller k; insufficient/nonseparable data use one cluster). No gain enters
feature distances, and no minimum positive-gain cutoff is fitted.

For u15, **11positive records from four distinct worlds** produce two clusters
(silhouette0.310 versus0.209 for k=3):

1. **Downstream acceleration with neighbouring traffic recovering** (10records):
   member mean speed10.83m/s, speed gain17.60m/s, realised acceleration+0.420m/s²;
   neighbours gain17.00m/s. Members are upstream in only0.83% of observed frames,
   with74.1% mean member visibility. This cluster overlaps gap-creation and
   follower-attenuation detections. Same-leader gap change is observed for only
   one ranked record and excluded from clustering; its imputed centroid must not
   be interpreted as a common physical opening.
2. **Near-stationary downstream queue creeping with a lane change** (one record,
   `fixed_u15_w4_k2`): member mean speed0.184m/s, displacement10.68m, realised
   acceleration+0.0402m/s² and speed gain2.41m/s; neighbours average0.106m/s and
   gain1.81m/s. The mean member lane-change count is0.5 and visibility100%.
   **No detector (1)–(4) overlaps this record**, so it is explicitly reported as
   an unanticipated-signature candidate. Its `C_G` is only0.0000296veh-s and
   `D_G=-0.000000954veh-s`; this singleton does not establish a novel beneficial
   cooperation mechanism.

The largest positive record, `fixed_u15_w1_k3`, has `C_G=2482.05veh-s` but
`D_G=-3413.52veh-s`: positive interaction does not imply positive group benefit.
The second-largest has `C_G=74.78veh-s` and `D_G=74.78veh-s`. These post-selected
training observations require independent deployed-policy replication. Cluster
overlap is descriptive co-occurrence, not proof that a detector caused the gain.

## Corrected S-COOPMEASURE v2 coverage (2026-11-02 recovery)

The producer's corrected full-prefix audit is complete. Its observable export is
local at `E:/agent-work/c4-coopmeasure/tools/coopaudit/evidence/v2_complete/`;
the contract is `tools/coopaudit/TRACE_HANDOFF.md` in that worktree. Recovery
verified **all 247 gzip SHA256s** against `COLLECTION.json`, and reconciled all
**231 production files** against `v2_analysis/TRACE_MANIFEST.csv` (SHA256, rows,
minimum/maximum clock). These cover **21 eligible production worlds** of the
28 prospective worlds, with seven missing groups; the other 16 files are the
separate mechanics canary. Receipt: `tools/coopbehav/evidence/COOPAUDIT_V2_COVERAGE.json`.

Across 528,644 vehicle-frame rows, **none contains vehicle length, per-row CAV
status or controller status**. All rows are frozen members or ramp-observed
vehicles; no additional mainline leader/follower rows are exported. In contrast
to historical v1, non-null issued-safe requests **are** present for MAPPO and
SUMOCF (production 6,609/6,611 rows respectively), as well as wall (32 rows).
These are vehicle-frame observations of requests, not independent control ticks
and not realised acceleration. The inherited scanner's obsolete v1/non-wall-
command blocker has been corrected for full-prefix v2.

Consequently yield-slot geometry, unaccepted gap openings, overlapping-body walls,
fixed follower chains and neighbour-complete discovery signatures remain
**unidentifiable from this crop**. The coverage receipt uses null detector counts,
never fabricated zero events or default lengths. Producer cost/credit evidence is
separate from geometric identifiability. A complete normalized ready index with
observable lengths, mainline neighbours and explicit release status can be consumed
by the existing watcher. No new simulator run is part of this analysis task.

The canonical SUMOCF index still selects the already-completed u15 TRAIN report.
A read-only inspection of published `COOPBEHAV_dev_u*_INDEX.json` files found
only the existing u03 and scope-corrected u03 full-fleet indices; no newer ready
deployed-policy behaviour export was present. Existing reports were retained
without regeneration, and no additional report worker was dispatched.

## Runtime, checks and continuing handoff

### Fresh-session reconciliation (2026-11-03)

Recovery from clean HEAD `eafce9db` preserved all completed work. The inherited
watcher remains live as PID29244 with the original creation identity, polling
every120s; its stderr is empty and no owned report is active in its status.
A read-only gpuR inventory again finds the same canonical31-pair u15 TRAIN index
and only the two already-reported u03 DEV indices, with identical SHA256s.
The corrected audit handoff still lacks the geometric observables listed above;
its live index is not ready. No completed report or coverage scan was rerun and
no watcher/report was dispatched. The existing59-test CPU result remains the last
implementation verification; this recovery changes only documentation/receipts.
Recovery receipt: `tools/coopbehav/evidence/RESUME_20261103.json`.

The 21:53 local fresh-session check inherited clean HEAD `8851a413` and again
verified PID29244's original parent, creation time, executable and polling command.
Its status has advanced since the prior recovery, stderr remains empty, and no
report is active. Read-only producer-index inspection found the same31-pair u15
index and the same two u03 DEV indices, with unchanged hashes; the audit index
remains not ready. Both producer trace handoffs retain the same evidence limits.
All completed analyses and the existing120s watcher were preserved. Receipt:
`tools/coopbehav/evidence/RESUME_20261103_2153.json`.

The 22:36 local continuation inherited clean HEAD `ba76743d`. PID29244 again
matches its recorded creation identity and command; polling status has advanced,
stderr is empty and no report is active. Direct read-only gpuR inspection confirms
the same canonical31-pair u15 TRAIN index, the same two u03 DEV indices and the
not-ready audit index, all with unchanged hashes. Producer trace contracts retain
the documented coverage limits. Receipt:
`tools/coopbehav/evidence/RESUME_20261103_2236.json`.

### Fresh-session reconciliation (2026-11-04)

Recovery inherited clean HEAD `8a95a631`. PID29244 still matches its recorded
parent, creation time, executable and polling command; its status has advanced
since the preceding recovery, stderr is empty and no report is active. Direct
read-only gpuR inventory confirms the same31-pair u15 TRAIN index and the same
two u03 DEV indices, all with unchanged hashes. The audit index remains not ready;
both producer handoffs retain the documented coverage limits. Completed reports,
the59-test verification and the existing120s watcher were preserved. No new
report or watcher was dispatched. Receipt:
`tools/coopbehav/evidence/RESUME_20261104.json`.

### Existing verification and continuing polling

- CPU suite: **59 synthetic tests** covering all four detectors, identity/slot
  recycling, native/policy-mean contracts, credit reconstruction, discovery,
  source precision/hashes, complete reports and advancing-index publication,
  plus corrected audit coverage, corrupted-file rejection, manifest rosters and
  restricted-seed preflight. The final full CPU suite passed in 2.70 s during the November 2 recovery;
  exact command/results in `evidence/RESUME_VERIFIED_20261102.json`. Prior 54-test
  verification remains in `evidence/RESUME_VERIFIED_20261031.json`.
- Final descriptive runs use the physical closure implementation committed as
  `05a2b210`; executed source and input hashes are embedded in every report.
  Earlier pilot/v2 outputs are preserved as QA history and are not the final
  counts above. QA caught recovered braking deficits being attributed to later
  leader-created gaps; the final closure rule and regression test fix this.
- cpuQ lacked Matplotlib. Plotting dependencies were installed only under our
  `/home/ubuntu/coopbehav20261031_analysis/deps` and used only by our reports.
  Existing interpreters were retained; no environment or simulator was created.
- Recovery independently reverified all four selected reports (180descriptive
  arm-episodes,2deployed DEV crops and31sampled TRAIN windows), including event/
  member counts, exposure, summary digests and all derived-output hashes. No
  completed report or simulation was rerun.
- Polling is active every120 s through the inherited own Windows watcher; receipt in
  `tools/coopbehav/evidence/WATCHER.json`, live state/logs in
  `runs/coopbehav_poll/`. It watches both design documents, explicit ready indices
  and the producer-selected gpuR ready index. The broad SUMOCF historical scanner
  is disabled in the live config to prevent mixing invalid old probes with
  corrected cohorts; COOPAUDIT coverage checking continues. Immutable index
  snapshots, one own paired report
  at a time, verified local publication and task-output-only step commits are
  implemented. All completed snapshots are inventoried even if the live index
  advances. Historical cumulative snapshots overlap; canonical corrected update
  cohorts remain distinct from deployed DEV and are never pooled indiscriminately.
- COOPMEAS2 has costs/hashes but no replayable frames. The corrected COOPAUDIT
  crop has issued-safe requests but lacks observable lengths, mainline neighbours
  and normalized per-row CAV/controller flags; its geometric analysis is blocked.
  A complete normalized ready-index handoff can be consumed automatically. The producer
  contract and commands are in `tools/coopbehav/README.md`.
- This task incurred **zero SUMO sim-s**. No peer process was signalled, and
  restricted source paths, truth/assignment files and prohibited seeds were not read.
