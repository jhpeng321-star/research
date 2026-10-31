# LC2013 precision iterations — resumed 2026-11-01 00:10

## Preserved starting point (127a106f physics, 6fc46b16 handoff)

Clean worktree on resume. PANEL2_COMPLETE.json records exited owned worker
214459; live identity-bound status confirmed exited/complete at 00:14.
160 FIT episodes, 3255 recorded events, 44333 predicted positive steps,
1136 exact matches: precision 2.5624%, recall 34.9002%. Return-right alone:
30116 proposals / 207 matches (0.6873% precision); overtake 10845/587;
merge 3372/342. Original receipts are immutable. No completed MOBIL scoring
will be repeated.

## Source audit hypotheses before the new measurements

1. `keep_request` is evaluated outside the source's `thisLaneVSafe - 5/3.6
   <= neighLaneVSafe` branch. A previously accumulated keep-right wish can
   therefore trigger even while the destination is much slower. Source
   MSLCM_LC2013.cpp:1710–1791 gates BOTH accumulation and the request.
2. Default through-lane route distances are 1e6 m. Actual frozen WM geometry
   ends at s=553.33 (batched.py:38–46). **Further source inspection:** SUMO uses
   `neighDist` (best-lane continuation LENGTH, not remaining distance) for
   keep-right time, anticipated speed and speed-gain 20s checks; it subtracts
   `posOnLane` only for strategic geometry. A naive replacement of all 1e6s
   by 553.33-s would therefore be wrong. Coordinate separation needs another
   source-backed iteration; it is not part of the first request-gate fix.
3. Source thresholds are already correctly scaled: 0.2/lcSpeedGain left and
   0.2/(lcSpeedGainRight*lcSpeedGain) right, fixed-point truncation each update.
   No outcome-based threshold adjustment is authorized or planned.

## Measurement addition

The preserved scores do not contain predicted motivation. A new diagnostic
pass is necessary to attribute the old candidate's false positives. Added
exact precision and recall contribution for strategic/cooperative/speed-gain/
keep-right and compressed ±2s runs. Recorded motion has no motivation labels:
recall per predicted reason is a contribution to all-event recall, NOT true
class-conditional recall. Baseline MOBIL is reused with original SHA gates.
One nice19/single-thread CPU only; load admission <20. At 00:14 cpuQ load21.4
blocked dispatch; no GPU present on cpuQ, no new job launched at that check.

## Candidate 4: source keep-right request gate

Only the definite request-branch discrepancy (item 1) is corrected. Mature
keep-right wishes persist during a slower-target interval, but do not fire
there. Synthetic regression isolates this from secure-gap safety and checks
that the wish resumes when the target lane clears. Helper `lcspec/DONE.md`
has appeared and was consumed; its §5 independently confirms this nesting.
Its ballistic/1s examples are assumptions, not our Euler/0.5s configuration;
probability updates follow source integer truncation, not helper prose rounding.

Baseline attribution (panel3, same physics and identical aggregate as panel2):
- strategic: 318/3013 exact, precision10.55%, recall contribution9.77pp;
- cooperative: 76/838, precision9.07%, contribution2.33pp;
- speed-gain: 539/12558, precision4.29%, contribution16.56pp;
- keep-right: 203/27924, precision0.727%, contribution6.24pp.

Keep-right generates 27721/43197 = 64.17% of all false positive steps.
Compressed keep-right ±2s runs are only73/1612 (precision4.53%), so this is
not explained solely by repeated positive steps under teacher forcing.
Panel3 source/aggregate/input/MOBIL/motivation-partition verification PASS;
worker299672 exited after136.11s. Gate-fix CPU suite23/23 PASS.

Candidate4 completed/verified (worker319957 exited): 44078 proposals,1136
matches, precision2.5772%, recall34.9002%. The gate removes255 false positives
and no true positives. Per predicted motivation (precision / recall contribution):
strategic10.554%/9.770pp (318/3013), cooperative9.069%/2.335pp (76/838),
speed-gain4.292%/16.559pp (539/12558), keep-right0.734%/6.237pp (203/27669).
Compressed-run precision16.663%, recall26.452%; this small first fix is
insufficient. Next investigate geometry/continuation length and lane-change
source evaluation gates rather than tuning the already-correct thresholds.

## Geometry audit before candidate5

Consumed completed obsstate helper's `netframe.py` geometry contract and
independently read ONLY the corridor net (SHAf58c4ef5…) into
`receipts/GEOMETRY_AUDIT.json`. The lane representation collapses THREE
different SUMO edges into each through lane. Best-lane LENGTH is therefore
3197.55m upstream,553.33m on the merge edge,477.89m downstream, not1e6m.
Each route length is the sum of edge lengths; SUMO uses the first lane's
length as edge length (MSVehicle.cpp:6163,6295; MSEdge.cpp:126). It excludes
internal links, as does the frozen model frame. Strategic distances must
subtract the **ego's current edge-local position**, unlike the accumulator
and anticipated-speed code, which consume continuation LENGTH directly.
Upstream left position is s+2645.67, right s+2644.22; merge position is s;
downstream position s-75.44. Ramp approach has no legal adjacent lane.

Read-only extra public source `MSEdge.cpp` from SUMO v1_27_1 downloaded into
NEW EMPTY `E:/Temp/opencode/sumo-1.27.1-lane-edge`, SHA256
f9677034237961789b9b9ebd9c837b0c1ce1704caaa58918b121d30a84eb7e58.
Never executed. No scenario routes/types/assignment files read. No behaviour
threshold inferred or tuned. Also identified source's both-best-lanes-zero
offset resolution (LC2013.cpp:1144–1155); resolving toward the neighbor must
not create a route-required urgent/early strategic request (1330/1874).

Candidate5 implements these finite raw lengths and the separate edge-local
position, including both-best-zero offset resolution. Optional remaining-
distance inputs retain their documented meaning; new `current_lane_position`
and `target_best_lane_offset` support explicit alternative route geometries.
No generic 553.33-s substitution is made in wish updates. CPU27/27 PASS,
including independent helper-vector3 numeric counter increments828571 at1s
and first keep-right wish atstep25, source5s limit for long continuations,
exact corridor segment positions/lengths, and neutral lanes near route end.
Candidate source hashes now include the geometry module and physics dependencies.

Candidate5 completed/verified (worker359119 exited,150.22s):1118/39090
exact, precision2.8601%, recall34.3472% (vs34.9002%). Predicted-reason
precision/recall contribution: strategic10.554%/9.770pp (318/3013),
cooperative9.069%/2.335pp (76/838), speed-gain4.419%/16.559pp (539/12196),
keep-right0.803%/5.684pp (185/23043). Geometry removes4988 proposal steps,
18 exact matches; retained because the source semantics are now correct,
not because of outcome selection. Still far too many false keep-right wishes.

## Candidate6: actual SUMO numerical precision

INT's handoff identified SUMO NUMERICAL_EPS=.001, not this subset's1e-6.
Independently downloaded/read config.h.cmake:236 to confirm (source SHA/path
in LC2013_SOURCE_NOTES.md). Use .001 in safe-stop margin, IDM gap clamping
(persist the clamped gap before internal substep subtraction), and the left
speed-gain positive-gain veto at1854. This source constant prevents an old
mature left wish from firing on vanishing anticipated advantages. A synthetic
regression has a safe target, mature wish and positive gain~.00033; SUMO's
veto rejects it. CPU28/28 PASS. Candidate5 is collected and verified, so the
next diagnostic can run without overlapping an owned worker; no threshold
fitted to data.

Candidate6 diagnostic extension (same fixed cohort): predicted-reason TP/FP
profiles by physical upstream/merge/downstream edge, mean observed and
anticipated speeds; recorded-event failures split into no-wish/front-block/
back-block/other safety. Proximity reports whether a proposal's vehicle and
directed transition has ANY recorded event in the **bounded60s window** and
the nearest signed timing error. Missing means window-censored, not proof it
never changes. Scoring-only5/5 CPU tests PASS, including censored-direction
and identity handling. These diagnostics do not change the physics decisions.

Candidate6 verified (worker392698 exited,141.12s):1112/39045 exact,
precision2.8480%, recall34.1628%. Source epsilon correction removes45
proposals but6 matches, so it is not an empirical precision improvement.
Predicted-reason precision/recall contribution: strategic10.554%/9.770pp
(318/3013), cooperative9.069%/2.335pp (76/838), speed-gain4.386%/16.375pp
(533/12151), keep-right0.803%/5.684pp (185/23043).

Remaining false positives are predominantly **free-flow upstream**, not just
jam merges:21566 keep-right false steps upstream, mean observed speed29.6m/s;
only211 on the merge edge and1081 downstream. 19123/23043 keep-right steps
have no matching vehicle/direction event anywhere in the scored60s window.
Recorded-event failures: return629 no-wish/44 follower-block/189match;
overtake922 no-wish/41 follower-block/48 leader-block/581match;
merge141 no-wish/195 follower-block/123 leader-block/342match.

A key source dependency is current-leader **desired speed** at1749–1752:
the cannot-pass-on-right cap only runs if that leader's desired speed is
lower than the ego's. Homogeneous population defaults make that comparison
always false among HDVs. This is a plausible parameter-information failure,
not permission to substitute an observed-speed heuristic or new wish cutoff.

## Next parameter sensitivity (no fitting)

Add support for the **already published**, motion-only S-RB-FIT pooled IDM
point used by INT's registered variantC (owner amendment5700). Validate its
exact FIT969400–969431/pre300 schema; retain SUMO LC defaults and CAV defaults.
For this new physical-parameter setting, score BOTH LC2013 and MOBIL afresh
once (old MOBIL scores cannot be reused across changed settings). This is a
new paired sensitivity, not retraining or a source-threshold sweep. Settings
artifact bytes are bound inside the immutable bundle and verified. Seven
validator CPU tests PASS, including rejecting post-start/out-of-FIT metadata.

Candidate7 sensitivity completed/verified (worker423311 exited,203.08s):
LC1263/62689 exact = precision2.0147%, recall38.8018%; newly matched-setting
MOBIL187/3752 = precision4.9840%, recall5.7450%. This pooled CF point improves
strategic precision but **worsens total over-proposal**. Per predicted reason:
strategic20.625%/15.607pp (508/2463); cooperative13.312%/3.902pp (127/954);
speed-gain2.967%/8.725pp (284/9571); keep-right0.692%/10.568pp (344/49701).
No fitted CF setting is substituted into the LC default implementation as a
result. This matters to INT'sC interpretation; the full integrated gate owns
adoption. LC style remained source defaults throughout.

## Candidate8: existing motion encoder, no new fitted setting

Fetched the registered WM2 checkpoint from its explicit gpuA model-asset path;
SHA b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436,
3,756,566bytes, receipt ENCODER_SOURCE.json. Copied into NEW EMPTY approved
temporary directory. CPU source inspection only; GPU0/no model rollout.
The checkpoint has physical-mode,60s/1s long-history encoder weights.

New `tools/rblane/encoder.py` strictly loads ONLY all `driver_encoder.*`
tensors into the exact existing DriverEncoder. Shared trainer history features
and source model interaction function, no learned motion/LC correction runs.
One inference at300 using only240..300 data; parameters cached by observed VID
for all scored frames. LC warmup replays pre-t0 observations with these static
inferred settings. Drivers absent at300 use the frozen encoder's zero-history
population centre (including future entrants); CAVs retain original defaults.
No post300 motion contributes to driver inference. Noncomparable inferred
MOBIL lane parameters never map into LC2013 style.

CPU extractor tests independently compare all9 outputs bit-exact to the
original full model's encode_drivers, mutate all post300 data (outputs stay
identical), check source-centre fallback/CAV defaults, and fail closed on a
missing weight or wrong checkpoint SHA. Combined encoder/scorer10tests PASS.
Candidate8 will score BOTH algorithms under this new frozen parameter setting
once; old baseline results are retained, not reused across different settings.

Real frozen-weight parity also PASS (`ENCODER_PARITY.json`): all171 source
model keys strictly loaded, all9 inferred outputs bit-exact against the full
source model. First constructor-only attempt stopped because its original
learned-yield architecture requires lane_learnable=true; setting that original
construction flag permits strict load, with no acceleration/LC forward run.
The extracted encoder retains the diagnostic's original physics defaults;
its parameters still match the source model exactly. No gate was relaxed.

Candidate8 completed/verified (worker462783 exited,267.49s):1249/58225 exact,
precision2.1451%, recall38.3717%; matched-setting MOBIL108/3121 exact,
precision3.4604%, recall3.3180%. Per predicted reason:
strategic24.215%/14.224pp (463/1912); cooperative15.483%/4.086pp (133/859);
speed-gain2.334%/10.384pp (338/14484); keep-right0.769%/9.677pp (315/40970).
The existing encoder improves strategic precision vs defaults, but does not
resolve free-flow keep-right over-proposal. No parameter mode selected by this
diagnostic; all results remain explicit sensitivities.

## Candidate9: SUMO lane coordinates and consecutive-edge visibility

Further source-backed coordinate defects found before scoring:
- Same upstream edge: raw position is s+2645.67 left or s+2644.22 right.
  Directly subtracting normalized s loses1.45m in neighboring gap checks.
- Consecutive edges: source gaps include internal links (main entry3m,
  ramp entry2.93m, downstream8m). SUMO only starts looking into a subsequent
  normal edge once the preceding normal lane's end is in braking range, but
  then inspects that edge's first leader regardless of its in-edge position.
  Source MSLCM/MSLaneChanger and MSLane.cpp:3036,3136–3161 support both rules.
- MSVehicle.cpp:5617 enterLaneAtLaneChange preserves RAW lane position;
  normalized upstream s must translate+1.45m for0→1 and−1.45m for1→0.
  The runtime adapter now returns shifted positions after joint arbitration;
  rejected moves restore their origin before a fixed-point recheck. Teacher
  CAV positions are already in their destination frame and are not shifted.
  Two tiny LC2013-only wms guards wire the returned position and prevent a
  second legacy arbitration. Three default-OFF cases (legacy, legacy-conflict,
  physical-long-history) compare weights/forward/step/rollout BIT-EXACT to
  preserved pre-coordinate commit b192e0c8 (DEFAULT_OFF_COORDINATES.json).

The validator also previously paired POST-change normalized s with PRE-change
lanes without undoing that origin jump. New snapshot contractv2 restores
post-motion coordinates to the source lane, and reports each hypothetical
proposal in its own destination origin. This uses the observed lane label
only to undo a coordinate transform, not as a motivation/driver input.
The0.5s movement-before-LC ordering was independently confirmed in public
MSNet.cpp:872/878/884; it remains unchanged.

**Scoring contract changes:** old panels1–8 remain immutable diagnosticv1.
Fresh MOBIL scoring is REQUIRED once per parameter mode underv2; old v1
baseline cannot be silently reused just because raw tape hashes match.
Subsequent compatible MOBIL results are auto-reused after exact physics,
parameter SHA, snapshot-contract, and MOBIL/dependency source-byte checks.
CPU45tests PASS before the final rejection-origin regression was added;
source reads only, no SUMO or fitting. Extra downloaded sources were stored
in NEW EMPTY directories and never executed:
MSLane.cpp SHA845efee31486644e24965462e2cec3ac63da00942db5f73897b535f341982f21
at E:/Temp/opencode/sumo-1.27.1-lane-neighbours;
MSNet.cpp SHA33c14a847ea1edf1897285e65236a53e59ff6d534cd807b92d23617dcf60b40b
at E:/Temp/opencode/sumo-1.27.1-lane-order.

Final candidate9 CPU gate:46/46 PASS, single-thread6.53s, persisted JUnit
`CPU_COORDINATES.xml`. The extra regression proves rejected front movers must
restore their original origin before a rear mover is admitted (otherwise a1m
post-resolution overlap is possible). All earlier raw/source/scoring receipts
and the exact real-encoder/default-OFF parity proofs are retained.

Candidate9v2 verified (worker561787 exited,314.94s): LC1134/40246 exact =
precision2.8177%, recall34.8387%; fresh matched MOBIL241/7623 = precision3.1615%,
recall7.4040%. Per predicted motivation (precision / all-event recall contribution):
strategic9.816%/9.831pp (320/3260); cooperative8.511%/2.335pp (76/893);
speed-gain4.524%/16.805pp (547/12092); keep-right0.796%/5.868pp (191/24001).
Compressed ±2s runs:835/4611, precision18.109%, recall25.653%, matched
timingMAE0.4132s/positionMAE7.149m. Exact-position MAE2.05e-5m is coordinate
rounding, not predictive longitudinal error. This corrected result is still
far from acceptable positive-step precision.

Verification first found ONLY a6e-12m aggregate position-sum difference between
local Python3.13 compensated sum and original Python3.10.16 sum. Original
receipts/results were preserved. Reassembly now additionally runs the immutable
archived scorer in its original runtime and is **EXACT** there. Local differences
are restricted to position sums/means within a row-count-derived float64ULP
bound; counts, precision, recall, timing, inputs, settings and sources remain
exact. Integrity test rejects even a tiny precision change or one-count change.
PANEL9_VERIFIED.json retains both differing local floating values and the exact
original-runtime proof; no metric/result bytes were corrected or rerun.

## Candidate10: signed current-leader cap and paired encoder ablation

Source1749–1752 divides by **vMax-currentLeaderSpeed** without clamping its
sign. The port's positive-denominator clamp removed negative feedback when
an observer-inferred desired speed is below the observed leader speed. Source
then increases the keep-right counter (delaying the wish), while the clamp
continued accumulating it negatively. A numeric case29/28m/s inferred desired
speeds vs30/31m/s observed speeds has(95−2.5)/(29−31)=−46.25s; the source cap
suppresses a mature keep-right wish. No inferred desired-speed clipping is added.

Also corrected1742's nesting: a faster target leader must not overwrite
fullSpeedGap used by the later current-leader cap. Independent synthetic case
with target35m/s, ego desired30m/s and current leader20m/s gives9.25s, matching
source arithmetic. Zero closing speed with positive gap stays finite via the
source infinity/minimum semantics. Persistent-state finiteness is now audited.

52CPU tests PASS (6.65s, CPU_SIGNED_CAP.xml), including the paired candidate
scorer with independent wish memories and identical source-frame inputs.
Source9 vs10 default-population equality:9600 synthetic float32 vehicle
decisions,259200 compared tensor elements, all old outputs BIT-EXACT
(POPULATION_EQ_SIGNED_CAP.json). This default-homogeneous result is retained;
the source change is relevant to heterogeneous/inferred settings.

Next run freezes **existing encoder A**, snapshotv2 and both source kernels
in the SAME pass: current signed-cap fix vs our pre-fix0a1e7e93 kernel (SHA
c1aa89fafb69ee18d34ba6b7c75445a6985a10096d41237439e8dd06189e1921), plus
new matched-setting MOBIL once. The earlier encoder8 used old coordinates and
cannot replace this paired ablation. Prior kernel is identity-pinned OWN
project code; no downloaded SUMO code is executed. All settings, current
geometry, history and observations are shared, and both kernels get independent
persistent state. Encoder-source fingerprints now protect MOBIL reuse as well
as checkpoint SHA, inputs, physics and scoring transform.

Candidate10 paired result VERIFIED (worker644043 exited,566.25s). Same encoder,
snapshotv2 and current geometry, before→after signed-cap:
- exact precision2.1475%→2.2490%, recall38.7097%→37.4194%;1260/58672→1218/54157.
- Removes4473 false-positive steps at the cost of42 exact matches (net4515
  fewer proposals). The signed-cap branch fires25160 scored HDV frames.
- Per motivation before→after precision / recall contribution:
  strategic23.461%/14.286pp→same (465/1982); cooperative14.549%/4.117pp→same
  (134/921); speed-gain2.347%/10.445pp→2.337%/10.445pp (340/14486→340/14550);
  keep-right0.778%/9.862pp→0.760%/8.571pp (321/41283→279/36704).
- Current compressed±2s runs914/3586: precision25.488%, recall28.080%,
  timingMAE0.3972s, positionMAE6.053m. Paired MOBIL108/3118 exact:
  precision3.464%, recall3.318%. New matched-mode baseline retained.
- Source/encoder/input/motivation-partition and original-runtime exact
  reassembly PASS. Non-return transition controls (merge/overtake) are EXACT
  between ablation kernels, as predicted by the targeted source change.

This source fix improves total positive-step precision slightly, but **does
not resolve the keep-right problem**. Individual keep-right precision decreases
because some true positives are removed too. No distributional adoption claim.

## Candidate11: exact represented ramp endpoint / runtime motion consistency

A regression reproduces a real arithmetic legality bug introduced by this LC
port: float32(75.44)=75.44000244140625, so converting the represented physical
endpoint to double made it appear OUTSIDE the merge zone. Float64 merged;
the identical float32 scene stayed trapped. Input-only FIT audit (no WM run,
ENDPOINT_INPUT_AUDIT.json) counts5881 scored HDV endpoint frames including91
recorded merges, plus7465 warmup HDV frames/90 merges. All160 episodes and801
scored recorded ramp changes reconcile. Metadata has no population label;
no driver-assignment source was opened.

Snap ONLY the exact float32 representation (including already-promoted double
observation tensors) of the ramp endpoint back to
its geometric double value. One float32ULP farther remains illegal: no zone
extension or behavioural cutoff. The ramp end is topologically unique (no
continuation). Mainline edge-boundary ambiguity is not changed by this fix.

The now-possible merge also exposed a runtime inconsistency: LC decisions
were evaluated at a stopped/clamped position, then a successful merge released
the original untraveled overshoot and velocity. The adapter now returns its
actual clamped position AND speed even after merging, with the applied
acceleration recorded. Fixed teacher/controller CAV poses stay untouched;
their executed destination lane is used in the HDV decision geometry.
`step_output` now returns(next_lane,next_s,next_v), wired by one LC-only wms
tuple change. Synthetic tests cover retained stop motion and a fixed CAV
vacating the right lane before the HDV decision.

Targeted30CPU tests PASS, including the pre-fix failing float32 endpoint
regression. New DEFAULT_OFF_ENDPOINTS.json compares three full default-OFF
weight/forward/step/rollout cases BIT-EXACT to preserved b192e0c8. Previous
proofs are unchanged. Fresh default-mode FIT measurement will reuse the
already-completedv2 MOBIL9 score; no repeated MOBIL compute is necessary.

Final endpoint gate56/56 PASS, including float32 observations promoted to
double and rejection of the next genuine float32/float64 ULP beyond the end.
Final JUnit CPU_ENDPOINTS_PROMOTED.xml; initial endpoint gate and all earlier
receipts retained. No new scoring data or driver settings were fit.

Endpoint measurements VERIFIED, all160 FIT episodes, old MOBIL baselines reused:
- Default11:1235/41095 exact, precision3.0052%, recall37.9416%; vs9's1134/40246,
  2.8177%/34.8387%. Strategic TP rises320→411, consistent with the91 observed
  endpoint merges. Per motivation: strategic10.234%/12.627pp (411/4016),
  cooperative9.046%/2.826pp (92/1017), speed-gain4.486%/16.621pp (541/12061),
  keep-right0.796%/5.868pp (191/24001). Paired MOBIL241/3255 recall7.404%.
- Encoder12:1314/54904 exact, precision2.3933%, recall40.3687%; vs10's1218/54157,
  2.2490%/37.4194%. Per motivation: strategic20.574%/16.959pp (552/2683),
  cooperative14.995%/4.547pp (148/987), speed-gain2.306%/10.292pp (335/14530),
  keep-right0.760%/8.571pp (279/36704). Paired MOBIL108/3255 recall3.318%.
- Both execute endpoint canonicalization on5881 scored HDV frames, exactly the
  input-audit count. Both source/encoder/input/motivation and original-runtime
  reassembly checks PASS; both workers758018/762113 exited. No completed MOBIL
  inference was repeated. Keep-right remains the dominant precision failure.

## Cross-workstream kinematic identity check (no new WM inference)

Published `docs/LC2013_POSITION_HANDOFF.md` and shared
`rb-shared/LANE_TO_CF_POSITION_20261101.md` for CF/INT.2,346,379 valid same-edge
HDV FIT intervals show Euler displacement MAE3.49e-5m vs trapezoid0.0390m;
public MSVehicle.cpp:4954–4967 confirms the mode. Current inspected INT source
still used trapezoid; no CF-owned code was changed by LANE. Position audit
receipt/script committed separately; the lane diagnostic already teacher-forces
recorded position, so this does not explain away its poor keep-right precision.

## Published physical-inference sensitivity13 (frozen before scoring)

The source-correct keep-right cap depends on latent desired-speed differences;
existing encoder and homogeneous points leave many false proposals. Consume
the ALREADY-FINISHED S-RB-FIT finite_mixture_v4 point reader (owner physical-fit
amendment5700), not another fit. Exported3130 native driver points across32 FIT
worlds with MotionFit.idm(sample=False), plus its population fallback; source
reader SHA2c50c732… and artifact/provenance hashes retained in
MOTION_POINT_TABLE.json. All32 native input hashes match completed PANEL9
observations exactly. No new training, LC fit, optimizer or label-based tuning.

Freeze native-only32 episodes,240..300 warmup/300..360 score, source82894fab,
SUMO LC style defaults and original CAV defaults. Native-only avoids assuming
cross-arm numeric VID alignment. Missing/entrant IDs use the published population
point. This point is explicitly a sensitivity, NOT substituted into A/B/C or
claimed as calibrated uncertainty. New-setting matched MOBIL runs once; source,
arm subset, input SHA and table-reader identity protect future cache reuse.
Point-table tests cover identity/fallback/CAV defaults and reject foreign-arm,
out-of-FIT, nonpositive or mismatched-observation inputs.

Sensitivity13 VERIFIED: native-only32worlds/626 events, LC236/7627 exact,
precision3.0943%, recall37.6997%; matched MOBIL31/770, precision4.0260%,
recall4.9521%. Same native subset of default11 has236/8193, precision2.8805%
and identical total recall. Different physical inference removes566 proposal
steps with the same total match count; this is not proof of identical matched
identities or a distributional improvement. All identity/artifact/source and
original-runtime reassembly checks PASS; owned870050 exited. No fitting ran.
Per motivation (precision / native626-event recall contribution): strategic
14.808%/13.578pp (85/574); cooperative15.672%/3.355pp (21/134);
speed-gain2.343%/12.141pp (76/3244); keep-right1.469%/8.626pp (54/3675).

## Cooperative early-return ordering14 (source-frozen before scoring)

Source1643–1669 returns on cooperative clearing BEFORE1694–1826's wish updates.
The port had accumulated speed-gain wishes while returning COOPERATIVE, then
used updated counters in inconvenience. Reorder to compute the source condition
from pre-update wishes and freeze accumulation on that branch, including blocked
cooperative requests. Source1565's thisSpeed==0 inconvenience convention is also
retained. Remove the extra positive-parameter disable gate absent from source:
zero-cost cooperation atlcCooperative0 is permitted by the actual predicate.
The earlier avoid-overtake-right impulse remains, as it precedes the source's
cooperative return; only the later ordinary accumulation is skipped.

Independent synthetic cases have an urgent ramp request and its follower behind
a slow current leader: source returns COOPERATIVE with left wish still0, rather
than accruing a latent speed-gain manoeuvre. Targeted32physics/model tests PASS.
This changes no inferred parameters and targets a real accumulation/timing path;
before/after metrics and per-motivation precision/recall will be reported.

Final cooperative-order passes VERIFIED, all matched MOBIL reused:
- Default14:1235/41083 exact, precision3.0061%, recall37.9416% (12 fewer false
  proposal steps than11, same total matches). Per motivation precision/recall
  contribution: strategic10.234%/12.627pp (411/4016), cooperative9.046%/2.826pp
  (92/1017), speed-gain4.490%/16.621pp (541/12049), keep-right0.796%/5.868pp
  (191/24001).
- Encoder15:1318/54910 exact, precision2.4003%, recall40.4916% (4 more matches,
  6 more proposals than12). Strategic20.574%/16.959pp (552/2683), cooperative
  14.995%/4.547pp (148/987), speed-gain2.332%/10.415pp (339/14536), keep-right
  0.760%/8.571pp (279/36704).
- Native points16:236/7629, precision3.0935%, recall37.6997% (2 extra false
  proposals vs13). Strategic14.808%/13.578pp (85/574), cooperative15.672%/3.355pp
  (21/134), speed-gain2.341%/12.141pp (76/3246), keep-right1.469%/8.626pp (54/3675).
All input/artifact/source/original-runtime and exact motivation partitions PASS.
Owned910505/910966/911369 exited. This source-order correction is faithful but
only marginally improves these conditional metrics; keep-right remains unresolved.

## Final urgency / outward-message completion (no repeated traffic scoring)

Source1661/1861's mature left speed-gain/cooperative request now emits follower
help using source minNextSpeed slack; lcSpeedGainUrgency is active and urgency
flags distinguish strategic/cooperative/speed-gain paths. Source283–300 checks
RAW message bounds before foreign wanted-speed blending. Explicit message
presence/weight now preserves zero/positive advice instead of conflating it
with no message; new STATE_KEYS carry those fields, with old-state fallback.
IDM default passenger minNextSpeed slack1.5 and velocity floor are honored.

69CPU tests PASS (7.76s), including both sides of the urgency threshold, follower
deceleration, positive/zero/weighted messages and fitted low-decel admission.
9600 heterogeneous synthetic decisions/288000 non-message elements BIT-EXACT
to scored source52d2df5c;536 urgent stress frames exercised. Only outward
message semantics differ. Thus panels14–16 remain the conditional-decision
metrics, with no redundant traffic scoring. Changed autonomous feedback still
needs INT/EVAL smoke; no distributional gain is claimed. FINAL_DIAGNOSTICS.json
binds the verified score receipts and this explicit equivalence scope.
