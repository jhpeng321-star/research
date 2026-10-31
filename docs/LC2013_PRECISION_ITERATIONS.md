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
