# S-RB-LANE — precision continuation active (~03:08)

Branch `c4/rb-lane`, base `f0db3c82`; latest engineering source `52d2df5c`.
Coordinate lineage `0a1e7e93`, signed-cap source `92a3cd69`.
All prior work retained. Owner authority:
WM-PHYS REGISTRATION5683 plus00:10 precision addendum. Source-backed subset;
**current FIT precision remains poor, no distributional qualification claim**.

**CF/INT evidence:** actual FIT displacement is Euler(s+v_next*dt), MAE3.49e-5m
vs trapezoid0.0390m over2.346M valid intervals. Current INT source still shows
trapezoid. Read `docs/LC2013_POSITION_HANDOFF.md`; shared note published. This
does not alter the recorded-position lane diagnostic or legacy integrator.

## Switch / integration
- `ModelSpec(lane_model="lc2013", lane_head="off", merge_decision="hazard")`.
  Default unchanged; build a clean spec, disable learned acceleration/merge
  corrections. No new checkpoint weights; WM lanes left0/right1/ramp2.
- `lc2013.py`: strategic/cooperative/speed-gain/keep-right, fixed-point wishes,
  lookahead, IDM secure gaps, follower help, right-overtake prohibition.
  LC style uses SUMO defaults; only comparable motion-inferred settings map.
- `lc2013_geometry.py`: actual finite best-lane LENGTHS3197.55/553.33/477.89,
  edge-local position, both-best-zero route offset resolution, raw neighboring
  gaps, collapsed3/2.93/8m junction links and source consecutive-leader visibility.
- Candidate9 coordinate increment: upstream model s translates±1.45m on lane
  change to preserve SUMO raw lane position. Latest `step_output` returns
  **(next_lane,next_s,next_v)**; guarded wms edits consume motion and skip a
  duplicate legacy arbitration. Fixed CAV teacher positions are not shifted.
  Rejected moves restore their origin before a monotone joint-feasibility recheck.
- Step recomputes after longitudinal/CAV motion; ramp cannot pass its lane end.
  Carry `physics.lc2013.STATE_KEYS`; IDs `vehicle_id` or`vid`, reset absence/
  identity/observed change/newborn. Messages consumed next step. Optional route
  distances/offset/occupation plus raw-position and target-offset overrides.

## Verified precision iterations (160 FIT episodes,3255 events, v1 diagnostic)
- Original2: precision2.562%, recall34.900%,1136/44333 proposals; paired MOBIL
  recall7.496%. Historical1.4% is another panel/contract.
- Source gate4: 2.577%/34.900%,1136/44078 (255FP removed, noTP lost).
- Finite lengths5: 2.860%/34.347%,1118/39090.
- Correct NUMERICAL_EPS=.001 in6: 2.848%/34.163%,1112/39045.
- Published pooled CF point7: 2.015%/38.802%,1263/62689 (worse over-proposal).
- Existing frozen WM2 encoder8: 2.145%/38.372%,1249/58225. Strategic precision
  improves24.22%, but keep-right315/40970 is still0.769%. New-setting paired
  MOBIL recall3.318%. No LC fitting/threshold adjustment/black-box training.
- Per-predicted-motivation precision/recall contributions and TP/FP location/
  timing diagnostics: `docs/LC2013_PRECISION_ITERATIONS.md`, immutable panels3–8.
  Actual motivation labels unavailable; contributions use all3255 events.

## Candidate9 gate / scorer correction
- 46 lane/encoder/scorer CPU tests PASS. Three shared-WM default-OFF cases
  weights/forward/step/rollout BIT-EXACT to b192e0c8 (DEFAULT_OFF_COORDINATES.json).
  Real b8797242… encoder:171 source keys strict, all9 outputs bit-exact; inference
  uses only240..300 history, persistent VID settings, centre fallback for entrants.
- v2 validator undoes recorded upstream lane-origin shifts before decisions,
  then reports each proposal in its destination frame. Old v1 results retained.
  **Fresh matched MOBIL required once per setting**; auto-reuse thereafter needs
  exact input/physics/setting SHA/snapshot/MOBIL dependency bytes. Candidate9v2
  VERIFIED: precision2.818%, recall34.839%,1134/40246; paired MOBIL7.404% recall.
  Run±2s precision18.109%/recall25.653%. Still poor positive-step precision.
  Original-runtime exact reassembly PASS; local6e-12m float-sum roundoff retained.
- All owned workers through9 exited. Only six allowlisted observable tape files,
  published motion-fit artifact/frozen model weights, and road/source code read.
  No SUMO/GPU/training/signals/deletions/push. lcspec/obsstate/targets consumed.

## Candidate10 VERIFIED (source92a3cd69)
Source1749's current-leader denominator now preserves its SIGN; the old clamp
incorrectly removed negative keep-right feedback under inferred settings.
Faster target leaders no longer overwrite the later cap's gap. 52CPU PASS;
9600 default-population synthetic decisions remain bit-exact to9. Paired
encoder-v2 ablation: precision2.148%→2.249%, recall38.710%→37.419%; removes4473
FP at cost42 exact matches. Keep-right precision remains0.760%; NOT resolved.
Source branch fires25160 frames. Non-return transition controls are exactly
unchanged. All source/input/encoder/aggregate checks PASS, worker644043 exited;
no owned jobs remain. No setting/threshold fit. All original results preserved.

## Gaps
Synchronous proposals/messages vs SUMO serial processing; full leader
overtake/fall-behind/counterchange and follower branches, nonlegacy keep-right,
nonzero speed lookahead, nonhighway/emergency cases remain. See source notes
and `docs/paper_results/RB_LANE.md`; integrated CF+merge+LC gate is EVAL's.

## Endpoint repair VERIFIED (source82894fab)
Exact float32 ramp end75.44000244 was incorrectly outside double75.44.
FIT audit:5881 scored HDV frames/91 recorded merges affected (plus warmup).
Only the exact represented endpoint is canonicalized; nextULP stays illegal.
Runtime merges retain clamped stop position/speed instead of releasing free
overshoot; applied acceleration is consistent, fixed CAV poses/lanes preserved.
**Adapter API now(next_lane,next_s,next_v)**; merge its one guarded wms tuple
update. Final56CPU PASS (including promoted float32 input); default-OFF weights/forward/step/rollout bit-exact
in three cases (DEFAULT_OFF_ENDPOINTS.json). Default11: precision3.005%,
recall37.942% (1235/41095); encoder12:2.393%/40.369% (1314/54904). Both report
5881 corrected HDV frames, source/aggregate/input checks PASS, matching MOBIL
baselines reused. Both workers758018/762113 exited; no owned jobs remain.

Native-only physical-point sensitivity13 is verified: consumes the
already-published S-RB-FIT3130 causal driver points, exact32 native observation
SHA bindings; no fitting or LC-style change. Separate sensitivity, not an A/B
replacement. Native236/626 matches, precision3.094%, recall37.700%; default11
native has the same total matches but8193 vs7627 proposals. Matched-mode
MOBIL31/626. All checks PASS;870050 exited.

Next source fix: cooperative early returns now freeze wish accumulation as
SUMO1643–1669 requires, using pre-update inconvenience. An extra positive-
cooperation gate absent from source is removed. 60CPU PASS; default/encoder
and native-point measurements follow, with matching completed MOBIL reused.
Owned distinct-setting final passes:14/default910505,15/encoder910966,
16/native-point911369 oncpuQ; immutable PANEL14/15/16_DISPATCH receipts.
All reuse completed matched MOBIL; three admitted nice19/single-thread jobs,
loads16.8–18.2 on32CPUs. Preserve these dispatches; do not duplicate.
