# S-RB-LANE — precision continuation active (~01:25)

Branch `c4/rb-lane`, base `f0db3c82`; all prior work retained. Owner authority:
WM-PHYS REGISTRATION5683 plus00:10 precision addendum. Source-backed subset;
**current FIT precision remains poor, no distributional qualification claim**.

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
  change to preserve SUMO raw lane position. `step_output` now returns
  **(next_lane,next_s)**; two guarded wms edits consume positions and skip a
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
  exact input/physics/setting SHA/snapshot/MOBIL dependency bytes. Candidate9
  v2 FIT measurement pending; source/improved-metric claim awaits that result.
- All owned workers through8 exited. Only six allowlisted observable tape files,
  published motion-fit artifact/frozen model weights, and road/source code read.
  No SUMO/GPU/training/signals/deletions/push. lcspec/obsstate/targets consumed.

## Gaps
Synchronous proposals/messages vs SUMO serial processing; full leader
overtake/fall-behind/counterchange and follower branches, nonlegacy keep-right,
nonzero speed lookahead, nonhighway/emergency cases remain. See source notes
and `docs/paper_results/RB_LANE.md`; integrated CF+merge+LC gate is EVAL's.
