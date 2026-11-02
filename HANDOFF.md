# S-RB-LANE morning handoff — 2026-11-01

Branch `c4/rb-lane`, base `f0db3c82`. **Current engineering source80541278**;
scored decision source52d2df5c, decision-equivalent message completion certified.
All prior work retained. **Keep-right precision is NOT solved; no distributional
qualification. Prior inference jobs exited; publication-only observer completed.** Morning report:
`docs/paper_results/RB_LANE_20261101.md`; full iteration/receipts diary preserved.

<!-- LANE-FIT-PRECISION:BEGIN -->
Closed 2026-11-01T09:00:00.219934+08:00 at the09:00 deadline; 0new verified FIT summaries published.
No inference/fitting/remote dispatch. Keep-right qualification remains unresolved.
<!-- LANE-FIT-PRECISION:END -->

## Fresh-session reconciliation — 2026-11-02
- Resumed clean `8f229cd5`; all implementation, iterations and score receipts
  preserved. Local clock is already beyond the November 1 09:00 continuation
  deadline. Observer completion records zero new publications; exact local
  PID3316 is absent on the November 2 check. No observer restart is required.
- Consumed FIT's `lane_raw_5a5b15ef118c` publication (November 2 09:04:39+08):
  all three physical-source hashes match the current worktree. Receipt below.
  Existing panels14–16 and their per-motivation scores remain primary.

## Integration contract
- `ModelSpec(lane_model="lc2013",lane_head="off",merge_decision="hazard")`.
  Build a clean spec; disable learned acceleration/merge corrections. Default
  legacy/checkpoint behavior stays unchanged; no new weights or factory changes.
- `lc2013.py` + `lc2013_geometry.py`: strategic/cooperative/speed-gain/keep-right,
  source fixed-point wishes/lookahead, IDM secure gaps/assertiveness, finite
  route LENGTHS, raw lane origins/internal-link gaps, right-overtake prevention,
  cooperative early returns, urgency and source speed-message semantics.
- **step_output returns(next_lane,next_s,next_v)**. Merge its guarded wms tuple
  update and duplicate-arbitration exclusion. Post-motion clamp is retained even
  after merging; fixed CAV/controller positions AND destination lanes constrain
  HDVs. Rejected moves restore their coordinate origin before joint recheck.
- Direct step callers carry **all `physics.lc2013.STATE_KEYS`**, now including
  `lc2013_has_advice`/`lc2013_advice_weight`. Explicit zero/positive messages and
  foreign wanted-speed blending are preserved; old-state negative-only fallback.
  Numeric IDs:vehicle_id orvid. Reset absence/identity/lane change/newborn.
- LC style uses SUMO defaults; only directly comparable motion-inferred fields
  map. MOBIL politeness/threshold are not treated as LC2013 style parameters.
- Fresh05:00 continuation completes optional `lcSpeedGainLookahead` (default0)
  and nonlegacy `lcKeepRightAcceptanceTime` follower response (default-1).
  No fitted style; default panel decisions remain unchanged. Acceptance-time
  diagnostics added; existing STATE_KEYS and runtime ABI unchanged.

## Verified conditional FIT result (not autonomous qualification)
32 FIT worlds969400–969431;240..300 history,300..360 teacher forcing, snapshotv2.
- Population/160episodes:1235/3255 recall**37.942%**, precision**3.006%**,
  41083 positive steps. Matched MOBIL241/3255 recall7.404%.
- Frozen encoder/160episodes:1318/3255 recall**40.492%**, precision**2.400%**,
  54910 positive steps. Matched MOBIL108/3255 recall3.318%.
- Published causal physical points/native32:236/626 recall37.700%, precision
  3.093%; matched MOBIL31/626. Explicit sensitivity, not substituted into A/B/C.
- Dominant non-urgent keep-right FP:23810(population),36425(encoder); precision
  0.796%/0.760%. Per-motivation precision/recall contributions, timing/position,
  transitions and run-window metrics in **tools/rblane/receipts/FINAL_DIAGNOSTICS.json**.
  Actual motivation labels absent; contributions use all recorded events.
  Historical MOBIL1.4% has a different panel/contract.

### Latest verified FIT LC result (current source, not adopted)
- Reused completed `lane_raw_5a5b15ef118c` verified32native-world counts; no
  fitter, model inference or dispatch. Posterior435.655 expected positive steps,
  60.983 exact directed matches/626events: ratio-of-expectations precision13.998%,
  recall9.742%. Same-cache source-default263/12383:2.124%/42.013%.
- Source hashes for LC/geometry/config all MATCH current; pooled IDM/settings
  and native-only contract differ from the primary panels above. This remains
  a precision/recall tradeoff; EventBrier obscures the missed~90% of events.
  Learned population puts94.549% mass atlcKeepRight0, inferred upstream.
- `FIT_RAW_5A5B_PRECISION_20261102.json` pins checksums/world-sum arithmetic.
  Motivation counts absent upstream; do not invent them or replace panels14–16.
  Earlier different-source13.962%/9.721% receipt `FIT_RAW_C560_PRECISION.json`
  is retained separately. No qualification or mixture adoption is claimed.

## Engineering proofs / receipts
- **69 targeted CPU tests PASS**, CPU_URGENCY_MESSAGES.xml. Three default-OFF
  weight/forward/step/rollout cases BIT-EXACT (DEFAULT_OFF_ENDPOINTS.json).
- Exact float32 ramp endpoint was outside the double bound:5881 scored HDV
  frames/91 recorded merges affected. Only that representation is normalized;
  genuine nextULP stays illegal. Strategic matches rose91(default)/87(encoder).
- Final message completion:9600 heterogeneous synthetic decisions,288000
  non-message elements BIT-EXACT to scored52d2df5c,536 urgent stress frames.
  Autonomous advice feedback changes and still needs INT/EVAL smoke.
- Panels14–16 source/input/encoder/artifact/motivation/original-runtime exact
  reassembly PASS; completed matched MOBIL reused. Last owned910505/910966/
  911369 exited; all earlier workers exited. No SUMO/GPU/training/signals/push.
  Final identity audit:17 recorded workers,0 matching owned workers alive,
  0 signals (tools/rblane/receipts/OWNED_JOBS_FINAL.json).
- Optional-style completion: **71 targeted CPU tests PASS8.66s**;
  `OPTIONAL_STYLE_DEFAULT_EQUIVALENCE_FINAL.json` certifies46080 synthetic decisions/
  1658880 existing output elements BIT-EXACT to aefc4bd9, including5817 positive
  steps. No new FIT/GPU/SUMO jobs; old score receipts retained.
- `OPTIONAL_STYLE_FIT_SUPPORT_EQUIVALENCE.json`: same exact proof across ALL209
  published five-coordinate LC profiles (incl legal zero boundaries),4468
  positive steps. FIT watcher source jobs remain owned by FIT; no duplicate job.

## CF / INT coordination and scope
- Actual2.346M FIT motion intervals prove **Euler s+v_next*dt**, MAE3.49e-5m vs
  trapezoid0.0390m. Inspected INT still used trapezoid. Source/data note:
  `docs/LC2013_POSITION_HANDOFF.md`, mirrored inrb-shared. LANE did not change
  CF-owned integration; align SUMO-mode and executable CAV/entry plants.
- Remaining gaps: synchronous vs serial SUMO, full leader overtake/fall-behind/
  counterchange and every follower branch, nonhighway cases.
  Distributional CF+merge+LC/controller qualification belongs to INT/EVAL.
