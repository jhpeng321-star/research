# S-RB-MERGE — live handoff

Branch `c4/rb-merge`; base `f0db3c82`. Owner registration5683 / MAIN00:10.
Full earlier handoff preserved at `docs/rb_merge/HANDOFF_HISTORY_0100.md`.

## Integration-ready source
- Source advice now uses IDM's exact minNextSpeed bound
  `max(b,min(emergencyDecel,1.5))`, important for motion-inferred low b.
  Three source numeric anchors added; **44 local merge/native tests PASS /5.80s**.
- **Native-CAV blocker reproduced on CPU**: the frozen native head's no-leader
  dv feature gathers slot0 speed (`control/native_cav.py:65,80`), even when that
  car is on an unrelated mainline lane. At the observed ramp-head state, this
  produces a false standstill. A singleton fixture does NOT reproduce it.
- **d7be510a** NEW `model/sumo_native.py::native_physics_controller(dt_s=.5)` explicitly
  delegates the native arm to the existing physics execution ceiling and lane/
  merge streams, retaining the registered project_native executor bounds.
  Opt-in callback only; no shared model/factory change or retraining. INT should
  consider this for physics-native background rather than the learned head.
  Regression: old native head stays before s=0; physics-native crosses/merges
  into an empty mainline within10s despite the unrelated packed mainline car.
  **40 combined local CPU tests PASS /14.74s**. Fleet efficacy not yet measured.
- Matching `native_physics_accel(state, dt_s=.5)` is now exported for
  native-relative controller prediction, so native and zero-residual plans share
  one plant. Four native CPU tests PASS /5.50s, including exact zero-residual
  equality. Callback arithmetic is unchanged by this API extraction.
  Coordination pointer: `E:/agent-work/rb-shared/MERGE_NATIVE_FINDING.md`.
- **5870b556** follow-up source constants: eventual follower help uses POSITION_EPS=.1,
  not SPEED_EPS=.01; remaining-seconds denominator uses NUMERICAL_EPS=.001.
  Shared constants live in merge_network.py. **37 CPU tests PASS /9.95s**,
  including an independent near-boundary follower-help/fall-behind scenario.
- **3df73a27** latest mechanics: NEW `physics/merge_network.py`, local approach
  speed limit22.22m/s before s=0, acceleration-lane limit31.29 thereafter, and
  inferred speedFactor=v0/referenceLimit for departure/advice. Source geometry,
  not a fitted threshold. Insertion keeps INT's exact weighted BirthMix candidate.
- **0090c168** prerequisite leader negotiation: Euler brakeGap with tau,
  overtake time AND space checks, unblocked-leader follow-speed cap, optional
  persistent LC lookahead. POSITION_EPS=.1 agrees with INTf2d9a3ff; the helper's
  .01 was SPEED_EPS, and old results using it remain separately versioned.
- Switch `PhysicsConfig.merge_model='sumo'` (default `legacy`). Learned merge,
  courtesy, ramp adaptation, admission and motion corrections bypassed; no training.
  Driver settings only population defaults/existing motion encoder.
- MERGE owns ramp2->1, geometry, lane-end stopping and insertion; LANE owns
  mainline LC2013. Keep INT's reconciled adapter, with only TWO keyword hooks:
  merge_speed_advice gets `state.get('lc2013_lookahead')`; post-motion apply_merge
  gets `out.get('lc2013_lookahead')`. Do NOT replace INT's adapter with ours.
- **36 local CPU tests PASS /13.14s** for source mechanics/conservation/validation.
  Latest immutable combined runtime gate **33 PASS /1.83s** on cpuQ, including
  native CAV chunk parity in BOTH CF modes and observable LC-memory ID remapping.

## Full-fleet FIT diagnostics — all32 native worlds969400–969431, H300
- v3: **4f78ba0e**, INTff8a637f, legacy CF/cold LC, population defaults, native
  CAV executor, entrants + exact source ledger. PID338844 exited,32/32 complete.
  Ramp-origin road+source mean2366.57 predicted /2272.78 recorded veh-s;
  q10/50/90 predicted536/1091/5912 vs real1641/1966/3422; marginalW1=1336.91.
  t0 matches169/200; timingMAE35.86s /positionMAE16.61m. Six beyond-end merges
  show the legacy trapezoid/SUMO Euler-stop inconsistency. Not qualified.
- v4: **75a24397**, INTb0f66943 +0090c168 kernel/two hooks, SUMO CF/Euler,
  observable60s pre-t0 LC history, same settings/arrival stream190. Owned
  PID405098 **exited**,32/32 complete. No duplicate or running v4 dispatch.
  Ramp mean2165.68 predicted /2272.78 recorded; q10/50/90=499/844/5766.
  MarginalW1=1306.67; secondary pairedMAE1743.65. t0 matches173/200;
  timingMAE29.56s /positionMAE13.40m; zero beyond-end merge positions.
  **55 corridor-frame overlap pair-samples in worlds969402/969405**; these are
  geometric diagnostics, not SUMO collision logs. Internal connector geometry
  is omitted by the corridor projection, so physical interpretation needs care.
  Correct mean does NOT certify the still-wrong delay distribution. Not qualified.
- Both panels: all32 exact road/source conservation PASS, no demand discarded,
  no slot pressure. Ramp source waiting means9.69/9.53veh-s; bias mainly on-road.
  Historical562.4/1265.4 is a different panel/contract; recorded per-world clocks
  are primary. One future/native only does not qualify conditional distributions.
- Hash-verified raw receipts `docs/rb_merge/fullfleet_v{3,4}/`; compact scores
  `fullfleet_v{3,4}_summary.json`. Four receipt-audit tests PASS (hash, complete
  panel, duplicate/missing world, broken source conservation).

## Completed diagnostics v5 / v6
- **v6 complete32/32**, owned cpuQ PID**522021 exited**, source **3fd7ab03**.
  Validator SHA `d35a8a99790696cef19a47a0e460a36b9eb7d4748c63999aa3563c040f2342a1`.
  Receipt `/tmp/rb-merge-fullfleet-v6/LAUNCH.json`; v5 exited first. One CPU,
  nice19/CUDA-hidden, launch load7.27/32. No active prediction dispatch remains.
  Mechanical check969400/CAV187: v6 merges at t326.5/s52.884m, while v5 never
  merges it and remains at s=-1.464m/v=0 through t600.
  Full score `docs/rb_merge/fullfleet_v6_summary.json`: **200/200 t0 merges
  matched**,202 predicted; full-fleet merge mean33.16 vs33.06 recorded.
  TimingMAE29.42s, positionMAE13.45m (matched sets differ across versions).
  Ramp mean**1185.32 vs2272.78 real**: false gridlock removed, now over-clears.
  MarginalW1=1087.46, secondary pairedMAE1243.40;46 corridor overlap samples.
  All32 road/source gates PASS. Still **NOT distributionally qualified**.
- **01:04:36 +08 dispatched ONCE**, owned cpuQ PID**468961**, validator SHA
  `f68014b8b0586db5fdd329996c214fefcf7fb61c1e73aed5b4104ab5db21940b`.
  Receipt `/tmp/rb-merge-fullfleet-v5/LAUNCH.json`. PID468961 now **exited**,
  all32 worlds complete; all v3/v4/v5 dispatches are terminal.
- Source **7d4240ef**, same INTb0f66943 + mechanics3df73a27. Same default settings,
  stream190, SUMO CF and causal LC history; isolates the new network-limit rules.
- Immutable `/tmp/rb-merge-fullfleet-v5`;33CPUtests and all32 source preflights
  PASS. Ten-second front-ramp traces retain IDs/speed/gaps/required gaps/CAV flag
  without influencing motion. Score `docs/rb_merge/fullfleet_v5_summary.json`:
  ramp mean2363.87 predicted /2272.78 recorded; marginalW1 **1197.36** (v4=1306.67),
  secondary pairedMAE1651.77; timingMAE30.36s (v4=29.56), positionMAE13.03m.
  All32 source gates pass; same55 corridor overlap samples. Still not qualified.
- Tool: `py -3.13 tools/rbeval/remote_full_fleet.py status|launch|poll|collect`.
  Explicit `--version v5` inspects v5. Stage/dispatch/results
  refuse overwrite. Single CPU nice19/CUDA-hidden, resource admission load<28/32.
  Never relaunch a completed version or overwrite a running snapshot.
- Early trace identifies a concrete blocker in969400: initial CAV187 stalls
  at s=-1.464m/v=0 before the legal merge zone, from320s through600s despite no
  ramp leader. Native-head packing bug reproduced on CPU (above); v5 sources
  and results remain unchanged. Next v6 stages physics-native delegation plus
  EPS5870b556; **39 remote CPU tests PASS /2.12s**, all32 preflights PASS.
- v6 immutable runtime is staged at `/tmp/rb-merge-fullfleet-v6`; direct CLI
  `--native-model sumo` requires `--cf-model sumo_idm`. Old head remains the
  direct validator default; the versioned v6 runner selects physics-native.
  Model IDs in each row/receipt prevent pooling different native-law contracts.

## Next v7fast prepared — 600s frozen-motion-encoder diagnostic
- Sequential merge arbitration now recomputes geometry ONLY for the candidate
  at that rank, reducing O(ramp_count*slots²) to O(ramp_count*slots). Bit-exact
  full-vs-selected geometry tested with heterogeneous parameters/multiple batch
  rows; mixed ramp-count occupancy reservations preserved. **46 local tests PASS**.
- NEW `/tmp/rb-merge-fullfleet-v7fast` passes **51 CPU tests /2.64s** and all32
  preflights. Earlier v7 engineering stage retained; no v7 prediction launched.
- New `/tmp/rb-merge-fullfleet-v7`: INT**4015b641** (includes LANE811cd7e8)
  plus current MERGE kernel/network/insertion/native callback. H600,512slots,
  FIT32/native, same arrival stream190. No prediction launched yet.
- Reuses LANE's **b192e0c8 FrozenMotionEncoder**, exact registered WM2 SHA
  b8797242…; producer ENCODER_PARITY reports all9 settings bit-exact to the
  strict full model. No duplicate extractor or checkpoint fetch; no training.
- Preserves the trained encoder's legacy feature/default geometry; rollout
  uses SUMO geometry. Initial HDV settings follow observed VID, CAV defaults
  unchanged, entrants use the frozen zero-history centre. Actual admissions
  and newborn settings share one ordinal-verified point sampler.
- Source-by-source runtime gate **49 CPU tests PASS /2.37s**, all32 input/strict
  encoder preflights PASS. Additional8 receipt/birth tests PASS locally.
  Captures120/300/600 clocks in one600s rollout to assess clearing, not only
  t0 merges. Parameterized motion/lane corrections still forbidden by hooks.

## Interfaces / tests / known gaps
- Entries MUST use `TorchEntryProcessV2(conserve_queue=True, couple_draws=True,
  row_seeds=...)`. Existing FIFO, birth ordinals and SourceLedger conserved.
  Demand from identity-hashed observable start.pt pre-t0 insertions +t0 queue;
  future recorded states never feed the prediction. No assignment/truth opened.
- Full-fleet clock matches reported-boundary algebra:31.29m/s reference, no road
  charge on birth tick, exiting speed carried, ramp origin retained after merge,
  fractional source waiting separate. Mainline/ramp insertion counts scored.
- CPU: `PYTHONPATH=src;.` OMP/MKLthreads1, then
  `py -3.13 -m pytest tests/unit/test_rb_merge.py tests/unit/test_rb_merge_validation.py tests/unit/test_wmfix_admission.py -q`.
  Integrated validator tests need the pinned integrated runtime imports.
- `tools/rbeval/summarize_full_fleet.py` verifies collection hashes, exact FIT32
  identity and source conservation before distributions; no outcome-based tuning.
- Highway subset, not full SUMO parity: serial LC ordering, complete blocker/
  stopped-leader negotiation, continuous LC, internal-junction state, nondefault
  actionStep and full inferred-population efficacy remain open. LC/fleet posterior
  and final distributional qualification belong to INT/UNC/FIT/EVAL.
- Source ledger: `docs/rb_merge/SOURCES.md`. Public downloads read as text only
  in new empty temp directories; no downloaded code executed. No GPU/SUMO/training,
  peer PID changes, push, reset/clean/deletions/new venvs in this continuation.
