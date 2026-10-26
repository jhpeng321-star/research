# S-NATIVELANE — native CAV lane baseline (2026-10-26)

## Status

Implemented on `c4/nativelane`, starting at requested `01f590e9`.
**No adoption decision yet; gpuR MAIN14 arm remains untouched.**
This report is committed before the 05:30 laptop deadline; the server continuation
will write the completed scientific decision and operational receipt independently.
The previous **228/228** exploratory qualification jobs had **no PASS→FAIL
sign-unit transition**; that receipt is retained as prior-order evidence.
ON qualification is being repeated after the final longitudinal/lane-substep
ordering regression; it must finish before any adoption decision.

## Inference switch

`[policy14] native_lane_change = true` (default **false**), also exposed as
`make_cav_controller(..., native_lane_change=True)` and
`make_controller_factory(..., native_lane_change=True)`.

The bridge advertises its native-only mask before each WM forward pass. Only those
CAVs join the existing HDV lane-decider mask; the original learned lane head,
MOBIL/mandatory feasibility, recurrent lane state, paired random draws, learned
driver settings and first-passage merge/conflict rules determine their lanes.
The bridge takes those lane targets instead of overwriting them with “stay”.
The longitudinal learned native-CAV head remains the native plant. Policy-acting
CAVs retain their lane head. Entrance-zone release and all/base/group/flip ownership
are recomputed from the current observed/predicted state, without driver labels.
No retraining or new behavioural thresholds. Flag-ON policy checkpoints also
carry the lane switch into confidence-gate/real-credit imagination; their
arm dispatcher preserves native lane and ramp delegation across batched arms.

Files: `src/wms/control/{bridge,dreamer,policy14,policy14_cli,policy,confident_gate}.py` and
`src/wms/model/{cav,wms}.py`; regression cases in `tests/unit/test_native_lane.py`.

## Verification before inference

- cpuX initial integrated CPU suite: **351 passed**; final expanded suite on
  gpuB: **367 passed**, including calibration dispatch and adoption-rule vetoes.
- Flags OFF versus `01f590e9`: **55/55 tensor bit identities** under legacy
  ramp behaviour and **55/55** under LOOP14C `wm_law`, including sampled and
  deterministic policy/all/counterfactual rows.
- Real 13E b2 FINAL, flag ON, two-second/two-future inference preflight passed.
- Ruff passes on changed source/tests/measurement scripts.

## Registered measurement and adoption

Append-only shared `REGISTRATION.md`: **S-NATIVELANE-1**, registered before
measurement. Frozen WM SHA256
`952ec059de5a9c0285d2327bcea57c836fd5d736c9975592ffbb932a506e84c8`.
No new SUMO or evaluation-seed outcomes.

- Rerun OFF/ON on the 33 replay-matched COOPMEAS 60 s starts and 12 existing
  300 s starts, four paired futures. Reuse the original MAPPO harness, tapes and
  frozen behaviour detector. Report lane-vacate contrast, metering ratio, gain
  sign agreement, actual native ordinary lane transitions and their exposure.
- Exploratory pulse/coop qualification on the existing TRAIN tapes, FINAL WM,
  six paired future seeds, both original three-seed sets, frozen TOLERANCES_v2.
  The copied FINAL needs the committed stage-B physics config explicitly;
  startup failures from its absent adjacent sidecar are retained in logs and
  retried without changing the checkpoint or scoring rule.
  The pre-forward native mask now receives the execution tick, so scripted
  pulse targets release to native after 3 s consistently in lane inference and
  execution. Superseded ON tapes from the missing-tick harness are retained in
  `superseded_tickmask_attempt1/`; ON qualification is rerun, OFF successes retained.
  A second structural regression ensures a native mainline insertion competes
  with a ramp merge exactly like an HDV, rather than inheriting policy-command
  reservation priority. Superseded ON 60 s/qualification tapes are retained at
  `superseded_priority_attempt2/` and rerun before any decision; OFF is unchanged.
  Final sequencing regression: native head and safety execute in the current
  lane, then the HDV lane proposal is committed/vetoed. This prevents a proposed
  (possibly cancelled) native change from affecting the head one substep early.
  Superseded ON tapes are retained at `superseded_longitudinal_attempt3/`; all
  affected ON starts/samples rerun. Default-OFF trajectories remain exact.
  The shared C1 audit then identified three old C/300 s low-share runs as only
  60 s long. This task disables that training-only cap in its measurement
  launcher for both arms, archives short OFF outputs in `superseded_short_clock/`,
  repairs precisely the short starts, and requires 301/61 1-Hz frames for every
  300/60 s measurement. Main-loop training defaults/curriculum stay as registered.
- C1 PI u_same native/PI ranking is secondary, entries OFF and inherited
  snapshot-history/longitudinal plant retained; run only if required work finishes.
- Paired whole-world percentile bootstrap: 2,000 resamples, seed 0.

Adopt iff the primary **60 s** lane-vacate contrast agrees in sign with SUMO
**or** its absolute WM–SUMO gap shrinks with lower 95% CI >0; **and** native
ordinary lane-change counts move toward SUMO; **and** no exploratory pulse_m or
gap_open sign unit flips PASS→FAIL in either seed set. Missing required evidence
means no adoption. Other metrics remain descriptive.

## Detached continuation and owner handoff

cpuX `~/nativelane`: driver **PID/SID 156201**, qualification retry **213000**
(3 workers), continuation **213054**, `setsid nohup`, numerical
threads=1, nice19, CPU only. MAPPO pool <=6 plus qualification pool <=6;
after qualification completes, its six slots may transfer to the MAPPO pool (<=12 total).
Detached repair driver **198518** retains the valid three true OFF/300 s repairs.
The superseded ON/60 s pool has finished and is archived; the released three
slots run final ON qualification beside the 6-worker original OFF long-rollout
pool and 3-worker clock-repair pool (<=12 total). ON/300 s waits for the exact
repair PID incarnation, repairs ON/60 s, and takes 12 slots only after qualification
exits; otherwise it stays at <=6. No pools overlap above 12.
secondary C1 uses <=12 only after those pools exit. New inference is barred
after **09:00 UTC+8**. Only receipt-owned PID incarnations are inspected.

Code: `tools/nativelane/{drive,finish,launch,analyze,c1}.py`.
Durable outputs: `driver.log`, `finish.log`, `CHECKS.json`, `QUAL_TASKS.json`,
`DECISION.json`, `RESULT.md`, `NATIVE_LANE_COUNTS.json`, `QUAL_NATIVE_LANE.json`,
optional `C1_RANKING.json`, per-unit tapes and logs under `off/`, `on/`, `logs/`.
The continuation retries only missing infrastructure-task markers and reruns the
registered analysis. A registered, hash-pinned adoption package and a gpuR
in-memory watcher handle conditional adoption without a reverse SSH key. The
watcher reads cpuX using gpuR's existing key and writes **no armed file** unless
the scientific adoption rule has passed. Polling logs and its PID receipt live
on cpuX. Required prearm tests/smoke finish before `STOP_ARM` is touched.

If adopted before 09:00, MAIN14 amendment must preserve its newer allocator and
readiness fixes, use only documented `main14_arm/STOP_ARM`, preserve the previous
installation, append the registration amendment before re-arm, pass integrated
tests plus a two-update real-WM CPU prearm smoke, and retain every existing launch
gate. The later MAIN14 readiness/allocator changes have been retained by
cherry-picking their committed fixes onto this branch. If not adopted,
incomplete, already launched or late: arm untouched.

The initial conditional package at `18008a5f` and watcher 652131 were superseded
before any decision or MAIN14 write, following the mainline-priority regression.
That package is retained on cpuX as `rearm_package_superseded_72a0db3862dd.tar.gz`.
The source/config/registration hashes of its replacement are registered before
installing the replacement watcher; scientific adoption criteria are unchanged.

Conditional runtime 1619544e and watcher 652555 were also retired before any
MAIN14 write when the final sequencing regression was added. That package is
retained as `rearm_package_superseded_5bc09110ba9e.tar.gz`. Its verified receipt
proved original arm SHA256 `5c237f95...` and waiter 649847 remained untouched.
The replacement source/config/registration is pinned before installing its watcher.
Watcher events are on cpuX `gpuR_watch.jsonl`; final installation/failure receipts
will be `ADOPTED.json` or `ADOPTION_FAILED.json` there, with full gpuR evidence.
