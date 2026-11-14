# WM-43 REGISTRATION — WM-40 wall mechanics fixture (prepared; not launched)

- **ID:** WM-43 (registration for the fixture proposed in
  `output/action_model_v1/REPORT.md`, section "Proposed mechanics fixture").
- **Date prepared:** 2026-09-26. **Zero simulation performed or authorized by
  this registration.** Launch requires (1) the seeds diff applied, (2) an owner
  authorization file carrying this file's SHA-256, (3) the preflight passing.
- **Kind:** physical-path integration evidence only (mechanics). NOT efficacy
  evidence: short mechanics runs are not efficacy evidence; no controller
  comparison, no acceptance-sealed data, no fitting, no training.

## Exact new question and changed variable

Do the WM-40 dispatcher wall groups (`traffic_shepherd/mixed_assignment.py`,
mixed_assignment/3 + `traffic_shepherd/wall_group.py`) survive the REAL
TraCI/ownership write path — speedMode31 single-writer ownership, issued wall
accelerations, assembly/hold/release lifecycle, declared safety aborts with
same-tick native handback — on the preserved network? The changed variable vs
the WM-40 zero-simulation state is ONLY the addition of a fixture package
under `output/action_model_v1/fixture/` (runner, plan, checks, registration)
plus an integrator-applied seeds block; no active module is edited.

## Closest prior (novelty check, `tools/experiment_history.py closest`)

- **WM-40** (this package's parent, zero simulation): implemented
  mixed_assignment/3 with mocked-TraCI tests only; its report proposes exactly
  this fixture. This registration prepares that proposal.
- **WM-41 campaign pilot** (8401-8406): the runner-discipline template
  (authorization digest gate, new-output-dir preflight, allocation rows before
  launch, zero retries, one child process per run). WM-43 reuses the
  discipline for a micro-roster mechanics fixture; no saved-state, no
  efficacy arms.
- **OE-02 / OE-07 / OE-10, CV-32** (old `finite_lifecycle_wall/2` era):
  efficacy studies of the PREVIOUS wall implementation with fixed-entry
  gating. WM-43 runs the NEW planner-driven dispatcher wall on a 2-CAV
  micro-roster and scores nothing; comparators differ in kind.

Exact unanswered difference: no registered run has ever exercised
`MixedAssignmentSession` wall commands on the real ownership path.

## Source identities (at registration, worktree checkpoint 4c4fb2f)

- `traffic_shepherd/mixed_assignment.py` VERSION `mixed_assignment/3` (uncommitted
  WM-40 state, read-only for this task; digest recorded at preflight).
- `traffic_shepherd/wall_group.py` — pacing/assembly/abort constants reused as-is
  (SAFE_SPEED_MIN_MPS=2.0, ABORT_SPAN_M=150, ASSEMBLY_SPEED_TOL_MPS=1.0,
  ASSEMBLY_SPAN_TOL_M=6.0, REAR_GAP_M=10, REAR_SPEED_EXCESS_MPS=2).
- `traffic_shepherd/ownership.py` `CommandOwnership` (sole longitudinal writer,
  speedMode31), `traffic_shepherd/damping.py` `CirclesController` +
  frozen model `baselines/circles/model.onnx` (adapter digest
  `762fde0763c693c0e88f41b98b22fc0a11135a312b78c4c567e6211a3035ccee`),
  `traffic_shepherd/sensing.py`, `traffic_shepherd/scenarios.py`
  (template `scenarios/idm/driver_routes.rou.xml`, read-only),
  `traffic_shepherd/simulation.py`.
- Network: `assets/research/highway_merge_shepherd_long/osm.net.xml.gz`
  (bytes untouched; mainline `1191554636` = 2 lanes @ 31.29 m/s, ~2644 m).
- Environment: CPython 3.10.20
  `C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe`, SUMO 1.27.1.

## Seeds (development-mechanics block; fresh identities)

- Block `WM43_WALL_MECHANICS_SEEDS` = 920244-920248 (5 identities), each
  accepted ONLY with its registered fixture loading **2000** (registry key
  only — the micro-roster generates no demand streams), role `exposed`
  (WM-41 pilot convention), penetration 1.0 (all roster CAVs).
- Freshness: successor region of the audited 92xxx fresh-world series (first
  contiguous run after the WM-34 block 920144-920243). NEVER the 60 WM-28
  TEST worlds (920044-920123), never any registered final-test seed. Exposure
  audit: `EXPOSURE_AUDIT_WM43.json`, audit sha256
  `5d178eff3549efde967b77ac4302faf4064c314d1b1b8afee49edfabc9ae1a05`
  (identity-aware `git grep -n -w`: the only prior token is the WM-34 range
  END BOUND `range(920144, 920244)`, a declared non-use).
- Delivery: `seeds_py_wm43.diff` (sha256
  `4a4b0afa4fee5e4a9d95affa16e8092bb1bfcd3216874dacf310a807334a426f`),
  dry-apply-verified by preflight against the live `seeds.py`; **the
  integrator applies it** — this package does not edit `traffic_shepherd/`.

## Splits, comparators, budget

- Split: development-mechanics evidence only; excluded from any prospective
  (final-test) registration; never quoted as efficacy.
- Comparator: each arm against its OWN registered telemetry expectations
  (the per-arm check lists below). No controller-vs-controller comparison.
- Budget: **3 arms × 5 seeds × 600 s simulated = 15 runs, 9000 simulated
  seconds**, single replay each, **zero retries**, zero training, ≤ **6
  workers**, wall-clock request ≤ 1 h (micro-roster expected faster). New
  output directory `runs/active/wm40_wall_mechanics_v1/` (never pre-created;
  the runner creates it only at launch after all gates pass).

## Arms and schedules (plan: `PLAN.json`)

- **A1 planner wall lifecycle** — 2 members (`wall.0` lane 0, `wall.1` lane 1,
  200-260 m, 19-21 m/s, ≤ 4 m apart); single plan at t=0: wall at 8 m/s,
  `release_at_s=90`.
- **A2 wall + damping non-member** — same members; `follow.0` 70-100 m behind
  at 22-26 m/s, assigned `damping` from t=0; `release_at_s=300`.
- **A3 induced safety abort** — same members; `release_at_s=600` (never
  reached). From t=90 s a **scripted actuator** pins the front member
  (`wall.1`) to 25 m/s AFTER the tick's single ownership write; span grows
  ~17 m/tick; the declared `member_span_exceeded` abort (span > 150 m) fires
  (< 200 s) and members hand back to native in the same tick. The actuator
  is a logged fault injector (`scripted_actuator` telemetry rows), NOT a
  control write: every control command still flows through
  `CommandOwnership` alone. This disclosure is part of the registered design.

## Per-arm machine-evaluated checks (implemented in `wm43_checks.py`)

Common (all arms): `C01_run_complete` (600 ticks, complete receipt),
`C02_zero_collisions`, `C03_zero_teleports`, `C04_single_writer_path` (one
ownership record per live CAV per tick; wall rows exactly the registered
member set), `C05_speed_mode_31_contract` (wall speed 8.0 m/s identical
across plan/events/telemetry, strictly above the 2.0 floor; ownership
speedMode31 validation held every tick), `C06_roster_and_wall_identity`.

- **A1:** `A1_1_assembly_start` (exactly once at t=0, 8.0 m/s),
  `A1_2_assembled_event` (once, both members, ≤ 30 s),
  `A1_3_hold_speed_tolerance` (held members within 1.0 m/s of 8.0; ≥ 20
  response samples/member), `A1_4_released_event` (once, handback=native,
  at 90 s), `A1_5_native_handback_receipts` (same-tick owner=native +
  `released_to_native` + `selected_native`), `A1_6_no_wall_after_release_no_abort`,
  `A1_7_wall_pacing_identity` (issued acceleration == clip((8−v)/1, −3, +1.5)
  and target == max(0, v+a) per frozen wall_v2 pacing, exact to 1e-3).
- **A2:** `A1_1`, `A1_2`, `A1_3`, `A1_7` plus `A2_1_no_abort_release_at_300`
  (no safety_abort ever; single release at 300 s), `A2_2_follower_damping_path`
  (every `follow.0` row owner=damping), `A2_3_surveillance_no_region_entry`
  (per-tick `wall_proximity` rows; the (gap ≤ 10 m ∧ speed > wall+2) abort
  region never entered; min rear gap recorded), `A2_4_cross_path_single_writer`
  (members wall+locked, follower damping+free every active tick),
  `A2_5_handback_receipts_at_release`.
- **A3:** `A1_1`, `A1_2`, `A1_3` (hold tolerance on the pre-injection window only), `A1_7` plus `A3_1_injections_logged` (all on the
  declared member at 25 m/s, first ≥ 90 s), `A3_2_abort_reason_span`
  (`member_span_exceeded`, 90 s < t ≤ 200 s), `A3_3_same_tick_handback`,
  `A3_4_injections_cease` (no injection at/after the abort tick),
  `A3_5_hold_until_injection` (hold tolerance before t=90).

## Launch procedure (after owner sign-off, in order)

1. Integrator applies `seeds_py_wm43.diff` to `traffic_shepherd/seeds.py`
   (conflicts: re-run the exposure audit against the newer file first).
2. `python output/action_model_v1/fixture/run_fixture.py preflight` — must
   exit 0 (read-only; no SUMO start; creates nothing under `runs/`).
3. Planner writes `AUTHORIZATION.json` next to this file (schema
   `wm43_authorization/1`, carrying this file's SHA-256; template provided).
4. Launch (exact command; repo root `D:\research-dense`; ~1 h bound, 6 workers):

```powershell
C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/run_fixture.py launch --plan output/action_model_v1/fixture/PLAN.json --registration output/action_model_v1/fixture/REGISTRATION.md --authorization output/action_model_v1/fixture/AUTHORIZATION.json --workers 6
```

The runner then creates `runs/active/wm40_wall_mechanics_v1/`, writes an
allocation row BEFORE each child, dispatches exactly 15 child processes
(never overwrite, never retry), and writes `LAUNCH_SUMMARY.json` with budget
reconciliation. Any failed run remains on disk with its failure receipt.

## Refusal conditions (registered)

Launch refuses: missing/stale authorization digest; preflight failure
(including the pending seeds diff); existing `runs/active/
wm40_wall_mechanics_v1/`; workers > 6. No run is ever retried or overwritten;
a changed registration is a different experiment.

## Close-out duty (integrator)

After the runs: append the outcome to
`protocols/experiment_history.json` per `docs/EXPERIMENT_RECORD_TEMPLATE.md`
(15/15 receipts, checks verdicts, budget charged vs actual), update
OPEN_QUESTIONS/ROUTE/INDEX, regenerate RESULTS_SUMMARY, run `validate`,
scoped `verify-sources`, `summary --check` and the ledger tests. This
preparation ships `LEDGER_DRAFT_WM43.json` for that entry.

## Amendment A1 (2026-09-27): r1 launch failed at scenario load; remediated; relaunch is r2

**What happened.** The owner-authorized r1 launch into
`runs/active/wm40_wall_mechanics_v1/` FAILED on all 15 runs at scenario
load, before any physics: SUMO rejected every route file with
`Error: The vehicle type 'wm43_cav' for vehicle 'wall.0' is not known.
Quitting (on error).`  Simulated seconds consumed ~ 0 (LAUNCH_SUMMARY:
complete 0 / failed 15 / retries 0).  The r1 evidence stays untouched at
`runs/active/wm40_wall_mechanics_v1/` as a failure receipt.

**Root cause.**  The scenario builder copied the template `normal_car`
vType attributes INCLUDING `id` and merged them as
`{"id": "wm43_cav", **template_attrs}` — in a python dict merge the LATER
binding wins, so the generated route file declared `normal_car` while all
roster vehicles referenced `wm43_cav`.  The r1 TraCI-shape audit could not
catch this: it audited call shapes, not generated XML content.

**Remediation (this amendment).**
1. Builder fixed: the template identity is dropped before the merge; the
   generated route file declares exactly one vType `wm43_cav` (carFollowModel
   preserved, `speedDev="0"`) and every vehicle references it.
2. NEW static validator `wm43_validate.py` (pure python, zero SUMO) now runs
   in the preflight dry-run for EVERY planned run AND in every run child
   immediately after the scenario build and BEFORE any SUMO start.  Rules:
   V0 generated files present and parseable; V1 every sumocfg-declared input
   path resolves; V2 network parses (gzip-aware) and exposes edges/lanes;
   V3 every route edge exists in the network; V4 every vehicle's type is a
   DECLARED vType (the SUMO default type is not accepted for this fixture);
   V5 exactly one `wm43_cav` vType carrying `carFollowModel` and
   `speedDev="0"`; V6 no duplicate ids across vTypes/routes/vehicles; V7
   vehicle routes exist with nonempty edge lists; V8 departLane < lane count
   of the route's first edge; V9 departPos within the depart lane's length;
   V10 numeric departSpeed within the depart lane speed limit; V11
   scenario_manifest.json agrees with the route file (ids, lanes, positions,
   speeds) and every roster lane fits the assembly edge.  The validator
   REJECTS the actual failed r1 scenario files (test, on a read-only copy)
   and ACCEPTS all 15 regenerated scenarios.
3. Runs root parameter updated: the relaunch writes to the NEW directory
   `runs/active/wm40_wall_mechanics_v1_r2/` (registered below); the failed
   r1 directory is never reused or overwritten.

**Relaunch authorization.**  This amendment changes the registration bytes,
so the r1 `AUTHORIZATION.json` no longer matches and the runner refuses
until the planner re-issues it against the new digest below.  Everything
else in the launch procedure is unchanged except the runs root.

## Amendment A2 (2026-09-27): standing PRE-LAUNCH SMOKE + smoke-gated launch

Owner decision (2026-09-27, pending-final in the morning): before EVERY
launch of this package, a standing SMOKE exercises the real pipeline once
and its verdict gates the launch.  Registered shape (PLAN.json 'smoke'):

- ONE run: arm A1, seed 920244, capped at t=70 s (70 simulated seconds,
  inside the 80 s cap) - just past the arm's critical point: scenario/vType
  load, dispatcher wall assembly start (assembly_start at t=0, assembled
  well inside the cap) and telemetry rows all occur through the REAL TraCI/
  ownership path via the SAME run-child code (`_exec_smoke`), never a
  parallel implementation.
- Separate smoke runs root `runs/active/wm40_wall_mechanics_v1_r2_smoke`
  (registered runs root + `_smoke`); write-once, never reused; smoke runs
  are NEVER data: they feed no analysis, no comparison, no statistics.
- Zero retries.  Peak RSS of the child is sampled and recorded as evidence
  metadata (not a gate).
- The smoke needs no owner AUTHORIZATION.json (it is part of this
  registered procedure), but it refuses unless the full preflight passes
  (including the seeds gate and the static scenario validator).
- Result: a write-once `SMOKE_VERDICT.json` beside this registration with
  passed=true bound to (a) the sha256 of the CURRENT REGISTRATION.md and
  (b) the package code digest (sha256 over the sorted package *.py bytes)
  plus the smoke's run/root/horizon/ticks/peak-RSS.  Any registration or
  code edit invalidates it and forces a re-smoke.

**Launch gate (order).**  Budget guards -> SMOKE_VERDICT.json (passed,
current registration digest, current package code digest) -> owner
AUTHORIZATION.json (registration digest) -> preflight -> dispatch.  The
launch refuses with the exact remediation when the verdict is missing,
failed, or stale.  The verdict file is write-once: after a failed or stale
smoke it is deleted consciously, never overwritten.

**Commands (planner/owner, in order; cwd D:\research-dense).**

1. Preflight (read-only, must exit 0):
   `C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/run_fixture.py preflight`
2. Smoke (70 simulated seconds; writes SMOKE_VERDICT.json):
   `C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/run_fixture.py smoke`
3. Launch (unchanged, now smoke-gated):
   `C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/run_fixture.py launch --plan output/action_model_v1/fixture/PLAN.json --registration output/action_model_v1/fixture/REGISTRATION.md --authorization output/action_model_v1/fixture/AUTHORIZATION.json --workers 6`

Smoke budget accounting: each smoke consumes 70 simulated seconds into the
smoke root; the registered 9000 simulated seconds of the campaign remain
unchanged and are consumed only by the launch into the registered runs root.

## Amendment A3 (2026-09-27): r2 smoke failure diagnosed (pre-insertion abort); fixture plan deferral; roots -> _r3

**What happened.**  The approved r2 smoke ran its 70/70 ticks cleanly (zero collisions, zero teleports, single-writer path green) but S05/S06 failed: `assembly_start events=0`, `assembled events=0` - the dispatcher never started the wall.

**Root cause (evidence: runs/active/wm40_wall_mechanics_v1_r2_smoke/a1_seed920244/telemetry.jsonl, read-only, untouched).**

- The ONLY wall event in the whole run:
  `{"kind": "wall_event", "wall_id": "wall_a1", "event": "safety_abort", "timestamp_s": 0.0, "reason": "member_frame_missing", "handback": "native"}`.
- The plan WAS submitted and applied: all 138 mixed_assignment rows carry `plan_id "a1_seed920244-p0"`.
- At simulation time 0.0 - BEFORE the first `simulationStep()` - SUMO has not yet inserted the depart=0 vehicles, so the loop observed an empty roster; `wall_group.WallGroupSession.tick` evaluates `_safety_abort` BEFORE emitting `assembly_start`, and a member without a frame is an emergency abort in every phase.  The wall therefore aborted `member_frame_missing` at t=0, was handed back to native in the same tick, and the session was popped.  From t=1.0 both CAVs were live and ran native for the remaining 69 ticks (mixed rows: action `native`, wall_id null; command rows: owner `native`, fallback `selected_native`) - 69 ticks x 2 CAVs = 138 rows.

**Why the mocks missed it.**  The fake TraCI pre-inserted the roster at t=0, so `getIDList()` was never empty in tests.  The fake now models real insertion (vehicles are TraCI-visible only from the first step at/after their depart; `build_fake_session(..., insertion_delay_s=...)`), and a REPLAY test re-evaluates the recorded r2 files through the real checks and reproduces the exact recorded verdict (S05+S06 failed, nothing else), with the evidence byte-identical after the replay.

**Remediation.**

1. FIXTURE (applied, no dispatcher change): the loop now submits the `AssignmentPlan` on the FIRST TICK WHERE THE FULL ROSTER IS LIVE and emits `plan_deferred` / `plan_submitted` telemetry rows documenting the decision.  Staging is planner-side behavior under the WM-40 contract; the dispatcher's `member_frame_missing` semantics are untouched.  A delayed-insertion test (roster visible from t=2) proves: plan deferred at t=0/t=1, submitted at t=2, `assembly_start` at t=2, assembled inside the cap, S01-S06 green.  A never-live test proves the smoke still fails honestly (no wall, S05/S06) if insertion never completes.
2. S05 re-scoped: `assembly_start` must fire exactly once inside the cap at the registered wall speed - not necessarily at t=0 (the start tick now equals the roster-live tick).
3. DISPATCHER PATCH PROPOSAL (NOT applied; integrator decision): `wm43_dispatcher_patch_proposal.diff` (sha256 a5df3d4af13722c0f7f80896b106a8a171a643588c88974f1013032f800f36bb) proposes that `wall_group.WallGroupSession.tick` treat missing member frames as a DEFERRAL - not the declared emergency abort - only in the window before assembly has ever started (`assembly_since is None`); after assembly_start the abort semantics are unchanged.  The fixture does NOT depend on this patch (deferral in the planner suffices); applying it changes registered WM-40 semantics and requires its own registration amendment plus the mocked-TraCI mutation checks.

**Roots -> r3.**  Registered full runs root `runs/active/wm40_wall_mechanics_v1_r3`; smoke root `runs/active/wm40_wall_mechanics_v1_r3_smoke`.  The r2 smoke evidence directory and its write-once SMOKE_VERDICT.json (passed=false) stay in place untouched; the planner archives the r2 verdict, after which the fresh smoke writes a new one (write-once discipline).

**Commands (planner/owner; cwd D:\\research-dense).**

1. Preflight (read-only, must exit 0):
   `C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/run_fixture.py preflight`
2. Smoke (70 simulated seconds into the r3 smoke root; writes a fresh SMOKE_VERDICT.json):
   `C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/run_fixture.py smoke`
3. Launch (unchanged CLI; smoke-gated + authorization-gated; runs into the registered _r3 root):
   `C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/run_fixture.py launch --plan output/action_model_v1/fixture/PLAN.json --registration output/action_model_v1/fixture/REGISTRATION.md --authorization output/action_model_v1/fixture/AUTHORIZATION.json --workers 6`

Smoke budget unchanged: 70 simulated seconds per smoke, never data.

## Amendment A4 (2026-09-27): r3 full-run check failures split (3 check defects fixed; 1 real dispatcher defect diagnosed); roots -> _r4

**What happened.**  The r3 launch completed all 15 runs at the full 600 s horizon with zero collisions and zero teleports, but the acceptance checks failed per the recorded matrix: A1_1 + A1_7 in all 15 runs; +A2_2 in all 5 A2 runs; +A2_1 + A2_5 in a2_seed920246 only.

**Per-failure verdicts (evidence: runs/active/wm40_wall_mechanics_v1_r3/<run>/, read-only, untouched).**

- **F1 (A1_1, 15/15) - CHECK DEFECT.**  Every run: plan_deferred at t=0.0, plan_submitted at t=1.0, assembly_start at t=1.0 (the A3 deferral stages the wall on the first full-roster-live tick; SUMO inserts depart=0 vehicles during the first step).  A1_1 still hardcoded t=0.0 - the A3 re-scope had been applied to S05 only.  Fixed: A1_1 now requires assembly_start exactly once at the plan_submitted tick (telemetry cross-check), inside the assembly deadline, at the registered wall speed.
- **F2 (A1_7, 15/15) - CHECK DEFECT.**  The recorded violation tuples (wall.x, 1.0, -3.0, -3.0) compare the ISSUED vs EXPECTED acceleration (equal); the actual mismatch was the target term.  Arithmetic on the recorded rows (a1_seed920244): issued_target=18.885 at t=1.0 and the response 20.385 at t=1.5 reconstruct the TRUE frame speed v=21.885 (= 20.855 depart + 1.03 of un-commanded IDM acceleration during the deferred tick) - the real command obeyed a=clip((8-21.885)/1)=-3.0, target=18.885 exactly.  The check's v-reconstruction had no same-timestamp response at t=1.0 (no responses exist before the CAVs are commanded) and fell back to the stale depart speed.  Fixed: A1_7 verifies the identity ONLY on ticks with a same-timestamp measured speed (verified 176 rows, skipped 2 evidence-free rows at t=1.0, both members, declared in the detail).  Not a weakening: the identity is proven everywhere measurable and the hold-speed check constrains the wall independently.
- **F3 (A2_2, 5/5 A2) - CHECK DEFECT.**  follow.0's COMMITTED assignment stayed 'damping' on all 320 mixed rows in every A2 run (the dispatcher never reassigned a non-member).  11 command rows per run were owner=native with fallback_reason 'missing_or_uncovered_leader' - the frozen damping controller legitimately refused around the t=295-320 window (the members' span-abort/release churn removed its leader from measurable range) and the ownership contract releases a refused damping decision to native for that tick.  Fixed: A2_2 now asserts (a) committed action == damping on every mixed row and (b) native-owned rows carry a non-'selected_native', non-None documented refusal reason, with both counts reported.
- **F4 (A2_1 + A2_5, a2_seed920246 only) - REAL DISPATCHER DEFECT (diagnosed, patch PROPOSED, not applied).**  The abort at t=295.0 was `member_span_exceeded` - NOT rear-end surveillance: the recorded follower geometry never entered the surveillance region (A2_3 green; minimum rear gap 14.67 m at t=12; gap 27.51 m at constant 8.00 m/s around the abort).  Both members held 8.0 m/s through the abort (measured speeds at t=295.0) with a true span of ~3 m: the front member crossed from the 2644.22 m pre-merge edge onto the merge edge at ~t=295 (this seed drew the deepest roster position, 262.8 m), and wall_group._member_positions compares LANE-LOCAL positions, which are meaningless across edges (2530 vs ~0 -> >150 m phantom span).  Remedies: (a) PROPOSAL wm43_dispatcher_patch_r3.diff (sha256 below; integrator applies with a backup): sensing.SensorFrame gains an optional route_position_m (vehicle.getDistance, route-absolute) and wall_group._member_positions prefers it when present - older frames keep the registered behavior; skipped-until-applied tests ship in tests/test_dispatcher_patch_r3.py; (b) FIXTURE (applied): the A2/A1/A3 roster start band tightened to 120-170 m (+<=4 m member offset, follower 60-90 m behind, >=30 m start) so the worst-case front-member absolute position at the t=300 release is <=2618 m - >=26 m inside the edge end - making the registered arms well-defined under the CURRENT dispatcher too.  A2_1/A2_5 are NOT relaxed: the r4 replay of a2_seed920246's corrected geometry is what re-validates them.

**Roots -> r4.**  Registered full runs root `runs/active/wm40_wall_mechanics_v1_r4`; smoke root `runs/active/wm40_wall_mechanics_v1_r4_smoke`.  The r3 evidence directory stays untouched.

**Smoke coverage statement.**  The F1/F2/F3 check fixes are fully smoke-covered at the registered 70 s cap (the defects and their fixes live at t=1.0/t<=320 damping fallbacks: the plan_submitted tick, the skipped pacing row, and the follower's committed assignment are all exercised by t=70; the smoke checks S01-S06 now also run against the corrected semantics).  F4 is different: the false span abort fired at t=295 s, i.e. a smoke can show that the wall ASSEMBLES and HOLDS through 70 s but CANNOT prove the full-hold edge-boundary behavior - the r4 full A2 runs (release at t=300) are the evidence for F4's fix, and until the dispatcher patch is applied the F4 signature is expected to be PREVENTED only by the tightened geometry bound.

**Commands (planner/owner; cwd D:\\research-dense).**

1. Preflight (read-only, must exit 0):
   `C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/run_fixture.py preflight`
2. Smoke (70 simulated seconds into the r4 smoke root; writes a fresh SMOKE_VERDICT.json):
   `C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/run_fixture.py smoke`
3. Launch (unchanged CLI; smoke-gated + authorization-gated; runs into the registered _r4 root):
   `C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/run_fixture.py launch --plan output/action_model_v1/fixture/PLAN.json --registration output/action_model_v1/fixture/REGISTRATION.md --authorization output/action_model_v1/fixture/AUTHORIZATION.json --workers 6`

Smoke budget unchanged: 70 simulated seconds per smoke, never data; campaign budget unchanged: 9000 simulated seconds.

## Amendment A5 (2026-09-27): F4 remedies REDONE after planner review (merge-relative coordinate; loud failure; geometry crossing band); roots -> _r5

**Planner review outcome.**  F1/F2/F3 accepted as check defects (F3 verified: the 11 native follower rows carry fallback_reason missing_or_uncovered_leader).  BOTH F4 remedies as written in A4 were REJECTED and are redone here:

- REJECTED 1 - the A4 dispatcher patch: traci vehicle.getDistance() returns the ODOMETER (distance driven since insertion), not a route-absolute position - members inserted at different departPos carry offset odometers, so the span would be wrong (odometer-offset trap); and the bare silent exception fallback is unacceptable in a safety check.
- REJECTED 2 - the A4 120-170 m roster bound: it HIDES the pre-merge/merge boundary instead of testing it.

**Redone remedy 1 - the coordinate is the registered merge-relative axis already present on every frame.**

SensorFrame.distance_to_merge_m (traffic_shepherd/sensing.py: observation_map defines base[lane] - pre_merge/ramp 0.0, merge -len(merge_0), downstream -len(merge_0)-len(lane) - and SumoLocalSensor.capture adds lane_length - position) is CONTINUOUS across the mainline and defined for EVERY edge a wall can occupy: pre_merge [0, 2644.22] decreasing, merge [-75.44, 0], downstream [-553.33, -75.44]; the boundaries match exactly (verified from the network: 0 -> 0 at the pre-merge/merge end, -75.44 -> -75.44 at the merge/downstream end).  NEGATIVE post-merge values are valid positions, explicitly handled.  The same axis is what the reconstruction grid uses for longitudinal alignment (output/worldmodel_spatial_v1/reconstruction/spatial_grid.py edge offsets).  This supersedes the rejected odometer approach AND the silent fallback: wm43_dispatcher_patch_r5.diff (sha256 0df3bc83184bf400a23cfff1a8a736df7482cbe0122df439eeac09e36d30068d; single diff against the CURRENT traffic_shepherd/wall_group.py; supersedes and replaces the rejected wm43_dispatcher_patch_r3.diff) makes _member_positions REQUIRE distance_to_merge_m, moves the non-member rear-end surveillance onto the same axis, and adds the declared LOUD abort reason member_position_unavailable (handback native) when the coordinate is missing or nonfinite - no silent fallbacks.

**Patch tests (always-on, against a patched temp copy imported from the diff - no skip markers).**  tests/test_dispatcher_patch_r5.py: same-edge span; ~3 m spans straddling the pre-merge/merge AND merge/downstream boundaries (no abort); the odometer-offset trap (members at one position, different insertion points -> span 0, the true geometric span); a genuine 152.22 m cross-boundary violation (declared member_span_exceeded abort, safety semantics unchanged); loud member_position_unavailable for missing AND nonfinite coordinates; non-member rear-end surveillance on the same axis; and a REPLAY of the recorded a2_seed920246 state at the false-abort tick (identical member speeds 8.0 m/s + the manifest's 2.964 m longitudinal offset -> true span 2.964 m; the old lane-local comparison yields a constant phantom span of ~2641 m): the patched span does NOT abort.

**Redone remedy 2 - the geometry now GUARANTEES the boundary crossing instead of hiding it.**

The A4 120-170 m bound is REVERTED and REPLACED by the registered A5 roster band 240-300 m start (follower 70-100 m behind, >= 140 m start; seeds and derivation unchanged): with <= 304 m front-member start and the registered hold profile, the predicted crossing ticks of the pre-merge/merge boundary during the hold are 288.7 s (920244), 289.4 s (920245), 288.7 s (920246), 291.4 s (920247), 292.6 s (920248) - ALL 5 A2 seeds cross deterministically during the hold, >= 6.5 s before the t=300 release (requirement was >= 2/5; stated seeds: every seed crosses, first 920246).  A2_1/A2_5 are unchanged and are re-validated by the r5 full runs WITH the crossing exercised - under the un-patched dispatcher these runs would abort (the false positive the patch fixes), under the applied patch they must release at t=300 with handback receipts; that contrast is the F4 re-validation.

**Smoke coverage statement.**  The 70 s smoke covers the patch's same-edge span, loud-failure wiring and the A3 deferral; it CANNOT reach the boundary crossing (~t>=288 s) or the t=300 release - the F4 re-validation is the r5 full A2 runs' job, as stated above.

**Roots -> r5.**  Registered full runs root runs/active/wm40_wall_mechanics_v1_r5; smoke root runs/active/wm40_wall_mechanics_v1_r5_smoke.  The r3/r4 evidence and the write-once SMOKE_VERDICT.json stay untouched; the planner archives the verdict before the r5 smoke.

**Commands (planner/owner; cwd D:\\research-dense).**

1. Apply the reviewed patch (with backup): output/action_model_v1/fixture/wm43_dispatcher_patch_r5.diff to traffic_shepherd/wall_group.py.
2. Preflight (read-only, must exit 0):
   C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/run_fixture.py preflight
3. Smoke (70 simulated seconds into the r5 smoke root; fresh SMOKE_VERDICT.json):
   C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/run_fixture.py smoke
4. Launch (unchanged CLI; smoke-gated + authorization-gated; runs into the registered _r5 root):
   C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/run_fixture.py launch --plan output/action_model_v1/fixture/PLAN.json --registration output/action_model_v1/fixture/REGISTRATION.md --authorization output/action_model_v1/fixture/AUTHORIZATION.json --workers 6

Budgets unchanged: 70 simulated seconds per smoke; 9000 simulated seconds campaign; never data; zero retries.

## Amendment A6 (2026-09-27): r5 patch REJECTED on a sign-convention safety defect; sign-audited r6 patch; roots -> _r6

**What happened.**  The planner rejected wm43_dispatcher_patch_r5.diff: distance_to_merge_m INCREASES UPSTREAM (it counts down toward the merge point: pre_merge [0, 2644.22] decreasing, merge [-75.44, 0], downstream [-553.33, -75.44]), but wall_group's comparisons assume a DOWNSTREAM-INCREASING coordinate: `rear = min(positions.values())` and the window `rear - REAR_GAP_M <= pos <= rear`.  The r5 patch switched member AND non-member positions to distance_to_merge_m without touching these, so `rear` became the FRONT member and the rear-end surveillance silently inspected vehicles AHEAD of the wall - an inverted safety check.  The r5 same-axis test missed it because it encoded the same wrong assumption (it placed the fast vehicle at d2m 94, numerically below the members, and called it 'behind').

**Sign audit (every position use; coordinate = s_m = -distance_to_merge_m, downstream-increasing).**

| # | Use (traffic_shepherd/wall_group.py) | Sign dependence | r5 | r6 |
| --- | --- | --- | --- | --- |
| 1 | _member_positions coordinate choice | picks the axis | raw d2m (inverted) | s_m via the new module-level downstream_position_m(frame) = -distance_to_merge_m (None if missing/nonfinite) |
| 2 | _safety_abort span = max-min | sign-independent | ok | s_m |
| 3 | rear member = min(positions) | SIGN-DEPENDENT (r5 = front member) | inverted | min(s_m) = most upstream member |
| 4 | rear-end window rear-REAR_GAP_M <= pos <= rear | SIGN-DEPENDENT (r5 = looked ahead) | inverted | behind-window on s_m |
| 5 | non-member coordinate for surveillance | MIXED AXIS (position_m vs member d2m) | inconsistent | same downstream_position_m; unplaced non-members are skipped (conservative) |
| 6 | _assembly_complete span = max-min | sign-independent | ok | s_m |
| 7 | member ordering (plan members = lane order) | no position-based ordering exists | n/a | audited |
| 8 | pacing (direct_target) | speed only | n/a | audited |
| 9 | traffic_shepherd/mixed_assignment.py | grep: no position use at all | n/a | audited |

**The r6 patch** (wm43_dispatcher_patch_r6.diff, single diff against the CURRENT traffic_shepherd/wall_group.py; sha256 3a6e4a5f97c000c987b1ac0783214d42c9341a9c53c97138ef33a6d3cc20f815; supersedes and replaces the rejected r5 diff) introduces the one documented helper downstream_position_m(frame) = -distance_to_merge_m (None for a missing/nonfinite coordinate), routes member positions AND non-member surveillance through it, keeps every existing comparison's semantics on s_m, and keeps the A5 loud contract: a member without a finite coordinate aborts with the declared member_position_unavailable (handback native); an unplaced non-member is skipped by the surveillance (conservative, documented).

**Patch tests (always-on, against a patched temp copy imported from the diff; no skip markers).**  tests/test_dispatcher_patch_r6.py: a fast non-member directly BEHIND the rear member (s in [rear-10, rear], speed > wall+2) trips non_member_rear_end_risk; the SAME vehicle directly AHEAD of the front member does NOT; both cases on the same edge AND straddling pre-merge/merge AND merge/downstream; the helper's sign (d2m -75.44 -> s +75.44) and None/NaN handling; every r5 geometry test re-run (same-edge span, both boundary straddles at ~3 m, odometer-offset trap -> true span 0, genuine 152.22 m cross-boundary violation aborts, loud failures); and the recorded a2_seed920246 replay at t=295.0 with the follower geometry: no member_span_exceeded and no rear-end abort (follower behind at a 27.5 m safe gap per the recorded wall_proximity rows), while a genuinely unsafe follower (inside the 10 m gap, faster) still aborts.

**Roots -> r6.**  Registered full runs root runs/active/wm40_wall_mechanics_v1_r6; smoke root runs/active/wm40_wall_mechanics_v1_r6_smoke.  The r3/r4/r5 evidence and the write-once SMOKE_VERDICT.json stay untouched.

**Commands (planner/owner; cwd D:\\research-dense).**

1. Apply the reviewed patch (with backup): output/action_model_v1/fixture/wm43_dispatcher_patch_r6.diff to traffic_shepherd/wall_group.py.
2. Preflight (read-only, must exit 0):
   C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/run_fixture.py preflight
3. Smoke (70 simulated seconds into the r6 smoke root; fresh SMOKE_VERDICT.json):
   C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/run_fixture.py smoke
4. Launch (unchanged CLI; smoke-gated + authorization-gated; runs into the registered _r6 root):
   C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/run_fixture.py launch --plan output/action_model_v1/fixture/PLAN.json --registration output/action_model_v1/fixture/REGISTRATION.md --authorization output/action_model_v1/fixture/AUTHORIZATION.json --workers 6

Budgets unchanged: 70 simulated seconds per smoke; 9000 simulated seconds campaign; smoke never data; zero retries.  Smoke coverage statement unchanged from A5: the smoke covers the patch wiring and same-edge assembly; the boundary crossing (~t>=288) and release are the r6 full A2 runs' re-validation.

## Amendment A7 (2026-09-27): r6 full-run loud aborts at the pre_merge/merge crossing - internal-lane coordinate coverage; roots -> _r7

**What happened.**  The r6 full run (with wm43_dispatcher_patch_r6.diff applied) passed 11/15 runs with zero collisions; 4/5 A2 runs aborted LOUDLY with member_position_unavailable at t=290-295 - exactly when the front member crossed pre_merge -> merge (a2_seed920244 t=295.0, 920246 t=290.0, 920247 t=293.0, 920248 t=294.0).  920245 released correctly at t=300: its crossing happened to complete between control ticks, so no control-tick frame ever sampled an internal lane.

**Root cause (coordinate coverage, not a check or dispatcher-semantics defect).**  traffic_shepherd/scenarios.observation_map maps only NORMAL lanes ('<edge>_<i>'); a member traversing the junction is on an INTERNAL lane (':<junction>_<n>_<m>'), whose distance_to_merge_m is None - and the r6-patched wall group correctly (per A5) raises the loud member_position_unavailable abort instead of silently skipping. Verified against the recorded telemetry: the abort ticks match the modeled boundary-crossing ticks per seed, and 920245's crossing fell between control ticks (hence its clean release at t=300).

**Fix (at the SOURCE of the coordinate - the mapping, not the abort).**

scenarios.observation_map is EXTENDED (normal-lane values bit-identical) to also map every internal junction lane of the mainline path (pre_merge -> merge and merge -> downstream), derived from the REAL network by a raw gzip-XML read of the preserved net file (no sumolib, no live TraCI): each <connection> between mainline edges names its via internal LANE; all lanes of that internal edge map to the collapsed boundary value their junction shares (pre_merge/merge -> 0.0; merge/downstream -> -75.44).  The four mapped internal lanes: :1780739017_1_0, :1780739017_1_1 (-> 0.0) and :13638027693_0_0, :13638027693_0_1 (-> -75.44).  The junction interior is collapsed onto the boundary point - monotone with travel, comparable with both neighbouring edges; the loud member_position_unavailable abort is kept for any lane STILL outside the map (genuinely unmapped lanes fail loudly; tested).

**Sensor-construction audit (repo-wide SumoLocalSensor call sites).**
- traffic_shepherd/workflow.py:255 - WITH observation_map -> covered.
- output/campaign_pilot_v1/suffix_runner.py:96 - WITH map -> covered.
- the fixture loop (wm43_loop.py) - WITH map -> covered.
- traffic_shepherd/mechanics.py:73 - SumoLocalSensor(api, config) with NO map: distance_to_merge_m is always None there, so a wall run through mechanics.py WOULD hit member_position_unavailable.  mechanics.py runs no walls (bounded mechanics CLI only); documented, not changed (out of this fixture's write scope).
- archive/circles_era/* mechanics scripts - historical, no walls.

**Tests (all offline).**  From the REAL network file: every internal lane on the mainline path is mapped with a coordinate CONTINUOUS with its neighbours (0.0 at the pre_merge/merge boundary on both sides; -75.44 at the merge/downstream boundary on both sides) and monotone along the path; normal-lane values bit-identical to the registered map; the recorded r6 abort ticks replayed with the member ON the internal lane -> no abort (true span ~3 m preserved); a genuinely unmapped lane still aborts loudly.

**Roots -> r7.**  Registered full runs root runs/active/wm40_wall_mechanics_v1_r7; smoke root runs/active/wm40_wall_mechanics_v1_r7_smoke.  The r6 evidence stays untouched.  Budgets unchanged (70 simulated seconds per smoke; 9000 simulated seconds campaign; zero retries; smoke never data).

**Commands (planner/owner; cwd D:\\research-dense).**

1. Apply this diff (with backup):
   C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/apply_patch_r7.py --repo-root D:/research-dense
2. Preflight (read-only, must exit 0):
   C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/run_fixture.py preflight
3. Smoke (70 simulated seconds into the r7 smoke root; fresh SMOKE_VERDICT.json):
   C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/run_fixture.py smoke
4. Launch (unchanged CLI; smoke-gated + authorization-gated; runs into the registered _r7 root):
   C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe output/action_model_v1/fixture/run_fixture.py launch --plan output/action_model_v1/fixture/PLAN.json --registration output/action_model_v1/fixture/REGISTRATION.md --authorization output/action_model_v1/fixture/AUTHORIZATION.json --workers 6
