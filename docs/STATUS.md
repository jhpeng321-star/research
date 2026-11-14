# Current status — 2026-09-08

**PPO interface line (latest)**: the ONE authorized 120 s mechanics
verification of the wall_ppo/2 duration semantics **PASSED** (ledger PP-10,
runs/active/ppo_mech_v2: 12/12 checks incl. admitted H30 EXTEND, exact 0.5 s
clock, clean handback). The wall_ppo/3 causal current-report selector is
implemented; PP-11's mocked-verification claim was corrected (CORR-19:
WAIT-only path only) and the four integration defects are REPAIRED in PP-12
(selector causal_selector/2, zero simulation): offer bound before
observations, execution commits exactly the approved offer, named
phase-aware comparator, populated mocked-environment chain verification
(876 tests + Ruff pass) — see
[output/ppo_selector_repair_v1/REPAIR_REPORT.md](../output/ppo_selector_repair_v1/REPAIR_REPORT.md).
The ONE authorized LIVE selector mechanics allocation then PASSED
(ledger PP-13, runs/active/ppo_mech_v3: 16/16 checks — live 4-pair
search, exact offer-to-admission binding, full H30 delivered,
clean handback, 0 collisions/teleports). Execution is verified;
benefit evaluation design lives in the OE-03 probabilistic
protocol; no training or corridor campaign is authorized.
Training and acceptance evaluation remain paused; any further simulation
needs new authorization. See the [current route](ROUTE.md) current-state
block and the ledger for details.

The latest owner-authorized A–D execution is complete. The original map suffices;
the finite gentle lifecycle passes both mechanics questions and delivers full
H20 walls in4107/4108 after report-based upstream selection. Two matched H5
release branches also deliver correctly. All retain−0.3m/s² voluntary braking,
0.15m/s³ ramp, identical damping handback and full2400s physical accounting.

Five distinct new corridor traces across two exposed seeds yield four harmful
outcomes and one nonbeneficial refusal; none qualifies or improves all continuous
costs with acceptable safety/service. H5 versusH20 costs28.03s/request in4107
while improving TTC/source/collisions, and barely changes4108's harm.642 tests,
Ruff and exact saved CIRCLES verification pass. Total new allocation:12480
simulated seconds including the preserved mechanics failures. Part E replication,
F fitting/extra data and G prospective efficacy were not activated. Acceptance
is sealed, training paused, and no successor runs remain active. See the
[complete findings](../output/lifecycle_repair_v2/REPORT.md) and [current route](ROUTE.md).

## Previous completed blocks

The owner-authorized [braking-limit/HOLD factorial](../archive/circles_era/docs/BRAKE_HOLD_TEST.md) is complete:
eight new 2400 s episodes, two 120 s mechanics fixtures, two reused cells and two
replay-equivalent cells. All 12 cells fail the unchanged physical criteria; only
4107/strong/H20 delivers its complete registered maneuver. The -0.3 limit passes
command checks but is not a guaranteed physical floor or consistently better.
HOLD has large, cap-dependent effects in 4107; 4108 aborts before HOLD and all
same-cap duration cells have exact actuator equivalence. Neither nominal H40
maneuver completes. These are two independent seeds, not twelve replications.
487 active tests, Ruff, exact saved replay, archive verification and mechanics
passed. See the [completed report](../output/brake_hold_v1/REPORT.md). The entire
19,440 simulated-second budget is finished; no successor run or training began.

Previous block: the separately authorized replacement gentle-wall block completed
both 2400 s cases, reusing verified damping arms. Neither qualifies. Seed 4107
delivered a 20 s HOLD, then aborted at 632 s; it saved 8.042 s/request but increased
TTC exposure by 0.020495 vehicle-seconds/request and recorded two collision
involvements. Seed 4108 aborted during assembly at 631 s, without HOLD, and worsened
braking, effort, service and TTC. All requested trips arrived; neither treatment
teleported. Actual wall-command bounds and handback contracts passed, which does
not establish physical safety or benefit. See the
[completed comparison and mechanism diagnosis](../output/gentle_wall_completion_v1/REPORT.md).
The prior failed allocations, metadata-only logging repair, **431 passing active
tests**, Ruff result and two passing 120 s mechanics fixtures remain preserved in
the [earlier record](../output/gentle_wall_v1/REPORT.md). No extra ablation or training ran.

**The formation/paired system works in bounded physical tests, but the intended
escalation benefit is not validated.** The fixed CIRCLES actor is unchanged.
All14 registered natural runs completed:8wall attempts,5full20s HOLDs,0attempts
meeting every predeclared benefit/safety condition. Later penetration, learning,
calibration, prospective and robustness stages remain gated and unexecuted.

The initial natural-panel suite had **344 passing tests** and full Ruff passes. Original
characterization, failed fixtures, repaired integration failures, exact full
baseline replay, every natural negative result and checkpoints are preserved.
No acceptance data was opened. The scheduler is implemented but abstains without
qualified artifacts. See the [current execution record](../archive/circles_era/docs/ESCALATION_EXECUTION_STATUS.md)
and [physical decision report](../output/escalation_execution/REPORT.md).

## What is installed

Active implementation version `traffic_shepherd 1.0.0` has one actor, one adapter,
one sensor contract and one CAV command owner. The permanent
[model and provenance](../baselines/circles/provenance.json) identify repository
`CIRCLES-consortium/algos-stack`, commit
`3b3ee3b91afdae93075c8b90334d65748cd9ff74`, source
`controllers/09_04_rl_finaldecn3p0_maxacc0p25/super_resolution.onnx`.

Whole-model SHA-256:
`762fde0763c693c0e88f41b98b22fc0a11135a312b78c4c567e6211a3035ccee`.
Adapter schema is `circles_direct_adapter/1`; its checksum is
`f027176fe59d255377a5e143d006eaf9f6e3024f63ff9ae54cc8de5e0a2652de`.
The exact MIT license, weights, embedded wrapper and singleton raw-SI behavior
are retained. Exact correspondence to a publication's evaluated checkpoint is
unverified; do not make that attribution.

The [adoption manifest](../baselines/circles/adoption_manifest.json) records the
reviewed active source/config/test identities without creating a git commit or
discarding the prior dirty worktree.

Native fallback, temporary wall ownership and return to identical damping are
implemented. HDVs retain IDM. The RSU accepts only CAV-origin reports and explicit
receipt/history metadata. The active formation/paired runner uses explicitly
registered experimental assignment; the complete repeated-opportunity scheduler
abstains until qualified feasibility and benefit artifacts exist. No manual
benefit score is represented as a trained gate. The earlier compact-only
`wall.py` diagnostic interface remains separate from the current formation path.

## Existing physical evidence

The [completed transfer report](../archive/pre_circles/docs/PUBLISHED_IDM_FIT_CHECK_2026-09-06.md)
and [raw paired outcomes](../archive/pre_circles/results/runs/probe/published_idm_fit_check_2026-09-06/summary.json)
are historical evidence for this adoption. The constrained adapter performed
poorly, particularly at higher penetration. It must not be silently restored.

For the direct p20 controller versus matched native:

- Seed 911001, mainline 2000 veh/h: severe braking **−18.76%**, acceleration
  effort **−5.29%**, service **+2.084 s/request**, source waiting **+0.0343 s/request**,
  TTC<3 s exposure **+53 vehicle-seconds**.
- Seed 911003, mainline 2200 veh/h: severe braking **−4.63%**, effort **+0.94%**,
  service **+1.474 s/request**, source waiting **+0.0427 s/request**,
  TTC<3 s exposure **+55.5 vehicle-seconds**.

Every requested vehicle arrived and no collision/teleport was recorded in these
cases. This is mixed development evidence from one seed per demand, not a safety
guarantee or generalization result. Only direct p20 was tested. The constrained
5–30% panel cannot establish direct cross-penetration performance.

## Adoption verification

- Pre-change full suite: **1,146 passed**. The original 184 historical test files
  remain unchanged with their original implementation in the archive.
- Active suite: **105 passed**, with Ruff passing; the final full-suite log is
  [postchange_pytest.log](../output/adoption_cleanup/postchange_pytest.log).
- [Exact numerical replay](../output/adoption_cleanup/full_direct_replay.json):
  all 32 permanent fixtures and **36,200** decisions across both exposed direct
  trajectories reproduce exactly, maximum absolute error **0**.
- [Physical ownership check](../runs/active/adoption_mechanics_2026-09-06/summary.json):
  two matched 10 s synthetic IDM runs pass damping/wall/native/re-engagement,
  unavailable-frame release, HDV isolation and safety-mode checks. After an
  identical prefix, cancelling an unfinished command changed acceleration at
  4 s from **−0.8 to +2.412 m/s²**. This demonstrates release changes actual physics;
  it is not evidence of traffic benefit or a smooth/safe wall release in general.
- [Real workflow smoke](../runs/active/adoption_workflow_smoke_2026-09-06/summary.json):
  one separately budgeted 30 s exposed case. Its [saved-prefix comparison](../output/adoption_cleanup/workflow_prefix_verification.json)
  matches raw inputs, model/issued actions, braking, effort, source waiting and
  TTC exactly; service differs by at most **1.1×10⁻¹⁴ vehicle-seconds** per interval.
  Only three active damping decisions occur in this prefix. No full-episode
  refactor-equivalence or efficacy claim follows from this short run.

The new mechanics allocations total **50 simulated seconds**, with no simulation
retry. A wrong seed-role argument was rejected before SUMO; the
[preflight amendment](../output/adoption_cleanup/runner_smoke_preflight_amendment.json)
preserves that failure and correction. Existing historical regression integration
tests are separately accounted by the archive verification, not counted as new
scientific samples.

Cross-review corrected integer seed identity, network-content hashing, ownership
of existing TraCI connections, wall authorization snapshot binding and taper
bounds. Active runtime imports exclude old learners. No acceptance outcomes were
read; archive integrity reads use opaque file hashes only.

## Archive and retained dependencies

[Archive index](../archive/pre_circles/ARCHIVE_INDEX.md) and
[complete relocation manifest](../archive/pre_circles/manifest.json) preserve
**59,846 files / 21,446,210,494 bytes**, including checkpoints, logs, negative
results, old configuration, original tests and 213 historical documents.
There is one physical archived data copy; large files were moved, not duplicated.

Historical `config`, `env`, `idm`, `policies`, `counterfactual`, `experiments`,
`tools` and root training/evaluation launchers moved under `archive/pre_circles/code/`.
New active modules directly contain useful extracted sensing, IDM scenario,
accounting, seed and startup logic; they do not wrap/import those packages.

Nineteen explicit NTFS aliases preserve concrete frozen-reader paths; they are
listed in the manifest and excluded from active navigation. One archive-only
test fixture copies five shared input files totaling 5,053 bytes to restore the
old relative-path regression assumption without editing tests or shared assets.
Legacy Ray/PyTorch requirements are retained only for the historical suite.
The active actor depends on its permanent model, local IDM template and shared
long SUMO network. Read-only full-trajectory verification intentionally reads
the two archived exposed trajectories; ordinary inference does not.

The first relocated historical suite had **1,143 passed / 3 path failures**;
its failed log is retained. After restoring the relative-path test fixture, the
unchanged full suite passed **1,146 tests**. Its three pre-existing SUMO tests
each run 37 s including warmup (**111 simulated seconds**, separately from the
50 s new mechanics checks). See the [historical regression verification](../output/adoption_cleanup/archive_regression_verification.json).
No original assertion was weakened. Post-test hash verification also detected
one test rewriting a generated path manifest; the [exact-byte restoration](../output/adoption_cleanup/archive_fixture_restoration.json)
preserves the altered test output separately and restores the original inventory
checksum rather than changing the frozen archive record.

The final [archive integrity report](../output/adoption_cleanup/archive_verification.json)
passes all **59,846 original file identities**, all 19 aliases and the small
copied shared-input fixture. It transparently combines the complete post-suite
SHA scan with the exact restoration and recheck of the sole changed file; the
failed scan remains preserved. Future historical tests use
[run_regressions.py](../archive/pre_circles/run_regressions.py), which copies small
writable source/scenario fixtures and retains one raw-data copy. Its targeted
two-test check passes with no archived input changes. Do not run tests directly
inside the frozen archive. Thirty-two orphaned root bytecode files were also
moved intact under `archive/pre_circles/transient/` with a separate cache manifest.

There are no remaining verification failures. This statement concerns software
and preservation checks; the traffic-benefit, sensing-realism and prospective
scientific questions above remain open.

## Next scientific task

Owner clarification: escalation can improve damping at any traffic state;
unresolved congestion and standalone damping superiority are not prerequisites.
Follow [the current route](ROUTE.md) and [ordered test plan](../archive/circles_era/docs/ESCALATION_TESTS.md).
The new escalation study tools register a bounded characterization panel and
evaluate report geometry and physical evidence. Poor damping results are retained
as characterization, not a trigger for another learner. Formation, wall effects,
both gates and prospective evaluation still need their physical evidence.
The catalog distinguishes specified future experiments from implemented tests.

Escalation test-build verification: **161 active tests passed**, Ruff and
model/import/document checks passed. One new bounded 30 s SUMO check found exact
telemetry and physical-summary equivalence with the saved reference after adding
offline density collection. The eight-job initial panel is registered but none
of its full trajectories have run. See the [build report](../output/escalation_test_build/BUILD_REPORT.md).
