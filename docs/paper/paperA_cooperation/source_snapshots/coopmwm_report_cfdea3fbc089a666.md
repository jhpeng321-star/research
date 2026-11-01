# S-COOPMWM — live checks

## OWNER19:34 — acceleration-space restart HOLD (fresh recovery)
- Verified cpuQ supervisor2146725 boot/start identity and SIGSTOPed that owned
  supervisor. A reached update19; eighteen update20 jobs remain incomplete at the
  snapshot and may drain. All A/B set-point results are superseded engineering runs.
- **811,640sim-s charged/678 reserved jobs**, retained in full. B/R remain held;
  no new physics or duplicate dispatch. Receipt `HOLD_ACCEL_1934.json`.
- New common A/B/R policy must use1s acceleration residuals on frozen MAPPO,
  unchanged safe_accel/CommandOwnership; B needs committed group acceleration plans.
- Probe6 trailing-slot trimming FAILED forecast parity: max1912.604veh-s on gpuA;
  both failed receipts retained. No trimmed implementation is admitted to training.
- Restart awaits corrected batch parity, CPU qualification, measured plan capacity,
  and a prospective exact budget/quota registration including superseded A charges.

## 19:14 — A resumed independently; B/R volume qualification remains held
- CPU25 **55PASS/zero skips/1.72s/zero SUMO**, including private multi-window
  forwarding, exact gate-to-closed-episode matching, R regime quotas, p=0 own-route
  density, restart checkpoint isolation, and never-attempted precharge adoption.
- Registered A-only amendment; **5,005,200 allocated/6,000,000 cap**,994,800 free.
  The extra86,400 retains all original B-u1..3 training charges; no refund.
- Owned old supervisor quiesced after all attempted jobs closed. Twelve reserved
  A-dev jobs had never CLAIMed and were adopted exactly once without recharge.
  Owned A-only supervisor2146725 now continues from existing checkpoints; B/R held.
- B/R quota/mixture/matching code is CPU-qualified, but guided release remains gated
  on measured GPU capacity and prospective numeric registration. No efficacy claim.
- Fused1600-slot two-window K8 parity passes both hosts (max0.0000749veh-s), but
  capacity is31.06/28.65queries/hour. gpuR four-windowK8 OOM is retained.
  A960-slot source-capacity candidate changed paired costs by3.506veh-s and was
  **rejected before training**. Next registered probe retains source draws/capacity,
  trims universally absent trailing slots only during the one-step model forward,
  and restores them before inherited insertion/observer. Original deadline retained.

## OWNER18:54 — guidance-volume HOLD (18:55 recovery)
- Verified owned cpuQ supervisor1945256 boot/start and paused it before B update4.
  A has update4; B updates1–3 are superseded engineering runs (all bytes/charges retained).
  R has not started. The shared supervisor pause temporarily also holds A scheduling.
- Charge307,640sim-s/258 reserved jobs at hold;24 already-reserved A dev jobs may drain.
  No new SUMO dispatch. `HOLD_1854.json` records the outstanding job identities.
- Both existing owned GPU services remain (gpuA520375/gpuR954996); no duplicate service.
  The preserved K8 whole-future batch probe passed max cost error0.00065279veh-s,
  peak allocated2.43GB. Multi-window throughput qualification/gating is next.
- Latest18:54 amendment supersedes the earlier continuation: B/R restart at MAPPO u0
  only after registered execution-count gating, measured batch capacity and matched R counts.

## DIAGNOSIS — 2026-11-01 18:47, OWNER18:38 fast path
**No action-injection/roster/reversion bug found; release the engineering hold.**
Window01 `.139` has target-minus-MAPPO speed **+5.70434m/s**, tau−0.000249751s;
`.154` −0.076919m/s, tau−0.202225s. Window00 `ramp.72` −3.53868m/s,
tau+0.279726s; `.214` +0.477607m/s, tau+0.00685912s. Thus targets are not uniformly
tiny. The decoded requests reach the inherited WM callback, with both members
present/CAV. For `.139`, initial bounded raw action2.60000 vs MAPPO2.59707m/s²
becomes **the same safe action−0.265855m/s²**; all traced12s positions/speeds
remain identical despite differing raw requests. This is safety/plant saturation,
not a missing action. Non-requested CAVs run the frozen deterministic MAPPO
bridge, not native/IDM; a real CPU callback test verifies standalone MAPPO equality
and nonzero speed/position response to an unsaturated braking set-point.
Inherited available O1/O3/O4 proposals DO alter `.139` trajectories (maximum12s
position difference8.569/8.228/8.228m); H60 single-draw costs are distinct from
policy/MAPPO (O3/O4 legitimately coincide in this geometry). O2 is unavailable
here and retains its availability mask. Window00 also responds (ramp12s maximum
position difference1.146m), with no inherited upstream option available.
`ACTION_DELTAS_1838.json`, `ACTION_PROBE2_RESULT_00/01.json`, `DIAGNOSIS_1838.json`
retain decoded requests, every0.5s accel/speed/position and before/after-safety
values. **CPU WM regression1PASS/0skips, CUDA disabled, SUMO forbidden.** CPU22's
wrong-runtime fixture failure and first focused fixture failure are preserved;
the qualified WM library fixture passes in `CPU_WM_ACTION_RESPONSE2_1838.json`.
Exact twins and original K8 results are reused; these diagnostic rerolls add0SUMO.
The late exact-SUMO divergence (H60/H120 near0; H300/H600 +1699/+5723veh-s)
is compatible with amplification of small execution differences, but two windows
do not establish an empirical variance reduction or WM accuracy/scale CI.
Conditional WM variance alone cannot support the paper's noise-reduction claim.

### Suggestion-rate diagnosis and prospective release
Original B-u1 used1/900 dense and1/3600 ordinary query probability, before
availability was known: **1 query,0 accepted,0 executed**. Both GPUs' real available
H300 K8+canary calls took213.67/234.79s (aggregate32.18queries/hour), explaining
the throughput constraint. Outcome-independent numeric fit raises the shared
B/R dense rate to**0.0022449262636300934**, ordinary**0.0005612315659075234**
(2.02× historical rates). It uses309dense+219ordinary eligible B-u1 windows and
274.0465s original48-episode span scaled4→12workers; no efficacy-based sampling.
Numeric registration/binding completed; **18:48 same owned supervisor resumed**
(`RELEASE_1838.json`, identity checked, stateS), preserving original u1
optimizer/RNG/checkpoints and existing GPU queues. Exact allocation remains4,918,800/6,000,000sim-s,
1,081,200unallocated; current charge83,200. All final-test/checkpoint rules remain.

## OWNER18:34 — engineering HOLD (fresh-session recovery)
- Verified supervisor PID1945256 boot/start identity; paused it with SIGSTOP at the
  update-2 sampling gate. B/R are held at update1/not-started respectively. The
  interleaved supervisor also pauses A until safe arm-independent continuation exists.
- All70 jobs are closed;83,200sim-s charged. No new SUMO, duplicated job or checkpoint
  replacement. Existing GPU inference/exact receipts are retained for wiring diagnosis.
- `coopmwm/HOLD_1834.json` records the owned pause. No numeric sampling binding will
  release learning before action-response regression and a registered repair amendment.

## 2026-11-01 18:31 — source-prior ABI qualified; primary checkpoints preserved
- CPU21: **50PASS/zero skips/1.71s/zero SUMO**, including non300-clock integer source
  sufficient statistics, real frozen MAPPO parity and per-future proposal parity.
- First availability-qualified H300 probe exposed a pre-rollout integer-count API fault;
  failed dual1 services ended. Both remote inventories verified no surviving task GPU process.
- Fresh immutable dual2 services: gpuA ownedPID520375/gpuR ownedPID954996; original
  GPU deadline retained. They now qualify/timestamp an actual available K8 H300 proposal.
- Source encoding uses rounded FIT-prior rate×current-clock pseudo-counts, NOT measured
  source requests. This current-clock pseudo-exposure is weaker than retaining the
  donor posterior exposure; no causal/full-history source-conditioning claim is made.
- All u1 weights/Adam/RNG, original queue, completed forecasts and exact twins are kept.
  **83,200sim-s charged /70closedjobs**, no physical duplicate; u2 remains gated on
  numeric registered sampling. Allocated4,918,800/cap6,000,000 is unchanged.
- Owned30s publisherPID1116 will publish and commit E2 windows/full agreement CIs,
  every UPDATE, every4-update TRAIN check, dev/final and STOP receipts for up to24h.

## 2026-11-01 18:10 — owner17:55 dual-GPU amendment active
- Registered unchanged **4,918,800 allocated /6,000,000 cap**,1,081,200 unallocated.
- CPU18/19: **47PASS/zero skips/zero SUMO**; finalCPU19 2.87s. Design remains70lines.
- Both immutable FULL-v4 services are ready: gpuA ownedPID516897, gpuR ownedPID951313;
  each has an8GiB cap. Original failed bridge/startup receipts remain untouched.
- Same registered first-window K8 diagnostic is running on both GPUs for cross-host
  parity; next is one H300 proposal timing per host and outcome-independent sampling fit.
- cpuQ will adopt all62finished jobs/74,400charged;12nice19training workers plus2E2
  workers, shared serialized accounting. E2 now runs beside PPO and cannot gate learning.
- A/B/R final checkpoints, primary-first once-only test and remaining seed2 plan unchanged.

### 18:15 — first-update ETA
- Both primaryu1 checkpoints/UPDATE receipts exist; supervisor ownedPID1945256 is alive.
  The62closed jobs were adopted without new SUMO;74,400sim-s remains charged.
- B u1 query1/accepted0/executed0; no learned-cooperation efficacy claim.
- Initial48-episode rollout closes span254.33s with4workers;12worker CPU scaling plus
  PPO/check overhead gives a provisional **1–3h after numeric u2 sampling release**
  for both primaryu24 checkpoints. This is a planning range, not measured end-to-end
  throughput; update after u2 elapsed time and WM queue latency are available.
- u2 cannot launch physics until the outcome-independent sampling fit is numerically
  registered and hash-bound. First dual-GPU E2 forecast is still running; no agreement result yet.

## 2026-11-01 18:01 — fresh recovery and frozen reducer repair
- Recovered all62 closed jobs /74,400sim-s; no live task supervisor or GPU service.
- Prior GPU startup failed because `inspect` resolved an exec-created reducer against
  the producer's shared synthetic filename. Explicit qualified22cc32f6 reducer source
  is now frozen and bytecode-matched before adapting only the horizon tuple.
- CPU17: **44 PASS / zero skips /1.72s / zero SUMO**, including the actual frozen
  reducer, incorrect-source rejection and MAPPO command parity. Failure receipts retained.
- Owner17:55 dual-GPU service/sampling amendment is next; no finished physics repeated.

## 2026-11-01 17:48 — recovery and random-guide amendment
- Registered owner17:42 arm R before any UPDATE/E2 outcome; registration and exact quote
  preserved in `BUDGET_AMENDMENT_1742.json` and commit `bb9d6fdb`.
- Revised allocation **4,918,800 / 6,000,000 sim-s**, unallocated1,081,200; A/B seeds1+2,
  R primary only, fixedu24; seed3 dropped. Primary A/B test still precedes R/replications.
- Recovered owned supervisor1877910; let all **62/62 jobs close** before controlled adoption.
  **74,400 sim-s charged** includes smoke,12TRAIN-check controls and48initial A/B episodes.
- At the E2 forecast wait, verified boot/start identity and no active/queued physics;
  stopped only that supervisor. Original jobs/checkpoints/charge ledger/WM queue survive.
  Receipt `QUIESCE_1742.json`; existing GPU bridge retained, no duplicate dispatch.
- R implementation/CPU qualification underway; no training update or R physics yet.

### 17:53 — arm R CPU qualification PASS
- CPU15/16 both **43 PASS / zero skips / zero SUMO**; final CPU16 took1.66s.
- Covers actual frozen MAPPO parity; uniform available inherited options without WM;
  mixture mass/imitation/SUMO-only reward; per-type/regime trust; safe completed-job reuse;
  fixed R final-test one-use gate, reused primary controls and paired B-R bootstrap.
- Final70-line design SHA `02879dd48ed26b466575891df920c1cbfad9da848a841c26af56c26777466cea`.
- Noise diagnostics report C_G, each D_i, sum D_i and D_G variance from the original K8
  draws. No superseded reward-combination weight is used; no inference is repeated.
- GPUA snapshot17:50:1,359MiB/49,140MiB,32%util; original service waiting on admission,
  WMPLAN priority retained. E2/first-update results and a measured ETA are still pending.
- Exact registered allocation remains4,918,800/6M; `READINESS_1742.json` binds CPU/design.

## 2026-11-01 17:41 — full-episode zero-update smoke PASS
- Owned cpuQ smoke PID1872644 finished; both registered1200s episodes closed.
- A and B metrics exactly equal reused bitwise MAPPO baseline at J600/J900;
  A/B final full-state fingerprints equal. **2,400 sim-s charged**, new MAPPO simulation0.
- Collisions0, teleports0 in both episodes; this is mechanics/parity, not learned performance.
- One-use CPU dispatch, GPU bridge PID41056 and immutable bindings are preserved;
  no training update yet. Primary paired training dispatch follows this receipt.
- E2 agreement/noise is diagnostic only; it will use preselected initial TRAIN windows.

## 17:27 refinement — CPU qualification and prospective registration
- Recovered at17:29: no task runtime/jobs/services on cpuQ/gpuA/gpuR; no duplicate dispatch.
- CPU13: **40 PASS, zero skips, 1.36 s, zero SUMO**; CPU11/12 failed receipts preserved.
- Eight option-type x observable-regime trust buckets use normal-means empirical Bayes:
  between-bucket variance and bootstrap uncertainty determine shrinkage toward pooled evidence.
- Independent random query precedes winner-type selection; execute with that bucket's beta.
  PPO uses the logged query measure, including rejected/declined query atoms; reward remains SUMO-only.
- Observable merge-dense rule projects the union of LEVERPROBE's four recorded upper-quartile
  bands, with ramp and cross-lane CAV pair; no true demand/share input or new behavioural cutoff.
- Trust evidence is a critic-based selected-type residual contrast against own actions in the
  same regime. It is **not** an exact type-specific causal effect; nonquery winner labels are unobserved.
- Cap remains **6,000,000 sim-s**, allocated5,738,400, unallocated261,600. Three paired training
  seeds and all primary/replication final-test liabilities remain protected.
- Source/design hashes are frozen in CPU13; exact amendment: `coopmwm/BUDGET_AMENDMENT_1727.json`.
- CPU14 repeats40PASS/0skips/1.35s; deployment now copies the tested immutable snapshot,
  avoiding concurrently changing peer tooling. Initial v1 staging failure was before binding/SUMO.
- Bound cpuQ `/home/ubuntu/coopmwm20261101_v2` and gpuA `/root/coopmwm20261101_v2`;
  source/cache/runtime/checkpoint hashes verified;4nice19workers, task simulation still0 at binding.
- Next: one full-episode zero-update parity smoke per arm, then primary training/E2.

## 17:12 redesign — implementation CPU-verified before physics
- Arm B now uses WM-guided exploration and **SUMO-only team reward/advantage**.
- FULL-v4 K8 H300 compares inherited O1–O4 set-point proposals with policy/MAPPO.
- Logged random routing permits exact importance correction on the augmented
  behaviour mixture, including deterministic proposal atoms; imitation weight is beta.
- Beta starts at1 and uses a monotone, scale-free world-bootstrap SUMO-evidence rule.
- Primary A/B episodes share an interleaved admitted worker pool; fixed primary u24
  test precedes replications, with MAPPO/native final-panel controls reused.
- CPU9: **36 PASS, zero skips, 1.32 s, zero SUMO**, including actual frozen MAPPO
  parity, proposal-mixture mass, SUMO reward isolation, adaptive trust and leakage checks.
- CPU8's failed syntax/roundoff checks are preserved; fixed before CPU9 and physics.
- E2 agreement/variance remains diagnostic; historical reward-credit gates below are superseded.
- Registration/runtime deployment is next; no task simulation or update dispatched yet.

## 17:07 amendment — registered before physics
- MAIN excludes unreachable cpuP; existing registry/protocol and cpuQ/gpuA/gpuR audits qualify.
- Recovery found no task runtime, training jobs or GPU services on admitted hosts; SUMO remains zero.
- Stopped the superseded metadata-only waiter by verified owned PID46700.
- Primary A/B train first in parallel. Run primary four-arm final panel once at both u24 checkpoints;
  train replications afterwards and evaluate A/B once per seed with MAPPO/native reused.
- Added replication-test liability172,800: allocated **5,738,400 / 6,000,000**, unallocated261,600.
- `BUDGET_AMENDMENT_1707.json` and the shared registry record this prospective amendment.
- Historical entries below describe the prior gate/order; the 17:07 amendment supersedes them.

## Recovery and prospective registration — 2026-11-01
- Preserved recovered work in `0a3df807`; previous CPU5 sources matched 108 staged files.
- No COOPMWM runtime or processes existed at recovery; task SUMO/training updates: zero.
- CPU6: **29 PASS, zero skips, 1.35 s, zero SUMO**, with simulator subprocess creation forbidden.
- Includes real speed-only MAPPO169 `d9042cc5` command parity; WM/exact credit fixture algebra;
  fixed ramp; observable-only/no-leakage checks; dual seed guards and three-seed/final-panel roles.
- Exact prospective plan appended to the shared `REGISTRATION.md` before any outcome.
- Three independent training seeds fit: 24 PPO updates per arm per seed; 4,147,200 training sim-s.
- Total allocated 5,565,600 / 6,000,000 sim-s; 434,400 remains unallocated, not automatically spent.
- Final-test liability **172,800 sim-s precharged**: fresh 989600–989635, 36 worlds, four arms.
- Panel canonical SHA256: `69b8a300eab45fb8208bd4bc66cefd32c6ebd16c71c23b5aac8567d41ee3c182`.
- Final checkpoints fixed to primary-seed update24; additional seeds are replications, never selectors.
- Panel stays unopened for outcomes until all training ends; one complete evaluation, no reruns.
- Registry/protocol audit: 538 safe files, no conflicts. Available execution-host filename audits:
  cpuQ 337,849; gpuA 895,554; gpuR 328,139; all zero exposures.
- Additional cpuP audit could not complete: SSH connection closed. cpuP is not admitted;
  full cross-host exposure qualification is pending resolution before physics.

## Registered credit/noise check
- FULL-v4 `22cc32f6`, original WMTEACH K8 paired-future executor, H60/120/300/600.
- Twelve randomly selected initial TRAIN windows, one per cell, selected before credit outcomes.
- Report D_i/C_G sign agreement, world-bootstrap 95% CIs and learned credit scale immediately.
- Every horizon must pass both sign lower bounds >0.5; primary H300 if admitted.
- No passing horizon stops B. Calibration/weights remain frozen for all three training seeds.
- Compare per-window K8 variance of the same joint-action team-return advantage with paired replay
  credit variance, in veh-s². A fixed critic baseline does not change the conditional variance.
- CRN pairing may cancel shared hidden-driver/arrival noise; this is a hypothesis to measure,
  not an observed noise reduction. Conditional WM variance is not empirical SUMO variance.
- Report actual across-TRAIN PPO advantage variance separately; no noise/performance result yet.

## Next checks
- Immutable final runtime/CPU/source binding and complete exposure admission.
- One full-episode smoke per arm, against the reused bitwise MAPPO cache.
- WM/exact gate, then every-four-update TRAIN-cell checks and first registered dev check.

## CPU7 — final workflow implementation
- **30 PASS, zero skips, 1.28 s, zero SUMO**; receipt `coopmwm/CPU_coopmwm20261101_cpu7.json`.
- Runner now namespaces all three training-seed checkpoint/job/check receipts and fixes
  seed0/update24 for the final test. The final-test gate requires all training completions;
  missing B finals produce an explicit incomplete test, never a fallback checkpoint.
- Native reference reuses the genuine native plant. Final paired bootstrap supports all36 worlds.
- H600 uses the original producer with H600/1600 entrant capacity and four indexed K2 batches
  assembled into complete paired K8. Draw identities/executor/reducer are retained; actual GPU
  memory/forecast parity remains a pre-inference check, not established by structural CPU tests.
- Actual team-only GAE variance is logged separately from conditional WM credit noise.
- cpuP SSH still closes at the second bounded metadata-only probe. No runtime binding,
  owned training/GPU service, smoke, credit agreement or learning update has been dispatched.

## Live publication — 2026-11-01T10:15:00.034770+00:00
- Charged 74,400/6,000,000 sim-s; closed jobs 62/62; pending WM 1.

### S0_A_UPDATE_001.json
- A update1, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.66493e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_001.json`.

### S0_B_UPDATE_001.json
- B update1, seed20261101; SUMO-only reward; queries/accepted/executed 1/0/0; next pooled beta 1.0.
- Per-bucket trust: O1|merge-dense=1; O1|ordinary=1; O2|merge-dense=1; O2|ordinary=1; O3|merge-dense=1; O3|ordinary=1; O4|merge-dense=1; O4|ordinary=1.
- TRAIN GAE variance 4.69666e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_B_UPDATE_001.json`.

## Live publication — 2026-11-01T10:20:31.236333+00:00
- Charged 79,200/6,000,000 sim-s; closed jobs 66/66; pending WM 7.

### CREDIT_WINDOW_00.json
- E2 window world986000 t600: exact quartet complete.
- H120 conditional WM K8 variance: joint-team advantage 801277, paired C_G 0 veh-s²; not empirical SUMO noise.
- H300 conditional WM K8 variance: joint-team advantage 9.8999e+06, paired C_G 0 veh-s²; not empirical SUMO noise.
- H60 conditional WM K8 variance: joint-team advantage 86872.7, paired C_G 0 veh-s²; not empirical SUMO noise.
- H600 conditional WM K8 variance: joint-team advantage 5.84757e+07, paired C_G 0 veh-s²; not empirical SUMO noise.

## Live publication — 2026-11-01T10:29:51.501844+00:00
- Charged 83,200/6,000,000 sim-s; closed jobs 70/70; pending WM 7.

### CREDIT_WINDOW_01.json
- E2 window world986007 t400: exact quartet complete.
- H120 conditional WM K8 variance: joint-team advantage 265240, paired C_G 0 veh-s²; not empirical SUMO noise.
- H120 WM/exact C_G 0/-7.09977e-07; D_i [0.0, -0.001434922218322754]/[-0.07634644157951698, 7.099770300555974e-07]; zero C_G is a tie, not sign evidence.
- H300 conditional WM K8 variance: joint-team advantage 3.5524e+06, paired C_G 0 veh-s²; not empirical SUMO noise.
- H300 WM/exact C_G 0/-7.52552e-07; D_i [0.0, -0.001447141170501709]/[1698.600333714603, 7.525522960349917e-07]; zero C_G is a tie, not sign evidence.
- H60 conditional WM K8 variance: joint-team advantage 31878, paired C_G 0 veh-s²; not empirical SUMO noise.
- H60 WM/exact C_G 0/9.01442e-07; D_i [0.0, -0.0004132091999053955]/[0.11916434977320023, -9.014420356834307e-07]; zero C_G is a tie, not sign evidence.
- H600 conditional WM K8 variance: joint-team advantage 4.016e+07, paired C_G 0 veh-s²; not empirical SUMO noise.
- H600 WM/exact C_G 0/-7.52567e-07; D_i [0.0, -0.001447141170501709]/[5723.486351683328, 7.525668479502201e-07]; zero C_G is a tie, not sign evidence.

## Live publication — 2026-11-01T10:49:25.085948+00:00
- Charged 145,600/6,000,000 sim-s; closed jobs 98/122; pending WM 1.

### CREDIT_WINDOW_02.json
- E2 window world986012 t600: exact quartet complete.
- H120 conditional WM K8 variance: joint-team advantage 1.02526e+06, paired C_G 1.81899e-12 veh-s²; not empirical SUMO noise.
- H120 WM/exact C_G 4.76837e-07/0; D_i [0.0046305060386657715, 0.0]/[-0.008384575048694387, 0.0]; zero C_G is a tie, not sign evidence.
- H300 conditional WM K8 variance: joint-team advantage 1.10728e+07, paired C_G 1.81899e-12 veh-s²; not empirical SUMO noise.
- H300 WM/exact C_G 4.76837e-07/0; D_i [0.004627645015716553, 0.0]/[-0.008384783359360881, 0.0]; zero C_G is a tie, not sign evidence.
- H60 conditional WM K8 variance: joint-team advantage 96105.6, paired C_G 1.81899e-12 veh-s²; not empirical SUMO noise.
- H60 WM/exact C_G 4.76837e-07/0; D_i [0.004109740257263184, 0.0]/[0.00042830735037568957, 0.0]; zero C_G is a tie, not sign evidence.
- H600 conditional WM K8 variance: joint-team advantage 7.85469e+07, paired C_G 1.81899e-12 veh-s²; not empirical SUMO noise.
- H600 WM/exact C_G 4.76837e-07/0; D_i [0.004627645015716553, 0.0]/[6.07445183393429, 0.0]; zero C_G is a tie, not sign evidence.

## Live publication — 2026-11-01T10:50:27.621315+00:00
- Charged 145,600/6,000,000 sim-s; closed jobs 122/122; pending WM 1.

### S0_A_UPDATE_002.json
- A update2, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.70697e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_002.json`.

## Live publication — 2026-11-01T10:51:00.705174+00:00
- Charged 203,200/6,000,000 sim-s; closed jobs 128/170; pending WM 1.

### S0_B_UPDATE_002.json
- B update2, seed20261101; SUMO-only reward; queries/accepted/executed 2/0/0; next pooled beta 1.0.
- Per-bucket trust: O1|merge-dense=1; O1|ordinary=1; O2|merge-dense=1; O2|ordinary=1; O3|merge-dense=1; O3|ordinary=1; O4|merge-dense=1; O4|ordinary=1.
- TRAIN GAE variance 4.77848e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_B_UPDATE_002.json`.

## Live publication — 2026-11-01T10:52:34.609181+00:00
- Charged 203,200/6,000,000 sim-s; closed jobs 170/170; pending WM 1.

### S0_A_UPDATE_003.json
- A update3, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.429e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_003.json`.

## Live publication — 2026-11-01T10:53:06.635788+00:00
- Charged 260,800/6,000,000 sim-s; closed jobs 172/218; pending WM 1.

### S0_B_UPDATE_003.json
- B update3, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 1.0.
- Per-bucket trust: O1|merge-dense=1; O1|ordinary=1; O2|merge-dense=1; O2|ordinary=1; O3|merge-dense=1; O3|ordinary=1; O4|merge-dense=1; O4|ordinary=1.
- TRAIN GAE variance 4.50958e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_B_UPDATE_003.json`.

## Live publication — 2026-11-01T10:54:40.121508+00:00
- Charged 264,440/6,000,000 sim-s; closed jobs 222/222; pending WM 1.

### CREDIT_WINDOW_03.json
- E2 window world986019 t310: exact quartet complete.
- H120 conditional WM K8 variance: joint-team advantage 966381, paired C_G 7.21099e-12 veh-s²; not empirical SUMO noise.
- H120 WM/exact C_G -1.19209e-06/0; D_i [8.344650268554688e-07, 4.76837158203125e-07]/[0.0, 0.0]; zero C_G is a tie, not sign evidence.
- H300 conditional WM K8 variance: joint-team advantage 1.51445e+07, paired C_G 7.21099e-12 veh-s²; not empirical SUMO noise.
- H300 WM/exact C_G -1.19209e-06/0; D_i [8.344650268554688e-07, 4.76837158203125e-07]/[0.0, 0.0]; zero C_G is a tie, not sign evidence.
- H60 conditional WM K8 variance: joint-team advantage 55320.5, paired C_G 7.21099e-12 veh-s²; not empirical SUMO noise.
- H60 WM/exact C_G -1.19209e-06/0; D_i [8.344650268554688e-07, 4.76837158203125e-07]/[0.0, 0.0]; zero C_G is a tie, not sign evidence.
- H600 conditional WM K8 variance: joint-team advantage 1.31309e+08, paired C_G 7.21099e-12 veh-s²; not empirical SUMO noise.
- H600 WM/exact C_G -1.19209e-06/0; D_i [8.344650268554688e-07, 4.76837158203125e-07]/[0.0, 0.0]; zero C_G is a tie, not sign evidence.

## Live publication — 2026-11-01T10:55:12.537480+00:00
- Charged 278,840/6,000,000 sim-s; closed jobs 225/234; pending WM 1.

### S0_A_UPDATE_004.json
- A update4, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 5.03716e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_004.json`.

## Live publication — 2026-11-01T11:15:29.022889+00:00
- Charged 336,440/6,000,000 sim-s; closed jobs 258/282; pending WM 0.

### S0_A_CHECK_004.json
- A_minus_MAPPO: {"bootstrap":"10000 whole-world paired resamples; RNG20261101; negative delay is better","learned_minus":"mappo","metrics":{"600_collisions":{"ci95":[0.0,0.0],"mean":0.0},"600_hard_brake":{"ci95":[-2.7510416666666657,7.5],"mean":2.2083333333333335},"600_team_delay_vs":{"ci95":[-1249.1101470572473,3372.3958383866448],"mean":1203.3873745997926},"600_teleports":{"ci95":[0.0,0.0],"mean":0.0},"600_ttc_below_3_vs":{"ci95":[-11.709375,97.91770833333332],"mean":44.0},"900_collisions":{"ci95":[0.0,0.0],"mean":0.0},"900_hard_brake":{"ci95":[-7.5,4.583333333333333],"mean":-1.875},"900_team_delay_vs":{"ci95":[-1868.5913824377792,4332.582430582843],"mean":1372.7795884038114},"900_teleports":{"ci95":[0.0,0.0],"mean":0.0},"900_ttc_below_3_vs":{"ci95":[-55.21041666666667,109.5],"mean":28.833333333333332}},"worlds":12}

### S0_A_DEV_004.json
- A_minus_MAPPO: {"bootstrap":"10000 whole-world paired resamples; RNG20261101; negative delay is better","learned_minus":"mappo","metrics":{"600_collisions":{"ci95":[0.0,0.0],"mean":0.0},"600_hard_brake":{"ci95":[-4.645833333333333,2.625],"mean":-0.875},"600_team_delay_vs":{"ci95":[-2008.6695155343664,953.7011788286733],"mean":-526.333169436575},"600_teleports":{"ci95":[0.0,0.0],"mean":0.0},"600_ttc_below_3_vs":{"ci95":[-90.14583333333333,9.750520833333326],"mean":-40.354166666666664},"900_collisions":{"ci95":[0.0,0.0],"mean":0.0},"900_hard_brake":{"ci95":[-4.083333333333333,4.416666666666667],"mean":0.2916666666666667},"900_team_delay_vs":{"ci95":[-2525.7345253020353,2240.0123362352256],"mean":-184.73109833748217},"900_teleports":{"ci95":[0.0,0.0],"mean":0.0},"900_ttc_below_3_vs":{"ci95":[-107.08333333333333,23.75104166666665],"mean":-43.104166666666664}},"worlds":24}

## Live publication — 2026-11-01T11:16:31.748572+00:00
- Charged 365,240/6,000,000 sim-s; closed jobs 282/306; pending WM 0.

### S0_A_UPDATE_005.json
- A update5, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.64449e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_005.json`.

## Live publication — 2026-11-01T11:18:05.307340+00:00
- Charged 394,040/6,000,000 sim-s; closed jobs 307/330; pending WM 0.

### S0_A_UPDATE_006.json
- A update6, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.62821e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_006.json`.

## Live publication — 2026-11-01T11:19:08.132519+00:00
- Charged 422,840/6,000,000 sim-s; closed jobs 330/354; pending WM 0.

### S0_A_UPDATE_007.json
- A update7, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.70447e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_007.json`.

## Live publication — 2026-11-01T11:20:10.926285+00:00
- Charged 437,240/6,000,000 sim-s; closed jobs 354/366; pending WM 0.

### S0_A_UPDATE_008.json
- A update8, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.90423e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_008.json`.

## Live publication — 2026-11-01T11:21:13.725511+00:00
- Charged 466,040/6,000,000 sim-s; closed jobs 368/390; pending WM 0.

### S0_A_CHECK_008.json
- A_minus_MAPPO: {"bootstrap":"10000 whole-world paired resamples; RNG20261101; negative delay is better","learned_minus":"mappo","metrics":{"600_collisions":{"ci95":[0.0,0.0],"mean":0.0},"600_hard_brake":{"ci95":[-1.9166666666666667,8.0],"mean":3.2083333333333335},"600_team_delay_vs":{"ci95":[-2134.4230452651373,2514.0648813466746],"mean":366.9245178487299},"600_teleports":{"ci95":[0.0,0.0],"mean":0.0},"600_ttc_below_3_vs":{"ci95":[-39.376041666666666,84.3770833333333],"mean":23.458333333333332},"900_collisions":{"ci95":[0.0,0.0],"mean":0.0},"900_hard_brake":{"ci95":[-6.375,8.0],"mean":1.1666666666666667},"900_team_delay_vs":{"ci95":[-2310.9142790688265,4254.126301910809],"mean":1194.8769365963044},"900_teleports":{"ci95":[0.0,0.0],"mean":0.0},"900_ttc_below_3_vs":{"ci95":[-61.833333333333336,125.25],"mean":36.375}},"worlds":12}

## Live publication — 2026-11-01T11:22:16.980124+00:00
- Charged 494,840/6,000,000 sim-s; closed jobs 390/414; pending WM 0.

### S0_A_UPDATE_009.json
- A update9, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.50355e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_009.json`.

## Live publication — 2026-11-01T11:23:19.724350+00:00
- Charged 523,640/6,000,000 sim-s; closed jobs 414/438; pending WM 0.

### S0_A_UPDATE_010.json
- A update10, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 5.15375e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_010.json`.

## Live publication — 2026-11-01T11:24:56.588634+00:00
- Charged 552,440/6,000,000 sim-s; closed jobs 442/462; pending WM 0.

### S0_A_UPDATE_011.json
- A update11, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 5.05183e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_011.json`.

## Live publication — 2026-11-01T11:25:59.568360+00:00
- Charged 566,840/6,000,000 sim-s; closed jobs 462/474; pending WM 0.

### S0_A_UPDATE_012.json
- A update12, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.69977e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_012.json`.

## Live publication — 2026-11-01T11:26:31.329316+00:00
- Charged 595,640/6,000,000 sim-s; closed jobs 474/498; pending WM 0.

### S0_A_CHECK_012.json
- A_minus_MAPPO: {"bootstrap":"10000 whole-world paired resamples; RNG20261101; negative delay is better","learned_minus":"mappo","metrics":{"600_collisions":{"ci95":[0.0,0.0],"mean":0.0},"600_hard_brake":{"ci95":[-3.2083333333333335,4.583333333333333],"mean":0.375},"600_team_delay_vs":{"ci95":[-661.2338917042277,2949.737879335951],"mean":1282.3524654967944},"600_teleports":{"ci95":[0.0,0.0],"mean":0.0},"600_ttc_below_3_vs":{"ci95":[6.833333333333333,98.54270833333332],"mean":50.083333333333336},"900_collisions":{"ci95":[0.0,0.0],"mean":0.0},"900_hard_brake":{"ci95":[-6.208333333333333,4.6260416666666515],"mean":-1.125},"900_team_delay_vs":{"ci95":[-41.61367904337277,4606.31203827328],"mean":2363.594791228685},"900_teleports":{"ci95":[0.0,0.0],"mean":0.0},"900_ttc_below_3_vs":{"ci95":[-31.46145833333333,117.54166666666667],"mean":42.791666666666664}},"worlds":12}

## Live publication — 2026-11-01T11:28:05.670888+00:00
- Charged 624,440/6,000,000 sim-s; closed jobs 499/522; pending WM 0.

### S0_A_UPDATE_013.json
- A update13, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 5.2456e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_013.json`.

## Live publication — 2026-11-01T11:29:09.162719+00:00
- Charged 653,240/6,000,000 sim-s; closed jobs 522/546; pending WM 0.

### S0_A_UPDATE_014.json
- A update14, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 5.09244e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_014.json`.

## Live publication — 2026-11-01T11:30:12.024242+00:00
- Charged 682,040/6,000,000 sim-s; closed jobs 546/570; pending WM 0.

### S0_A_UPDATE_015.json
- A update15, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.82592e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_015.json`.

## Live publication — 2026-11-01T11:31:47.424047+00:00
- Charged 696,440/6,000,000 sim-s; closed jobs 570/582; pending WM 0.

### S0_A_UPDATE_016.json
- A update16, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 5.15962e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_016.json`.

## Live publication — 2026-11-01T11:32:19.359481+00:00
- Charged 725,240/6,000,000 sim-s; closed jobs 582/606; pending WM 0.

### S0_A_CHECK_016.json
- A_minus_MAPPO: {"bootstrap":"10000 whole-world paired resamples; RNG20261101; negative delay is better","learned_minus":"mappo","metrics":{"600_collisions":{"ci95":[0.0,0.0],"mean":0.0},"600_hard_brake":{"ci95":[0.041666666666666664,9.25],"mean":4.125},"600_team_delay_vs":{"ci95":[-78.65468623970786,3459.082650857894],"mean":1686.0698296149446},"600_teleports":{"ci95":[0.0,0.0],"mean":0.0},"600_ttc_below_3_vs":{"ci95":[-10.626041666666666,101.25],"mean":43.291666666666664},"900_collisions":{"ci95":[0.0,0.0],"mean":0.0},"900_hard_brake":{"ci95":[-3.3333333333333335,7.2093749999999845],"mean":1.25},"900_team_delay_vs":{"ci95":[331.8738344183094,6951.7146298653915],"mean":3346.086848854511},"900_teleports":{"ci95":[0.0,0.0],"mean":0.0},"900_ttc_below_3_vs":{"ci95":[-27.0,99.08437499999998],"mean":29.291666666666668}},"worlds":12}

## Live publication — 2026-11-01T11:33:53.736644+00:00
- Charged 754,040/6,000,000 sim-s; closed jobs 612/630; pending WM 0.

### S0_A_UPDATE_017.json
- A update17, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 5.19562e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_017.json`.

## Live publication — 2026-11-01T11:34:58.261814+00:00
- Charged 782,840/6,000,000 sim-s; closed jobs 631/654; pending WM 0.

### S0_A_UPDATE_018.json
- A update18, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 4.75876e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_018.json`.

## Live publication — 2026-11-01T11:36:01.920162+00:00
- Charged 811,640/6,000,000 sim-s; closed jobs 654/678; pending WM 0.

### S0_A_UPDATE_019.json
- A update19, seed20261101; SUMO-only reward; queries/accepted/executed 0/0/0; next pooled beta 0.0.
- Per-bucket trust: .
- TRAIN GAE variance 5.30598e+08 veh-s²; cooperative-move residual/SNR and PPO details in `S0_A_UPDATE_019.json`.
