# S-COOPMWM — live checks

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
