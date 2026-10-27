# S-LOOPRESUME — amendment installed; paired-critic data blocker

## Status

**Cycle 1 is NOT launched.** The attempted detached continuation stopped
`OPERATIONAL_PAUSED` during paired export: all four cycle-0 worlds are assigned to
S-PAIRCRIT's existing development partition, leaving **zero fitting worlds**.
The original attempt and three operational retries are exhausted and preserved.
No partition was changed and no extra world was collected.

The separately authorized **cycle-0 WM fine-tune is running detached on gpuA**,
followed automatically by the **unchanged acceptance check**. Its coordinator
cannot launch an actor or collectors, advance the loop, or promote a candidate
into shared state. Completion/acceptance is pending.

## Installed amendment and checks

- Appended the five prospective owner decisions and exact pre-launch bindings to
  `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md`; appended the same
  amendment to the loop JSON before execution. Twin-readable artifact:
  `gpuA:/root/autodl-tmp/c4/restart20261026/AMENDMENT_C1.json`, SHA256
  `049abecbf6063ad5ecef2dcc4d95eaa60d91c11e51b41c5771ae3a9f2ca43403`.
- Cycle 0 remains historically **INCONCLUSIVE**, policy `1b9949cc...` retained.
  Its collection is closed at worlds968144–968147; nine reservations968148–968156
  stay unconsumed. Original stop/state/config/registration/journal/PID/log retained.
- Cycles>=1 use the same four-world bootstrap and zero margin; only lower CI>0
  stops collection. Inconclusive continues without being labelled PASS. Both-arm
  applicability is explicit; the twin was not contacted.
- Integrated `c4/paircrit@f8374e1f`; paired critic ON in the boundary config,
 100 critic-only warmup updates; longcritic/absolute-share selection replaced.
  Breakdown/conformal OFF. Fit/development RMSE monitoring was wired but could
  not run on an eligible cohort. Any next-boundary reversion remains unexercised.
- First cycle-1 phase is configured to measure150 new updates, select the best
  eligible frozen16-panel native-relative J at the unchanged10s scoring clock,
  and retain SHA/WM/config vetoes. Added phase-relative update10 progress gating
  and subsequent measured-count wiring. None of those new actor updates ran.
- Final offline checks: **422 passed, 2 skipped** (410 integrated18.53s plus
  12 paired9.66s). Skips are optional historical exports absent on gpuA;
  current-policy exports were independently checked against real common starts.
  Ruff passes. Final staged39-file delta-manifest SHA:
  `90cc238df1445ae37bb7f4e4fdc46c52ba52d394dfe558ce45c57f6b67c84852`,
  identical on gpuA/cpuP/cpuQ. The measured-count follow-up was staged after the
  pause; future continuation requires fresh bindings, not the initial launch receipt.

## Why the actor cannot proceed

The registered rule is `sha256("965140:" + world_seed) % 5 == 0` for development.
Remainders for968144/145/146/147 are **0/0/0/0**. All16 converted episodes and
eight matched300s POLICY/B pairs are present; both observable initial arrays and
raw common-start proofs match. The exporter then refuses `no paired fitting worlds`.
The resulting partial proof is evidence of common starts, **not fitting eligibility**.

Driver **62093/start841844830** exited. `OPERATIONAL_PAUSED.json` and
`LOOPRESUME_PAIRING_BLOCKED.json` record the blocker. There is no new scientific
WAIT_OWNER verdict and no fourth exporter retry is armed. A new owner-authorized
fitting-data/partition amendment is required to enable the paired cycle-1 actor;
development labels and the nine reserved worlds remain untouched.

## Independent WM and gates

External coordinator **62645/start841871931**, trainer **62910/start841872654**,
started2026-10-26T18:15:05Z. Script SHA
`32fd991c8659c691e8f935610498565c6f9678dfd23cb6a65820a5a425c11d66`.
It reuses verified attempt1 arrays, without retrying the exporter or repairing the
failed shared phase. Original R16 config resolves40 steps,4.5h;72 tensors/core
parameters frozen, no unmatched freeze prefixes,171 checkpoint tensors loaded
with no new/mismatched/unused tensors. Loop pairs supply304 windows at the
original0.15 mix; original sources remain included. Same acceptance runs afterward.
At2026-10-26T18:23:01Z, step1 is finite: loss181.57535, grad norm81.53729,
242.623s, peak allocated1888.71MiB; coordinator incarnation is live. Shared
loop state is byte-identical to its independent-WM starting snapshot. This is
progress evidence, not a completed40-step fine-tune or acceptance verdict.

Existing confidence-table recalibration completed: reference64 samples,
in-sample Brier0.001510,1 open block; mean comparison56 samples,
Brier0.0002973,0 open blocks. These are in-sample diagnostics, not efficacy or
deployment qualification. No different confidence rule was enabled.

Evidence stays under `gpuA:/root/autodl-tmp/c4/restart20261026/`:
`LOOPRESUME_{BINDINGS,SOURCE,TESTS_PASS,LAUNCH,PAIRING_BLOCKED,WM_PID,WM_ONLY}.json`,
`loopresume_wm_only.log`, `arm/loop/wm/c00{,.log}` and subsequent checks.
Small snapshots are in `tools/loopresume/receipts/`.

**Charged simulation remains102,607/210,000; remaining107,393.** No new simulation,
actor updates, evaluation seeds, MAPPO initialization, GPU-R writes, restricted
data access, peer-process signals or push. cpuP/cpuQ source/configuration is staged;
cycle-1 collectors have not started.

Reproduce the local handoff checks with:
`python -B tools/loopresume/verify_receipts.py`.
Read current main progress with `python -B tools/loopresume/operate.py status`;
the independent WM's authoritative progress is `LOOPRESUME_WM_ONLY.json` on gpuA.

## S-C1FIX — policy recovered; collection deployment/allowance pause (2026-10-27 05:04 UTC+8)

- Driver64792 was healthy and had automatically replayed from cycle0 in `c01.a2`.
  It was never signalled/restarted. Cooperatively stopped only that owned replay,
  retaining its files; existing retry logic launched `c01.a3` from original best
  full checkpoint170/SHA`2463e550...`. Config-only policy cap is now0.65 (~30.8GiB),
  leaving7.1GiB even at the WM's full0.20cap. Original WM62645/62910 untouched.
- **150 logical updates completed through180; native-relative gate PASS,16/16
  improved, mean deltaJ=-0.3837442398**, unchanged10s frozen panel/WM/config vetoes.
  Selected checkpoint `arm/loop/policy/c01.a3/policy_latest.pt`, SHA256
  `64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c`.
  Retained original140updates through170 and reran171–180; no repeated warmup/share
  selection, no bit-identical uninterrupted-run claim. **72 pre-retry tests PASS;
  final73 PASS** including an explicit pre-training corpus-SHA veto; Ruff PASS.
- **Collection did launch on both CPUs, but all12 dispatched world jobs failed
  before SUMO**: missing successor-runtime `baselines/circles/adapter.json`.
  First failed worker PIDs cpuP589756/589471, cpuQ582909/583052; **none active**.
  Identical original adapter now installed on both (SHA`8a925e9e...`); missing
  network/template metadata links repaired without opening scenario assignments.
  This is deployment preflight only, not a successful collector run.
- Automatic attempts1–3 retained34,092charged sim-s/7,200long-credit sim-s.
  Driver exited naturally `OPERATIONAL_PAUSED` at04:59:13; cycle caps36,933/8,000
  cannot fund another four-world dispatch. **136,699/210,000 charged**,0completed
  cycle1worlds; convert/WM1 pending, independent WM0 remains autonomous.
  Owner amendment is required before further dispatch: same13worlds, no refunds,
  cycle attempt allowance>=71,025 and long-credit allowance>=15,000 would fund
  one complete remaining collection at173,632total; these increases are **not installed**.
  **ETA blocked pending that amendment.**
- Registration/events append-only; authoritative GPU receipts
  `C1FIX_{SOURCE,TESTS,RECOVERY,VERIFIED,COLLECTION_BLOCKED}.json`, repaired current
  `C1_STATUS.json`; CPU `C1FIX_COLLECTOR_REPAIR.json`. Compact local result:
   `tools/loopresume/receipts_c1fix/RESULT.json`. No push.

## S-C1COLLECT — amendment installed; single canary failed, OPERATIONAL_PAUSED (2026-10-27 05:34 UTC+8)

- Installed main05:20 owner act-on-recommendation amendment prospectively in the
  shared append-only register and loop registration/events: **cycle1 attempt
  allowance71,025; included long-credit allowance15,000; total210,000 unchanged**.
  Same13 reservations968157–968169, no refunds, existing SHA-bound policy/native
  veto and four-world HARMFUL-only guard retained. The historical **34,092charged
  sim-s lost to12 pre-SUMO failures** remain explicit, including7,200long-credit.
- Added serial per-box startup barriers and per-submission accounting. Exactly
  **ONE world** dispatched: seed968158 oncpuP, world attempt4 (peer alternation),
  2,841charged including600long-credit. Driver73820/start843042499 ran
  21:30:10–21:30:23Z; worker623495 exited. **No cpuQ canary or remaining world
  dispatched. No active main collector on either box; no simulation retry.**
- Actual collector blocker: missing
  `runtime_c1/baselines/circles/model.onnx`. Trace reaches `env.reset` line205
  after successful SUMO startup/TraCI setup, but before recorder initialization
  or stepping. Summary: **0simulated seconds,0branches,0completed episodes**.
  This is a failure before first verified outputs, not proof of working SUMO
  collection. The earlier adapter/network/template repair was insufficient.
- A separate orchestration error passed configured SSH argv as a scalar to the
  startup probe. Driver's recorded phase error is therefore TypeError; the
  worker's retained traceback independently identifies the missing model.
  Probe transport corrected offline after pause, with no redispatch. Original
  sources/tests/logs retained. Prelaunch67testsPASS; final **69testsPASS8.88s**,
  including SSH argv, sequential canary/no-remainder, charge-before-dispatch,
  cycle-scoped allowance and failure-propagation checks. RuffPASS.
- **Charged139,540/210,000; remaining70,460.** Cyclecharged36,933/71,025;
  long-credit7,800/15,000 (included in total). **Real-SUMO per-world policy-minus-
  native delta (veh-s, entrance waiting included), mean,95%CI and guard verdict
  are UNAVAILABLE:** no completed paired starts. Imagination gate remains the
  historical16/16PASS/-0.383744, not a real-SUMO guard result.
- Main priority predicate was active while collect status wasrunning for13s;
  it is nowinactive after failure. No claim that a watcher sampled that short
  window. Convert/WM1/check remainpending. Independent WM0 coordinator62645/
  start841871931 and trainer62910/start841872654 verifiedlive and untouched.
- Authoritative receipt `gpuA:/root/autodl-tmp/c4/restart20261026/C1COLLECT_RESULT.json`,
  SHA256`b266ac4b557e6a9791a2d65e7aff718c883b977989d800b2f50a9c79363fc54f`;
  `C1COLLECT_{SOURCE,TESTS,BINDINGS,LAUNCH,ATTEMPT,FINAL_TESTS}.json`,
  `arm/driver_c1collect.log`, CPU retained `c01.a4/collect/collect_log.jsonl`.
  Final registrationSHA`24bd54391a61c85f873ba3a91a4f49ecc6d34d7f42b1bbe8be194161669b35c4`.
  Compact local receipt `tools/loopresume/receipts_c1collect/RESULT.json`.
  **Renewed collection authority consumed: OPERATIONAL_PAUSED; main decides
  repair/further attempt and accounting. No additional launch is armed.**

## S-C1COLLECT2 — exact asset repair; preflight checker failed, OPERATIONAL_PAUSED (2026-10-27 05:47 UTC+8)

- Main05:45 amendment appended to shared register and loop events/registration:
  cycle1 attempt allowance **73,866** (+2,841), included long-credit15,000,
  total210,000, no refunds; same13 reservations968157–968169 and policy64f70af4...
  preserved. One conditional canary-first collection allowed only after parity and
  both preflights PASS. **No canary/remainder dispatched; no launch armed.**
- Successful cycle0 receipts968144/cpuP and968145/cpuQ identify the original
  `restart20261026/runtime`. Compared all accessible runtime/input trees, including
  symlinked assets; installed **480 missing permissible files/312,914bytes per CPU**
  from identical originals with SHA verification. CIRCLES modelSHA
  `762fde0763c693c0e88f41b98b22fc0a11135a312b78c4c567e6211a3035ccee`,
  baseline sidecars and PI dependencies restored. Source-overlay differences retained;
  protected paths/route assignments pruned, template checked by same-original
  filesystem identity only; generated runs/bytecode excluded. Identical CPU parity
  receiptSHA`0796259a7b69a9e096e660e21756c5f992db41543746332ef7ea931db0b4b0a2`.
- **cpuP preflight FAIL; cpuQ NOT RUN after fail-stop.** Collector import completed;
  adapter/model/network hashes verified. Checker then incorrectly demanded
  `/home/ubuntu/c4/main14/restart20261026/runtime_c1/assets/research/migration_manifest.json`.
  That file is absent in both original/successor runtimes on both CPUs, but is
  **unused by this adapter's ordinary direct network path**. This failure is a
  checker bug, not proof of another collector-required missing asset. ONNX and
  learned-policy loads were not reached. No corrected preflight/retry performed.
- **Charged139,540/210,000 unchanged; additional0; remaining70,460.** Cyclecharged
  36,933/73,866; included long-credit7,800/15,000. Guard unavailable; convert/WM1/
  check pending. Main priority inactive because collection never launched. SSH argv
  fix was already installed. WM0 coordinator62645/start841871931 live; trainer62910
  no longer live at observation; neither was signalled or altered by this task.
- gpuA `C1COLLECT2_RESULT.json` SHA
  `1a77f0837f2c9942194183640313e2eb978a28475966dd254cf23c779bac58aa`;
  registrationSHA`45b05c95741957f2459a16786fe00c343574c48f0c1b9e05e1e995600e09292e`.
  Source/preflight/parity receipts and old state/pause/register retained; compact
  local `tools/loopresume/receipts_c1collect2/RESULT.json`. **OPERATIONAL_PAUSED;
  main decision required before corrected preflight/launch.** No push.
- Handoff integrity check PASS: `python -B -m tools.loopresume.verify_c1collect2`
  verifies source syntax/SHA, remote result/register/state bindings, unchanged charges
  and the paused driver; it does not rerun preflight or simulation.

## S-C1COLLECT3 — preflights/canary/early guard PASS; long-credit cap pause (2026-10-27 06:12 UTC+8)

- Installed identical original `assets/research/migration_manifest.json` on BOTH
  CPU successor runtimes: git blob`dd6dc191f0de71d6056a0ccb820a808ab77fe403`,
  198,743bytes/box, SHA256`fe722b8941f50ff23463f3d673d0e07fa5e826c2a6e85ea46eac8108167f7fcb`.
  Corrected preflight checks collector-resolved direct-path assets; the unused
  legacy migration fallback is no longer an artificial dependency. **cpuP PASS,
  cpuQ PASS**, including collector import, CIRCLES ONNX CPU-session load and exact
  native-residual learned-policy load. No SUMO execution/charge during preflight;
  protected template checked by original-file identity only. Both receiptSHA
  `c8f5bb5e4496a147236aecf71c4a5c881e08d4b02083dd1defab6ca4a0d40273`.
- Main06:00 authority retained cycleallowance73,866, includedlong-credit15,000,
  total210,000/no refunds and all13 reservations968157–968169. Bound one cpuP
  startup barrier through a narrow remote-only configurable-box overlay; local
  pre-existing source changes retained. Initial driver77640/start843244579 refused
  the old failed phase **before any charge/dispatch**; archived its source/claim/
  PID/log/pause, corrected the driver to enter through `retry()` once, and rebound
  source/tests/register. **72testsPASS8.87s**, including one-canary/no-remainder
  barriers, charge-before-dispatch and the failed-phase driver transition. RuffPASS.
- Successor driver **77914/start843255163**,22:05:37–22:09:46Z. Exactly ONE
  canary: **968160/cpuP/worldattempt4**, verified22:06:23Z with1,323 finite,
  increasing frames through661sim-s and an owned live SUMO descendant. Only then
  dispatched968157/cpuQ/a4,968158/cpuQ/a5,968159/cpuQ/a4. All **four completed**;
  numerical output verified on both CPUs. Main-priority predicate was active during
  collection; retained `C1COLLECT3_RUNNING.json` SHA`077c8defab98737a9981e8c7a6a5dad2e2bd9a987de1ed7411573b311cb1e52f`.
- **Real-SUMO four-world early guard PASS**, policy-minus-native veh-s including
  entrance waiting, summing both paired starts/world:968157 **-1508.832184**,
  968158 **-54.638037**,968159 **-1259.776429**,968160 **-3626.275887**;
  mean **-1612.380634**,95%CI **[-3034.651022,-418.186574]**. Original bootstrap,
  zero margin, confidence procedure and HARMFUL-only stopping rule retained.
- **OPERATIONAL_PAUSED**, exact driver refusal:
  `long credit attempts exceed 15000 additional sim-s/cycle (no refunds)`.
  Historical7,800 + new2,400 = **10,200** included long-credit; remainingnine need
  5,400, projecting **15,600**,600 above the unchanged15,000 cap. This is an
  accounting blocker after successful collection/guard, not preflight/canary failure.
  **No remainingnine dispatched; no active main collector or driver; no retry armed.**
  Main priority is nowinactive; convert/WM1/check remainpending behind collection.
- **150,904/210,000 charged**, additional11,364, remaining59,096; cycle48,297/73,866.
  Main decision is needed for included long-creditallowance15,600(+600); total cap
  need not increase. Preserve the four completed artifacts and PASS guard. A future
  authorized resume must handle the consumed canary/empty completed-bootstrap batch
  rather than blindly selecting another startup canary. No policy/WM/veto changes,
  process signals, restricted-file reads, eval seeds or push. WM0 exact PIDs62645/
  62910 no longer live at final observation; neither was signalled by this task.
- Authoritative `gpuA:/root/autodl-tmp/c4/restart20261026/C1COLLECT3_RESULT.json`
  SHA`c37ba32bb426fdbc252af832f0e5e6ad6841856c4e8821ab55183d6846b84122`;
  final registrationSHA`44a95b56934b0f596c103e933fc0dc8728c935e94fff598a893fedd30ddb97f3`.
  `C1COLLECT3_{PREFLIGHT,SOURCE,PATCH,BINDINGS,RECOVERY_SOURCE,RECOVERY_TESTS,RECOVERY_BINDINGS,LAUNCH,ATTEMPT,RUNNING,STATUS}.json`,
  `arm/driver_c1collect3.log` and `c1collect3_before/` retain deployment and failed
  transition history. Compact local receipt:`tools/loopresume/receipts_c1collect3/RESULT.json`.
  Verify without dispatch: `python -B -m tools.loopresume.verify_c1collect3`.

## S-C1COLLECT4 — remaining nine running; detached WM/check continuation (2026-10-27 06:29 UTC+8)

- Installed main06:20 exact-shortfall amendment prospectively in append-only shared
  `REGISTRATION.md`, loop registration/amendments, C1events and cycle ledger:
  included cycle1 long-credit **15,000 -> 15,600 (+600)**, cycleallowance73,866,
  total210,000 unchanged, no refunds. Policy64f70af4... and all native/WM/config
  vetoes, four completed artifacts, original four-world **PASS** guard retained:
  mean-1612.380634veh-s,95%CI[-3034.651022,-418.186574].
- Narrow remote-only orchestration repair skips canary selection for the empty
  completed-bootstrap batch; consumed canary/attempt bindings retained. **Exactly
  nine worlds968161–968169 dispatched once**,22:25:21Z; no bootstrap redispatch.
  Detached driver **80364/start843373550** live. At22:27:19Z collectors cpuP
  **680115/680062/679776/679975/679220**, cpuQ **671328/671847/672084/671803**;
  **both boxes verified collecting with actual owned SUMO descendants and1,323
  finite/increasing frames0–661s**. Proof worlds968165/cpuP and968162/cpuQ,
  SUMO682402/start6791283 and674109/start6791230. Main-priority predicate **active**;
  evidence does not claim an independently observed peer-watcher sample.
- At22:28:56Z **8/13 worlds complete**, collectionrunning, convert/WM1/checkpending.
  **176,473/210,000 charged**, +25,569, remaining33,527; cycle73,866/73,866;
  includedlong-credit15,600/15,600. Every world charged before dispatch; all failure
  charges retained. No new science gate, process signal or peer-process modification.
- Driver automatically continues **convert -> reconcile completed/ACCEPTed WM0 ->
  existing next-boundary decision -> wm_ft -> unchanged check**. No actor launched
  by this continuation. WM0 candidate0044eddb... SHAverified; resolved40-step WM
  recipe validationPASS, all7data roots present, acceptance entrypoint present,
  10.318GiB free. Historical40steps summed8,531.35s (~2h22m). **Estimated collection
  end06:32–06:40; WM1fine-tune end09:00–09:30 UTC+8**, plus subsequent unchanged
  acceptance check. Estimates are not completion receipts.
- One durable non-scientific operational retry maximum across this continuation;
  **unused** at observation. Official fresh-attempt recovery preserves outputs/
  charges; accounting/scientific/provenance/safety refusals fail closed. Full cycle
  and long allowances are now consumed, so additional simulation is not fundable
  under these cycle caps. **77testsPASS8.95s**, including exact nine-world resume,
  ledger-before-dispatch, one-retry and refusal/scientific no-retry regressions;
  initial offline fixture-budget failure/source/log retained. RuffPASS.
- gpuA receipts under `restart20261026/C1COLLECT4_{SOURCE,TESTS,BINDINGS,LAUNCH,
  ATTEMPT,RUNNING,CONTINUATION,STATUS,RESULT}.json`; originals in`c1collect4_before/`.
  Source manifestSHA`66866c8cbe53f0900cac95d910755fbac9a59d923af87eac22990a41bcf9b17b`;
  initial registrationSHA`3b78b63ccb4f50b0b02c794e9e087157afebf88f7142a6aa34962d1061fef0ac`;
  runningSHA`120a585fc3eeb3b4f78f3687dc186eb3f4dd7b02aae5175bc2d36e2f8382cf9d`;
  resultSHA`fe3157e056ebb2281461911a4017aeec4d8f297de459c0768277bfa48648669d`.
  Compact local`tools/loopresume/receipts_c1collect4/RESULT.json`; no-dispatch handoff
  check`python -B -m tools.loopresume.verify_c1collect4`; live phases/proofs via
  `python -B -m tools.loopresume.c1collect4 status`. Local commit follows; no push.

- **06:33UTC+8 progress supersedes the running-collection snapshot:** all13worlds
  completed22:29:54Z (**06:29:54UTC+8**), convertcompleted22:30:05Z. Driver80364
  automatically reconciled accepted WM0 and entered WM1 at22:30:05Z; exact owned
  trainer **81201/start843401980** live at22:32:39Z,171tensors loaded without new,
  mismatched/unused tensors or unmatched freeze prefixes. First optimizer step
  stillpending at that observation; unchanged check ispending and automatic after
  WM1. **WM1 endETA09:00–09:30UTC+8** retained. No operational retry consumed.
  CPU main-priority predicate is nowinactive during GPU-only WM; it was verified
  active on both collecting boxes and reactivates forcheck. Charge176,473 and
  includedlong-credit15,600 unchanged. Post-boundary registrationSHA
  `1a6cf335fd34571bc28bdc6a125f44b1fadd594228bd3711e77f3517d9934c99`;
  gpuA`C1COLLECT4_PROGRESS.json`SHA
  `c38f89b29be12d09ac4b3c011180a1b95e9f8b07580cde8a3a447e5ab858a246`,
  compact local`tools/loopresume/receipts_c1collect4/PROGRESS.json`.

## S-C2H60 — 60 s cycle-2 replacement (2026-10-27 11:20 UTC+8 authority)

- Main waived S-C2PREP's three-hour time projection. Accepted WM1 SHA
  `5257a64c7cbb6a8cb828cb26230e97c686d7715cc33825bd3e1ca95f7bf51d8e`
  passes the unchanged60s horizon gate: sign0.963157895,95%CI
  [0.935483871,0.989361702], threshold0.868421053. Original60s smoke was
  finite and used7.816GiB allocated. Time does not select the scientific horizon.
- Cooperatively stopped only coordinator85784/start844014193's child90493/
  start844953642 through `arm/loop/policy/c02/STOP` at03:21:20Z. Child saved
  and coordinator exited naturally03:21:37Z, final update**233**. Entire30s
  phase,53new updates since180, curves/log/checkpoints and original resource
  receipts remain on disk, superseded. Actual full resumable warm start:
  `arm/loop/policy/c02/policy_latest.pt`, SHA256
  `ed0d4242493ba3012d440db484cb837ee785836fed31cfbd41e1e613fcc5e283`.
  C1fallback64f70af4 was unnecessary. Persisted anchor0; no actor reset.
- Detached coordinator**91911/start845196315**, child**91912/start845196330**,
  started03:29:09Z (**11:29:09UTC+8**). Output `arm/loop/policy/c02.h60`,
  log `c02.h60.log`; coordinator log `arm/driver_c2h60.log`. Config
  `configs/c4_policy14_c2h60.toml` has knots[0,150,233], pads[32,83,149]:
  new234–383 are150logical updates under gate-qualified60s stage2, retaining
  the original10s low-share mask. Paired criticON, existing100critic warmup,
  correct C1paired-label source64f70af4/partition, cap0.65, no time cutoff.
- Unchanged frozen16 native-relative scoring at10s, phase-relative update10
  progress gate, earliest eligible minimum checkpoint selection and all
  WM/config/panel/checkpoint vetoes remain active. WM1 is explicitly bound as
  `policy_wm` for eventual gated adoption. Final native-relative gate remains
  pending while training runs.
- Automatic post-phase stop **WAIT_OWNER_BUDGET**: charged176473/210000,
  remaining33527; full collection36933 projects213406, shortfall3406. Collection/
  convert/wm_ft/check remain pending. A future owner-approved Markdown budget
  raise must also be prospectively bound into the loop registration, using
  amendment ID `S-C2H60_OWNER_BUDGET_RAISE`, `owner_approved=true`,
  `collection_authorized=true`, the matching raised `sim_s_total` and
  `markdown_registration_sha256`; the coordinator rechecks that registration
  after the native gate. Only then does it enable serial two-box canary-first
  collection -> convert -> wm_ft -> unchanged check. No automatic refund/retry.
- First coordinator91792 exited before any actor/output on exclusive log-path
  collision with existing `c02.log`; original30s log untouched. Archived source/
  binding/tests/state/log in `c2h60_attempt1_no_actor/`; append-full-name fix
  applied, then **60offline testsPASS3.77s**, RuffPASS. No scientific retry or
  simulation charge. SourceSHA
  `5096e2d9f6f0fc5142616f626b3d25f1ddab592365f55e75ef70048153f27e63`;
  bindingSHA`b999ef1b0e41bcb35a87fac0e2aa91ada46571e26273d23684bfad84f25783e5`.
- Live progress: `python -B -m tools.loopresume.c2h60 progress`.
  No-dispatch handoff verifier: `python -B -m tools.loopresume.verify_c2h60`.
  Compact verified receipt goes to `tools/loopresume/receipts_c2h60/VERIFIED.json`;
  authoritative receipts are `gpuA:/root/autodl-tmp/c4/restart20261026/C2H60_*.json`
  and10s own-PID memory samples `C2H60_MEMORY.jsonl`. Registration/events are
  append-only. Existing S-C2PREP local work remains present; task commit includes
  its unchanged small config prerequisite, not its broader unfinished edits.
- **11:43UTC+8 live verification PASS:**12new updates234–245,10actual60s
  rollouts plus2original low-share10s masks, all under60s stage2. All actor/
  critic losses and gradient norms finite. Unchanged phase-update10 progress
  gate at243 **PASS16/16**, mean10s-panel J change **-0.042008400** relative to
  warm start; this is not the final native-relative/real-SUMO verdict. Own-PID
  memory afterupdate5 ranged9498–12376MiB and plateaued **12376MiB(12.086GiB)**
  for more than3minutes through verification, below0.65allocator cap; no OOM.
  Measured rate **49.89updates/hour**,72.16s/logical update including startup/
  scoring (training-only mean55.75s). **Phase-end ETA2026-10-27 ~14:30UTC+8
  (06:30Z), allow ~30min variability**, then native gate and budget stop.
  Authoritative `C2H60_VERIFIED.json` SHA
  `155ae0f95939f898ab5863db0f4667ef8c3fbcdec5d655f8d64e42f023d35172`;
  local compact receipt includes prospective Markdown amendment, source/tests/
  launch/stop/checkpoint hashes, own-PID memory samples and progress gate.
  Coordinator91911/actor91912 live,176473charged and allCPU phasespending.
  Append-only shared registration and loop events updated. Local commit follows;
  no push. Exit with detached phase running; completion/native verdict pending.

## S-C2PREP — prepared/verified; superseded by S-C2H60 (2026-10-27 11:40 UTC+8)

- Prospective main08:05 amendment appended to shared register and loop JSON:
  C1checkpoint64f70af4, cap0.65, paired criticON100warmup, breakdown/conformalOFF;
  only the60s curriculum knot changes300->190 (verified update180/anchor0),
  retaining fixed10s scoring and all original native/WM/config/panel vetoes.
  **74unit/integration checks PASS7.73s; RuffPASS.** Stale8,000-vs8000 test-message
  assertion repaired; failed test/source receipts retained. Original WM/check
  driver80364/trainer81201 were never signalled or modified.
- **WM1 unchanged check ACCEPT02:12:10Z**, candidateSHA
  `5257a64c7cbb6a8cb828cb26230e97c686d7715cc33825bd3e1ca95f7bf51d8e`;
  paired-difference error170.211vs255.496veh-s, sign66.67%vs62.5%. Current-WM
  horizon gate30/60PASS. Three isolated60s updates finite:119.889/50.440/62.559s,
  peak7.816GiBallocated/9.316GiBreserved under30.793GiBcap. Registered max-time
  projection17983s>10800s selected30s fallback before fullphase;30s smoke
  58.341/30.295/29.977s, peak3.080/4.072GiB, projection+smokes9257sfits.
- Detached **85784/start844014193**, actor**90493/start844953642**, started fullphase
  02:48:42Z. Completed53newupdates181–233 on acceptedWM1; phase-update10progress
  gatePASS, mean-0.010866642. Latest measured native panel at230PASS16/16,
  mean-0.426156998; **no final deployment gate/selected adoption was produced**.
  New main11:20 **S-C2H60** cooperatively STOPped the30s actor at03:21:20Z;
  checkpoint saved233, both own processes exited03:21:37Z. Original outputs intact:
  `arm/loop/policy/c02/policy_latest.pt`, SHA
  `ed0d4242493ba3012d440db484cb837ee785836fed31cfbd41e1e613fcc5e283`.
- Paired labels retain11fitting/2developmentworlds (22/4starts), C1policy source.
  Fit/development RMSE beforewarmup1497.178/1971.144; after100warmup1443.975/
  2040.975; independent CPU-only checkpoint233diagnostics1449.158/1997.145.
  Zero-difference baselines1497.178/1971.144: fit improves, development isworse.
  Diagnostics are off-policy after actor updates; retain registered next-boundary
  monitoring, no unilateral reversion/calibration enabled.
- **SUPERSEDED_BY_S-C2H60**, not a scientific gate failure or completed budget stop.
  Separate successor91911/start845196315 (actor91912/start845196330),60s,
  nowowns continuation; no successor process/state/config changed by this task.
  **176473/210000charged,33527remaining; no C2collection.** Exact nextcollection
  need36933, projected213406, **additional cap3406**. Successor finalgate/policy
  completion and WAIT_OWNER_BUDGET remain its responsibility.
- Authoritative `gpuA:/root/autodl-tmp/c4/restart20261026/C2PREP_SUPERSEDED.json`
  SHA`918c2b7e9057cafc0f7117f82f239afc8a6b75dbf8c106868224e148dc23dd89`;
  compact local`tools/loopresume/receipts_c2prep/SUPERSEDED.json`. Source manifest
  `8e08c5e5afe090e8e29926b9625d772a0ab861aaae6d82555c4b9d5b06e0afcb`.
  No-dispatch verification:`python -B -m tools.loopresume.verify_c2prep`.
  Original standby85660 was replaced childlessly before WMcompletion to enforce
  non-resource-smoke fail-closed handling. Local task commit only; no push.

## S-C2H60B — operational OOM recovery (main 2026-10-27 12:45 UTC+8 authority)

- Original91911/91912 failed03:51:26Z after logged249: allocator0.65 allowed
  30.79GiB,30.38GiBallocated,16.16GiBphysicallyfree. This is an operational failure;
  the11:43 short12.1GiBplateau was historical, not a later peak/memory guarantee.
  Complete original30s and failed60s outputs/charges/source/logs remain on disk.
- Latest original full60s checkpoint was **240**, `c02.h60/policy_native_best.pt`,
  SHA`21c04bdc0f3821c4c05e9ce0bf85db919848fb39ca57d2cfff1fd80082ceead4`.
  Preserve7logicalupdates234–240 and nativecurve233/240; unsaved241–249 are replayed
  with their failed-attempt rows retained, never double-counted. Same original
  **150logicalupdates234–383**, not a new150-updatephase. Cap is now**0.92** in
  both TOML/environment; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` explicit.
  Batchstarts unchanged; if resourceOOM persists, register a starts-batch reduction
  before retry, retain60s. Per-update full saves and allocated/reserved peak telemetry
  are operational additions. Pairedcritic/partition/C1source64f70af4, acceptedWM1
  5257a64c, low-sharemask, native16panel/10sclock/selection/vetoes stay bound.
- First94725startup rejected143remaining as a non-multiple of10 before output/CUDA;
  validator now checks original150phase length. Artifacts `c2h60b_no_actor/` retained.
  Next94910/94911 was proactively STOPped through its own `c02.h60b/STOP`, saving
  replay241 and exiting naturally04:53:08Z. Same-phase recovery must retain original
  phase-start233 progress baseline when saved240 precedes phasegate243; baseline
  handling corrected, predicate/panel/clock unchanged. No process signals.
- **Actual successor warm state241**, `arm/loop/policy/c02.h60b/policy_latest.pt`,
  SHA`d5d2f28060a927b1d0b692eb9b2397f40c9231c6e6d24af9e7edd7fda7c1d77b`;
  all8logicalupdates234–241 retained, **142remaining242–383**. Separate original
  native-best240 kept byte-for-byte. Full actor/group/critic/target/optimizers/duals/
  RNG/samplers/pairedstate restored; no repeated warmup, target reset or share fit.
  No bit-identical uninterrupted-run claim. Own administrativeSTOP summary and
  originalOOM cause independently SHA-bound; all prior source/launch/memory receipts
  archived in `c2h60b_gate_baseline_stop/`.
- Detached coordinator **95283/start845712191**, actor **95284/start845712207**,
  started04:55:07Z (**12:55:07UTC+8**), output `arm/loop/policy/c02.h60b.r2`,
  driverlog `arm/driver_c2h60b.log`. **72offlinechecksPASS3.86s**, RuffPASS;
  source manifestSHA`cdbbed49de8bffda32d8818bc233ad9723227c6a70f1cd7d36a502dde3555dc1`,
  bindingSHA`0d76a70f9cfd2850cc37b6a8a497f4b993b8721c4d397c991582ab62cf5519d9`.
  Five-second own-PID watcher `C2H60B_MEMORY.jsonl`; allocator lifetime telemetry
  `c02.h60b.r2/cuda_peaks.jsonl`. Fresh>=15finiteupdates AND>=20min watch required
  before verified handoff; initial05:03:44Z snapshot6newupdates242–247 is provisional.
- After383 the driver automatically verifies the unchanged native-relative gate,
  adopts only its best eligible checkpoint, then **WAIT_OWNER_BUDGET** at
  **176473/210000charged**. Collection36933 would project213406; no dispatch under
  current cap. Only owner-approved Markdown raise prospectively bound through
  `S-C2H60_OWNER_BUDGET_RAISE` permits canary-first collect->convert->wm_ft->check.
  Registration/loop events append-only. Progress:
  `python -B -m tools.loopresume.c2h60b progress`; read-only handoff verifier:
  `python -B -m tools.loopresume.verify_c2h60b`. Verification/ETA receipt follows.
- **13:19UTC+8 verification PASS:**17fresh updates242–258 finite (25logical phase
  updates including retained8); **1447.416s/24.12min**,287five-second own-PID samples.
  The previously failing update250 completed. Observed process peak **37388MiB =
  36.512GiB**; allocator lifetime peaks **37201868800bytes =34.647GiB allocated**,
  **38757466112bytes =36.096GiB reserved**, under0.92cap (~43.58GiB). NoOOM;
  batchstarts remains1, no batch/horizon reduction. This is an observed peak,
  not a guarantee about later starts. Watcher remains detached with the driver.
  Measured **42.177logicalupdates/hour**,85.355s/update including startup/scoring
  and the large250update; **phase-end ETA~16:17UTC+8/08:17Z2026-10-27**, allow~1h
  start/workload variability, then finalnativegate/budgetstop. Original progress
  gate243PASS16/16, mean-0.042008400 vs233; finaldeploymentgate stillpending.
- At13:22, full per-update checkpoint260was verified saved: SHA
  `ddfecfc3b62a560bd4afe3efd0b46e32c3f67f6763142f0b6b04d199d9d1ec3b`,19newupdates
  (18actual60s/1inherited10smask), stage2; nativecurve250/260PASS, mean delta
  -0.505407810/-0.542551517. These are intermediate critic-free panel measurements,
  not finalreal-SUMO evidence. Exact coordinator95283/actor95284 incarnations live;
  176473charged, allCPUphasespending. VerifiedreceiptSHA
  `9bb56469bd12153f36ddb2b34182284061608fd1c91869c8c4fb77d12e116516`;
  compact source/tests/recovery/bindings/launch/verification/Markdown snapshots in
  `tools/loopresume/receipts_c2h60b/`. Read-only handoff verifierPASS;72tests/RuffPASS.
  Scoped attributes retain exact transported source bytes; WindowsCRLF is explicit.
  Shared registration and loop verification events appended; localcommit follows,
  no push. Final native-relative gate ->WAIT_OWNER_BUDGET remains automatic.
- Concurrent main append records owner approval210000->280000 for main/twin.
  At05:26:02Z remote cap is still210000,charged176473, with no installed
  `S-C2H60_OWNER_BUDGET_RAISE`. An independently authorized autonomy task is preparing
  that binding; the driver rechecks after its final native gate and follows the
  registered owner authorization if then installed. This task did not dispatch SUMO.
- Final13:28:30snapshot: **25freshfiniteupdates242–266**,33logicalphaseupdates;
  **1999.23s/33.32min**,396memorysamples, same36.512GiBprocess/34.647allocated/
  36.096reserved peaks, both ownPIDs live. Rate**44.940updates/h**,80.107s/update;
  central endpoint383ETA**16:05UTC+8/08:05Z**, allow~1hworkloadvariation. Compact
  `receipts_c2h60b/CURRENT.json` retains this later observation separately from the
  SHA-bound13:19verification; the native gate/budget decision is still pending.
