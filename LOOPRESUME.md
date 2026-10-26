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
