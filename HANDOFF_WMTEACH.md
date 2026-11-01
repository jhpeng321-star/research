# S-WMTEACH resume / ownership

## CURRENT: PAUSED by MAIN 2026-11-01 17:35 — supersedes all launch instructions below
- Resume authority is ONLY a new MAIN `S-WMTEACH RESUME` entry. Paper A/S-COOPMWM
  has GPU priority. Do not start a collector, teacher, student or live job while held.
- Pause/code/first inventories committed `1df3e615`. Durable local hold:
  `docs/paper_results/wmteach/OWNER1735_PAUSE.json`; both collector entry points
  and automatic/direct live launch return `PAUSED_BY_MAIN` before remote work.
- Local collector43128 was verified by exact command + creation epoch1793511929
  against its receipt and stopped. No new local collector/dispatch is running.
- Verified own gpuA453587/start889174202 and453608/start889174685/boot
  a81555d7-176e-4ed8-bc7d-896fc89cc914, then wrote ONLY their roots' cooperative
  STOP and owner-pause receipts. Both exited at existing task guards; no GPU signals.
  `FAILURE.json` in these roots records the intentional `own STOP`, not a new model bug.
  Former quality453566 was already complete/absent and received no stop action.
- Final read-only checkpoint1793525989: all three PIDs absent, none in GPU accounting;
  other workers remain untouched. Live root absent; teacher SUMO0; student not fit.
- FULL-v4 `/root/autodl-tmp/wmteach_scale20261101_v3`:5120 states/64 complete
  lineages;47 fully saved K8 paired-future caches with BOTH H120/H300 labels.
  Next sorted unlabelled state `w969400_line0_t1695`; no partial label directory.
- FULL-v5 `/root/autodl-tmp/wmteach_scale_v5_20261101_v2`:4993 states/62 complete
  lineages;1 complete K8/H120+H300 label cache. Partial lineage `w969430_line1`
  has33 saved states through547.5s (next planned saved clock555s);969431/line1
  not generated.127 states remain. First sorted unlabelled `w969400_line0_t1005`.
- Completed v5 quality `/root/autodl-tmp/wmteach_v5quality20261101_v2`:
  96 futures +24 horizon labels; retain and reuse, never forecast them again.
- Exact resume inventory: `wmteach/OWNER1735_CHECKPOINT_1793525989.json` plus
  `_SUMMARY.json` (under docs/paper_results). The full receipt binds all10503
  saved cache files by relative path/bytes/SHA256, source identities, owner receipts,
  quality, corpus, next label positions, partial lineage and intentional STOP tracebacks.
  Scale files are `states/<world_line>/*_t*.json` and
  `labels/<state>/{FUTURES,LABEL_H120,LABEL_H300}.json`; quality is `tasks/<state>/`.
- Existing immutable `scale.execute` is NOT a restart entry point: exclusive writes
  and terminal STOP/FAILURE remain. After authority, continuation must validate/reuse
  completed caches without recomputation in a separately registered successor.
  The v5 partial lineage saves observable states, not a serialized latent/controller
  generator state; exact internal-state continuation is not currently implemented.
  Preserve this limitation and the prefix; do not claim a bitwise generator restart
  or silently replay the prefix. No resume or new job was attempted during pause.
- CPU pause/live checks5PASS. Read-only recheck only:
  `python -B -m tools.wmteach.pause` (never invokes live/collector/dispatch).

## Historical recovery milestone — superseded by MAIN1735 pause above
- SOLE CURRENT COLLECTOR: local43128, exact command
  `C:\Users\jpeng\AppData\Local\Programs\Python\Python313\python.exe -B -u -m tools.wmteach.scale_watch`,
  CIM creation2026-11-01 13:45:29+08 / receiptstartedUnix1793511929.2292252.
  OS-exclusive SCALE_WATCH.lock; every20min, up to168h. First full collection and
  owned auto-commit succeeded. Do not launch another collector/GPU dispatch.
  Prior local41120/9796 collectors were exact-owned stopped and remain historical.
  Snapshot1793511933:245generated v4states/1labelled; v5quality8futures/2labels;
  v5scale waits its quality parent. Auto-live only sees the complete v5 scale policy.
- ACTIVE verified gpuA (common boot a81555d7-176e-4ed8-bc7d-896fc89cc914):
  v4scalev3 PID453587/start889174202, v5qualityv2 PID453566/start889173543,
  v5scalev2 PID453608/start889174685 (waits own v5 quality, no CUDA allocation).
  Dispatch once only; all older attempt roots remain terminal.
- BATCHED CANARY PASS on first new FIT state:8complete futures, finite/conserved,
  source-inclusive team costs, max batch-vs-single error1.52587890625e-05veh-s
  (<1e-3 gate). First production result reused once; snapshot1793511674 shows
  23generated states/1labelled state and owned process R. Full5120corpus still pending.
- V5 new quality snapshot1793511702:3/96complete futures, owned process R,
  ~49s/future; no v5 quality/student/live result claimed yet. Teacher SUMO0.

### Earlier canary/staging history — preserve, not active launch instructions
- Latest operational canary repair (new designacf6f48dbd4b): v4scalev2 EXITED0states
  at observer due_area initialization after its raw process was unwrapped; keep
  InstrumentedEntries attached. V5qualityv1 EXITED0forecasts because a no-CAV
  roster can still have future CAV arrivals; add frozen plant CAV prior then v5
  registered execution settings. V5scalev1 stopped on failed-quality-parent guard.
  All three old claims/tracebacks retained, no completed task replayed.
- Current staged roots: v4scalev3 identityc7678363/431bindings/9CPU;
  v5qualityv2 identity69f975d0/446bindings/10CPU (LC/CAL+empty-CAV regression);
  v5scalev2 identity7c9e636a/485bindings/9CPU. Every older root stays terminal.
- Repair/successor code staged and CPU-checked under designa3684b70abda,
  `WMTEACH_SCALE_V2_DESIGN.md`. All three prospective registry receipts committed
  before dispatch. Original failed roots are immutable and remain terminal.
- `tools/wmteach/marginal.py` removes saved per-ID posteriors while preserving
  released LC population, CAL hidden/arrival wrappers and registered CAV settings.
  Shared FIT-start generator now uses this population contract rather than the
  restricted native exporter. Real released-type LC/CAL CPU regression PASS.
- Staged v4repair root `/root/autodl-tmp/wmteach_scale20261101_v2`, identity43bf5570,
  431bindings/9hostCPU PASS. Reuses completed v4quality, never its failed FIT jobs.
- Staged v5quality root `/root/autodl-tmp/wmteach_v5quality20261101_v1`, identity69603de6,
  446bindings/10hostCPU PASS (including real LC/CAL regression), quality ONLY.
- Staged v5scale root `/root/autodl-tmp/wmteach_scale_v5_20261101_v1`, identity6183ab2f,
  485bindings/9hostCPU PASS; waits v5quality. Existing local Python31327CPU PASS.
- Latest releasedv5 fixed as sole live teacher BEFORE its new quality forecasts;
  v4repair is engineering comparison only. FIRST complete v5student_scale_5120,
  its own helpful-quality PASS and original final frozenpolicy/budget/guards gates.
- FULL-v4 quality COMPLETE: receipt1793510517/12states/96futures; H3008/12argmax,
  helpful4/8argmaxhits/6positivechoices, exactchosenD+680.3/regret593.0veh-s/state.
  Helpful-quality gate PASS. Shared REGISTRATION result line appended.
- Original437822 EXITED at firstFIT969400 loader guard; scale447666 EXITED at
  same guard before any imagined state/rollout (SCALE_SNAPSHOT1793510546).
  Frozen native exporter only accepts969420..969439, not allFIT32. All roots,
  quality futures and failures preserved; NO restart/replay of either root.
- retire_small correctly refused absent/changed ownership; no STOP/signal sent.
  Exact-owned local collector9796 stopped before source/bridge repair.
- FULL-v5 READY now published: sourceb2df7458a41f5f4d3a048532bc40d8d4435ec381,
  gpuA `/root/autodl-tmp/c4/rb2int20261101/phys_code_b2df7458a41f`,
  recipe023fa74a4d3059115ec0a5f28d39030a76137d871db5798eb1d23a13fc1e32c5.
  Shared `rb-shared/RB2_INT_FULL_V5_READY.md`/two-world allfivearm receipt verified;
  current-jobs older prose is superseded by explicit READY+completion evidence.
  LP449208 is PEER-owned/live, never duplicate/signal it. Final qualification pending.

## OWNER12:53 recovery amendment
- Fresh recovery read clean `e6ba1bde`, registration and all ownership contracts first.
- Snapshot1793508966: GPU437822 identity MATCH/R,64 complete futures,16 horizon
  labels; quality/student still pending. Preserve this running initial experiment.
- Verified local collector41120 by exact command and creation Unix1793507667;
  stopped ONLY that owned collector to prevent its obsolete first32world-student
  automatic live launch. GPU and peer processes untouched; no live root/result.
- New owner priority: report helpful-state exact teacher quality FIRST, generate
  >=5000 distinct FIT/imagined decision states, then fit the scale student. Live
  requires meaningfully nonzero helpful-state agreement plus original approvals.
  The earlier first-small-student live contract is superseded before any dispatch.
- FULL-v5/FULL-v5-LP explicitly NOT READY in rb2-int/HANDOFF.md; retain FULL-v4.
  WMDATA2 969550..969599 is quarantined by INT, so new scale inputs use FIT32 only.
- Scale code/design: `tools/wmteach/scale.py`, `scale_ops.py`, `scale_watch.py`,
  `docs/paper_results/WMTEACH_SCALE_DESIGN.md`; prospective registry receipt
  `wmteach/SCALE_REGISTRATION.json`, design7f3bb98d3c5f. Target5120DISTINCT states,
  FIT32 x2O0imagined paths x80clocks307.5..900, one prechosen setting/state,
  eachK8xavailableO0..O4/H120+H300. Futures batched in one physical rollout.
- Descendant adapter explicitly uses frozen FIT joint population/cold motion and
  controller/causal imagined request counters+queues; generator latents excluded.
  Existing roster-only exact quality is reused, not fully conditioned qualification.
- Shared student refactored through `train_records` without changing model grid,
  feature ABI, confidence selection or whole-source-world nested CV. Fit scale
  only after complete5120corpus. `auto_live.py` now requires this scale snapshot
  and helpful-quality gate; old `student_setting0_0` cannot trigger live.
- Helpful H300 live gate fixed before quality: >=3helpful states,>=2argmaxhits,
  >=20%agreement, Wilson95lower>5%, positive mean chosen exact saving on helpful.
- CPU27PASS using existing Python313/sklearn (default Python314 lacks sklearn;
  initial test invocation failed only at that missing dependency; no environment created).
- Scale continuation DISPATCHED ONCE: gpuA447666/startticks888983366/boot
  a81555d7-176e-4ed8-bc7d-896fc89cc914; immutable
  `/root/autodl-tmp/wmteach_scale20261101_v1`, identity1f465057363203d6,
  429source bindings, host9CPU PASS. Snapshot1793509643 verifies identity/S,
  waiting for initial12state exact quality; generation/labels0, SUMO0.
- Do not stage/launch this root again. Collect with
  `python -B -m tools.wmteach.scale_ops collect`. Any FAILURE.json is terminal;
  preserve partial work and diagnose. The original437822 remains independent.
- Read-only partial quality report1793509786 (no repeated forecasts):10/12states,
  H300argmax70%/sign61.5%/rank0.485/exactchosenD+802.0/regret313.8veh-s;
  helpful4/7argmaxhits,5/7positivechoices. PARTIAL, cannot arm live. Full12state
  estimate still awaited. `python -B -m tools.wmteach.quality_report` reads completed
  predictions before their exact targets and writes a bounded receipt.
- New exclusive scale collector local9796 runs every20min, maximum168hours;
  `wmteach/SCALE_WATCH_PROCESS_9796.json` and `SCALE_WATCH_1793509708.log`.
  It commits own progress and gates the approved scale live once; do not duplicate.
- Fixed-milestone resource decision BEFORE full quality closes: after all12exact
  diagnostics finish, retire the obsolete128context/small-student continuation
  cooperatively at its next future boundary, regardless of quality outcome.
  `python -B -m tools.wmteach.retire_small` verifies exact ownership +24quality
  rows before writing ONLY its own STOP/retirement receipt. Preserve every prior
  future/label; no GPU signal, replay or relaunch. Scale5120 job remains active.

## Historical initial ownership — superseded by latest milestone
- Branch `c4/wmteach`; recovery `9f92c488`, shared student `ee47f395`
  (original S-DISTILL `4b139e5f`), teacher `d1584ddc`, canary/dispatch `b919f4bf`.
- Active GPU job: gpuA PID437822, startticks888636829,
  boot `a81555d7-176e-4ed8-bc7d-896fc89cc914`, immutable
  `/root/autodl-tmp/wmteach20261101_v3`. Own strict8GiB GPU cap/nice19/one CPU thread.
- Full H300 canary34.248s passed finite/conservation/full-team-source arithmetic;
  completed future0 is reused. Never redispatch this root or repeat completed tasks.
- v1 failed before any rollout (hash-helper namespace); v2 never dispatched.
  All predecessor files remain. No process signal/deletion/new environment/push.
- Pinned sklearn/scipy dependencies live in v1/deps; v3/deps links there.
- Read status: `python -B -m tools.wmteach.ops status`.
- Save bounded receipt: `python -B -m tools.wmteach.ops collect`.
  This is one-shot read-only SSH, not another observer/dispatcher.
- One local20minute collector may be running: inspect `wmteach/WATCH_PROCESS_*.json`
  and exact local command/identity first. Its exclusive WATCH.lock prevents duplicates.
  It snapshots/updates its marked report section/commits owned receipts and gates
  ONE MAIN-approved live test after the first genuine WM student is complete.
  It never retries a launch or dispatches another GPU teacher. It stops after
  teacher/live terminal states, changed source, foreign staging or24h. No duplicates.
  Original local34576 was exact-own stopped after first committed receipt60ee8a16
  to fix Windows GBK Git decoding; `WATCH_ENCODING_REPAIR.json`, GPU437822 untouched.

## Pipeline already armed
- Twelve EXAM quality states (one/cell, selected by observed candidate availability,
  then clock/world), each K8 ×H120/H300. Exact outcomes opened after predictions.
- EXAM only contains roster/features, not causal motion/source counters. Diagnostics
  explicitly use fitted population +32FIT source-state priors and cold controller
  memory. They are weaker-input diagnostics, NOT fully conditioned FULL-v4 qualification.
- After all twelve complete, job writes QUALITY.json/sign/rank/choice/regret/coverage.
- Then32FIT969400..969431 ×4settings, K8, two horizons; no reserved8 or never-fit DEV.
- After each complete32world pass, job invokes shared GBT/MLP/classifier implementation,
  nested whole-world CV/confidence grid, fixed H300 WM-only targets. Saved snapshots
  `student_setting0_0` through `_3`; no exact labels or quality outcomes fit the student.
- Initial training contexts are t300 starts under four intervention settings. Later
  imagined-state augmentation is still needed for broader time/state coverage; do not
  describe these as a completed large-scale imagined-descendant corpus or live success.

## Live / successor work
- MAIN11:48 approval IS present: S-WMTEACH cap100000sim-s, same S-DISTILL panel.
- Shared panel DEV984000..984023, two/all12cells, high-share first; canonical SHA
  `72776ff122956f021d1caf2db9622674b0cde2a74af8bd9f20c3de5ba0a383a6`.
  Source: `E:/agent-work/c4-distill-1101/HANDOFF_DISTILL.md`.
- Initial live continuation is `tools/wmteach/auto_live.py`, invoked by the collector:
  FIRST complete `student_setting0_0` ONLY, no exact-result-based snapshot selection;
  original-pickle SHA/232input portable parity +3CPU/24guard gate, automatic durable
  final design/policy/exactbudget registration BEFORE one-use launch. Design
  `WMTEACH_LIVE_DESIGN.md`:58800sim-s=57600full paired+1200extra bitwise mechanics,
  first own MAPPO baseline reused once, no external baseline reuse/action pruning.
  Efficacy-only, no member-alone C_G branches;41200 remains unallocated.
  Own immutable live root `/root/autodl-tmp/wmteach_live20261101_v1`, <=2CPUworkers.
  Source-copy inherits verified S-DISTILL executor/replay/metric/runtime/checkpoint;
  no peer mutation. Resource wait does no physics. Any terminal launch stays terminal.
- Re-read `E:/agent-work/c4-rb-int/HANDOFF.md` and the new S-RB2-INT handoff before a
  successor. MAIN11:58 round2 is preparing FULL-v5, report due Nov2~10:00. Preserve v4.
- Numpy-only `tools.wmteach.portable` supports BOTH selected GBT and MLP (and unseen
  option constants), preserving H300 metadata; export each completed policy in its
  original sklearn1.7.2 environment and verify its predictions/actions before live.
- Actual published S-RB-DECIDE source parity PASS on first two complete contexts;
  rerun `python -B -m tools.wmteach.audit` after more complete contexts, no inference.
- Deliverable `docs/paper_results/WMTEACH.md`; local CPU25PASS/remote17portablePASS.
- Registry is shared/ignored by main Git: own prospective line is already appended;
  preserve other workers' edits. Committed own receipts live under `wmteach/`.
