# Authoritative owner17:35 continuation — 2026-11-01 ~17:56 +08

## Active dispatch ownership (never redispatch)

- **gpuA capped supervisor513448/start890664618**, same boot as original.
  Handover COMPLETE17:52:42: all13prior claims validated/preserved; original473053
  actually EXITED; exclusive dispatch lock acquired. Exactly two new children
  started: v5_969434/514547 and v5_969435/514556. Remaining reserved969439 is next.
- **gpuR coordinator948148/start4215302457**, same original boot, still waits
  for last original WM2_969550 child at17:54:48. Other old children finished;
  original924719 remains paused/T. Zero new admission while draining. On final
  child exit, validate complete K8, retire only drained CPU dispatcher, acquire
  lock and continue at **one** evaluate process. Remaining reserved-v5 is first.
- Both capped coordinators use exact committed source87ef677d, scriptSHA
  9150d74817f8c908f3c915a25e40d3098a0ed9a4657e611af743024b277e1ed0.
  Stage `<final_root>/owner1735_control` on each host; old frozen source/model/
  recipes/plan/claims/results remain preserved. New model remainsb2df7458.
- Read `OWNER_1735_*_CAPPED_PROCESS/STATUS*.json`, DRAIN receipts and remote
  OWNER_1735_CAPPED_HANDOVER/STOP. Use `tools.rb2int.capped_ops status`, not the
  old final_ops supervisor-alive field, which intentionally refers to the retired
  original process. No evaluate process/group or peer PID was signalled.

## Sole current collection/publication owners

- Collector25240/create_filetime134379970282244431, runs/rb2int_finish_v3.
- Publisher44988/create_filetime134379973214244749, runs/rb2int_interim_v3.
- Both exact incarnations verified LIVE17:54. Existing cache/tasks/scores reused;
  no duplicate collection/publication. Next regular scored-cell update~19:57;
  reserved8 publishes immediately on completion, then~3h per view.
- Existing publisher checks the report's byte hash. Do not externally edit
  WMPHYS_V5.md or watched analysis files; owner priority is recorded in HANDOFF,
  this receipt, OWNER_1735_ADMISSION.md and the shared registration line instead.
  Latest report remains corrected2world INTERIM, not a model-selection result.

## Later-delivery standbys (not additional active dispatch/collection)

- Remote late waiter gpuA514854/start890706569 and gpuR949247/start4215344299,
  immutable source0b3a652e, scriptSHA
  4e6a72774f8a9021ecae9bd3700efe4cb46e9f3c1193cceeda8d0ae156b1da74.
  Stage `<final_root>/owner_late_control`; only after the respective current capped
  supervisor naturally emits a clean cutoff terminal AND actually exits can its
  late waiter acquire both locks and admit only unclaimed jobs, at the same2/1.
  All failures/incomplete claims/STOPs require explicit recovery, never retries.
- Local standby9724/create_filetime134380004640417822, runs/rb2int_late_wait_v1,
  LIVE/first STANDBY verified. No SSH/inference/signals or active collection.
  It waits for BOTH current local deadline-incomplete receipts AND exact process
  exits, then exclusively owns late collection AND publication. If the original
  final completes, it exits without taking over. Unexpected STOPs stop safely.
- Original Nov2 cutoff/attrition is preserved. Prospectively registered finite
  late admission boundNov4 09:00+08; collection/publication09:55, with immediate
  delivery sooner at216complete new worldjobs. This is a bound, not a final ETA.
  The frozen58world/fivearm/K8/fourmodel/primary54/gate/CI design is unchanged.
- Read OWNER_LATE_*_PROCESS/STATUS receipts and local START/STOP/COMPLETE before
  any future launch. Do not duplicate these standby processes.

## Verification / scientific status

33CPU guard/claim/priority/late-takeover/original-worker/collector checks PASS1.34s;
Ruff PASS. Original-stage GPU queries checked before intervention and after
handover; peer PIDs untouched. Common rules, morning report/noon provenance and
all four current worker handoffs read. Later adverse LANE and retired-CAV-binder
evidence remains disclosed, without changing final candidates after outcomes.

FULL-v5 and FULL-v5-LP remain engineering-ready for S-WMTEACH's independent quality
rerun. No calibrated action/cooperation/planner switching qualification follows
from the existing2world interim or from readiness. Teacher's own owner17:35 pause
is respected. Full final gate/short decision-quality report remains pending.
