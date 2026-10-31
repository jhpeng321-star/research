# RUNA owner-side noon continuation

## Resume audit — 2026-11-01 02:09 +08

- Preserved clean HEAD `b3deb25b`, original watcher PID44796 (created
  2026-10-31 23:58:14), all claims, failed A/C receipts and live A2 dispatch.
- Read rb_common, previous handoff/log/receipts, INT/EVAL handoffs and owner's
  `INT_NOON_SCHEDULE_REQUEST.md`; registration5736 supersedes07:00.
- Watcher polled every600s through02:08:14 without error. A2 launched once01:38;
  C2 remains next, unclaimed. B2/FULL-v1 are peer-owned and retained.
- gpuA02:09:23: RTX4090D,4706MiB used/43805MiB free,99%; load16.72/14.87/14.46.
  A2 workers335002/335003/335004 remain live; each shard has4 completed futures
  (12/160, no complete K8 world). INT workers333463/333464/333465 untouched.
- Original A2 supervisor334996 and local watcher captured07:00 at startup.
  Changing files alone cannot extend them.

## Authorized continuation procedure

Owner's02:08 instruction explicitly extends supervision: interim04:00/08:00,
candidate freeze09:30, evaluation through11:30 (+08), report11:45.

Implement a separate, immutable CPU-only A2 continuation monitor. It verifies
the original dispatch and every worker's boot/start-tick incarnation, captures
existing task/scorer hashes, and transfers ONLY the owned supervisor334996.
It never launches inference, repeats a failed task, changes live sources, or
signals a peer. Existing workers continue unchanged. The original exclusive
supervisor lock must transfer before monitoring/scoring begins. Preserve the
original dispatch and append a handover receipt. Guard own GPU use at18500MiB;
verify all160 completion hashes before the original frozen full scorer runs once.

Replace the local owned watcher under its exclusive lock with an explicitly
documented continuation through11:30, admission freeze09:30. Preserve claim-safe
list ordering, one owned active variant, ten-minute INT/EVAL polling, and failure
markers. New variants use the noon supervision deadline. Interim scores use
only complete K8 ensembles and the existing scorer, with sparse evidence caveats.

Execution/verification receipts will be appended after implementation.

## Implementation verification

- `continue_noon.py`: original-dispatch/worker incarnation checks, parent-only
  transfer, exclusive lock, preserved-artifact hashes,18500MiB own guard,
  complete-K8-only04/08 frozen scorer,160-task full scorer and11:30 hard stop.
  It contains no inference launch path. `deploy_noon.py` stages immutable helper
  copies and refuses a second dispatch or a terminal original candidate.
- New watcher admits before09:30 and polls/collects through11:30. New candidate
  remote supervisors retain the same3x5GiB budget with the11:30 deadline.
- `py -3.13 -B -m unittest tools.rbruna.test_runner -v`: **11 PASS /0.233s**.
  Added checks cover exact-parent-only transfer, unchanged dispatch/workers,
  dead/reused/foreign-PID refusal, peer exclusion from GPU accounting and
  post-freeze result collection without admission. Existing claim/concurrency,
  READY ordering,160-task hashes, builder binding and reused-PID tests pass.
