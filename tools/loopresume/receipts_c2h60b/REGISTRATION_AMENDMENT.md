# S-C2H60B registration excerpt and verification

Authoritative append-only register:
`E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md`.
Prospective full-register SHA at the productive launch:
`0107403ebcfc8540d4c79420a530e6fbfd52121192994c873833fbf5e7207ee4`.
This compact task-only excerpt is not a byte-identical copy of the whole register.

## Main 2026-10-27 12:45 operational fix-and-retry authority

- Record original91911/91912's allocator-cap CUDA OOM after249 as an operational
  failure. Preserve all failed/superseded outputs and charged176473sim-s.
- gpuA dedicated: policy cap0.65->0.92, explicit
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Horizon60s, original low-share
  mask, paired critic ON, fixed10s/frozen16 native-relative gates and vetoes unchanged.
- Latest original full saved state240/native-best SHA
  `21c04bdc0f3821c4c05e9ce0bf85db919848fb39ca57d2cfff1fd80082ceead4`.
  Keep7logicalupdates234–240; original unsaved241–249 must be replayed, with failed
  rows preserved and never double-counted. Original phase ends383 (150logicalupdates).
- Per-update full saves; no repeated warmup/target reset/share selection. If resource
  OOM persists, record a starts-batch reduction before retry rather than horizon.
- First pre-output startup rejected143remaining as non-multiple10; corrected
  validation to original150phase length. Preserve `c2h60b_no_actor/`.
- Own94910/94911 cooperatively STOPped to correct original233progress baseline
  before replaygate243; saved241 and exited naturally. Latest actual resume is
  `arm/loop/policy/c02.h60b/policy_latest.pt`, SHA
  `d5d2f28060a927b1d0b692eb9b2397f40c9231c6e6d24af9e7edd7fda7c1d77b`.
  Keep8logicalupdates234–241, resume142updates242–383 in `c02.h60b.r2`; separately
  retain original native-best240. Bind own STOP summary and original OOM cause.
- After native-relative gate, WAIT_OWNER_BUDGET at176473/210000 unless an explicit
  owner-approved raise is recorded and prospectively bound by the existing
  `S-C2H60_OWNER_BUDGET_RAISE` protocol. Then serial canary-first collection ->
  convert ->wm_ft ->unchanged check. No refunds, peer signals or push.

## Verified 2026-10-27 13:19 UTC+8

17freshfiniteupdates242–258;24.12min/287five-second own-PID samples.
Peak process37388MiB(36.512GiB), lifetimeCUDAallocated37201868800bytes(34.647GiB),
reserved38757466112bytes(36.096GiB); noOOM, batchstarts1 unchanged.
Rate42.177updates/hour; ETA16:17UTC+8/08:17Z, allow~1hworkloadvariation.
Coordinator95283/start845712191, actor95284/start845712207 live.
72testsPASS3.86s, RuffPASS, read-onlyhandoffPASS.
VerifiedreceiptSHA`9bb56469bd12153f36ddb2b34182284061608fd1c91869c8c4fb77d12e116516`.

## Concurrent budget amendment observed

The shared register now contains a separate owner approval raising main/twin
210000->280000. At05:26:02Z gpuA still had cap210000,176473charged and no installed
`S-C2H60_OWNER_BUDGET_RAISE`. Driver rechecks after the final gate. The independently
authorized autonomy task may install that prospective binding; this task has not
installed a raise or dispatched simulation.
