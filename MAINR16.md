# S-MAINR16 — cpuQ checkpoint staged; WAIT_OWNER after prearm invocation failure

## Current continuation receipt — 2026-10-26 16:36 UTC+8

**WAIT_OWNER. Checkpoint staging succeeded; no cycle-0 migration or launch occurred.**
My prearm command incorrectly combined synthetic `--smoke` with the frozen `--gate-panel`
argument. `load_gate_panel` requires real sampler metadata; `SmokeSampler` has none, so it
exited before the first optimizer update with
`AttributeError: 'SmokeSampler' object has no attribute 'metadata'`.
The owner's any-failure stop rule was applied immediately; the corrected smoke was not retried.

The owner explicitly authorized substituting cpuQ's copy for the unreachable gpuB source.
The substitution was appended to the shared registration **before any restart launch** and
recorded in the successor loop registration JSON. The original verbatim decisions remain.
The checkpoint is now on gpuR at **`/root/main14_loop/mainr16_base/wm_s_best.pt`**:

- SHA256 **`5b2569418bf651b8ef2c3a2abdc110ac6860c32dd92dca8cdf3b2f48b1c8cbb2`**
  matches cpuQ's copy-time `~/autopilot_r10/runs/r16/provenance/SOURCE_SHA256.json`, where
  it is recorded as the original gpuB round-16 stage-B FINAL `wm_s.pt`.
- `physics.json` SHA256
  `7d3ec19c04e418e61078ea673db702cb12d7765bf4a4e58a3b26f63f5eaa48b1`;
  `config.resolved.json` SHA256
  `d218190193dcab5881b307c8dab7165b16bd3f6d618baae7d909268607e78bf1`.
  Both match the provenance SHA256s. All three files also match `ck/md5.txt`.
- `RUN_SUMMARY.json`, `TRAIN_RECEIPT.json`, `INIT_FROM.json` were copied from the sibling
  `r16/provenance/` directory, and each matches SOURCE_SHA256. Provenance records
  `lane_calib.json` as absent on gpuB; no calibration sidecar was invented.

Successor arm directory **`/root/autodl-tmp/c4/v5/mainstart/mainr16/restart`** contains
`REGISTRATION.json`, `CONFIG.json`, `ARMED.json`, `STAGED_BASE.json`, `TESTS.json`,
`prearm_smoke.log`, and **`WAIT_OWNER.json`**. Its registration is a prepared,
**uninstalled** snapshot: SHA256
`4a2596fd8cdc4522d9b7651185ce869fd0e7eb815ff9c1f829fa02d31ae31c6d`;
config SHA256 `50abcea3fabda8458d07ce18fe0e6ab7680cf03fce0012a6ac8a5c5bafb1fead`.
It records the **210,000 total** and the restart's arithmetic cycle0 combined allowance
**87,399 = 58,266 prior + 29,133 fresh**, without refund or automatic retry.

The same integrated round-16/native-lane/FASTLOOP runtime was deployed and SHA-verified to
gpuR `restart/runtime` and both CPU collector roots
`/home/ubuntu/c4/main14/mainr16/runtime`. Source manifest SHA256:
**`69a67f3e4c67247dbd8021a4710ba7187e6417856f20b980fe885e5a542effdb`**.
Both CPU boxes pass the real frozen-model/ONNX/SUMO-version/reset-import readiness probes.
**463 integrated CPU tests PASS, zero skips**, including a new pure restart regression
that preserves charges/seeds/history, clears adoption of the old policy, preserves attempt
numbers for distinct output paths, and rejects a scientifically stopped loop.

The live `LOOP14_STATE.json` remains byte-identical to the previous task's snapshot:
**58,266 charged / 0 actual**, no new seeds or SUMO seconds. The old policy's SHA256 was
reverified. Staging/model prearm reached no training updates. `MIGRATED.json` and
`LAUNCH.json` do not exist. Driver/trainer/collector PIDs from this continuation: **none**.
Update-10 and both-box collection are not verified because no productive run started.
Final output free bytes **12,745,072,640**; scratch free bytes **2,274,299,904**.

Local continuation receipts and source: `tools/mainstart/mainr16_resume/` and
`tools/mainstart/mainr16_resume.py`; state migration helper:
`scripts/c4/mainr16_restart.py`. The prepared migration archives the failed world journal,
retains append-only seed/charge ledgers, retains the superseded policy/cycle history, clears
policy adoption, and starts the new policy in `policy/c00.a2`. It was unit-tested but has
not been applied remotely.

The invocation correction is prepared locally: remove `--gate-panel` and `--horizon-gate`
from **synthetic prearm only**, hash-check the unchanged frozen panel separately, and keep
both arguments for the real policy and update-10 gate. The helper refuses prearm/migration/
launch while the new WAIT_OWNER receipt exists. No productive policy/scientific gate changed.
Further execution requires owner direction out of WAIT_OWNER.

Read-only monitoring on gpuR:
```bash
A=/root/autodl-tmp/c4/v5/mainstart/mainr16/restart
cat "$A/WAIT_OWNER.json"
cat "$A/prearm_smoke.log"
cat /root/main14_loop/LOOP14_STATE.json
```
No process needs stopping. For a later authorized launch, the scoped cooperative stop
markers are `$A/STOP_ARM`, `/root/main14_loop/STOP_DRIVER`, and the new policy's
`/root/main14_loop/policy/c00.a2/STOP` update-boundary marker. These were not written.

## Original integration receipt — preserved history

**Status at 2026-10-26 16:24 UTC+8: WAIT_OWNER; cycle 0 was not restarted.**
The read-only gpuB prerequisite probe failed with
`Connection closed by 172.19.0.169 port 16573` before producing the FINAL's SHA256.
The owner's “on any failure stop in WAIT_OWNER” rule therefore stopped staging/launch.
No background process was started by this task. No new seed or SUMO second was consumed.

## Owner amendment / accounting

Owner approval: 2026-10-26 ~16:15 UTC+8. The **verbatim owner decisions** are in
`tools/mainstart/MAINR16_OWNER_DECISIONS.txt`, appended to the shared
`E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md` before any restart,
and recorded as `mainr16_amendment.owner_decisions_verbatim` in gpuR
`/root/autodl-tmp/c4/v5/mainstart/mainr16/REGISTRATION.json`.

- Total cap **210,000 charged sim-s** supersedes all earlier totals, including the
  S-COLLECTFIX 233,064 cap. **58,266 charged / 0 actual**, 26 seeds **968103–968128**,
  remain recorded without refund. Remaining total allowance: **151,734 charged sim-s**.
- Registered block **968100–968227** and probe exclusion **968100–968102** remain fixed.
  Restart must continue its own allocation rules without reuse; no allocation occurred here.
- Selected base: round-16 FINAL on gpuB,
  `/root/autodl-tmp/c4/v5/runs/c4_wm_v10_round16_b/wm_s.pt`.
  **Checkpoint SHA256: UNVERIFIED / null**, because the source probe returned no digest.
  Planned staging uses the loop/root filesystem, not the space-constrained scratch volume.
- `E:/archive_cpuN/QUAL_r16.md` supports selection under the registered base rule:
  no new HDV-reaction FAIL versus 13E; strict TTC **11.3x**, versus 14b **12.1x** and
  13E **16.3x**. Overall qualification is **NOT-QUALIFIED**, with safety_abs FAIL;
  base-rule eligibility is the owner's specified criterion.
- Native-lane stays **ON**. SAFE SPEED2 policy-only flags are
  `WMS_SPEED_PIVOT_DIAG_OFF=1`, `WMS_SPEED_GRAD_AUDIT_EVERY=10`.
  Deterministic alternating cpuP/cpuQ collection retains seven workers/box,
  capped at **12/box**, numerical threads 1; readiness/resource checks remain required.
- Round-16 model flag `ramp_speed_adapt_learnable=true`; physics:
  `emergency_braking`, `secure_gap_veto`, `lane_change_secure_gap`,
  `ramp_lane_end_stop`, `ramp_risk_retain` true; `emergency_onset="driver_b"`;
  `merge_end_wait=false`. Driver settings and adaptation scalars are learned.

The new JSON is an **uninstalled amendment snapshot**. The existing failed loop state
still references its preserved S-COLLECTFIX registration. No registration hash/state
migration or cycle-budget rewrite was attempted after the source failure. A continuation
must explicitly reconcile cycle-0 accounting/cap with the authorized restart and retain
all prior charged attempts; changing only the total cap is insufficient.

Snapshot SHA256: `bb0456caedf5b9e908baecde88b0f3b8cde2233b97efdfd76a4f9ae8a52fc741`.
Shared register SHA256 at amendment capture:
`49097b5efc3aa656f956c3254ab08aef85d35adf4fc98224400a230d719bc3a4`.

## Merge notes

Merged `c4/v10r16` **3b3c201df9dc04b11bc2f647e52fc508bb7a6bce** into the
`c4/mainstart` line based on **c38c7ec3**, which already includes **a5b76fcf** FASTLOOP
and collector dependency/PATH fixes. Three conflicts resolved:

1. `src/wms/model/wms.py`: retain both native-lane decider-mask output and round-16
   learned ramp-adaptation output in the same forward pass.
2. `src/wms/train/trainer.py`: support both `C4_GPU_MEM_FRACTION` and
   `WMS_GPU_MEM_FRACTION`; apply the stricter configured cap using an indexed CUDA device.
3. `tests/test_loop14_emergency_onset.py`: retain loop-FT and round-14 onset checks.

`configs/c4_wm_loop14_r16_ft.toml` preserves the original 40-step loop FT recipe and
enables round-16 learned ramp adaptation, with its stage-B scalars frozen. It continues
residual/courtesy/yield training. Regression checks compare every other flattened config
key to the existing loop recipe. The recipe is inlined onto the same parent as the
existing loop preset because one additional inheritance layer exceeds the loader limit.
The config is prepared only; the live launcher has not been repointed to it.

## Tests and receipts

On gpuR, isolated verification release:
`/root/autodl-tmp/c4/v5/mainstart/mainr16/runtime_v2`.
**462 CPU tests PASS, zero skips**, including the integrated main/native-lane/FASTLOOP/
collector/policy/cooperation gates, round-16 model/gate tests, emergency onset and allocator
merge tests. One pytest warning: cache_dir config is unknown when cacheprovider is disabled.
An initial **461 passed / 1 failed** run caught the config inheritance-depth issue; its
receipt and original release are retained. The fixed release passed the entire suite.

Verified 712 deployed files, source manifest SHA256:
`11e46c88347a046484e102e5bc7b127e1fabde98d4434fff4a9668fd99ebfa8f`.
Local receipts: `tools/mainstart/mainr16/{BLOCKED,FINAL,STAGE_v2,TESTS_v2}.json`;
initial verification receipts: `STAGE.json`, `TESTS.json`.
Remote logs: `mainr16/integrated_tests.log` (initial) and
`mainr16/integrated_tests_v2.log` (passing). No driver/trainer/collector launch log exists
for S-MAINR16. Real-checkpoint prearm smoke, policy progress, update-10 gate and both-box
collection verification were **not reached**; the earlier 13E gate is not an r16 receipt.

Final disk free: output filesystem **12,752,224,256 bytes**, scratch **2,319,978,496 bytes**.
The live state is byte-identical to `mainr16/STATE_BEFORE.json`. The superseded 13E policy
remains on disk, SHA256
`63219be447762da33b3ceef634092bbfde1324d9217e35cde8735b96a0577866`, reverified at finish.
No CPU host, scoring process or gpuB file was modified.

## PIDs, monitoring and stop commands

S-MAINR16 detached driver/trainer/collector PIDs: **none**.
Exact predecessor receipts checked read-only: mainstart driver **929014 / start4161848089**
and collectfix driver **87024 / start4161998640** are both no longer live.
`mainr16/WAIT_OWNER.json` is the durable failure receipt; `FINAL.json` confirms preservation.

Read-only status on gpuR:
```bash
A=/root/autodl-tmp/c4/v5/mainstart/mainr16
cat "$A/WAIT_OWNER.json" "$A/FINAL.json"
cat /root/main14_loop/LOOP14_STATE.json
```

No PID needs stopping now. Existing cooperative stop-marker commands for this owned loop,
if a later authorized continuation is installed (not executed by this task):
```bash
touch /root/autodl-tmp/c4/v5/mainstart/mainr16/STOP_ARM
touch /root/main14_loop/STOP_DRIVER
# Only for the current owned c00 policy path, after confirming its launch receipt:
touch /root/main14_loop/policy/c00/STOP
```

Continuation requires restored/confirmed gpuB access and owner direction out of WAIT_OWNER,
then SHA-verified checkpoint plus physics/config staging, tested restart/state/accounting
migration that never collects the superseded policy, fresh prearm smoke, and a recorded
setsid/nohup launch. All subsequent phases still require the existing gates and 210,000 cap.
This task's helper `tools/mainstart/mainr16.py` has no launch action.
