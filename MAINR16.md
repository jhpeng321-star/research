# S-MAINR16 — round-16 integration; WAIT_OWNER before checkpoint staging

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
