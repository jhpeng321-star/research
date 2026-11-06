# S-MAINSTART — main coop-ON policy started on 13E

## S-COLLECTFIX (2026-10-26, owner13:30)

**Current status: BLOCKED / WAIT_OWNER after the explicit retry. Collection is not progressing.**
Implementation commit `a5b76fcf` integrated the gated FASTLOOP changes. Detached driver87024
(SID87024/start4161998640) invoked the official retry at13:43:25; it exited at13:43:54 after
all13 fresh worlds failed before simulation time advanced. cpuP then exposed missing
`output.controller_verification_v7` transitive modules; cpuQ lacked SUMO on its noninteractive
SSH PATH. The original and retry outputs are retained. `PROGRESS.json` confirms no owned
collector processes or completed worlds remain on either box.

**Final accounting: 58,266 charged / 0 actual sim-s; 26 consumed seeds968103–968128; no refund.**
The cycle0 combined cap58,266 is exhausted. Another full journal retry would charge29,133
and requires an owner-approved cycle0 budget amendment (at least87,399 combined); none was
inferred or applied. The existing total cap233,064 still applies. The loop stays failed,
with its policy/WM hashes preserved and no automatic retry.

The follow-up fix includes the three required `output/controller_verification_*` source
modules, pins collector PATH to the interpreter bin, checks the actual SUMO executable
(`--version`, no simulation), and exercises eager reset controller imports before allocation.
It is staged separately under `mainstart/collectfix/preflight_v2/runtime` and CPU
`~/c4/main14/collectfix/preflight_v2/runtime`, preserving the failed release. Both hosts pass
real model/ONNX/SUMO-version/reset-import probes, **421 integrated CPU tests PASS, zero skips**,
and the one-world synthetic mechanics dry run passes again with **0 actual SUMO seconds**.
Latest source manifest SHA256 **80729af793dcda78835a38ed7e72e080be6c5e5b3e89d323b6ccc4f736edd170**.
No live arm/state was repointed to this preflight-only release. Before any owner-authorized
next retry, migrate its tested source/config/PATH hashes into a new recorded amendment and
keep the existing world journal968116–968128 (peer replay, not fresh allocation).

### First repair / explicitly authorized retry history

Cycle0 policy completed150 updates at13:18. Collection attempt1 failed before the first
SUMO step: all13 reset tracebacks name missing `baselines/circles/model.onnx`. The original
source-only deployment omitted that binary; PROJ diagnostics were warnings, not this failure.
**29,133 sim-s charged / 0 actual sim-s; seeds968103–968115 remain consumed, no refund.**

Integrated FASTLOOP72300236/native-lane/SPEED2/mainstart code passes **419 CPU tests, zero
skips**. The documented **one-world mechanics dry run** passes real conversion/merge/return
export with a synthetic collector endpoint, **0 SUMO seconds**. Both CPU runtimes now contain
the unchanged frozen model SHA762fde0763c693c0e88f41b98b22fc0a11135a312b78c4c567e6211a3035ccee;
readiness checks its digest and actual ONNX session construction before allocation.

The append-only S-COLLECTFIX registration enables deterministic alternating cpuP/cpuQ
collection, **7 one-world workers/box**, numerical threads1, resumable per-world artifacts,
and SAFE SPEED2 for subsequent policy phases only. Other FASTLOOP switches remain false.
The registered combined caps are **58,266 sim-s/cycle / 233,064 total**, including attempt1;
all scientific gates and base-boundary rules remain fixed. The first explicit retry used
fresh eligible worlds968116–968128; later journal retries reuse only their own missing worlds.
The accounting regression reconciles original batch charges as well as new world charges.

Scoped runtime/arm: gpuR `/root/autodl-tmp/c4/v5/mainstart/collectfix/` (`runtime/`, `CONFIG.json`,
`REGISTRATION.json`, `ARMED.json`, `driver.log`, `PID.json`, `ACCOUNTING.json`); CPU owned root
`/home/ubuntu/c4/main14/collectfix/`. Original failures and predecessor registration are retained.
Local code/deployment/test receipts: `tools/mainstart/collectfix.py`, `tools/mainstart/collectfix/`.
The amendment was appended **before** registration migration/retry. See launch/progress receipts
below for the detached runtime's observed state.

```bash
A=/root/autodl-tmp/c4/v5/mainstart/collectfix
/root/miniconda3/bin/python "$A/runtime/scripts/c4/loop14.py" status --config "$A/CONFIG.json"
tail -F "$A/driver.log"
cat "$A/PID.json" "$A/PROGRESS.json"
# Explicit owner retry used by the detached wrapper:
/root/miniconda3/bin/python "$A/runtime/scripts/c4/loop14.py" retry --config "$A/CONFIG.json" collect
# Graceful stop of this successor:
touch "$A/STOP_ARM" /root/main14_loop/STOP_DRIVER
```

**2026-10-26 12:21 UTC+8 verification:** cycle-0 policy is running on gpuR, **17/150 updates**;
the frozen update-10 gate **passed, 16/16 starts improved, mean delta J −1.0110969543**.
Driver **822919**, trainer **823090**. Base is **13E b2 FINAL**, native-lane **ON**.
Collection has **zero allocated seeds / zero charged SUMO seconds**. cpuP collector readiness passes.

Branch `c4/mainstart`, minimal worktree `E:/agent-work/c4-mainstart`, created from `c4/main14`.
Native deployed code `9cc39f15` is preserved. Wrapper launch commit `b9a9b5e0`, collector/base
handoff `dfc91946`, tested boundary-continuation correction `3270f631`; no push.

## Registered amendment text and receipts

The following was appended to the shared register **before launch**, and incorporated into the
actual `/root/autodl-tmp/c4/v5/main14_REGISTRATION.json` before loop initialization:

> Owner request ~12:00: “start the policy training already while we wait for the scores so the
> GPUs don't sit idle”. Launch C4-LOOP14-COOPON-20261026 cycle 0 immediately from **13E b2 FINAL**,
> SHA256 `952ec059de5a9c0285d2327bcea57c836fd5d736c9975592ffbb932a506e84c8`, because cpuX,
> the scoring host, was spot-reclaimed. This supersedes initial waiting for 14b/round-16
> verdicts and the original 13:00 arm expiry. Native-lane ON; from-scratch policy; seeds
> 968100–968227, excluding 968100–968102; four cycles, 150 policy updates and 40 WM steps/cycle;
> all existing acceptance, incident, consecutive-rejection, update-10, horizon and budget gates.
>
> Collect on **cpuP, fallback cpuQ**, at most **12 SUMO workers**, numerical threads 1. Before
> phase entry or seed allocation/charge, require the selected host's exact collector source,
> dependencies, existing fixture and disk/RAM readiness. Missing readiness waits fail-closed.
> Record the host; no host switch after charging that collection attempt.
>
> At an **unstarted cycle>=1 policy boundary**, switch to the best eligible complete verdict
> among **{14b, round16}** only if it introduces **no new FAIL** in aggregate/per-set HDV-reaction
> sign, late-sign or magnitude items versus 13E, and its strict all-pair TTC ratio
> `fid.all_pairs.ratio` is **<=13E**. Lower eligible ratio wins; tie 14b. Require scored checkpoint
> SHA256 to match the actual FINAL. Missing/incomplete/hash-mismatched evidence selects nothing.
> Record every boundary decision and prior/new WM provenance. Current-policy return provenance,
> WM-specific horizon measurement and subsequent WM acceptance remain mandatory.
>
> Policy GPU allocator fraction **0.45**, batch 1, two futures, checkpointed imagination;
> allow only exact registered PAIRFT incarnations/their descendants alongside policy, refuse
> unrelated CUDA processes. WM fine-tune remains **exclusive-GPU** and waits on any CUDA user.
> Preserve the 1.5 GiB phase disk floor; put owned loop outputs on gpuR's root filesystem if
> protected scratch occupancy prevents safe attainment of the 6 GB scratch target, requiring
> >=6 GiB on the actual output filesystem before launch. Archive only dependency-cleared,
> SHA-verified inactive scratch work to gpuB's authorised archive. **No SPEED2 flags**: FASTLOOP's
> integration was uncommitted at launch.

Shared register:
`E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md`, S-MAINSTART amendment and subsequent receipts.
Filled JSON/config/arm and original installation are committed under `tools/mainstart/`.

- Actual registration SHA256: `8f80a023e2fdb230808be13a3f046694013bae5b7a7e92d0654bcec09817c6ea`.
- Registered config SHA256: `8f3c72cd5a9b27b91f1418a75858ad067ab1dc7d4f17b2c232e91fbc7a108fc7`.
- Frozen panel SHA256: `3eaa69a170c17302660d12dd4d7fddb4657e43fa334eece17844dd0fa6c0411e`.
- Original waiter **657216 / start ticks 4159403954**: only documented
  `touch /root/autodl-tmp/c4/v5/main14_arm/STOP_ARM`, then verified absent; its own log records
  **`disarmed_by_marker` at 12:00:17**. Original arm/source/stop marker remain intact.
- New installation keeps original arm/config/registration as `mainstart/PRIOR_*.json`;
  `tools/mainstart/PRIOR_STATE.json` is the local snapshot.

## Safe space recovery

`tools/mainstart/inventory.json` lists candidates/sizes, denied paths and symlinks. Inventory
included old smoke runs (~12–16 MB each), superseded repo copies (e.g. repo_r12 ~315 MB), and
old round runs (~3–40 MB each). Copies containing restricted paths/XML were not archived.
The two largest top-level check trees were ~900 MB each; only their inert work children moved.

Before any move/removal, searched candidate paths in both hosts' active process code/config
roots and operational gate/check/waiter/loop sources. Final exact work-path search scanned
**2,967 gpuR + 4,274 gpuB files, zero matches**. Two additional work directories were retained
because a retained waiter script names them. `chk_r10b` and all RESULT/gate directories remain.
Evidence: `grep_gpu*.json`, `work_grep_gpu*.json`.

All destinations below have prefix
`gpuB:/root/autodl-tmp/c4/archive_gpuR/S-MAINSTART-20261026/`.
For each: rsync, compare **every file's SHA256 and the full file set**, rehash the unchanged
source, enumerate symlinks again (**none**), then remove only that verified local work tree.
No unarchived deletion; no protected process signalled.

| Source relative to gpuR v5 | Allocated bytes before move | Files verified | Manifest SHA256 prefix |
|---|---:|---:|---|
| `chk_r14/a/midrun_work` | 66,420,736 | 115 | `dcae029b5f30` |
| `chk_r14/a/postrun_work` | 382,713,856 | 120 | `c069c9157c82` |
| `chk_r14/b/midrun_work` | 66,486,272 | 115 | `68b249369d4b` |
| `chk_r14/b/postrun_work` | 384,344,064 | 120 | `eb67caad061a` |
| `chk_r13e/a/midrun_work` | 65,884,160 | 117 | `8813982513d5` |
| `chk_r13e/a/postrun_work` | 382,005,248 | 122 | `1dbd23027e58` |
| **Total** | **1,347,854,336** | **709** | Full hashes in archive receipt |

Measured net scratch free-space gain was **1,347,178,496 bytes** while PAIRFT continued writing:
**1,189,695,488 → 2,536,873,984 bytes**. The requested **6 GB scratch target was not attained**:
protected `data/` alone occupied ~47.0 GB of the 53.7 GB volume; the other protected files and
referenced gates prevent enough safe reclamation. This limitation is explicit in the
prelaunch amendment, rather than moving protected inputs.

Actual loop outputs are **`/root/main14_loop`**, on the separate root filesystem:
**12,780,822,528 bytes free before launch**, **12,759,285,760 bytes at final verification**.
The actual output filesystem passes the >=6 GiB launch check; scratch is still above its
original phase floor. `ARCHIVE_RECEIPT.jsonl` retains every source/destination/hash/count/free-space receipt.

## Detached launch and gates

Launch used **setsid nohup**, stdin `/dev/null`. Driver's final PPID/SID are **1/822919**;
stdout/stderr are `mainstart/driver.log`; its nonblocking duplicate-lock probe confirms the
exclusive driver flock is held. All 1,879 native source files and registration/config hashes
were reverified. **358 integrated CPU tests passed**, then the actual 13E/native-lane-ON
two-update CPU mechanics smoke completed. **9 focused amendment tests pass locally**,
including foreign/PID-reused GPU refusal, scientific/hash vetoes, no-charge unavailable
collection, and preservation of fine-tunes after an external base has already been adopted.

Policy phase began **12:11:59 UTC+8**. Trainer **823090 / start ticks 4161450603**, actual env
`C4_GPU_MEM_FRACTION=0.45`, OMP/MKL threads 2. At the 12:21 verification it used ~3,068 MiB;
owner-replaced PAIRFT **823269** used ~4,018 MiB with its separately registered 0.4 cap.
No peer PID is signalled or managed here. The wrapper's old PAIRFT incarnation allowance does
not admit arbitrary replacement PIDs; later WM training still waits for exclusive access.

### Successful-policy-boundary continuation (queued, not yet executed)

The registered correction persists the selected external FINAL and ratio: accepted loop
fine-tunes must not be reset back to that same FINAL at every later cycle. A later external
replacement must have a **strictly lower eligible ratio** than the previously selected
external base. Commit **3270f631** adds this correction and a guarded `--resume` mode.

Supervisor **875528 / SID 875528 / PPID1 / start ticks4161499060** is detached and waiting for
the exact old driver incarnation. This task's own `mainstart/STOP_ARM` asks driver822919 to
exit **after the current policy phase**; it does not interrupt trainer823090. Only if that
phase succeeds, collect is still pending/uncharged, and no owner/scientific/failure stop
exists, the supervisor preserves the old marker as `STOP_ARM_v1_completed` and launches
`mainstart_v2.py --arm ARMED_V2.json --resume`, again setsid/nohup. Failed phases are never retried.
The successor updates `PID.json`, `CONTINUED.json`, `CONTINUATION_INSTALLED.json` and
`driver_v2.log`. Original v1 and its arm remain immutable.

The handoff helper **823326** prepared cpuP, then self-exited on that same graceful marker;
the supervisor reinstalls it after successful takeover. Its source remains pinned to
`dfc91946` (`handoff.py` SHA256 `05b74eadf1b07f7009f3801911bb703a5569b456a8f8d6f9907898651c26fba3`).

## Collection / later base handoff

- cpuP `/home/ubuntu/c4/main14/runtime`: all **1,879 source files verified**; torch
  **2.12.1+cpu**, numpy **2.2.6**, SUMO modules import; configured network/template presence
  checked through an existing fixture-directory link. Preparation never opens assignment
  files and runs no SUMO. Receipt `~/c4/main14/COLLECTION_READY.json`.
- cpuQ was still completing its scoring environment at first preparation and remains a
  fail-closed fallback until its independent receipt/dependency/fixture checks pass.
- Readiness is checked again immediately before collect. Workers are <=12, additionally
  bounded by available RAM using S-NEWCPU's **1.8 GiB/worker +12 GiB floor**. No registration
  seed is consumed merely by waiting. After charge, failures retain the charged seeds and
  require owner review; no silent reallocation or host failover.
- The helper polls complete cpuP `r13e`/`r14b` and cpuQ `r16` qualification outputs. It stages
  explicit FINAL/config/physics files under `mainstart/bases/`, compares 14b to gpuR FINAL and
  round16 to gpuB FINAL, and atomically publishes `BASE_CANDIDATES.json` only with complete
  reference/candidate evidence. No verdict was available or presumed at this handoff.
- The driver reuses the frozen MAIN14 scientific comparator and writes
  `BASE_BOUNDARY_cNN.json` plus cycle-ledger provenance before any boundary swap.
- No phase overlap/two-box simultaneous collection/SPEED2 flags were adopted. Converted
  caches are retained; this wrapper does not recursively remove data directories.

## Monitor and stop

All paths below are on gpuR; `A=/root/autodl-tmp/c4/v5/mainstart`.

```bash
A=/root/autodl-tmp/c4/v5/mainstart
cat "$A/PID.json" "$A/CONTINUATION_WAITER.json"
tail -F "$A/driver.log" "$A/continuation.log" "$A/driver_v2.log" "$A/handoff.log"
tail -F /root/main14_loop/policy/c00.log
/root/miniconda3/bin/python /root/autodl-tmp/c4/v5/repo_main14/scripts/c4/loop14.py status \
  --config /root/autodl-tmp/c4/v5/main14_launch_config.json
cat "$A/WAIT_OWNER.json"                 # only present after a failed phase/launch

# Stop the planned continuation, helper, and loop after the current phase:
touch "$A/STOP_CONTINUATION" "$A/STOP_HANDOFF" /root/main14_loop/STOP_DRIVER
# Optional: stop the current policy safely at its next update boundary:
touch /root/main14_loop/policy/c00/STOP
```

Do not rely on `mainstart/STOP_ARM` alone during the queued takeover: it is already the
task-owned continuation marker. `STOP_CONTINUATION` plus `STOP_DRIVER` prevents replacement.
The original `main14_arm/STOP_ARM` remains in place. No kill-by-name, foreign PID signal,
automatic failed-phase retry, seed refund, or push is used.

Full final evidence: `tools/mainstart/EVIDENCE.json`; installation, launch, continuation,
registration/config and archive receipts are adjacent. Operational logs stay on gpuR.
