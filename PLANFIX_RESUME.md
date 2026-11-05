# S-VARIETY resume — 2026-11-05 16:12 +08

Recovered the original f631ce90 collector source and existing local operator payloads without changing numerical code. The completed 301-test/9-subtest qualification and 11 Linux executor regressions are retained in the run's PLAN_FIX_READY.md. No qualification is inferred for gpuA.

- cpuV v3 bank dispatch: PID 248230, start_ticks 5405052, seven untouched inputs. Original none-labelled 986052 startup failure remains excluded and charged 2,400 sim-s. Do not repeat or dispatch this job again.
- Recovered the old watcher's successful 16:09 read: v3 had exited BEFORE charging its first wave because the pace comparison included B's owner/safety pause (observed 4,712.51 s versus baseline 987.57 s). This is a pacing hold, not a native SUMO failure. Phase spending was still 2,400. Preserve its STARTED/FROZEN_SNAPSHOT/verification claim; a new namespace requires an independent zero-physics/zero-claim proof and prospectively registered inactive-training pace handling.
- Original local reconnect watcher PID 4160 (creation 15:48:54) is being stopped using its own durable STOP file, without signalling any remote workload. Its direct Windows SSH transport does not meet F6.
- Its 16:13:17 exit notice is durable. No replacement background watcher has been launched.
- Fresh read-only access at 16:09 failed on the old WSL socket; socket check confirmed absent. A shared F6 pool attempt at 16:12 failed during SSH key exchange, before any remote command. Current bank progress and phase spending are therefore unverified.
- The latest registration records MAIN's clean2 restart override. The independent plan_fix reservation remains 600,000 on clean1's sole ledger; clean2 must protect its remaining allocation without double-counting prior charges.
- GPU package: `E:/Temp/opencode/planfix_f631_gpu_src_361a974fd6b9274a.z`. Existing GPU stage/worker roots and any key backup/diff must be inspected before staging. Recovery will use small content-addressed chunks, bounded logged attempts/backoff, SHA verification of every chunk before assembly/staging, and Linux OpenSSH multiplexing.
- World-model search remains required. No non-WM substitute and no automatic bank switch is authorized. The existing u12 switch remains a prepared-only target; MAIN owns any clean2 rebinding.

Authoritative run documents: `E:/research-dense/runs/active/C4-V7FT-20261017/{REGISTRATION.md,CLEAN1_STATUS.md,PLAN_FIX_READY.md}`. Current readiness is false pending actual bank/variety/all8 receipts.

## 16:46 checkpoint

- Found an actual F6 startup defect: `-o ControlMaster=yes -M` becomes `controlmaster ask` under the installed Linux OpenSSH. `planfix_cpuv_transport.py` corrects this in an operator subclass while retaining the identical shared socket, admission gate, rate and no-direct-fallback checks. Fresh multiplexed reads succeeded16:38. No shared master was signalled. One interrupted OWN local reservation (parent1680) was archived only after proving its parent/supervisor/session absent and deadline elapsed under the shared lock.
- Prospectively registered v4 pace handling in local REGISTRATION.md and remote AUTHORITY.md. All existing clean1 owners exited, exact MAIN restart stop exists, clean2 absent; any change stops our next wave. All v2/v3 evidence and failed986052 remain preserved.
- **v4 dispatched16:45:57: PID264975/start5820583, boot2c19986e-dd5e-4de9-95ca-3212b1ea0568,8workers.** Linux11tests passed2.73s, actual stopped-owner guard passed, SUMO1.27.1 bound. Remote BANK_DISPATCH_v4.json is authoritative. Do not repeat this dispatch; use planfix_probe_v4.py.
- GPU source rsync reached100%1,345,145bytes16:43, then timed out at transport completion. Stage attempt16:46 also timed out without verification/worker output. **Inspect stage1/worker1 and source files before any repeat.** No eight-gate/key success is inferred.

## 17:30 checkpoint — authoritative latest execution

- v4 failed before SUMO startup; retained phase4,800. Exact spawned-process zero-SUMO diagnosis found the missing staged `runtime_src/baselines/circles/adapter.json`. Placing the already-qualified physical `/home/ubuntu/e3final/runtime` before incomplete runtime_src imports fixed that startup contract; the exact spawn probe reached SUMO Popen with process creation explicitly blocked. No diagnostic SUMO ran.
- v5 PID271505/start5991182 ran four full branches, then final all8 publication hit the operator charge-row schema omission `actual_sim_s`. All measurements/9,600 phase charges were retained. Both native receipts/four RESULTs/snapshot/STARTED are SHA-bound by `ADOPTION_V6.json`.
- v6 added only the missing charge metadata, checked cached branch/window/SHA/sole-ledger identities, and completed native all8 publication with reservation and process-pool construction forbidden. The first measured source's two translated gains/lower bounds are0 and correctly rejected; original measurements unchanged, zero repeated physics. CPU11tests passed2.21s. Transport loss interrupted the final coordinator dispatch; independently proved no old owner/dispatch before finishing that step.
- **Current coordinator v6 PID274411/start6066509, boot2c19986e-dd5e-4de9-95ca-3212b1ea0568, dispatched17:26:56.** Uses bank_verification_v5, adopts its completed first result and processes five untouched inputs. Verified alive17:30. Do not duplicate it.
- Completed measured inputs so far: world986066/sourceCRN2 (original+1,066.808582),986068(+3,216.944238),986003(+2,909.829894), wall translations: gains0/lower0 on both native verification CRNs, all rejected. 986068 also locally ineligible. No survivor yet. Phase21,600/cumulative1,237,440 at17:30; later charges require fresh receipt. Failed986052 and986066/sourceCRN1 remain excluded, gainsunknown.
- GPU rsync progress100% was **not durable full-package evidence**: remote hash read showed the old source still294,912bytes. Switched to4KiB content-addressed pieces. 72 matching prefix pieces plus newly transferred pieces are independently SHA-verified. Dedicated bounded transfer owner **WindowsPID42852** (17:04:45) continues with six-failure backoff and one guarded staging attempt after all329pieces + whole SHA; no SUMO. Log `E:/Temp/opencode/PLANFIX_GPU_RESUME.log`, owner receipt `PLANFIX_GPU_RESUME_OWNER.json`. Do not duplicate it or change its immutable helper files while live.
- CPU endpoint prepared17:29:59: `root@planfix-gpua`, single owner's restricted warm_rpc identity, pinned known-host key, ControlMaster/ControlPersist600; config backup retained. No admin identity copied, no RPC attempted before GPU key/8gates. GPU authorized_keys backup/diff and worker qualification remain unconfirmed.
- `planfix_progress.py 6` reads current owner/results/ledger, appends PLANFIX status, and caches compact per-plan receipts. Any final bank view remains prepared-only; MAIN must prospectively bind the eventual clean2 update boundary and R's effective family distribution.

## 17:36 bank terminal receipt

V6 owner has exited and BANK_RESULTS.json is complete. Six source examples on worlds986066(sourceCRN2),986068,986003,986026,986036,986047 each completed both native verification CRNs: all twelve translated gains and lower bounds0, all rejected, **zero bank survivors**. Two startup-failed sources986052/986066(sourceCRN1) remain unverified/excluded with null translated gains, not scientific zeros. Original gains/family/move/context and all seals are retained in `E:/research-dense/runs/active/C4-V7FT-20261017/PLAN_FIX_BANK_RESULTS.json`; compact progress receipt `PLAN_FIX_BANK_PROGRESS_6eb6501567de.json`. No bank dispatch remains live; do not restart it.

Phase total33,600 =28,800 measured full-branch sim-s +4,800 retained startup charges. Cumulative including parity1,249,440/5M. Unspent plan_fix reservation566,400 must remain protected; global remainder3,750,560 leaves3,184,160 disjoint from that reservation for MAIN's clean2 planning (arithmetic only, no new launch authority).

GPU transfer owner42852 remains the only staging work. Its successful remote4KiB SHA receipts—not rsync sender progress—define coverage. No library example has been released for feed-in. F3's accepted handoff schema is `coopmwm-plan-fix-banks-v1`; until verified variety and the effective R teacher change are ready, `outputs` must be empty. F3 also requires matching verified B/R family support, which MAIN must reconcile with the requested R random-teacher distribution without substituting B's WM guidance into R.

## 18:07 latest owner override and fresh recovery

- MAIN18:03 supersedes GPU continuation: no further gpuA transfer attempts. Windows42852 is absent; durable `E:/Temp/opencode/PLANFIX_GPU_RESUME_STOP.json` now prevents accidental helper restart. All transferred chunks are preserved. No gpuA command was sent during this recovery.
- Read-only F6-multiplexed cpuV receipt at unix1793873244.6085885 confirms v6 exited, BANK_RESULTS terminal, plan_fix33,600 and clean1 including parity1,249,440. No bank job is redispatched.
- **clean2 root is absent at18:07.** Owner requires its launch before even one CPU WM search. Preparing code/zero-physics checks is allowed; actual search and SUMO remain gated.
- CPU WM search must use the same pinned f631ce90 WM/scoring, frozen MAPPOu0, at most19/48 CPU cores (<=40%), nice19/one-thread workers, and yield if clean2 update pace worsens. No non-WM proposals or automatic feed-in switch.

## 18:36 CPU source preparation checkpoint

656 immutable dependency files/8,065,516bytes are SHA-staged on cpuV under
`plan_fix/cpu_prepare_ea559c952ff0`; original WM2 checkpoint b8797242 is already
present and reverified. `PLANFIX_CPU_PREPARED.json` / `PLANFIX_CPU_HANDOFF.md`
record exact source hashes and missing/unmapped bootstrap dependencies. No
runtime bootstrap, startup gate, search or SUMO ran. clean2 remains absent.
The same-scoring port needs original32 frozen source-prior donors/config/export
mapping; do not fabricate substitutes or resume GPU transfers. Remaining
plan_fix566,400 remains reserved. See handoff for exact next authorized sequence.
