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
