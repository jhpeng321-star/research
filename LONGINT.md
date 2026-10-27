# S-LONGINT: cycle-3 policy phase on WM2 (2026-10-27)

## Merged and tested
- `c4/longint` = c4/restart + looplong 6a0f5dff, longmem c0b40eec, probe 030d2784, groupgrad acf7e308.
- Default-OFF flags `native_residual_train_group` and `native_residual_train_trunk` train the full actor (group, own, report and attention encoders).
- The residual output is still zero-initialised, so the policy starts exactly native. Tests show:
  - the initial WM rollout is bit-identical to native;
  - group and trunk gradients are nonzero after the first non-trivial update;
  - Adam state migrates on resume and restarts are exact.
- Per-module gradient norms, learning rate (5e-5) and residual size are logged every update.
- Focused gpuA suite: 132 passed. Full suite: 2425 passed, 35 failed, 14 errors, all outside the changed modules and apparently environmental (not checked against a baseline).

Three bugs on real data were fixed, each with a test:
1. Paired windows had no world metadata.
2. The cycle-2 pairs have no `entry_backlog` clocks. The CQR fit now refuses and gives zero credit beyond the trusted horizon.
3. `REAL_CREDIT_c02` (individual 0, coop 1) tripped the anti-gaming guard. Coop is now capped at the individual weight (reduced only, logged).

## Registration
- `REGISTRATION.md` appended before launch.
- Amendment `S-LONGINT-C3` added to `arm/REGISTRATION.json` (sha f61e6b58); the loop state is rebound to it.

## Horizon and long credit
- WM2 gate: unchanged registered source, 48 CPU shards.
- Sign agreement at 10/30/60/90/120/180/300 s: .921/.947/.963/.900/.763/.647/.637, threshold .868 → **trusted horizon 90 s**.
- The built curriculum runs at 60 s until update 463, then at 90 s.
- **CQR: not fitted, so not qualified** (no queue clocks). No credit beyond 90 s. Direct real 300 s pair credit is on; the critic bootstrap is off.

## Run (`arm/loop/policy/c03.a4`)
- Launcher PID 165047, trainer PID 165050; 150 updates (381→530).
- Updates 381–390 are all finite. Group-encoder gradient norm ranges 0.05–96 (pre-clip spikes of about 1.5e3, clipped at 100).
- Mean |residual| rose from 0.097 to 0.119 m/s² (different starts).
- Memory: 17 GB peak, stable. Disk: 7.8 GB free.
- Signal gate at update 390 passed: 16/16 starts improved.
- About 160 s per update. **ETA about 06:00 on 10-28.**

## Chain state
- Status `C3_POLICY_RUNNING`, then `WAIT_MAIN_C3_COLLECT`. `STOP_DRIVER` is kept; no SUMO collection is dispatched.
- Probes are not enabled (`probe_cqr.json` is missing).
- The converter must export `entry_backlog` clocks before the CQR or probes can qualify.

## S-C3RESUME — MAIN06:20 operational recovery registered (2026-10-28)
- Read LONGINT/AUTONOMY and live gpuA status first. c03.a4 failed CUDA OOM after saved update514; GPU idle (1 MiB). Original state/attempt/checkpoints retained; charged213406/280000.
- Append-only prospective MAIN amendment installed in authoritative REGISTRATION.md and remote registration/state rebound to `d8b5fde5`; failed-attempt and retry events appended to both ledgers. Recovery receipt: `arm/loop/longint/c3resume/RECOVERY.json`; u514 SHA `1c3dbff0`, WM2 `b8797242`.
- Same150-update phase381–530: full trainer/Adam/RNG recovery,16 remaining updates; full original curve and progress gate retained. Actual eligible saved best is u490 (`b06efb13`), rather than the preliminary u460 observation; unchanged selector remains authoritative.
- Memory-only retry: start microbatch1, checkpoint segment8 (was16), both host offloads ON, expandable segments, cap0.92. Boundary GC/unused allocator-cache reclamation changes allocation lifetime only. Per-update finite/allocator checks and5s owned-process GPU-memory sampling are required before collection.
- Recovery/LONGMEM focused suite:33 passed; one temp-path workspace refusal passed when rerun with workspace-local basetemp (34 distinct checks passed).
- New bounded coordinator `tools/longint/resume_c3.py` runs only policy retry → unchanged gate → registered zero-charge preflight/canary/guard collection → conversion → `WAIT_MAIN_C3_WMFT`, preserving STOP_DRIVER. MAIN explicitly authorized collection conditional on the gate; probes remain OFF.
