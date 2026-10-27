# S-LONGINT: cycle-3 policy phase on WM2 (2026-10-27)

## Merged and tested
- `c4/longint` = c4/restart + looplong 6a0f5dff + longmem c0b40eec + probe 030d2784 + groupgrad acf7e308.
- Two default-OFF flags, `native_residual_train_group` and `native_residual_train_trunk`, now train the full actor: the group encoder and the own/report/attention encoders.
- The residual output is still zero-initialised, so the policy starts exactly native. Tests show:
  - the initial WM rollout is bit-identical to native;
  - group and trunk gradients are nonzero after the first non-trivial update;
  - the legacy head's Adam state migrates on resume, and restarts are exact.
- Per-module gradient norms, learning rate (5e-5 for all modules) and residual size are logged every update.
- Focused gpuA CPU suite: 132 passed. Full unit suite: 2425 passed, 35 failed, 14 errors. All failures are outside the changed modules and appear environmental; not checked against a baseline.

Three bugs on real data blocked the launch; each is fixed and has a test:
1. Paired-branch start windows had no world metadata.
2. The cycle-2 pairs have no `entry_backlog` clocks, which crashed the CQR fit. It now refuses cleanly and gives zero credit beyond the trusted horizon.
3. `REAL_CREDIT_c02` (individual 0, coop 1) tripped the anti-gaming guard. The coop weight is now capped at the scaled individual weight (reduced only, never increased) and the cap is logged.

## Registration
- `REGISTRATION.md` appended before launch (sha ee3f7b06).
- Amendment `S-LONGINT-C3` added to `arm/REGISTRATION.json` (sha f61e6b58); the loop state is rebound to it.
- Config: `arm/CONFIG_C3_LONGINT.json`.

## Horizon gate on WM2
Unchanged registered gate, run in 48 CPU shards. File: `gate_wm2/HORIZON_GATE_WM2_REG.json`.

| Horizon (s) | 10 | 30 | 60 | 90 | 120 | 180 | 300 |
|---|---|---|---|---|---|---|---|
| Sign agreement | .921 | .947 | .963 | .900 | .763 | .647 | .637 |

Threshold .868 → **trusted horizon is 90 s**. The built curriculum runs at 60 s until update 463, then at 90 s.

## Long-credit (CQR) qualification
**Not fitted, so not qualified:** the real pairs have no queue clocks. There is no credit beyond 90 s. Direct real 300 s pair credit is on; the critic bootstrap is off.

## Run (`policy/c03.a4`)
- Launcher PID 165047, trainer PID 165050; 150 updates (381→530).
- Updates 381–390 are all finite.
- Group-encoder gradient norm ranges 0.05–96. Updates 383/384 had pre-clip norms of 1.2e3 and 1.5e3, bounded by the clip at 100.
- Mean |residual| rose from 0.097 to 0.119 m/s² (first vs last updates, different starts).
- Memory: 17.0 GB peak, stable. Disk: 7.8 GB free.
- The signal gate at update 390 passed: 16/16 starts improved, mean ΔJ −0.032.
- About 160 s per update. **ETA about 06:00 on 10-28.**

## Chain state
- Status is `C3_POLICY_RUNNING`, then `WAIT_MAIN_C3_COLLECT`. `STOP_DRIVER` stays in place; no SUMO collection is dispatched.
- Probes are not enabled: `probe_cqr.json` is missing.
- The converter must export `entry_backlog` clocks before the CQR or probes can qualify.
