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
- Retry detached at06:22:36UTC+8: coordinator244051/start851997043/SID244051; trainer244055/start851997060/SID244055. u515–516 completed with finite metrics; early sampled process peak9036MiB, allocator-reserved peak9028239360bytes. These are intermediate measurements; final16-update verification is still required. Three no-simulation boundary tests PASS, focused fatal-error Ruff PASS.
- **Schedule audit invalidated that intermediate attempt:** fresh `looplong.configure` shifted the original saved knots `[0,380,463]` to `[0,514,597]`, returning eligible starts to60s. Our own c03.a5 STOP marker was obeyed at06:29:28; coordinator/trainer exited without signals and without SUMO. Its low-memory figures are not90s evidence; all a5 outputs remain archived and excluded from selection.
- Fixed same-phase looplong recovery to restore the SHA-bound checkpoint's exact horizon/padding/exploration/RNG and original unqualified-CQR refusal, with no re-anchoring/refit. Three new regressions cover90s schedule/RNG/refusal parity and changed gate/boundary rejection;18 recovery/boundary tests PASS. Prospective append-only correction and remote `S-C3RESUME_SCHEDULE_FIX` event installed06:33:13; fresh c03.a6 is bound to ORIGINAL u514, not a5 state. R2 recovery/config receipts live under `arm/loop/longint/c3resume/r2/`; budget remains213406.
- Corrected retry launched06:34:00: coordinator246214/start852065490, trainer246217/start852065504. Actual u515 at90s completed in252.62s, finite actor loss0.0014353/gradient norm711.77 (pre-clip); u516 at the unchanged low-CAV10s cap completed in14.96s. Early process peak12624MiB; final verification pending. `r2/SCHEDULE_VERIFIED.json` independently confirms u516 saved looplong state exactly equals original u514 (knots/padding/exploration/boundary/CQR), memory segment8. Frozen native-best remains from the original phase.
- **Policy recovery COMPLETE07:52UTC+8:** all16 remaining updates515–530 finite; original150-update phase and `[0,380,463]` curriculum preserved. Peak allocated16778883072bytes (15.63GiB), reserved17219715072bytes (16.04GiB), sampled owned-process17666408448bytes (16848MiB/16.45GiB), all below46797168312.32bytes/43.58GiB cap0.92. Process sampling interval5s; allocator peaks are exact PyTorch telemetry.
- Unchanged frozen10s native-relative panel PASS16/16; full-curve selector chose **u530**, mean policy-minus-native J **−0.8336534500**, checkpoint SHA `0978aa7288cf9e36a0b61949b1e236b3fe48a0d220dada64f67a4797f969dc5e`, on WM2 `b8797242`. No a5 measurements enter selection. Full remote proof `r2/POLICY_VERIFIED.json`; compact committed copy `tools/longint/receipts_c3resume/POLICY_VERIFIED.json`.
- Both cpuP/cpuQ zero-charge preflights PASS on that exact policy (SUMO not started by preflight; CPU ONNX provider). MAIN's conditional GO dispatched the first serial canary at07:52:54; four-world HARMFUL-only guard and collection/conversion remain in progress, with WM-FT boundary still pending.
- Both serial canaries and four guard worlds completed. Guard968183/184/185/186 policy−native delay (entrance waiting included; sum of two paired300s starts/world): **−2208.3985/−687.2085/+1787.9815/−651.1103veh-s**; mean **−439.6839**, registered5000 whole-world bootstrap95CI **[−1819.0764,+1169.1840]**. **INCONCLUSIVE, not PASS**; MAIN's HARMFUL-only rule continued the remainder. Probes OFF. Exact compact receipt `receipts_c3resume/GUARD.json`.
- Collection operational disk refusal08:10:30: all13 reservations charged36933 (total250339), but five cpuP remainder worlds968187/189/191/193/195 refused BEFORE any CPU work/input/SUMO creation at the unchanged1.5GiB floor. Four cpuQ remainder worlds968188/190/192/194 completed with zero incidents and valid artifacts while coordinator unwound; all four were SHA/provenance checked and adopted without replay, leaving8/13 complete. Every failed reservation/charge retained.
- Prospective repair amendment and remote `S-C3RESUME_COLLECT_DISK_REPAIR` binding installed08:22:42. CPU-Q lossless hardlink dedup examined1232 owned immutable input/observable-array files, shared580 SHA-identical duplicates (214073818 logical bytes), preserved every pathname/content; free1953603584→2168516608bytes. No unrelated files/processes changed. Retry ONLY the five pre-SUMO failures once on cpuQ with concurrency1, startup canary first, original guard unchanged; new14205 reserve projects **264544/280000**,15456remaining. `tools/longint/continue_c3.py` enforces aggregate disk readiness and collect→convert→WAIT_MAIN_C3_WMFT only; full receipts under `r2/collect_disk/`.
- Initial repair launcher266297 exited before dispatch because the canonical two-box schema requires ordered host names even when selecting one healthy box. Schema corrected/rebound with a zero-charge event; log/PID/config retained. Validated continuation **266633/start852744813/SID266633** launched08:27:14; zero-charge whole-five-world disk check PASS2162647040free≥2034237440required bytes, including unchanged1.5GiB floor and serial temporary allowance. Only cpuQ selected; retry canary968187.a2 dispatched08:27:37. Three lossless-dedup/collect-only-boundary tests PASS; focused Ruff PASS.

## S-C3RESUME collection/conversion COMPLETE — WAIT_MAIN_C3_WMFT (2026-10-28 08:53:58 UTC+8)
- All13 reserved worlds968183–968195 completed exactly once successfully; original eight successes reused, only five pre-SUMO failures retried at attempt2. All13 durable observable artifacts hash-verified; zero branch collisions/teleports. Probes OFF, with no probe exports in artifact manifests.
- Final charged **264544/280000**, **15456remaining** = prior213406 + original36933 reservations +14205 replacement reservations. All five original failed charges remain; no refunds or successful-world replay. Full18-attempt ledger validated.
- Canonical conversion produced **52episodes,26paired groups,26policy targets**, matching native-return export and common-start pairing receipt. Paths/SHA256s are in `tools/longint/receipts_c3resume/COLLECTION_COMPLETE.json`; GPU data `arm/loop/data/pairs_c03.a1`, returns/pairing `arm/loop/returns/*c03.a1*`.
- **STOPPED at WAIT_MAIN_C3_WMFT**. WM fine-tune and check both pending/attempt0; STOP_DRIVER preserved. Exact-incarnation verification confirms trainer246217 and continuation266633 naturally exited. Main retains the WM-base decision. Full numerical/artifact verification `r2/RESULTS.json` SHA631de924708d7fa16cd70604f0ba9013d66d767a201dc53e8b4ac6d869c1d87c; boundary receipt `r2/FINAL_VERIFIED.json`.
