# R12C LAUNCH — S-R12C physics-only core calibration (round 12C)

**Status: LAUNCHED** 2026-10-24 (receipt written after the smoke gate passed and the production run started).

## What this run is

Round 12C (S-R12C) = round 12 with every learned acceleration emission disabled and the trainable set
restricted to the driver encoder (physical mode) + the shared IDM/MOBIL physics parameters, to test
whether the IDM core calibrates when residuals cannot compensate (round-12 mid-run hypothesis: the
residual/temporal paths absorbed the core error; encoded accel p50 1.52, decel p50 3.14; offline
physics-only fit a 0.87, b 2.02, T 1.11, s0 2.08 from BASE_FOLLOWER_DIAG.md).

- Code: branch `c4/v10r12c`, commit **8d6f8133** (= 1878f842 + the R12C commit b170634a + the gate-fix
  commit; worktree E:/agent-work/c4-v10r12c). gpuR repo `/root/autodl-tmp/c4/v5/repo_r12c_8d6f8133`
  (git archive, hash-verified by construction from the commit).
- Config: `configs/c4_wm_v10round12c.toml` (sha256 00f281155b5da8a6…): inline copy of round 12 with
  `model.temporal_accel_off = true`, **new** `model.residual_accel_off = true` and
  `model.follower_courtesy_off = true` (default off; bit-identity vs 1878f842 on the round-12 config
  verified: 60/60 forward output tensors over 4 teacher-forced steps, equal short-rollout loss
  94.4401626586914, 159/159 parameter-grad hashes identical; probe E:/Temp/opencode/bitid_base.json vs
  bitid_r12c.json against a clean 1878f842 worktree).
- Trainable set: `driver_encoder.head` + `driver_encoder.centre_logits` ONLY (5 tensors; freeze list =
  round 12's + residual_net / temporal / follower_courtesy; merge pivot frozen as in round 12).
- Init: exactly round 12's (round-9 final `wm_s_stage300.pt`, sha256 af9bc643…, + round-12 init
  transforms: centre driver init, residual accel re-init, fresh zero courtesy head). Driver prior OFF.
  Losses otherwise round 12's.
- Schedule: single 120 s stage, 30 steps, batch 4, `selection.val_max_windows = 4`, `max_hours 2.5`.

## Launch facts

- Gate/production PID (gpuR): **497807** (`gate_r12c.pid`; the launcher execs the gate, which execs
  production, so the PID is the training PID; production cmdline =
  `python -m wms train --config configs/c4_wm_v10round12c.toml --mode run --out
  /root/autodl-tmp/c4/v5/runs/c4_wm_v10_round12c … --set loss.driver_prior.weight=0.0`).
- Started: gate 2026-10-24 15:57:04; second smoke verdict `smoke: ok (round 12C)`; **production started
  2026-10-24 ~16:17** (config.resolved.json 16:16:51, first train step logged 16:21:29; step 1 loss
  620.9, `raw_residual_abs_mean_mps2 = 0.0`, centre a 2.445 / T 1.162 / s0 2.656 / b 4.141 / v0 30.73).
- Smoke: `runs/smoke_r12c` (2 steps, val 4 windows), log `$L/smoke_r12c.log`, verdict
  `wms.train.round12c_gate` (gate report `$L/smoke_r12c.gate.json`).
- Production output: `/root/autodl-tmp/c4/v5/runs/c4_wm_v10_round12c`.
- Post-run checks: waiter `/root/autodl-tmp/c4/v5/round12c_autoarm.sh`, PID **497860**
  (`chk_r12c/auto_arm.pid`), waits for RUN_SUMMARY.json + gate PID gone, then runs the round-12 BASE
  checks (tools/r12_checks base_calibration.py BASE-1..5) and the mid-run read (midrun_check.py) on
  runs/c4_wm_v10_round12c/wm_s.pt (else stage120/best) vs round 9 and the round-12 stage120 arm, into
  the NEW dir `/root/autodl-tmp/c4/v5/chk_r12c/` (results: BASE_R12C.md/.json, MIDRUN_R12C.md/.json,
  R12_ARM_READ.md). Nice 10, <= 20 workers; one-step on the filled replay, round-12C config, r9 legacy
  arm-sets + r12 emission flags restored via --arm-set.

## Smoke evidence (first run 15:30, verdict module defect fixed, artifacts re-run 15:57)

- Trainable tensor list printed in the smoke log (`WM-S optimizer groups`): exactly
  `driver_encoder.centre_logits`, `driver_encoder.head.{0,2}.{weight,bias}` (5 tensors, 5831 params;
  154 frozen tensors, no unmatched freeze prefix).
- `physics_share = 1.0`, `raw_residual_abs_mean_mps2 = 0.0`, `residual_abs_mean_mps2 = 0.0` (residual +
  temporal emissions exactly 0); `follower_courtesy_head_{min,max,mean} = 0.0` (courtesy offset exactly 0).
- `merge_pivot_grad_norm = 0.0` (pivot frozen), `merge_pivot_tf_zero_mass_events = 0`, scored seconds > 0;
  cut-in term absent; driver prior absent (weight 0); fix-D telemetry present
  (`follower_courtesy_replay_filled_extra_rows >= 0`).
- The centre moved within 2 steps (a 2.445 -> 2.400, T 1.162 -> 1.189, s0 2.656 -> 2.711) and the
  driver head rows 0..4 all moved; nothing outside `driver_encoder.{head,centre_logits}` changed
  (bitwise vs the round-9 init, with the residual_accel_reinit row-0 zeroing accounted).
- Verdict: `smoke: ok (round 12C)` on the fixed module, re-verified against the first run's artifacts
  before the clean relaunch (gate report: 4 changed tensors + centre, all inside the trainable set).

## Process notes

- First gate run (15:30) failed CLOSED on a defect in my own verdict module (the frozen-bitwise check
  did not account for the residual_accel_reinit row-0 zeroing of `residual_net.accel.2` vs the raw
  round-9 init). No production was started. Fixed in 8d6f8133, verified on the same smoke artifacts,
  12c artifacts cleaned, ONE clean relaunch (no retry loop). Registration amended accordingly.
- gpuR disk: freed only the two authorized stale repo copies (repo_r10_d19c2066, repo_r9_3e5610fe_fit,
  299M each) before building repo_r12c (~3.0G free maintained).
- Registration: appended to E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md before launch
  (S-R12C entry + amendment).
- Local tests: tests/test_r12c_physics_only.py 9/9 pass; regression suites (r13f, r13a, r12 base, r11,
  r10b, r5tools, round12 gate verdict, cutin, merge pivot/core config, r10 config) 175+89 pass.
