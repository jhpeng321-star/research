# S-R13C launch receipt

Status: **LAUNCHED**. Smoke passed; production and both waiters verified alive.

## Reproducible build

- Worktree `E:/agent-work/c4-v10r13c`, branch `c4/v10r13c`; anchor untouched.
- Base `15c430fe` merged minimally with `c4/v10r12c` (`c8dd7162`).
- Smoke source commit `bd3c16a8b8fce3f88e0e39ba44e492ee793d5b9f`;
  gpuR `/root/autodl-tmp/c4/v5/repo_r13c_bd3c16a8`.
- Production operational commit `14bdd8ec1cc22656d16a778c12f780febf055d67`;
  gpuR `/root/autodl-tmp/c4/v5/repo_r13c_14bdd8ec`.
- Code: branch `c4/v10r13c`, trained commit **14bdd8ec**. Subsequent receipt/handoff-artifact
  commits do not change the deployed model/config/trainer or production scripts.
- Both deployed copies come from git archives of runtime/config/tool/test files.
  Identical **src tree** `0c8de2a4c52def66da67f7111915da4d1b06d30f` and
  **configs tree** `521d06c6ee00976f89efb97a00b436103fe296e6`:
  the operational follow-up changes the waiter and launch handoff only.
- Config `configs/c4_wm_v10round13c.toml`, SHA256
  `8d3b156c2016040c0e975a25c7702a90d37d2198b8e78499d430c7d81b499f15`.
- Init `/root/autodl-tmp/c4/v5/runs/c4_wm_v10_round12c/wm_s.pt`, SHA256
  `c76c338dea6355f85cfaee63cdb9077fb129ed25db0ab582d77eda2961822091`.
  Loaded as-is: `driver_param_init=checkpoint`, `residual_accel_reinit=false`;
  only the courtesy head is fresh zero-initialized.

## Training contract

- Entire `driver_encoder` frozen, including learned physical centre and IDM/MOBIL settings;
  merge pivot/core and round-12 non-acceleration paths frozen. Plant physics is immutable config.
- Trainable: 84 residual attention/acceleration/conditioning tensors + 6 courtesy tensors
  (730,405 parameters); 69 frozen tensors (117,590 parameters). No unmatched freeze/LR prefixes.
- Temporal acceleration emission OFF, residual and courtesy ON; unused temporal MLP frozen.
- Repair A: state gradient on, causal pivot intent, absolute weight **4.525** per documented rule;
  courtesy rollout weights detached as in round 12, state adjoint preserved. Driver prior OFF.
- Exactly **40 steps at 120 s + 60 at 300 s**, LR `[5e-5,3e-5]`, batches `[4,2]`,
  truncations `[120,60]`, residual LR scale 0.5, courtesy LR scale 1000;
  validation cap 16 production / 4 smoke, wall-clock cap 8 h. Full pathwise adjoint retained.

## Gate and operational handoff

- One smoke only, child PID **522790**, started **2026-10-24 22:18:31 +08:00**;
  output `runs/smoke_r13c`, stdout `smoke_r13c.log`, schedule 2+2, batches `[4,4]`.
- Initial gate parent **522786** was SIGSTOPed at 22:24 before production, while smoke continued.
  Preflight found gpuR's Conda Python lacked native pidfd functions. The operational commit adds
  a Linux x86-64 syscall-backed implementation; harmless signal-0 preflight on an own process passed.
- Production handoff uses `scripts/round13c_production.sh` after smoke exit 0 and the unchanged
  fail-closed verdict. No smoke/training retry. Exclusive production lock prevents duplicate launch.
- One-shot operator handoff `R13C_HANDOFF.py`, PID **525703**, observed smoke exit **0**, retired
  only held gate **522786** using its pinned pidfd, ran the verdict once and launched production once.
  Both original gate/smoke and the completed handoff exited; production is independent of SSH.
- Verdict requires the round-12 objective/merge/courtesy/replay telemetry, exact optimizer list,
  three strict checkpoint reloads, frozen core bitwise vs S-R12C, both trained heads moved,
  repair-A flags and absolute weight, and an actual forward probe with temporal emission exactly 0.

## Registered monitors

- Registration appended before gate, operational amendment before production:
  `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md` (S-R13C, 22:18 / 22:30).
- Midrun: at `wm_s_stage120.pt`, recompute candidate / S-R12C / as-trained round 9 on paired panels;
  run `tools/r12_checks/midrun_check.py` + BASE checks. STOP own training PID if full-road lane-1
  HDV `a < -3` share in 60 s free rollouts on 24 TRAIN-corpus states is >2x S-R12C, OR all-lane1
  one-step MAE on 24 TRAIN-corpus windows exceeds S-R12C by >0.1 m/s². Measurement errors stop
  fail-closed. Legacy R12 decision is context only. SIGTERM uses PID/start-ticks/argv + pinned pidfd.
- State provenance verified: 24 fixed states, 12 native + 12 plain; all are internal validation-role
  states within TRAIN corpus seeds 963000..963959, not evaluation seeds. No restricted paths opened.
- Postrun: waits RUN_SUMMARY and own training exit, then full BASE checks including head-zero.
- Outputs `/root/autodl-tmp/c4/v5/chk_r13c/{midrun,postrun}`;
  large intermediates `/dev/shm/chk_r13c_{midrun,postrun}`. Eight CPU workers per serial phase,
  single-threaded numerical libraries, nice 15, lock serializes both monitor modes.

## Verification

- gpuR focused tests: **62 passed** (R13C config/optimizer/temporal probe, waiter control/metrics/
  pidfd fallback, R12C physics-only, R13F temporal-off, R13A courtesy).
- Bash syntax: gate, production phase and initial start wrapper passed.
- Local: **29 waiter tests passed**, Ruff passed; initial local model tests could not collect
  because local Python lacks torch, so model tests ran successfully on gpuR.
- Actual diagnostic runner CLI, array/file naming, JSON schemas, checkpoint flags/physics sidecars,
  and TRAIN panel provenance passed remote preflight.
- Disk 2.8 GB free after builds; no existing remote repo or other user's process removed.

## Final launch facts

- Production PID: **527550**, start ticks **4148134828**.
- Started: **2026-10-24 23:12:41 +08:00** (gpuR local time; `production_r13c.start`).
- Output: `/root/autodl-tmp/c4/v5/runs/c4_wm_v10_round13c`;
  stdout `/root/autodl-tmp/c4/v5/production_r13c.log`.
- Midrun stop waiter: **527557**; postrun BASE waiter: **527558**, both started at 23:12:41,
  verified alive at 23:16–23:17 and bound to production PID/start ticks in their `CLAIM.json` files.
- Smoke verdict: **PASS**, `smoke_r13c.gate.json`, no errors; stdout
  `smoke: ok (round 13C); temporal emission = 0` also present in `smoke_r13c.log`.
  Four optimizer steps, 81 positive TF events, zero TF zero-mass events; scored seconds
  3237 at 120 s and 1565 at 300 s; 4090 courtesy rows, 72,319 contrast controls,
  2535 filled-replay extra rows; peak **3314.04 MiB**.
- All three checkpoints (`wm_s_stage120.pt`, `wm_s_stage300.pt`, `wm_s.pt`) reloaded strictly,
  temporal emission **exactly 0.0**, frozen core **bitwise equal** to S-R12C.
  90 trainable tensors; changed tensors 83 / 85 / 85, all confined to the permitted paths.
- Actual production resolved config verified: stages `[40,60]`, validation cap 16, temporal OFF,
  checkpoint-mode driver init, residual reinit false, prior zero. Actual optimizer lists exactly
  90 residual/courtesy tensors and no driver/physics/pivot/temporal parameters; init SHA matches,
  160 tensors loaded as-is, six fresh courtesy tensors, no shape mismatch/expansion/unused tensors.
- Production **step 1 completed**: loss **269.3590208248735**, finite true,
  merge-pivot gradient **0**, TF zero-mass events **0**, peak **1773.98 MiB**.
- Training was launched about 83 minutes after the 21:50 restart; the single prescribed smoke
  consumed about 54 minutes including validation. No validation was skipped to shorten it.
- Remaining work is the registered asynchronous training/midrun/postrun checks; launch acceptance
  is complete. No blocker and no additional launch scheduled.
