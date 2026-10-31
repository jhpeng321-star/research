# S-WMFIX-PILOT — amended gate (option b), MAIN REVIEW DECISION 2026-10-31 10:40

Authorization: owner-delegated review, recorded in C4-V7FT-20261017/REGISTRATION.md
10:40 entry; replaces the unsatisfiable FD-agreement sub-gate of the BLOCKED real
feasibility (WMFIX_FEAS_BLOCKED.md sha 65771d02...). Implementation:
`tools/wmfixmerge/pilot.py` sha256 `07da9e206a1f350cf26b8af8e4f8995f42784ac2e132cf936d96ef9301eda144`
(git a9ef6bbb; a NEW file — implementation_hashes() unchanged, verified identical
digest 2909d4263f072c93 on local worktree and gpuA staged tree, 30 files 0 mismatches).

## Pre-registered protocol (unchanged in code)
- Start: zero-head entry-path step_000000 (fresh WM2 b8797242, conserve_queue +
  learned admission ON, merge head at zero init), exported immutably with the
  train.py export path (frozen-bitwise checked per export, canonical reload).
- 30 Adam steps on the 12 FIT groups, registered lr 3e-4, stage-0 horizon 60 s,
  group-then-arm cycling identical to train.py; only merge_response.* in Adam.
- Frozen-rerun noise: 3 extra FULL-panel reruns of step_000000 after the baseline
  panel (4 evaluations total); noise = sample SD (ddof=1) per horizon; baseline =
  mean of the 4 (recorded per-run too).
- Full 12x2x{120,300} s TRAIN panel paired-J MAE at steps 0/10/20/30.
- PASS = best of steps 10-30 improves BOTH horizons by strictly > 2x SD, all 30
  gradients finite, frozen tensors bitwise; PASS -> continue curriculum in place
  (stages [120,30]+[300,60]) within the <= 6 h budget counting pilot time;
  FAIL -> stop training, WMFIX_PILOT_FAIL.md, option (c) design.
- Local tests: tests/unit/test_wmfixmerge_pilot.py 5 passed (+ existing 10).

## Launch (gpuA free: 1 MiB used before start)
2026-10-31 10:45 +08 (RUN.json/ps; the "~11:2x" in the first draft was the
receipt-writing time, not the launch). CWD /root/autodl-tmp/c4/wmfixmerge20261031, out dir `train`
(existing, empty; registered verdict name), PYTHONPATH=src:.,
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, memory fraction 0.35,
`timeout --signal=INT --kill-after=30s 21570s`. Launcher PID 147313, python PID
147315; log pilot_gpu.log; PILOT_PID.txt written by the launcher. Inputs identical
to the registered training command (TRAIN_PREPARED.json 7bbd41f3..., config
c4_wm_loop14_r16_ft.toml, checkpoint wm_s.pt b8797242, W5_COMBINED DESIGN.json).

## Progress (updated as results land)
- [done 11:13] PANEL_step0_r0: 120 s MAE 283.288, 300 s MAE 2185.236 (1710 s/panel;
  matches FEASIBILITY.json pre-step panel exactly — same measurement, same weights).
- [done 11:41] PANEL_step0_r1: 28.5 min cadence steady.
- [running] r2 (~12:10), r3 (~12:38) -> STEP0 baseline+noise; then steps 1-30
  + panels at 10/20/30 -> PILOT.json decision
- [pending] PASS -> curriculum continuation in place (pilot.py does this itself)
  to FINAL.json / FAIL -> stop
