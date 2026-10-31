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
- [11:52 ADDENDUM] r1 == r0 EXACTLY (paired_J_MAE and every per_group value identical;
  only wallclock meta differs; local sha-verified copies). Frozen-rerun noise = 0 by
  determinism (all rollout randomness is per-call CRN-seeded). Killed my own PIDs
  147313/147315 at 11:57:21 mid-r2 (no partial r2 file; record_panel writes only at
  completion); r2/r3 skipped per MAIN. Step-0 baseline = identical r0/r1 mean, SD 0.
- [done 12:09] CONTINUATION launched (gpuA): PID 160005 (timeout 21570s fresh cap per
  addendum item 3; projected PASS at ~2.5 h leaves >= 3 h curriculum), python child of
  timeout, `tools/wmfixmerge.pilot_continue.py` sha256 1d8be175... (git d4476e43;
  box tests 8 passed). Steps 1-30 only on gpuA (no inline panels); step-10/20/30 full
  panels evaluated on gpuR via `panel_eval.py` (same code/inputs/CRN, hash-verified
  relay) + a gpuR-native step-0 baseline panel as a same-hardware guard;
  PILOT.json assembled by the unchanged pre-registered decide() (improvement > 2x SD
  with SD=0 => strict improvement at both horizons). PASS -> curriculum continues
  in place under the fresh cap; FAIL -> stop, WMFIX_PILOT_FAIL.md, option (c).
- [done 12:26] steps 1-30 DONE on gpuA (PID 160005 healthy; loss 0.0593@1 -> 0.0010@30;
  all 30 grads finite; STEP0_TERMS paired_share 0.957). Exports step_000010/20/30.pt
  sha-verified and relayed (gpuA -> local -> gpuR, WMFIX_PANEL_RELAY.md).
- [12:40] GUARD FIRED: gpuR step0 panel != gpuA r0/r1 bitwise (283.485/2186.628 vs
  283.288/2185.236; ~0.07% cross-hardware bias). Per the pre-registered guard rule
  all DECISION panels are gpuA-native (MAIN 12:45 parallelization): step20/step30
  PID 165896/165897 from 12:41:58, step10 PID 166449 from 12:44:09, direct into
  train/ with the identical panel_eval path; gpuR step10 kept as cross-check only.
  Decision expected ~13:12 (sequential watcher would have been ~14:05).
- [pending] gpuA-native panels 10/20/30 -> PILOT.json decision (decide() unchanged,
  baseline r0/r1, SD 0) -> PASS: curriculum continues in place; FAIL: option (c).
