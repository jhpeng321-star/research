# S-MAPPOSPEED — speed-only MAPPO (owner-approved 2026-10-30 16:05)
- Worktree `E:/agent-work/c4-mappospeed`, branch `c4/mappospeed`, base `9613f956`; never pushed.
- Mode: `[policy] lane_control="native"`; default `learned` preserves existing MAPPO.
- Native mode removes the lane head, lane action/log-prob/entropy; PPO joint log-prob = accel only.
- Rollout + exported executor pass `lanes=None`, not an all-`-1` tensor (which enables mode 513).
- CTRL0 proof: cpuQ `~/ctrlarm/amend2258/tools/ctrlarm/zero.py` requires `lanes is None`.
- `CavSumoEnv._control_tick` then leaves native laneChangeMode untouched; prior owned modes restore exactly.
- Init choice: `mappo_v5_final.pt`, SHA256 `aecd7d70ccce2995a10d03d9400f587040b1d3b435eebeab2561768ba57c3564`.
- Accel/lane heads are parallel: strip only four `lane_head.*` tensors, strict-load everything else.
- Accel inputs still include observed lanes and previous commands/intents; no lane-head output feeds accel.
- Retain checkpoint accel bounds; fresh optimiser/counter/scaler; exported actor means fresh critic/dual.
- CPU checks on cpuQ: 32 passed, 1 SUMO test skipped, 2 checkpoint-deletion tests deselected.
- Checked: accel parity/log-probs, zero lane entropy, spawned workers, strict init, resume, both loaders,
  SUMO baseline-arm loader/step, fake-TraCI no changeLane/get/setLaneChangeMode, original-mode restore.
- No SUMO has run yet. Allocation, launch commands/PIDs, ETA and first-two-iteration comparison follow.
- Evaluation plan: WM rank vs CTRL0 via `tools/setupaudit` (integration tree); SUMO arm of S-EVAL-A,
  with checkpoint hash/arm/budget amendment registered there before evaluation. No evaluation seeds train.
