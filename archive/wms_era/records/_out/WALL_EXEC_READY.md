# WM wall executor integration

`output/wd01/wm_n1/wms/wall_exec.py` provides `TorchWallExecutor` and the
independent `NumpyWallExecutor` reference. Each executor represents a batch of
independent rollouts, with one wall per batch row and a fixed `[B,K]` member
mask. It implements the copied v2.1 closure law and the v2.2 merge-release
variant. State inputs are predicted downstream-increasing positions in metres,
predicted speeds in m/s, predicted lane indices, and optional predicted
presence. Control step duration defaults to 1.0 s.

## Calls in each 1 s control step

1. At the start of the control step, resolve the planner's wall proposal and
   planner abort/release signals. Keep the wall member roster fixed once the
   executor is created. Save each member's controller id from the per-CAV
   action timeline immediately before ownership begins (`prior_controller`,
   timeline ids native=0, damping=1, PI external-U=2, direct=3).
2. After producing the current predicted state, call
   `executor.step(position_m, speed_mps, now_s=t, lane_index=..., present=...,
   abort=..., release=..., base_acceleration_mps2=...,
   base_target_speed_mps=...)`. The call is after state prediction and before
   the model's action-conditioned transition. `abort` is a per-batch boolean
   and immediately hands back during assembly or hold. `release` is honored
   only in hold, matching the simulator's planner-release rule. v2.2 also
   performs its merge-start release internally.
3. Install the returned `controller`, `acceleration_mps2`, and
   `target_speed_mps` for that control step in the existing action-timeline
   tensors. Owner/controller id 3 marks members actively controlled by the
   wall; nonmembers keep their ordinary timeline controller and command.
   For assembly/hold, the target is `predicted_speed + clipped_wall_accel *
   control_dt_s`, i.e. the simulator's `_pace` command translated to the
   direct timeline record. For a handback tick the wall emits no action and
   returns the saved pre-wall controller with the supplied ordinary timeline
   command.
4. Exclude every currently wall-owned member from per-CAV timeline control
   selection/controller evaluation for that step. Use the executor's wall
   action for those slots exactly once; do not apply a concurrent native,
   damping, PI, or direct timeline action. On release/abort, wall ownership
   ends immediately and the saved pre-wall per-CAV action resumes on that
   same step. This executor does not mutate the input timeline.

The result also includes `phase`, `phase_names`, `abort_reason`, and `events`
for rollout diagnostics. Planner abort reason is `planner_abort`; simulator
safety reasons (missing member, unavailable position, span, lane coverage,
rear-end risk, assembly timeout) follow the copied wall session vocabulary.
Only CPU execution is supported.

## Verification

Run from the workspace root with the requested interpreter:

```powershell
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q output/wd01/wm_n1/wms/tests/test_wall_exec.py
```

The tests load `_in/traffic_shepherd/wall_group.py` directly with synthetic
frames; no SUMO or traffic data is used.
