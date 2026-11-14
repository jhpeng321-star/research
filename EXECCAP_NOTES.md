# EXECCAP: SUMO-consistent execution limit of commanded CAVs in WM-S imagination (2026-10-14)

Branch `c4/execcap` (from `c4/int` a2d3716). Code, tests and a verification replay only. No GPU training,
no SUMO. Whether a policy is retrained with the switch on is the owner's decision.

## Why

The cmdreal diagnosis (`E:/agent-work/tmp/cmdreal/`) found the cause of MAPPO's over-credit in the decision
check. In SUMO (speedMode 27) a commanded CAV realises about min(issued command, its own vType IDM
acceleration): MAE 0.086 m/s^2 over 318k CAV-seconds, and the IDM cap explains 96 % of overrides. In WM-S
imagination, command-driven CAVs only passed the stopping-distance filter and the accel bounds. Imagined
CAVs therefore accelerated where SUMO's could not.

## What changed

* **Switch** `WmSModel.cav_exec_cap = "off" | "idm"`. It is a runtime attribute like `checkpoint_forward`
  and is never stored in a checkpoint. The default `"off"` is bit-identical to a2d3716 (see Verification).
* **Ceiling** `wms.model.cav.exec_accel_cap`, computed in `WmSModel.step` when the switch is on and any
  CAV is present:
  `cap = min(out["physics_accel_mps2"], (v0 - v)/dt, accel)`.
  * `physics_accel_mps2` is the physics core's IDM for every present slot, CAVs included. It uses the
    slot's driver settings for that step, incl. the accel-lane-end obstacle and per-driver accel/decel
    bounds. The settings are the supplied `driver_params`, else the ones encoded in this forward, else
    the defaults.
  * `v0` and `accel` come from the same settings.
  * CAV slots do get inferred settings. The trainer and coop-imagine encode every slot as a driver, the
    encoder reads only kinematics and history, and `is_cav` is not an encoder input.
* **Where it applies.** Every command executor applies `a_exec = min(a_cmd after the existing safety
  filter, cap)` (`wms.model.cav.apply_exec_cap`) to its commanded slots on each 0.5 s physics step.
  Next v and s are recomputed from `a_exec` by `direct_command_motion`. The executors:
  * `wms.control.bridge.make_cav_controller` (Dreamer trainer, RSU check, deploy candidates): acting
    slots only.
  * `wms.control.coop_imagine.CallableCavController` (coop-imagine and branch_horizon arms).
  * The trainer's inline integrator (`DreamerTrainer.imagine` without a controller factory).
  * Direct action timelines (`WmSModel._apply_direct_commands`).
* **Data flow.** The model hands the ceiling to its callback as `state["cav_exec_cap_mps2"]`.
* **Not capped:**
  * Recorded teachers (`cav_teacher`, and any callback that passes a teacher through).
  * Native counterfactual CAVs (their simple IDM baseline is unchanged, so the native arm is not
    weakened).
  * Braking. A braking command passes through unchanged when it is below the ceiling. When the
    ceiling is below the command, the ceiling binds, as SUMO's safe speed does.
* **Unchanged:**
  * Lane commands and the forced merge.
  * The command reported to other CAVs (`cmd_accel_mps2`) and the logged `safe_actions` stay the
    issued (post-safety) command, as in SUMO.
  * The trainer's recorded `action` is the executed acceleration (bridge output), as before.
* **One flag per path:**
  * Trainer: `[trainer] cav_exec_cap = "idm"` (`DreamerConfig`). It is applied to every ensemble member
    only for the duration of each imagination rollout, then restored, and is saved in the trainer
    checkpoint config. A resume uses the checkpoint's value; pre-EXECCAP checkpoints resume with "off",
    like every other config field.
  * `configs/c4_policy_final.toml` sets `"idm"`. No other config changes default.
  * Every tool that loads its WM through `wms.eval.backend.load_backend` (deploy, RSU check,
    coop-imagine, `scripts/horizon_check/*`, eval) follows the env var `WMS_CAV_EXEC_CAP=idm`
    (default off), or the explicit `load_backend(..., cav_exec_cap=)`.
  * `python -m wms.control.deploy --cav-exec-cap {off,idm}`, recorded in DEPLOY_SUMMARY.
  * `RsuChecker(..., cav_exec_cap=)` forces the switch for its rollouts. `None` keeps the model's own
    setting.
* **Gradient choice: min-gradient (exact), no straight-through.** `torch.minimum` routes the gradient to
  the binding argument.
  * A braking or non-binding command keeps d a_exec / d cmd = 1, so analytic policy gradients still
    learn from braking.
  * Where the IDM ceiling binds, d a_exec / d cmd = 0. A larger command changes nothing physically.
  * The gradient then flows through the ceiling, i.e. the IDM's dependence on the gap and speeds, into
    the state and hence into earlier commands.
* **Merge note.** All changes are on the longitudinal command path. The `WmSModel.step` edits are the
  cap computation before the direct and override calls plus two kwargs. They do not touch lane choice
  (c4-lanefix edits).

## Verification

### Bit-identity (switch off)
* `tools/execcap_golden.py` was run on an extracted `git archive a2d3716 src tests` and on this branch
  (tiny WM; bridge, coop callable, direct timeline, and a full `DreamerTrainer.update` with
  counterfactual rows, entries and lanes). All five digests are identical.
* The open-loop replay below reproduces the cmdreal "CMD" row exactly with the switch-off executor
  (38.6/53.5/55.6 veh-s; CAV |ds| 51.5 m).

### Executor replay
`tools/execcap_replay.py`, scratch output in `E:/agent-work/tmp/execcap/out/`, tables in
`EXECCAP_TABLES.md` there.

* **Data and models:**
  * Recorded SUMO MAPPO episodes `E:/agent-work/tmp/cmdreal/conv` (mappo_ft_it45), t0 = 480 s,
    p20/p40, 16 windows. The policy is `cmdreal/policy/mappo_ft_it45.pt`.
  * C4 = `fetched/gpuW/c4_wm_main/wm_s_best.pt`, V3 = `fetched/gpuB/c4_wm_v3/wm_s_best.pt`. Both
    physics.json files loaded with no default-physics warning (the tool stops if one appears).
* **How the capped row runs.** It goes through the real switch: the model computes the ceiling, and the
  executor applies `apply_exec_cap`.
* **Sign convention.** "Excess" = TEACHER total delay minus row total delay (HDV + t0 CAVs, veh-s). A
  positive value means imagination credits the commands with more than SUMO realised under the same
  policy, so 0 is the target.

**Gate 1: CAV acceleration vs SUMO realised.** Teacher-forced: from every recorded control tick, hold the
recorded issued command for 1 s and compare the realised (v[i+2]-v[i])/1 s, 31,853 CAV-seconds:

| executor | C4 MAE (bias) | V3 MAE (bias) |
|---|---|---|
| switch off (before) | 0.882 (+0.84) | 0.882 (+0.84) |
| **switch on, WM-inferred settings** | **0.305 (-0.00)** | **0.402 (-0.28)** |
| switch on, CAV settings = true vType (diagnostic only) | 0.079 | 0.080 |
| reference: min(cmd, simple IDM, true vType) | 0.084 | 0.084 |

The ≤ 0.1 target is met by the executor form, using the physics core with the true settings. It is
**not met with the WM-inferred settings**. The remaining 0.3-0.4 is CAV driver-setting identification.

* Substituting one true setting at a time (2 windows, C4) gives v0 0.29, decel 0.28, T 0.31, accel and
  min_gap about 0.38. Only all five together reach 0.09, so no single setting is to blame.
* The IDM near free speed is very sensitive to v0. For example, cautious CAVs are inferred near
  v0 ≈ 31.3-31.7 vs the true 28.8.
* V3's inferred settings are biased conservative (bias -0.28), so V3 under-executes commands.
* Using true or fleet vType settings for CAVs would be an owner decision under the "settings must be
  learned" rule, so it is not implemented.

**Gates 2/3: closed loop with the MAPPO policy acting on the imagined state.** Columns are excess in
veh-s at each horizon; CAV |ds| is the mean at 30 s.

| | 30 s off → on | 60 s off → on | 120 s off → on | CAV \|ds\| 30 s off → on (true-IDM ref) |
|---|---|---|---|---|
| C4 | +51.4 → **+4.5** | +103.7 → **+29.1** | +248.3 → +149.4 | 52.5 → **20.2** m (11.9) |
| V3 | +55.4 → **-11.7** | +112.8 → **-7.4** | +356.0 → +154.7 | 58.4 → **19.8** m (13.6) |
| C4, open loop on recorded issued commands | +38.6 → -5.1 | +53.5 → -11.8 | +55.6 → -10.8 | 51.5 → 19.2 m (11.2) |

* **Closed-loop excess.** At 30 s it is about 0 on both WMs: C4 removes 91 %; V3 overshoots slightly
  negative, consistent with its conservative settings.
  * At 60 s C4 keeps +29. The true-IDM reference row also keeps +22.7, so this part is not the
    executor.
  * At 120 s, 40-57 % is removed. The rest also remains with the true-vType IDM cap (+127 / +224), so
    it is downstream WM error (HDV reaction, lanes), not CAV execution.
* **CAV position error at 30 s.** It drops from 52-58 m to about 20 m. That is at the top of the
  12-20 m target, and true settings give 12-14 m.
* **HDV median |ds|** also improves: 30 s 14.9 → 8.4 m (C4), 20.4 → 10.6 m (V3).
* The numbers agree with the cmdreal CAPWM diagnosis (-1.2 / +16.4 / +121 veh-s, 20.8 m).

### Tests and smoke
* `tests/unit/test_execcap.py` has 15 tests:
  * The default is off and modes are validated.
  * Off never computes the cap or passes it to callbacks.
  * Off is bit-identical before and after toggling and equals a hand-written legacy executor.
  * +2.6 behind a slow leader realises exactly the IDM accel, and next v/s are recomputed.
  * Free road realises min(cmd, IDM ≤ accel setting), a small command binds, and above v0 the
    (v0-v)/dt term binds.
  * Braking passes through.
  * Recorded teachers are unchanged.
  * The direct timeline and the coop callable are capped.
  * Gradients: d/d cmd = 1 when braking, 0 when the cap binds, and flows to the state via the cap.
  * The policy gradient through capped imagination is finite and nonzero.
  * The trainer applies the switch only during imagination and restores it.
  * The inline trainer controller is capped.
  * `load_backend` env default.
  * The final config is on and the other configs are off.
* `OMP_NUM_THREADS=1`: test_execcap + test_c4_dreamer + test_c4_srcgrad + test_c4_accelcap + test_c4_rsu
  + test_c4_deploy + test_c4_futures + test_c4_core + test_import_graph = **321 passed**. `ruff` is clean
  on the changed files.
* **Smoke:** `train_cli --config configs/c4_policy_final.toml --smoke`.
  * WM: C4 `wm_s_best.pt`.
  * Env: `E:/agent-work/cfmem_env.sh`.
  * Output: `_out/execcap/smoke_final`.
  * Status completed with 2 updates and 2 validations. All metrics are finite. The summary config has
    `cav_exec_cap = "idm"`, and the imagined action max is 2.0.
