# ACCELCAP: policy acceleration cap 1.5 -> 2.6 m/s^2 (owner-approved 2026-10-13)

Branch `c4/accelcap` (from `c4/int` 39a576c). Hypothesis being tested: every SUMO head-to-head controller
raises entrance (source) waiting vs native SUMO driving, worse at higher CAV share. The suspected cause is
the controlled-CAV cap of 1.5 m/s^2, while human drivers accelerate at 2.0-3.0. A slow CAV at an
insertion point blocks arrivals. The owner rejected a control zone, so long-range actions stay possible.
Instead the owner approved raising the cap to the `normal_car` driver accel (2.6;
`scenarios/idm/driver_routes.rou.xml`, aggressive 3.0, cautious 2.0).

## Design: the bound is a stored property of the actor

* `CavPolicy(accel_min=-3, accel_max=...)` maps tanh to `[accel_min, accel_max]`. The bounds are stored
  in the persistent float64 buffer `accel_bounds`, so they travel in every `state_dict` and checkpoint.
  **A state dict without the buffer loads as the legacy `[-3, 1.5]`** (a `_load_from_state_dict` hook),
  whatever the constructor default. Phase-1..4, policy_check, MAPPO and smoke checkpoints therefore decode
  exactly as before through every loader (`load_policy_fn`, `load_policy_components`, trainer
  `--init-policy/--resume`, MAPPO resume / `--init-from`, MAPPO rollout workers, `eval_bridge`).
* Fresh actors default to `POLICY_ACCEL_MAX_MPS2 = 2.6`. Trainers set the cap explicitly in config:
  `[policy] accel_max_mps2` (new section in `train_cli`, allowed range (0, 3.0]). It is 2.6 in
  `configs/c4_policy_final.toml`. If a loaded checkpoint's stored bound differs from the configured one,
  **the checkpoint wins** and a WARNING is printed (trainer and MAPPO). To re-decode an old checkpoint at
  a new cap you need an explicit `policy.set_accel_bounds(...)`, which changes the action for the same
  latent.
* The checkpoint payload also records `accel_min_mps2` / `accel_max_mps2` (Dreamer `save_checkpoint` and
  MAPPO `export_payload`). If these fields disagree with the buffer, loading is refused.
* `contract.ACCEL_MAX_MPS2` stays 1.5 (= `LEGACY_ACCEL_MAX_MPS2`). It is only the legacy default. New
  constants: `POLICY_ACCEL_MAX_MPS2 = 2.6`, `NATIVE_CAV_ACCEL_MAX_MPS2 = 2.6`,
  `DIRECT_ACCEL_MAX_MPS2 = 3.0`.

## Every place that bounds the POLICY's commands now uses the policy's bound

| path | change |
|---|---|
| `safety.safe_accel(state, accel, *, accel_max=1.5)` | new ceiling keyword (a float, or a per-slot [B,K] tensor). The float path is the exact old op. `with_accel_max(fn)` adapts custom safety layers that lack the keyword. |
| `bridge.make_cav_controller` | safety ceiling, `target_v = v + cap*dt` and the `direct_command_motion` clamp use `accel_max` (default `policy.accel_max`, which wrappers expose as an attribute). Counterfactual native CAVs use `native_accel_max` per slot. |
| `dreamer.DreamerTrainer.imagine` | the bridge path and the internal (non-bridge) path both pass the policy cap and the per-slot native cap. `policy_proxy.accel_max` is set. |
| `policy.PolicyExecutor` (deploy / `collect --policy` / MAPPO eval) | safety ceiling and the exploration-noise clamp use the loaded policy's bound. |
| `rsu_check.default_candidates` (policy_check imagination) | every candidate wrapper (policy, group_k, sample_k, hold) exposes the policy's bound. |
| `deploy` hold_speed executor | receives the policy, used only for its bound, so it matches the imagined hold. |
| `coop_imagine.CallableCavController` | clamps at the callable's `accel_max` / `policy.accel_max` (legacy 1.5 otherwise). |
| `mappo.agent` | `build_networks(accel_max=1.5)`. `ActorStep` safety uses the actor's bound. `latent_log_prob(..., scale)` and `evaluate_actions` use the actor's tanh scale (`policy_scale`). |
| `mappo.config_helpers.build_trainer` | `[policy] accel_max_mps2`. **Default is the LEGACY 1.5**, so MAPPO behaviour is unchanged. |
| `traffic_shepherd.c4.baselines.ContractPolicyAdapter` | passes `policy.accel_max` to a safety layer that accepts it. |

## What is NOT the policy cap

* **Counterfactual native rows** (`counterfactual.idm_native_accel`). The IDM `a_max` and clamp are now
  `CounterfactualConfig.native_accel_max_mps2`, default `NATIVE_CAV_ACCEL_MAX_MPS2 = 2.6` (it was the
  policy cap 1.5). This applies to every run on this code, including fine-tunes of old checkpoints. Set
  `[counterfactual] native_accel_max_mps2 = 1.5` to reproduce older training.
* **CAV vType in SUMO (finding).** There is **no CAV-only vType**. `traffic_shepherd.scenarios.build_scenario`
  writes the three HDV driver types (from `scenarios/idm/driver_routes.rou.xml`), and a CAV keeps the
  driver type drawn for it (`fixed_explicit_driver_type_and_cav_identity_no_runtime_type_change`).
  CAV accel is therefore aggressive 3.0 (p 0.1), normal 2.6 (p 0.7) or cautious 2.0 (p 0.2). The type
  is hidden from the controller, so the native rows use the dominant normal_car value 2.6. With
  speedMode 27, bit 1 (respect the vType maximum acceleration) is set, so SUMO caps a commanded CAV at
  its type's accel: **cautious CAVs (~20 %) cannot exceed 2.0 in SUMO.** This is an HDV-shared
  parameter, so it was not changed (fixing it would also change native CAVs and the HDV types).
* **SUMO env command path (fixed, C4 direct mode only).** `ownership -> damping.direct_target` clipped
  every wall/direct request to the adopted adapter bound 1.5 (`load_config` enforces 1.5 for CIRCLES).
  `direct_target(..., accel_max=None)` and `CommandOwnership.wall_accel_max_mps2` now exist.
  `CavSumoEnv` sets the attribute to `direct_accel_max_mps2` (new kwarg, default 3.0) on `direct`-mode
  ticks only, including the `fallback="pi"` route through the PI adapter. It resets the attribute to
  None (adopted 1.5) on every other tick, so the wall law, damping, PI and native are unchanged. Legacy
  policies are unaffected because their commands are already at most 1.5 after their own safety filter.
  `mappo.fakeenv` uses the same 3.0 ceiling.
* Unchanged: `RandomAccelPolicy` (WM data collection, U[-2.5, 1.5]) and the WM physics clamp (3.0).
  **Caveat:** the WM's direct-output training data never contains CAV commands above 1.5, so HDV
  reactions to CAVs accelerating at 1.5-2.6 are an extrapolation for the learned residual. Native SUMO
  CAVs and HDVs do accelerate that hard in the data, and CAV motion itself is kinematic (exact).

## MAPPO

The running MAPPO run is not affected: its process has its code loaded, and its checkpoints have no
stored bound, so they load as 1.5 under the new code. A MAPPO comparison at 2.6 needs one of these:
* **Restart from scratch:** add `[policy] accel_max_mps2 = 2.6` to the MAPPO TOML. Exports then carry
  2.6, and deploy/collect decode them at 2.6. The SUMO env already admits 2.6.
* **Fine-tune from the current 1.5 checkpoint:** `--init-from` / resume keeps the checkpoint's 1.5 and
  warns. Moving to 2.6 needs an explicit `trainer.policy.set_accel_bounds(-3.0, 2.6)` after loading.
  The same latent then decodes to a different action (midpoint -0.75 -> -0.2, scale 2.25 -> 2.8), so the
  first iterations are a re-decoded, not equivalent, policy. Stored PPO latents from before the switch
  must not be mixed in. A CLI flag for this was not added; a fresh run is cleaner for fairness.

## Verification

* **Legacy bit-identity vs the pre-change code** (`git archive HEAD src/wms`, a one-off script outside
  the tests). Two real legacy checkpoints were checked: the MAPPO export
  `runs/active/WM-S-C4-20261013/fetched/cpuT_mappo_main/policy_latest.pt` and the Dreamer smoke
  checkpoint `runs/active/C4-SMOKE-20261012/dreamer_smoke/policy_best.pt`. Each ran through the deploy
  executor (accel + lanes, plus low-speed near-cap states), a seeded 4-step stochastic lane bridge
  rollout, and a MAPPO `ActorStep` (stochastic). **110/110 arrays were bitwise equal**, including bridge
  accelerations up to 1.497.
* `tests/unit/test_c4_accelcap.py` (8 tests) covers:
  * legacy state dict -> bit-identical actions vs the legacy formula and an explicit 1.5 actor;
  * legacy file through `load_policy_fn`;
  * new checkpoint round trip, payload mismatch refusal and a float32 cast;
  * a 2.6 policy reaching > 2.5 m/s^2 on a free road through the bridge (accel-only and lane paths) and
    `PolicyExecutor`, while a legacy policy stays <= 1.5;
  * the `direct_target` ceiling;
  * counterfactual native at 2.6 through the bridge;
  * `[policy]` config plumbing and `c4_policy_final.toml`;
  * MAPPO legacy default and legacy load, export bounds, log-prob consistency at 2.6, and the config.
* Updated existing tests. In `test_c4_core` the tanh-density test now pins `accel_max=1.5`, and the bound
  assert uses `actor.accel_max`. In `test_c4_policy2` the `_real_trainer` scenario pins the legacy actor
  cap, and the CRN test's native expectation uses the native ceiling.
* Test run (OMP_NUM_THREADS=1): accelcap + dreamer + policy2* + srcgrad + deploy + rsu + mappo* + core +
  import_graph = **335 passed**. `ruff` is clean on the changed files.
* **CPU smoke** (`train_cli --config configs/c4_policy_final.toml --smoke`, WM
  `E:/research-dense/runs/active/WM-S-C4-20261013/fetched/gpuW/c4_wm_main/wm_s_best.pt`, data env as in
  `cfmem_env.sh`, `C4_ALLOWED_ROOTS=E:/research-dense`; output `_out/accelcap/smoke_final`). It
  completed 2 updates and 2 validations with finite metrics. Summary `policy_accel_bounds = [-3.0, 2.6]`;
  imagined action max 2.54. The checkpoint payload has `accel_max_mps2 = 2.6`, buffer `[-3, 2.6]` and
  `native_accel_max_mps2 = 2.6`, and `load_policy_fn` decodes it at 2.6.
* **One short SUMO smoke** (seed 4107, 2400 veh/h, p = 0.4, 60 s native warmup, then 6 control ticks of
  a constant 2.6 command to every CAV; scratch only). Issued commands reached 2.6, so the env ceiling
  works. normal_car CAVs with room ahead realised up to 2.57 m/s per s, median 2.07. The aggressive CAVs
  sampled were leader-limited. No cautious CAV was sampled; its 2.0 cap comes from the speedMode/vType
  rule above.
