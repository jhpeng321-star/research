# S-POLAUDIT-T — candidate-A policy TRAINING-side audit (2026-10-28)

Scope: `python -m wms.control.policy14_cli` as launched for candidate A
(`resolved_A_MAIN1500_8e75184392b3`, source 8e751843, resume C1 64f70af4, WM3 41e18f09),
read on gpuA read-only (CONFIG/BOUNDARY/HORIZON_AUTHORITY/A_PID argv, DRY_RUN RUN_SUMMARY,
POLICY/A updates 181–185) and on `c4/c4prep` HEAD f5bbeec0 locally. Zero SUMO, zero GPU.
Executable checks: `tests/audit/test_polaudit_t.py` on branch `c4/polaudit-t`
(11/11 PASS, CPU, torch 2.14.1; run with `C4_ALLOWED_ROOTS=<tmp>`). Existing suites:
see "Verification" at the end.

## Findings (most severe first)

### F1 MAJOR (BLOCKER if any [trainer]/[reward] value must change for the new WM) — full resume silently replaces the run TOML's `[trainer]` and `[reward]` with the C1 checkpoint's
- `dreamer.py:2116-2118` `self.config = DreamerConfig(**{**payload["config"], **memory_request, **training_request, **anti_request})`; `:2141` `self.reward_weights = RewardWeights(**payload["reward_weights"])`; optimizer `param_groups` (lr) also come from the checkpoint.
- Only memory fields, truthy `native_residual_train_*`/`real_credit_individual_floor`, and (when pessimism/anchor on) `pessimistic_credit, native_anchor, native_anchor_calibration, future_samples, uncertainty_weight, entrant_control, deterministic_crn` survive. This is a sibling of the S-HOLD-1 "C1-config overwrite" defect: the fix was a whitelist, not a rule.
- Live evidence (DRY_RUN/A/RUN_SUMMARY `effective_config_before_updates` vs resolved TOML): `checkpoint_every` 1→**100**, `entropy_weight` 0.001→0.0003, `real_return_mix` 0.3→1.0. Consequence today: no per-update checkpoints (POLICY/A has no `policy_0001xx.pt`); a crash loses up to 99 updates (~28 h). Also any new-WM-dependent change to `horizon, entry_pad_max, cav_exec_cap, entry_insert_rule, wm_physics_overrides, imagination_grad_decay, grad_clip, actor_lr, low_penetration_horizon_cap, ttc_*` would be silently ignored (`wm_physics_overrides` is even applied at `__init__` from the TOML, then the recorded config reverts to C1's — physics/config split).
- Test: `test_full_resume_silently_replaces_run_trainer_config` (horizon 7→5, checkpoint_every 1→100, grad_clip 5→100, lr 1e-3→5e-5).
- Fix: for an explicit `--resume` into a NEW phase, take every `[trainer]`/`[reward]` key from the run TOML and keep only actor/optimizer state + counters from the checkpoint (or fail when TOML and checkpoint disagree on any non-whitelisted key); log the diff. Minimal fix for Thursday: assert `effective_config_before_updates == resolved TOML [trainer]` before the first update, and set `checkpoint_every=1` effectively (it is cheap: 1.2 MB/ckpt).

### F2 MAJOR — "individual" credit is exactly zero; 3 of 9 imagined rows are wasted compute
- `counterfactual.py:row_coefficients`: with w_ind = w_coop = 8.9 every individual row gets 8.9−8.9 = **0**. Effective objective per start: `1·D_all + 49.7·D_G − 10.3·Σ_i D_{G\i}` (group 9.9+8.9+3·10.3). Live log: `cf_coverage rows_total 9, rows_individual 3`. The real-credit floor that "preserves a nonzero individual baseline" (AMENDMENT2) therefore only keeps coop ≤ individual; the net individual weight is zero by construction.
- Only one 3-CAV group per update (`cav_in_group_fraction` 0.06–0.18): 49.7 of the 50.7 coefficient mass is on 3 of ~40 acting CAVs; the rest learn only from D_all (weight 1) and the −10.3 flip rows reward making the 2-member subteams worse.
- Tests: `test_registered_weights_give_individual_rows_exactly_zero_weight`, `test_pair_groups_also_zero_individual_and_no_team_term`, and `test_equivalent_config_without_zero_weight_individual_rows_gives_same_gradient` (w_ind=w_coop=0, w_group=18.8, cover_group_members=false: identical loss and actor gradient with 6 instead of 9 rows).
- Fix: either (a) keep the objective and drop the dead rows (equivalent config above, ~1/3 less imagination per update → ~29 h instead of ~43 h for 150 updates; q10 selection unchanged because zero-coefficient rows add 0 to the joint), or (b) if a positive individual term is actually wanted, register w_coop < w_ind. Re-label the weights in reports either way.

### F3 MAJOR — every selection/gate score is a 10 s imagination; training optimises 90 s
- `dreamer.py:_imagination_config` extends the horizon only when `self._training_imagination`. `independent_score` (per-update rescore, native-relative 16-start panel every 10 updates, phase-relative signal gate at phase update 10, and the final `select_update`) runs with the C1 base `horizon=10`. Evidence: panel J 128–3168 and `independent_score_before.J=709` (10 s) vs training `returns` −2035…−7870 (mean 90 s return-to-go). Test: `test_selection_and_gate_scoring_use_base_horizon_not_training_horizon`.
- The policy is therefore selected/gated on a window that cannot see 30–90 s consequences (metering, queue spill-back) — exactly the C3 failure mode. Current panel deltas are tiny (C1 vs native mean −0.37 veh-s/10 s).
- Fix: score the panel/gate at the trained horizon (90 s, same K8 futures, deterministic policy), or score both and require both; record `panel_horizon_s` honestly (today the receipt says 10).

### F4 MAJOR — no active guard against WM optimism on the policy's own actions
- Active: q10 over K=8 futures of ONE WM (driver-parameter/lane/entry draws; `ensemble_epistemic_available=false`), native anchor OFF (scale 0, verified), TTC `wm_safety_report_only=true` (penalty 0, dual frozen at 0.1), uncertainty_weight 0, no real-SUMO check inside the phase. q10/K8 captures aleatoric spread, not the systematic WM bias that produced C3's −0.834 imagined vs +8 veh-s real. Its direction is correct (F-OK1) but its efficacy against that failure is UNVERIFIED.
- The horizon trust is inherited from WM2 (`measured_wm3_gate: false`).
- Live telemetry already shows `metering_like_holds_ge30s` 7–9 of 8 futures, max hold 75–89 s for C1 (t0 upstream CAVs slower than their native twin ≥30 s). With a 0.06 m/s² residual this metric is sensitive, but it is the C3 pattern.
- Fix: require a paired real-SUMO check (S-POLAUDIT-D) before any adoption; add a 90 s native-relative panel (F3); consider re-enabling the TTC penalty or at least a gate on `ttc_excess` (currently 0.004–0.052 positive at every update).

### F5 MAJOR — the "10→90 s horizon curriculum" does not exist in this phase
- `looplong.configure` (auto_horizon flag) discards POLICY.toml `[horizon]` and builds knots `[0, P]` at the resume position P (=180): every phase update is 90 s (log: `imagination_horizon_s 90.0`, stage 1 from update 181). POLICY.toml `curriculum_updates=[0,1]`/`entry_pad_max=[32,288]` are dead (the computed pads happen to equal them). Test: `test_auto_horizon_replaces_the_toml_curriculum_with_an_immediate_jump`.
- Starts whose observed CAV share is ≤10% never get the long horizon (base 10 s, then capped ≤60 s): the 40% non-focus draws include p02/p05/p10 windows trained at 10 s. Test: `test_low_share_starts_never_get_the_long_horizon`.
- Same for exploration (rho 0.5 and 60 s persistence from the first update; C1 anchor 0 → position 180).
- Fix: if a ramp is wanted, set BOUNDARY `previous_horizon_s` < 90 or make configure honour the TOML knots relative to the phase start; otherwise rename it "fixed 90 s".

### F6 MAJOR (design, unregistered weight) — direct real-pair secant credit
- `looplong_real.RealPairCredit.loss`: 4 C3 (0978aa72) pairs round-robin (benefits 2155, 53, −3634, 275 veh-s; update 185 repeats 181), one first-step executed-acceleration contrast each, 300 s real benefit, off-policy for C1. Sign is correct (`test_secant_direct_credit_sign`; returns are −J). Its weight is implicitly 1/`return_scale` of the 90 s imagination; logged loss is the same order as the whole imagination objective (update 183: direct +0.052 vs actor_loss 0.048). Its gradient share is not logged (the actor-gradient audit omits this term) — UNVERIFIED.
- Fix: register an explicit coefficient, log its gradient norm, or disable it for candidate A.

### F7 MAJOR (structural) — imagination "native" is a learned head, not SUMO native
- Residual composes on `native_cav_head_v5.json` (fit RMSE 0.20 m/s²) while the learned residual is 0.056 mean |r| / 0.07 RMS. The policy can learn to correct head error that does not exist in SUMO. Also `cav_exec_cap="idm"` (all CAVs) zeroes the gradient of positive residuals whenever the ceiling binds (`test_exec_cap_blocks_positive_residual_gradient_when_binding`) — the actor can mostly only learn to slow down relative to native. Not a bug; mapping to pin for S-POLAUDIT-D below.

### F8 BLOCKER (operational, fail-closed) — a different WM cannot launch on the current authority
- `c4launch_authority.inherited_gate` hard-codes WM3; `read_boundary` then needs a measured `coopw_horizon_gate/1` for the new WM (sign agreement at 10/30/60 and PASS at 90) bound to the new WM config hash; `HorizonConfig.bind` likewise. A new BOUNDARY.json (new WM sha/config; pairs/returns can be reused) is required. Do not patch the WM3 constant.
- Otherwise resume onto a new WM is handled: loop-refresh path, world_model_ids recomputed, best scores reset, native head/obs normalisers are WM-independent, anchor artifact unused, real-credit multipliers have no effect (F10). `return_low/high` carry over from C1 (EMA 0.99) — harmless because the direct term and imagination term share the scale and Adam is scale-invariant.

### MINOR / INFO
- F9 q10 is selected per (time step, start), not per start as the docstring says (`test_q10_selects_the_bad_tail...`: h=0 picks futures {1,5}, h=1 picks {2,6}). Objective = mean_h q10 of the suffix return. Acceptable; fix the docstring.
- F10 `--real-credit REAL_CREDIT_c03.a1.json` is a no-op: individual 0.401 is floored back to 8.9, others 1.0 → effective = registered (`test_real_credit_file_is_a_noop_for_candidate_a`).
- F11 Dead config (set, never effective): `[horizon]` (F5); `[uncertainty]` (disabled); `[lanes]`, `[policy] lane_control/init_zero_accel`, `[policy14] policy_ramp_merge` (forced off by `--native-init`); `[policy14] sampled_marginal_lane_weight`, `critic_warmup_updates`, `critic_value_scale`, `entropy_floor/decay` (deterministic residual → entropy 0, critic off); `[paircrit]` installs an inert TailValue critic (critic step off under long credit); `[counterfactual] start_tilt` (replaced by focus) and `validation_uniform` (no validate call); `[run] seed/updates/validation_batches` (CLI args win); `[trainer] real_return_mix` (critic off). LOOPLONG_CREDIT.json = `null` is expected (target 90 = trusted 90 → no CQR).
- F12 `select_update` treats the resumed C1 (update 180) as "trained" (`update > 0`), so a phase can end selecting C1 unchanged; `recommended_updates_per_cycle` would be 0. Decide whether that is intended.
- F13 Observation side (training): report `cmd_accel`/`lane_intent` are forced to 0 in residual mode and report age is always 0 in imagination; deployment must match (S-POLAUDIT-D).
- F14 `metering_like_holds_ge30s` counts runs, so it can exceed `sampled_starts` (9 > 8 at 185); label only.

### OK-verified
- OK1 Signs: reward = −(delay+source_wait) (`objective14.authoritative_reward`); returns are reverse cumsums of −J; `difference = R_row − R_base(detached)`; loss = −Σ c·diff/scale. q10 interpolates the two lowest of 8 paired futures (0.3/0.7) of the signed joint, i.e. the BAD tail of policy-minus-native; only those futures get gradient. Real returns are strictly negative (−J) so `benefit=policy−native>0` = improvement.
- OK2 Counterfactual rows: all-CAV row (policy for t0 CAVs + entrant slots), base row (all native, entrants native), individual/group/flip rows toggle exactly the named t0 CAV slots; weights 8.9/9.9/8.9/10.3/1.0 applied each update (logged `cf_weights`, `real_credit_weights.effective`). Coop is nonzero only through D_G (see F2).
- OK3 Action path (training), for S-POLAUDIT-D: decision every 2 physics ticks (1 s), residual `r = 1.2·tanh(head([own,ctx,group]))` on CAV slots, held 1 s (new slots 0 until next decision); training exploration multiplies the latent by (1+ε), ε group-correlated AR(1) (rho 0.5, 60 s), off in scoring. Each 0.5 s tick: `cmd = clamp(native_head(state, current lane) + r, min(native,−3), max(native,1.2))` → `safe_accel(…, accel_max=1.2)` (soft-min stopping bound) → `apply_exec_cap` (WM IDM ceiling, all CAVs) → `direct_command_motion` clamp [−3,1.2]. Lanes: every CAV uses the WM HDV lane path incl. first-passage ramp merge; the actor has no lane head. Entrant CAVs (share from `entry_feat_control`, present in c4_episodes starts) act on all-CAV rows.
- OK4 Inputs: own sensor (≤100 m same-lane neighbours), CAV reports (≤ report range), RSU cell field, CAV identity; no driver types/hidden settings/future arrivals in actor or group encoder. The WM's inferred driver settings enter only the dynamics/exec cap.
- OK5 Gradients: residual head, own/report encoders, attention and RSU group encoder all trainable (`native_actor_training`: group/trunk true, 91,793 params; nonzero per-module grads every update 181–185). Absolute/lane heads frozen. C1's head Adam moments kept, new modules fresh Adam. Critic off; no detached path on the policy rows (base return detached by design).
- OK6 Native baseline = learned native head on every non-acting CAV + native lanes, never a policy or switching arm; anchor OFF verified (scale 0.0, penalty 0.0).
- OK7 CRN: futures/driver draws per start shared by all rows; merge-pivot, lane-uniform and entry draws keyed per start + sample index; policy noise drawn once per start and indexed per row; RNG restored per future.
- OK8 Memory modes: production K8/q10 full-actor exactness covered by existing LONGMEM/c4launch tests (CPU here, CUDA on gpuA per RESUME_AUDIT_CUDA_QUALIFIED); I did not re-run the CUDA parity.
- OK9 Focus sampling: exact 60% to {m2400,m2700}×{p20,p40} (5533/5326/5284/4949 windows), 40% window-uniform over the 41,116 rest; legacy tilt disabled.

## Verification
- `tests/audit/test_polaudit_t.py`: 11 passed (CPU).
- Existing: test_antiexploit + test_c4prep2_credit + nativeinit/entrant/looplong: 96 passed, 15 failed for local-environment reasons only (missing baseline-source env, `tools.*` not importable, workspace root, GBK decode).
- With PYTHONUTF8=1, C4_ALLOWED_ROOTS and repo root on PYTHONPATH: test_longmem + test_c4launch + test_c4prep + test_looplong + test_policy14_recovery = 116 passed, 4 skipped (CUDA-only), 901 s CPU. CUDA parity not re-run.
