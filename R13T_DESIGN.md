# R13T / R13F / R13M: temporal-emission fixes (prepared, NOT registered, NOT launched)

Branch `c4/v10r13t` from `079e39ff`. The cause is from BRAKE_DIAG.md: the learned temporal emission (`history_proj` + `k_embedding` + `temporal`, row 0) adds zero-mean, heavy-tailed acceleration noise in closed loop. Local CPU only.

## 1. Switches (default = round 12; omitted from the flat config at default)

- `model.temporal_accel_off` (**13F**): the emission is an exact 0 in training and rollout, the same tensor as BRAKE_DIAG `zerotemp`. It is saved in the checkpoint. In the hazard_v5 model the temporal MLP then gets no gradient.
- `model.temporal_accel_lane_bias` + `temporal_lane_bias_init` (**13M**, "keep the mean, remove the noise"):
  - The emission becomes a learned per-lane constant b[step-start lane] (pre-gate, 0 off road), new parameter `temporal_lane_bias`[3].
  - The init is measured (§3): the per-lane mean of the init checkpoint's emission. It then trains at the `temporal` lr scale (×5, i.e. near-fixed).
  - With b = 0 it equals 13F bit for bit.
  - Of the S-FACTORIAL candidates this is closest to a "mean-only" arm that replaces the emission by its mean; a state-conditional mean or a low-pass would be a different arm. The existing SLOWRES `temporal_residual_mode="slow"` is the low-pass option and was not configured here.
- `model.temporal_accel_reinit` (13T): an init switch that zeroes row 0 of `temporal[-1]` while keeping the gradient live, mirroring `residual_accel_reinit`.
- `loss.accel_tail.{weight, jerk_weight, table, lanes}` (13T): the tail-matching terms `accel_tail` and `jerk_tail` (§2).

## 2. Tail-matching loss (`losses/accel_tail.py`)

**Rows.** All rollout rows where the slot is present at step start and end, is an HDV, and its predicted lane is in [1]. The persistence terms supervise recorded onsets only.

**Cells.** Taken from the predicted state with the brake-persistence convention: gap × closing × leader-decel, 4×4×4. The cell code is shared through `brake_persist.leader_cells`, and a test confirms the old cell ids bit for bit.

**Variables.** a = Δv/dt (the braking terms' and BRAKE_DIAG's definition); |jerk| over three samples.

**Score.** For each cell, the predicted quantiles q̂(u) are scored with pinball ρ_u against the recorded cell quantiles, which act as pseudo-samples; the match constant is subtracted.
- dE ρ_u(y−q)/dq = F_rec(q) − u: the score is minimal exactly at the recorded quantile, and the gradient is bounded.
- The mean over the levels is the quantile form of the CRPS.

**Levels and support.** Levels u = 0.01…0.99; the diagnosed excess (4.0 % of rows with a < −3, against 0.18 % recorded) lies above 1 %. Support rule: a level is scored only if n·min(u, 1−u) ≥ 1; thin table cells use the pooled distribution.

**Table.** `tools/r13t_tail_table.py` reads recorded arrays after warm-up. **The loss refuses any table whose `split_roles` is not ["train"].** Only val-role episodes of the r9 TRAIN pool are local: 192 episodes, 10.5 M rows, a p1/p50/p99 = −2.07/0.06/1.45. That provisional table (uncommitted) was used only for measurements.

## 3. Rules and numbers

- **Tail weights** (`docs/r13t_weight_rule.json`): w = (G_brake/2)/G_term. G_brake = G(brake_persist) + G(brake_persist_quantile); each G is the geometric-mean accumulated gradient norm from the real `forward_backward`.
  - Init: round-9 final + round 12's transforms + reinit; 4 batches × 2 windows × 20 s.
  - Result: **accel 10.33, jerk 38.74**. Per-batch range: accel 5.6–17.7, jerk 19.9–170.
  - Re-measure at the round-12 final before launching 13T/13AT.
- **Absolute weight** (`docs/r13a_absolute_weight_rule.json`): 9.067 × G_contrast/G_abs at the zero-init head on the R11 TRAIN cache (207 windows) = **4.525**. Across seeds 4.13–4.78; intent zeroed 4.529.
- **13M init** (`docs/r13m_lane_bias_init.json`, `tools/r13m_lane_bias_init.py`): the mean pre-gate emission per step-start lane over learned on-road HDV rows of 60 s free rollouts. Rollouts start from the round-9 final with round 12's init transforms. Result: **__LB__**.

## 4. Configs and gates (prepared only)

- `round13f`: round 12 + repair A + off.
- `round13m`: 13F with lane-bias in place of off.
- For both, round 12's init and schedule are unchanged, so round 12 is the paired control.
- Gates: `scripts/round13f_gate.sh` and `round13m_gate.sh`. Each is the round-12 gate up to names, with prior off as round 12 was launched. The `wms.train.round13f_gate` verdict picks its mode from the config.
- `round13t`: round 12 + reinit + tail, init = round-12 final. `round13at`: 13A + the same temporal fix. No 13T/13AT gate yet.
- **13A defect fixed:** 13A re-applied round 12's init transforms (`driver_param_init=centre`, `residual_accel_reinit=true`) to round 12's final weights, zeroing its trained driver rows and residual read-out. Now `checkpoint`/`false`.

## 5. Tests

- New test files: `test_r13f_temporal_off` (8), `test_round13f_gate_launcher` (23), `test_round13m_gate_launcher` (23), `test_r13t_tail` (16), `test_r13m_lane_bias` (6).
- Bit identity (`tools/r13_bit_identity.py`, old vs new src): round 12 345/345 identical. 13F vs 079e39ff + `diag_zero_temporal`: 336/336 identical, including gradients.
- Suite: __SUITE__.

## 6. Launch condition

Launch only after the round-12 results, with registration + ack first. Run one fix against the current best: 13F or 13M are round 12's paired arms; 13T applies if round 12 becomes best. 13AT only if the owner or coordinator chooses the combined variant.

Before 13T/13AT, on gpuR: build `docs/r13t_tail_table.json` from the TRAIN-split roots, then re-run `tools/r13t_weight_rule.py` (+ `_combine`) at the round-12 final. Re-measure the 13M init on TRAIN-split windows.

## 7. Train the temporal path on closed-loop statistics?

Round 12 already trains through free rollouts, but its terms do not constrain the per-row acceleration distribution: position integrates a zero-mean pulse out, and the braking terms are onset-conditioned. `accel_tail` is such a closed-loop statistic on the rollout's own rows. Nothing more is built until reinit + tail is scored.

## 8. Risks

- **Lost temporal correction:** S-FACTORIAL shows that zeroing the emission raised blow-ups (12.5 → 40 %) and the delay ratio. 13F is therefore risky; 13M keeps only a lane mean, while 13T keeps the full path and penalises its tails.
- **Tail vs motion loss:** cell pooling can trade the position fit for the tail. Monitor term shares and the clip share.
- **Sparse, noisy gradient** from order statistics at batch 4; the weights vary about 3× between batches.
- **State-dependent cells:** a spurious leader brake moves its follower into a cell where hard braking is legitimate.
