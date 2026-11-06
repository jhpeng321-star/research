# R11_DESIGN - round 11: structural fix of the learned follower courtesy (one combined follower lineage)

S-R11, 2026-10-24. Branch `c4/v10r11` (from `c4/v10r10b` 8ee9341a). Source: R10B_REVIEW.md main finding and its
"Round 11" recommendation, R9_YIELD.md sections 5-6. Offline evidence: R11_REVIEW_PACKET.md, docs/r11_offline_fit.json.
Nothing launched; gpuR untouched.

## 0. Problem being fixed (R10B_REVIEW finding 2, finding 4)
1. **Tail confound.** The round-10b head is fitted to `a_rec - a_pre` on partner rows. In the last ~3 s before a merge
   the round-8/9 base over-predicts the target follower by +1.2..+1.6 m/s^2 (it accelerates into the gap that is being
   opened for the ramp car). Its inputs cannot separate "merge imminent" from "waiting at the lane end", and part of the
   base error is generic to the state (unpartnered cars in the same state are also over-predicted). The head absorbs
   both. Result: the head adds about -1.1..-1.8 m/s^2 on these rows, a matched excess 3-6x the SUMO one.
2. **Dead wait input.** Ramp cars never change into lane 2, so the lane-age wait is the 999 sentinel, which maps to 1.
3. **Attribution confound.** Fix-B control rows include the partnered rows, which see `a_pre + head`, so temporal is
   also trained to cancel head errors.

## 1. Inputs (model.follower_courtesy_version = 2)
The 10 round-10b inputs stay unchanged and in place. The dead wait column is kept as a constant bias input, so a
round-10b head zero-expands *exactly*. Three inputs are APPENDED (`COURTESY_V2_FIELDS`, `wms/model/follower_courtesy.py`):

| input | definition | causal source (same function in replay and rollout) |
|---|---|---|
| `a_ramp` | ramp partner's acceleration in its state at t / 3 | `kin[..., 5]`: recorded backward difference (the acceleration that produced row t) / the applied rollout acceleration (CAV teacher/controller acceleration for CAV partners) |
| `a_ramp_prev` | the same one step earlier / 3 | short motion-history slot t-1 (`history[..., -2, 1]`, accel/3), 0 where unobserved |
| `zone_wait` | `asinh(t_zone / 10 s)` | carried `courtesy_zone_s`: time since the ramp car entered the geometric merge zone `[merge_zone_s0_m, merge_zone_s1_m]` (the acceleration lane, i.e. road geometry, not a behaviour window); `+dt` while inside, reset outside / absent; t0 value from the observed pre-t0 long history (conventions of `initial_merge_wait`) |

The "current" acceleration of the ramp car at t -> t+1 is simultaneous with the follower's decision. A follower cannot
observe it, and the model's own predicted value would be wrong for CAV partners, which are overridden after forward.
So the latest observed accelerations (t and t-1) are used. No future quantity enters. The zone counter is carried by
`WmSModel.rollout` (initialised at the start and updated with the next state, like `merge_wait_s`; it survives chunked
BPTT through the returned state). It is also carried by the recorded-state replay (`one_step_replay`), with the same
functions. If a caller steps the model without the carry (belief/deploy propagators), the head sees the current state
only (`t_zone = dt`). This is documented and not used by training or scoring.

## 2. Loss: matched-control contrast (loss.follower_courtesy.contrast = true)
Same rows as round 10b: lane-1 HDV transitions whose follower has a ramp partner at t (geometric pairing). Same
prediction `clamp(a_pre.detach() + head)`. New target:

    target_i = a_rec_i - r_hat(phi_i),   r_hat = kernel-matched mean of (a_rec - a_model) over the UNPARTNERED
                                                 continuous lane-1 HDV rows of the same batch

So `head_i -> (a_rec_i - a_pre_i) - r_hat(phi_i)`. This is the part of the partner row's residual that a state-matched
unpartnered car does not have, i.e. the adjacency effect only (R9_YIELD's matched excess, as a training target).
- `phi` = R9_YIELD's matched-expectation covariates as continuous coordinates: own speed/10, asinh(time headway to the
  own-lane leader; 99 s without), closing speed/5, asinh((lane end - s)/20 m). These are recorded state at t, used only
  in the target, never as a model input.
- Kernel: Gaussian in coordinates standardised over the batch's partner + control rows. Scott's bandwidth
  `h = n_controls^(-1/(d+4))`. One pseudo-row at the batch control mean, so the estimate shrinks smoothly where support
  is thin. No bin, cell, window, threshold or tolerance. Without controls `r_hat = 0` (plain fit).
- Controls use the model's own plant-clamped acceleration. `r_hat` is recomputed every step with the live model, so as
  fix B fixes the base on unpartnered rows (`r_hat -> 0`), the head target moves toward the absolute residual. Absolute
  and excess fit then coincide. The two terms never compete for the same rows.
- Why headway and not gap: offline (packet section 4), matching on (v, gap) finds unpartnered cars in the partner
  state that the base predicts correctly (`r_hat ~ 0`), which gives the head target -1.2 in the last 2 s. Matching on
  R9_YIELD's (v, headway) gives -0.5..-0.7, consistent with the SUMO matched excess (peak -0.45). The registered reject
  checks (M1-M3) define "excess" by R9_YIELD's headway cells, so the loss uses the same covariates, continuously.
  A merge-zone occupancy coordinate (C3-like) was tried and dropped: it leaves ~1 effective control per partner row in
  the last 2 s, so the target collapses to the batch mean. This choice is a stated risk (section 6).
- Alternatives considered: (i) pair-removed counterfactual replay. The model's own "partner removed" prediction is
  exactly `a_pre` (the head is the only adjacency channel it is trained through), so it adds nothing without a matched
  recorded counterpart; the recorded side needs matched controls anyway. (ii) The round-10b option B (a learned
  regressor on unpartnered rows) was rejected there as not identified (it depended on training length). The kernel
  estimate is deterministic, has no training length, and its bandwidth follows a fixed rule.

## 3. Fix B: controls without partnered rows (loss.cutin_reaction.exclude_partnered_controls = true)
The control anchor drops every lane-1 row whose follower has a ramp partner at recorded state t. This is the head's
pairing function on the recorded state (`recorded_partnered`, tested equal). Event rows and all weights are otherwise
as in round 10b; the 1 s pre-context rows of events are unchanged (they are few, and the reviewer flagged only the
controls). Temporal/gate are then anchored only on rows where the head is exactly 0. Fix B is otherwise unchanged
(weight 2.174, control_weight 1, pre 1 s, <= 6 s, Huber 0.5).

## 4. Stochastic head: not added
A heteroscedastic/Student-t head would need a sampled offset in every training, scoring and planner rollout. That
means a new RNG stream in every rollout, a sampling-vs-mean decision for the planner, and calibration checks. This is
not cheap for one round. The heavy tail is mostly situational (R9_YIELD: situation R^2 0.56), which the new inputs
target. Deferred (risk 4).

## 5. Invariants kept
- Offset exactly 0 without a partner and on CAV slots (`torch.where(has_partner & learned, raw, 0)`, after the head).
- Unbounded linear read-out, no clamp/tolerance/floor on the head output, Huber 1.0.
- Identical function in training rollouts, recorded-state replays and every runtime/scoring rollout.
- `follower_courtesy_rollout_grad = false`: the head's weights are reached ONLY by the courtesy term (state adjoint kept).
- merge_pivot frozen bitwise. Round-9 decisions (TF + F2, counted exclusion, lane-end + risk-retain, state_st true,
  checkpoint_forward true) and the round-10b fix-B lineage are inherited unchanged (config pin test).
- Every new switch defaults to the round-10b behaviour. Flag off is bit-identical (tests).

## 6. Trainable set, weights, init
- Trainable (unchanged from 10b): `temporal.*` (lr x5), `gate_g_max_logit`, `gate_tau_raw` (lr x100),
  `follower_courtesy.*` (lr x s). Frozen bitwise: merge_pivot, merge_core, driver encoder/updater, lane gates, lane_v5,
  history_proj, k_embedding, residual_net.
- **Weight rule** (docs/r11_weight_rule.json): gradient-norm parity with the weighted fix-B term, as in round 10b:
  `w = G_cutin(2.174) / G_courtesy(w = 1)`. G_courtesy is the geometric-mean gradient norm of the unweighted contrast
  loss over the head at its round-11 INIT, on production-like batches (4 windows x 120 s) of local TRAIN data.
- **lr rule**: the scale in {1, 10, 100, 1000} with the lowest held-out (episode-disjoint TRAIN) contrast loss after the
  production schedule (30 x 4 windows @ 5e-5 s, 60 x 5 windows @ 3e-5 s, Adam), starting from the round-11 init.
- **Init**: round 10b final (`/root/autodl-tmp/c4/v5/runs/c4_wm_v10_round10b/wm_s_stage300.pt`). Its head is
  zero-expanded to version 2 (`expand_head_inputs`, in `_init_from_checkpoint`; reported as `expanded` and
  `follower_courtesy_init = "checkpoint"`). Switch `model.follower_courtesy_init_zero = true` (gate: `R11_HEAD_INIT=zero`,
  plus `R11_INIT=<r10b_headzero checkpoint>` if that is the chosen base) starts a fresh zero-init head instead.
  Fix-B tensors (temporal + gate) always come from the init checkpoint.

## 7. Risks
1. **Matching metric decides attribution.** Headway matching follows the registered evaluation. Gap matching would
   assign about twice as much to adjacency. The absolute partner-row error after round 11 equals the base's error on
   matched unpartnered cars, until fix B reduces that error.
2. **Positive head offsets where the base under-predicts its controls.** For partners alongside (dA < 6 m) and far from
   the end (d_end > 40 m), the contrast target is about +0.3 m/s^2 offline, because matched unpartnered cars are
   over-predicted by about +0.3 while the partner rows are not. One-step |model - recorded| on those groups can reach
   the M3 bound (0.3) until fix B removes the control bias. Watch M3 and insertion headway / TTC (M5).
3. **Init from 10b.** The 10b head carries the overshoot. The contrast loss must unlearn it within 90 steps (offline:
   packet section 5). The zero switch exists if the coordinator prefers the head-zero base.
4. **Heavy tail not modelled** (deterministic head).
5. **Train/runtime state shift** (rollout pairing, moving cohort), as in 10b.
6. **Dead head behind the plant clamp.** If an optimizer step drives the unbounded offset past the plant clamp on
   every row, its gradient is exactly zero. This happened in a mechanics-only micro-smoke at head lr 1.0. Production
   head lr is 3e-3..5e-3. No clamp or floor is added (owner rule); watch `follower_courtesy_head_min/max`.
