# R10B_DESIGN - learned pre-merge follower courtesy + fix B as one follower round (round 10b)

S-COURTESY, 2026-10-23. Branch `c4/v10r10b` (round-10 prep 4bdbbf18 + `c4/v10r9excl`). Owner-approved option A.
Evidence: R9_YIELD.md (S-YIELD), R9_STUCK_DIAG.md, R10_PREP.md, R10_REVIEW.md. Results: R10B_REVIEW_PACKET.md.

## 1. What the data require (R9_YIELD.md)
- SUMO: only the **target follower** (nearest lane-1 car behind or level with a waiting ramp car) slows before the merge.
  Matched excess -0.40 to -0.51 m/s^2 at dA 6-20 m, onset ~6 s before the merge, peak -0.45 at -1.5 s, ~0 at the merge row.
  Stronger near the lane end and for long waits; no detectable driver dependence; 54 % of merges (74 % of wait) follow it.
- WM (rounds 8/9): no pre-merge yield (+0.15 vs -0.40) and over-braking after the cut-in (follower min accel -3.2 vs -0.9).
- R9_STUCK_DIAG: residual stuck cars wait at v = 0 beside fast followers; without a follower channel the jam that licenses
  merges dissolves in rollouts (bistable: no jam -> no merge -> no jam).

## 2. Mechanism (`src/wms/model/follower_courtesy.py`, `WmSModel.forward`)
**Pair by geometry, no window.** For every ramp-lane (lane 2) car j its target follower is the nearest lane-1 car i with
`s_i <= s_j` (same relation as R9_YIELD). A follower that is the target of several ramp cars is paired with the nearest one.
A lane-1 car that is nobody's target follower has no partner; its offset is **exactly 0** (not a small number).

**Inputs** (state at the step start; identical function on recorded replay states and on rolled-out states; smooth maps only):
bumper gap `asinh(g/5)` (negative = overlap/alongside), ramp car distance to the lane end `asinh(d/20)`, follower and ramp
speeds, relative speed, ramp car wait `1-exp(-w/30 s)` (carried `merge_wait_s` if the model tracks it, else time in lane),
the merge head's detached P(merge) of the ramp car, the follower's own lane-1 leader (exists flag, `asinh(gap/10)`, closing
speed) so the head only has to learn what car-following does not explain. No thresholds, bins, "waiting"/"near end"
indicators, driver types, simulator cooperative bits or future quantities. The follower's own past acceleration is NOT an
input (R10_REVIEW persistence shortcut). The merge-pivot latent/margin is NOT an input: it is computed after the
acceleration (post-move geometry) and is not carried in recorded replays, so it would make training and runtime differ;
the head sees the geometry the pivot uses (gap, speeds, distance to end).

**Head.** MLP 10 -> 32 -> 32 -> 1 (GELU), last layer zero-initialised (exact no-op when enabled), **unbounded** output
(no tanh, no residual clamp); 1,441 parameters. The offset is added to the HDV acceleration before the plant clamp, outside
the residual gate (so fix-B's gate scalars do not rescale it), only on learned (HDV) slots. Deterministic: the heavy tail of
the SUMO response is largely situational (gap/end/wait explain R^2 0.56 of the encounter score) and a stochastic head would
need sampling in every training/scoring rollout (new RNG stream for all rollouts) - deferred, not justified for one round.

**Driver latent:** optional ablation input `model.follower_courtesy_embedding` (default false; R9_YIELD section 4: corr
0.04 [-0.01, 0.10] between encounters of the same driver).

**Where it acts:** inside `WmSModel.forward`, therefore in training rollouts, the recorded-state replays and every
runtime/scoring rollout identically. Flag off = no module, no tensor, no RNG draw (bit-identical; tests).

## 3. Losses (one round, two recorded-state terms)
| term | rows ("window") | prediction | trains |
|---|---|---|---|
| fix B `cutin_reaction` (unchanged, weight 2.174) | lane-1 follower after each recorded insertion (to first non-negative recorded accel, <= 6 s, 1 s pre) + all other lane-1 HDV transitions as controls | full model acceleration | temporal MLP + gate scalars |
| R10B `follower_courtesy` | every lane-1 HDV transition whose follower has a ramp partner at t (model's own pairing) | `clamp(a_pre(t).detach() + head(x_t))` | courtesy head only |

`a_pre` is the full model's acceleration without the courtesy offset at the recorded state. The frozen, state-conditioned
model is the **matched control at the identical state**: a learned expectation fitted on all TRAIN rows (no hand-set
cells). Offline (section 6 of the packet) its residual on C3-type non-adjacent controls is ~0 while on target followers at
dA 6-15 m it is ~-0.9 m/s^2 (the model accelerates into the gap the SUMO follower does not take). Partner rows where SUMO
followers do not slow (far partners, ramp car far from the end, fast free gaps) are in the same loss and anchor the head
at the model's residual there. Lane-1 cars without a partner contribute nothing (offset identically 0). Huber delta 1.0
m/s^2 (closer to the conditional mean than fix-B's 0.5, because the SUMO response is a mean deficit with a heavy tail).
Recorded future quantities enter only as the target a[t+1]; the replay runs without gradient.

**Gradient scope.** The head's weights are reached ONLY by the courtesy term: `follower_courtesy_rollout_grad = false`
evaluates the head inside forward with detached parameters (`torch.func.functional_call`), so rollout losses and the cut-in
term cannot train it, while the state adjoint through the head is kept. Fix B and all round-9 rollout terms train only the
round-10 set. Merge_pivot and everything else frozen bitwise (as round 10).

**Shared replay.** With both terms on, the cut-in replay also collects the (detached) courtesy inputs and `a_pre`; the
courtesy term then needs no second replay (values identical; test). If the cut-in term has no rows in a batch the courtesy
term runs its own no-grad replay.

**Merge-event rollout term:** not added. It needs a separate rollout from t_m - 12 s per recorded merge with the merge
likelihood, i.e. a second objective on the frozen pivot and a large extra cost; the functional role (opening the gap) is
left to the existing rollout terms acting through the state adjoint. Candidate for a later round if the courtesy appears
in one-step fit but not in rollouts.

## 4. Trainable set and weights
Trainable: `temporal.*` (lr x5), `gate_g_max_logit`, `gate_tau_raw` (lr x100) - round 10 unchanged - plus
`follower_courtesy.*` (lr x s, s from the offline pilot). Frozen bitwise: merge_pivot, merge_core, driver encoder/updater,
lane gates, lane_v5, history_proj, k_embedding, residual_net.

**Weight rule (docs/r10b_weight_rule.json):** the head is trained only by its own term and Adam is per-parameter scale
invariant, so the weight acts only through the global grad-norm clip (1.0) and the logged total. Gradient-norm parity with
the weighted fix-B term: `w = G_cutin(2.174) / G_courtesy(w=1)`, G_cutin = 2.174 x 0.3392 = 0.737 (docs/r10_weight_rule.json),
G_courtesy = geometric-mean gradient norm of the unweighted courtesy loss over the zero-initialised head on production-like
batches (4 windows x 120 s) of local TRAIN data (tools/r10b_courtesy_offline.py). Not tuned on outcomes.
**lr scale rule:** the scale in {1, 10, 100, 1000} with the lowest held-out (episode-disjoint TRAIN) courtesy loss after
the production schedule (30 steps x 4 windows at 5e-5 s, 60 steps x 5 windows at 3e-5 s, Adam), head alone, offline.

## 5. Expected effects
- One-step: target-follower acceleration at dA 6-15 m drops from the model's +0.8 toward the recorded ~0 (excess vs own
  controls toward the SUMO -0.4); event profile gains a pre-merge deficit; head ~0 for far partners.
- Rollouts: lane-1 followers of waiting ramp cars hold back, the jam near the merge persists, more merges are licensed by
  the pivot's follower-side boundary (R9_STUCK_DIAG); fewer stuck ramp cars; fix B reduces post-insertion over-braking.

## 6. Risks
1. **Followers yielding to cars that are not waiting** (e.g. fast ramp cars far from the end): the head sees speed, distance
   to end and gap and is anchored by such rows, but rows are dominated by jammed states; check the negative controls
   (partner dA > 30 m, d_end > 40 m, alongside) in rollouts.
2. **Base errors absorbed by the head.** The target is recorded minus the model; where the frozen model errs for reasons
   tied to the pair geometry (e.g. it over-brakes when the ramp car is alongside, +0.24 m/s^2 offline), the head corrects
   that too. This is fitting recorded accelerations, not a yield rule, but it means "head ~0 where SUMO does not yield" holds
   only where the base model is unbiased.
3. **Interaction with the merge rule.** More follower slowdown means more licensed merges (desired) but also possible earlier
   merges into small gaps, with fix B now softening the post-insertion reaction: watch collisions/TTC and the minimum gap.
4. **Train/runtime pairing shift.** In rollouts the partner set depends on rolled-out lanes; a mis-predicted lane-1 car can
   pick up the offset. Same function both ways, but the state distribution differs (moving cohort, as round 9).
5. **Weak wait input.** Without `merge_impatience` the model does not carry `merge_wait_s`; time in lane saturates for most
   ramp cars (lane 2 includes the approach), so urgency is carried mainly by distance to end, speeds and gap.
6. **Attribution.** One round with two new terms: the courtesy term moves only the head (pre-merge, partner rows); fix B moves
   temporal + gate (post-insertion rows + controls); rollout terms move only temporal + gate. Read windows separately:
   pre-merge [-6, 0) s target-follower excess -> head; post-insertion [0, +6] s min accel -> fix B.
