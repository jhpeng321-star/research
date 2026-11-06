# R13A — minimal courtesy pair-contract repair (prepared, NOT launched)

Branch `c4/v10r13a`, based on `c4/v10r12` @ `1d801beb`. It implements astra STRUCTURE_ADVICE §1 (the two interface defects) and the minimal part of §2 option A. Everything ran locally on CPU against synthetic states (no GPU, SUMO or eval data).

## 1. Defects verified on round-12 code

- **(1) No state derivative through courtesy: YES.** I built the round-12 config model with a randomised head. Its `courtesy_accel_mps2` has no graph, and d(follower accel)/d(ramp-car state) = 0. With `rollout_grad` on there is a graph to the weights only. Cause: `follower_courtesy_pairs` returns `features.detach()`, so the state adjoint promised by `live=False` did not exist.
- **(2) Stale intent: YES.** The follower's `intent` input equals the lane-v5 P(merge) of its partner. Forcing the lane-v5 merge logit from −8 to +8 moves that input from below 0.1 to above 0.9. Meanwhile `merge_pivot_c` and the ramp car's `next_lane` stay bit-identical, so the pivot decides the merge and lane-v5 (frozen) only feeds courtesy.

**Round-12 training impact.**
- The courtesy head's TF loss is unaffected by (1): its inputs are recorded and detached by design.
- The pivot TF/F2 terms are unaffected: the pivot is frozen and the geometry is detached.
- The rollout motion objective is affected: its pathwise adjoint misses ∂courtesy/∂state, which biases the gradients to the driver head, residual and temporal path. A test shows the same loss value but different gradients once the term is restored. The bias grows as the head grows.
- C4 policy gradients lose the CAV → follower-response path.
- (2) affects what the round-12 head learns, but the input is consistent between training and rollout.

## 2. Changes (all flags default to round-12 behaviour)

- `model.follower_courtesy_state_grad` (false): pair features keep their continuous derivative w.r.t. state. Partner and leader selection stays discrete and non-differentiable. Counters and history slot t-1 stay detached.
- `model.follower_courtesy_intent` (`lane_v5`):
  - `pivot` = `sigmoid(c_t/tau)`, the pivot's own boundary on the **step-start** state. This avoids an algebraic loop and does not use the latent. Veto/outside the zone gives 0; an unconstrained car gives 1. Pivot parameters are detached.
  - `none` = an ablation that sets the input to 0.
  - `pivot` is refused without first-passage.
- `loss.follower_courtesy.absolute_weight` (0): adds a head-only Huber loss to the recorded follower acceleration on the same partnered rows, alongside the contrast target and matched negative controls.

The input layout is unchanged (13 inputs). The flags are plumbed through spec, config, legacy_flat (omitted at default), settings and compose.

## 3. Round-13 config (`configs/c4_wm_v10round13a.toml`, prepared only)

The config is an inline copy of round 12 (which is at the extends cap). It changes only:
- `state_grad = true`
- `intent = "pivot"`
- `absolute_weight = 9.067` (equals the contrast weight; **untuned**, needs a documented rule before registration)
- `init_from` = round-12 final checkpoint, plus `name` and `out_dir`

It keeps a fresh zero head, because the intent meaning changes. The frozen set, pathwise adjoint, hybrid pivot, overlap veto and IDM/MOBIL are untouched. There are no thresholds, timers, padding or floors.

## 4. Tests

- `tests/test_r13a_courtesy.py`: **15 passed**. Coverage:
  - both defects;
  - non-zero, finite gradients that agree with central finite differences (h = 1e-2, rel 3e-2, fixed seeds) for partner s/v/a and follower s/v;
  - the same check for the pivot intent, including the ramp car's leader;
  - CAV-partner gradient reach, while CAVs are never offset;
  - pivot intent unchanged by the head output and by lane-v5;
  - exact limits;
  - the absolute term (off gives a bit-identical loss; on is head-only);
  - config deltas.
- **Round-12 identity across code versions:** same seeds and a non-zero head, run on the old and new `src`. Forward plus a 6-step rollout gives **59/59 tensors bit-identical**.
- Full suite: see the hand-off report.

## 5. Risks

- **Universal yielding:** the absolute term lets the head absorb base error on partnered rows, which is what the R11 contrast prevents. Keep the contrast on. Monitor the head on far/alongside/near controls and the contrast control residual (registered rule: below −0.3).
- **Residual compensation:** with the derivative restored, motion-loss credit flows through courtesy into the residual, temporal path and driver settings. Decompose core, temporal, courtesy and final acceleration as in the round-12 mid-run check.
- **Gradient scale:** the pivot intent adds 1/tau slopes near the boundary. Log the gradient norm and clip share early.
- **Pivot straight-through credit bias** is unchanged.

## 6. Launch condition

Launch is conditional on the round-12 results. Round 13A is one fix compared with the current best (one fix at a time): use it only if round 12 passes and becomes current best, otherwise re-base onto whichever model is best. It must be registered (ID, budget, sources) and acked before dispatch. Not pushed.
