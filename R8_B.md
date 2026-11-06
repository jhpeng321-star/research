# Round 8 agent B — recorded-state sequence TF

## Files

- `src/wms/train/losses/merge_core_seqtf.py` (new)
- `tests/test_merge_core_v2_seqtf.py` (new)
- `R8_B.md` (this note)

No other file was edited.

## `build_recorded_scene(targets, *, merge_lane, zone_s0_m, zone_s1_m, k, target_lane=1)`

Computed once, detached, over the whole unroll (`N = targets.n_steps`):

- `gap_l`, `gap_f`, `v`, `v_l`, `v_f`, `a`, `a_l`, `a_f`, `has_l`, `has_f` —
  `wms.model.merge_core.core_features` on **all present recorded vehicles**
  (`loss_valid`), so CAV neighbours on the target lane contribute gaps.
  CAVs are therefore scene neighbours but never supervised targets.
- `feasible` = present & ramp lane & in-zone (`zone_s0_m <= s <= zone_s1_m`).
  Missing gap sides are free space, matching the runtime `require_both = None`
  path. The runtime additionally gates on the learned `lane_v5_acts` envelope,
  which is a rollout-head output and cannot be reconstructed from recorded
  states; the recorded side therefore uses the physical presence/zone part only.
- `supervised` = HDV (`~is_cav`), observed (`loss_valid`) and on the ramp.
  Entrants have `loss_valid = False` in the target tensors, so they are excluded.
- `rec_event` = first absolute step `t` with `lane[t] = merge_lane` and
  `lane[t+1] = target_lane` and a valid next observation (`-1` if none).
- `last_obs` = last observed non-event ramp step (right-censoring point), not the
  disappearance step.
- `identity` / `entry` / `run_start` / `has_exposure` — the scored car is the
  contiguous presence run containing its event, else containing `last_obs`;
  `identity` is that run's first step, `entry` its first supervised ramp step,
  and `run_start` is the per-step run id used to terminate/slot-reuse safely.
  Because the contract fixes `identity` to `[B, K]`, at most one scored identity
  per slot is represented; earlier runs in a reused slot are not scored (they are
  outside the target window by construction of `last_obs`).

## `seq_tf_chunk(scene, c0, c1, *, s0, T, b, tau_a, temperature, sigma, n_quad, log_post, require_both=None)`

For each car `i` and quadrature node `q`:

```
alpha_q = exp(clamp(sigma * z_q, -8, 8))          # population location mu = 0
h_itq   = sigmoid(_core_margin(...)_itq / temperature)   # only feasible & supervised
log S_iq(t) = sum_{u < t} log(1 - h_iuq)          # infeasible steps contribute 0
event at e:    loglik_iq = log S_iq(e) + log h_ieq
censored at c: loglik_iq = sum_{u <= c} log(1 - h_iuq)
```

- `_core_margin` is imported from `merge_core`; residual is always `None`.
- `temperature` uses the runtime `clamp_min(1e-3)`; it may be frozen by the caller.
- A recorded event is supported only when its own step is feasible & supervised.
  An infeasible event gets **no** event factor: only the preceding supported
  non-events are scored and `unsupported_events` is incremented.
- No survival rows at/after the event. A censored car accumulates through
  `last_obs` inclusive. Masks use stable `logsigmoid`; no `0 * -inf` product.
- Persistent posterior: `A_q = log_post_q + g_chunk_q`,
  contribution `-logsumexp_q(A)`, `new_log_post = A - logsumexp(A)` detached.
  `log_post = None` starts from `log w_q`; an identity change resets to the prior.
  Summing the per-chunk contributions telescopes to one mixture likelihood.
- Returns `(nll_sum, n_cars, new_log_post, metrics)` with `n_cars` = eligible
  identities whose lifetime intersects the chunk; metrics contain `events`,
  `censored`, `scored_rows`, `zero_exposure`, `unsupported_events`, `n_cars`.
  `scene["n_cars"]` is the whole-unroll identity count for normalisation by the
  caller.

## Not implemented (and why)

- Alpha location `mu` is fixed at 0 as specified; the learned location head stays
  in the on-policy term (agent A).
- Margin residual is `None` per the contract (round-8 run freezes it at zero).
- No time-window/CRPS alternatives: ROUND8_ADVICE defers both; this is the exact
  single-interval discrete-time survival likelihood.
- Runtime `lane_v5_acts` envelope parity is not reproducible on recorded states
  (no rollout head); recorded feasibility uses presence + ramp lane + zone only.
  Margin/missing-side semantics do match the runtime `_core_margin` exactly.
- Slot reuse supports posterior reset and termination, but the fixed `[B, K]`
  identity can represent only the scored run per slot (see above).

## Tests (`tests/test_merge_core_v2_seqtf.py`)

Scene event/right-censor bookkeeping; event and censored NLLs; infeasible event
(no event factor, counted); passed feasible gaps contributing survival; chunk
split equality at fixed parameters with nonzero sigma; mixture mass
`sum_t f(t) + S(end) = 1` on a fully supported path; no post-event survival;
hazard-direction checks for survival and event factors; finite/nonzero gradients
to `s0`, `T`, `b`, `tau_a`, `sigma`; identity-change posterior reset; CAV
neighbours in gaps but CAV targets unsupervised; margin parity with the runtime
`_core_margin`; zero-exposure counted once.
