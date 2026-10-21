# R9K handoff: cut-in supervision and leader-change history

## Status and commits

Implementation and focused CPU checks are complete in the K-owned files. Both
switches remain default-off. Overall acceptance is **partial / not launch-ready**:
two inherited round-8 config tests fail, coordinator feature/history hooks remain,
and this workspace does not contain the measured component attribution or pilot.

- Base: `6fa9d020`; inherited untested draft: `7d9946be`.
- `a39c27c4`: rewrite loss, repair tracker, add config/settings/composer plumbing
  and the first 27 passing focused checks.
- `9279eaa8`: causal recurrent replay, 33 focused checks, round-8 gradient/update
  reachability, explicit prefix and slot-reset checks.
- This document is committed separately after those implementation commits.

Changed files: `src/wms/train/losses/cutin_reaction.py`,
`src/wms/model/leader_change.py`, `src/wms/config/train.py`,
`src/wms/config/legacy_flat.py`, `src/wms/train/settings.py`,
`src/wms/train/losses/compose.py`, `tests/test_cutin_reaction.py`, and this file.

## Loss contract

`[loss.cutin_reaction]` accepts `weight=0`, `control_weight=1`, `pre_s=1`,
`max_window_s=6`, `huber_delta_mps2=0.5`. The control coefficient is **relative**:
`weight * (event_mean + control_weight * control_mean)`. `weight=0` disables the
whole term, including controls, and the composer never builds/calls replay. A
positive control coefficient is required so event supervision cannot be enabled
without its ordinary-braking anchor. All numeric fields are validated as finite.

The term consumes the trainer's actual `UnrollTargets` and is differentiable once
per unroll. `UnrollRunner._finish` already invokes its backward at lines 560-567;
no new loss-backward hook is needed. Empty enabled batches return a backward-safe
zero. Each evaluation creates a fresh graph, independent of chunk backward and of
other unroll terms; no encoder graph from `lane_event` is reused after backward.

### Events, timing, windows and normalization

- An event is a continuous valid recorded slot going from ramp lane 2 to lane 1.
  Its timestamp `e` is the **first lane-1 observation**, matching
  `scripts/c4/wm_discharge_check.py:145-181`. Inserters may be CAVs.
- The follower is the nearest front position behind the inserter at `e` on lane
  1. A nearer CAV follower is not skipped in favor of a more distant HDV; that
  event is not supervised. Only continuous HDV follower transitions are scored.
- SUMO rows are post-step: input recorded state `t`, target recorded acceleration
  at `t+1`. The first prediction using inserted geometry is at `t=e`; the
  transition ending at `e` can also be scored as pre-event context. No future
  response labels, event flags or window lengths are passed to the predictor.
- The post window ends at the first nonnegative recorded follower acceleration
  **after** `e`, including its recovery transition. The documented maximum is
  **6 seconds** (the supplied C11 response horizon); configurable only downward.
  Optional pre-context is at most **1 second**. Step counts use floor, so a
  nonintegral horizon cannot exceed its seconds cap. End-of-unroll, absence,
  follower lane change, CAV status change and identity replacement censor a
  window. A missing recovery is counted as censored, not fabricated recovery.
- Each scored event has unit mass, uniformly distributed over its valid rows;
  average those event masses across the batch. Overlapping events retain their
  independent masses rather than being collapsed to one event. The control mean
  has separate unit mass over valid lane-1-to-lane-1 HDV transitions outside the
  union of follower event windows. Controls include negative recorded
  accelerations. Huber uses the stated delta in acceleration units.
- `last_stats` reports scored event count, unique event rows, control rows,
  censored windows and separate event/control loss values. Counts describe this
  unroll, not a full-world event census.

`UnrollTargets` contains slots, not vehicle IDs. Absence breaks continuity.
Immediate replacement without absence must be supplied as the explicit `reset`
mask described below; no position-jump or behavioral identity heuristic is used.

### Recorded-state prediction

Replay steps through recorded states, with scalar absolute unroll `k_steps=t`
(valid under `WmSModel.forward`'s scalar-or-`[B,K]` contract). It builds the same
rolling motion features as runtime from **current/past recorded** data, carries
slow residual state, lane-v5 accumulators, blocked counts, negotiation history and
merge-wait state, and resets reused slots. Driver parameters are freshly encoded
on an independent graph at t0 and for replacement occupants. Targets and all
recorded inputs are detached. Learned slow state remains differentiable.

Optional `targets.prepared['cutin_reaction_inputs']` fields:

- `history`, `history_mask`: the runner's rolling pre-t0 prefix, including t0.
- `driver_history`, `driver_history_mask`: the same causal observed driver prefix
  the rollout uses. They are observations, never driver truth tables.
- `lane_age0`: optional initial lane age; otherwise the model's history-based
  initializer is used, with its unknown sentinel when unavailable.
- `reset`: bool `[B,n_steps+1,K]`, true on a new occupant's first observation.

Without supplied prefixes replay explicitly cold-starts; long driver history is
zero-valued with an **all-false observation mask**. It does not invent observed
history. This is runnable with ordinary `UnrollTargets`, but is not a claim of
warm-start parity with a real training batch. **The history hook below is
required for the round-8 S2 pilot**, particularly for IDM gradient reachability.

## Leader-change features

`model.leader_change_features=false` reaches nested config, legacy flat and
`TrainSettings`; its default is omitted from legacy receipts. The tracker is a
pure opt-in helper; runtime consumption is deliberately left to the coordinator.

`LEADER_CHANGE_INPUTS` declares 14 ordered channels: the ten named physical/flag
features followed by four validity bits. `feature_matrix()` zeros missing values
and appends the masks; a true zero gap is distinguishable from unknown geometry.
Reference scales are 10 s, 50 m and 10 m/s. Raw named outputs retain physical
units. These are documented engineering scales, not fitted TRAIN statistics.

Association is the nearest front position on the same lane within a 150 m bumper
gap (feature association range), with no CAV exclusion. The age saturates at 30 s
only as a feature bound, never an acceleration/merge rule. Change priority is own
lane change, observed new-leader lane entry, then previous-leader exit. Other
observed re-associations have known age but no cause flag. A newly appearing
neighbor with unknown prior lane is not labeled a cut-in. Cause flags and change
geometry persist until the next change, including across chunks. The previous
leader gap is its **last observed** gap, not an invented current gap after exit.

`state_dict`/`load_state_dict`, `detach_`, and `reset_slots` support chunk carry.
Resetting a leader slot invalidates dependent follower history as well. Missing
lanes/absences clear history, and window-start age/change geometry have false
validity masks. `recorded_leader_change(..., tracker=..., reset=...)` uses the same
step implementation and must receive non-overlapping chunks.

## Exact coordinator hooks (line numbers against unchanged base files)

### Runtime feature consumer: currently not installed on this branch

1. `src/wms/model/spec.py:55-58,243-251`: add validated
   `leader_change_features: bool = False` to `ModelSpec`; its field-based
   `from_mapping` will then retain the new legacy key. At present `from_mapping`
   ignores this key, so setting it in TOML alone does not enable model features.
2. `src/wms/model/wms.py:173-179` and the **end** of `__init__`: store the flag,
   and only when enabled attach a zero-initialized 14-to-`d_model` linear input
   adapter to `residual_net`. Construct it after all existing model modules to
   preserve their RNG initialization. With the flag off create no parameters and
   execute no extra tensor arithmetic. Old checkpoints load strictly unchanged
   when off; enabled loading must allow only the specifically new adapter keys
   to remain at zero, not broadly ignore missing checkpoint weights.
3. `wms.py:1328`, immediately before `forward_kwargs`, in `step`: if enabled,
   build a local tracker from the incoming state's saved tracker tensors (fresh
   unknown history when absent), reset explicit new-occupant slots, step once on
   the **current** `state_kin`, and put its matrix in a copied forward state as
   `leader_change_features`. Do this outside activation-checkpointed `forward`;
   it must not mutate model-global state on checkpoint recomputation. Attach the
   detached saved tracker tensors to the step result under a unique prefix.
4. `wms.py:1974-1977`, alongside slow-state carry: copy those prefixed tensors
   from the step result into `current`; use top-level tensor keys so existing
   TBPTT detachment at `2017-2020` applies. The saved tracker describes the last
   decision input; the next chunk's first decision advances it once on the new
   current row. Do not advance twice at boundaries. At `1878-1881`, retain the
   actual `entries.insert(current)` new-slot mask for reset; real ID changes
   must also reset slots, including replacements that stay present.
5. `wms.py:813` and `_residual_prediction` at `664-672`: pass the supplied matrix
   through the residual helper (including its checkpoint call) to the residual
   net. `forward` only consumes it; the teacher-forced helper already supplies
   it when `model.leader_change_features` is true and must not advance a second
   tracker inside `forward`.
6. `src/wms/model/residual_net.py:348-351,369-389`: add the optional feature kwarg
   and, only when its adapter exists, add the masked adapter output to the
   per-vehicle embedding before attention. Preserve the seven kinematics
   channels rather than widening/misinterpreting `veh_kin`. Register the causal
   inputs and mask ordering in `input_schema`; include the new parameters in
   optimizer membership. Enabled-checkpoint/init parity needs coordinator tests.

### Warm-start loss history: currently requires one runner insertion

At `src/wms/train/unroll.py:180`, after `self.composer.prepare(targets)` and before
any backward, add the following guarded preparation (detached raw observations,
not shared encoder outputs):

```python
if 'cutin_reaction' in self.composer.names:
    targets.prepared['cutin_reaction_inputs'] = {
        'history': inp.hist, 'history_mask': inp.hist_mask,
        'driver_history': inp.observed_hist,
        'driver_history_mask': inp.observed_hist_mask,
    }
    # Add reset if the dataset permits immediate slot replacement without absence.
```

Presence-only datasets with stable slots do not need an explicit reset mask. A
dataset that reuses slots must derive reset from its identity/continuity metadata
and pass the same identity semantics to rollout and recorded replay. The loss
handles absent prefixes without error, but silently treating cold replay as the
same warm-start real-data objective would be incorrect.

## Reachable parameters and S2 interpretation

On a synthetic recorded sequence, the **round-8 ModelSpec and optimizer config**
(a fresh model, default PhysicsConfig, not a trained checkpoint), with two valid
causal pre-t0 observations, produce finite nonzero cut-in-loss
gradients in these parameter families:

- `driver_encoder`: inferred physical driver parameters/embedding (the IDM
  acceleration path is trainable when populated history supports it).
- `residual_net`: acceleration head, attention/trunk and conditioning path.
- `temporal`, `history_proj`, `k_embedding`.
- Saturating gate `gate_g_max_logit`, `gate_tau_raw`.
- `merge_core` trainable acceleration/yield-related parameters. This loss can
  therefore alter shared core dynamics; it is not an isolated B parameter branch.

`build_optimizer` uses the round-8 freeze prefixes/LR scales and verifies frozen
temperature and legacy core residual tensors stay bitwise unchanged after an
update. A separate small-model slow test reaches `temporal_tau_raw`; the standard
hybrid test also reaches the encoder. There are no new trainable parameters in
the loss or tracker themselves. The coordinator's enabled adapter will be new.

Important limits: with round-8 long history entirely unknown, shrinkage can make
the inferred IDM parameters population defaults and yield zero encoder gradients.
Residual-only mode does not retune fixed IDM constants. Final plant clamps and
residual tanh/soft-bound saturation can also kill a path's useful gradient. These
synthetic init checks do **not** establish reachability at the trained stage300
checkpoint, successful compensation of an IDM-dominant response, or sufficient
gradient share under the combined losses. S2 still needs per-component gradients,
an actual update/pilot and non-insertion/TTC controls at that checkpoint.

S2 component selection is unresolved here: the supplied evidence path is outside
the permitted workspace. No decomposition for rounds 2/7b/8, matched-cell counts,
episode-bootstrap excess CI, before/after response or TTC improvement is claimed.
Keep both switches off until that evidence selects a supported mechanism. Loss
weight zero at inference cannot undo shared-weight training: preserve the round-8
baseline and matched A-only/B-only training controls for attribution.

## Slow residual assessment

`temporal_slow.py:38-49` implements
`r_t = r_prev + dt/(tau+dt) * (target_t-r_prev)`, optionally slew-limited.
`wms.py:508-545,817-829` applies it only to the temporal acceleration emission;
the attention residual, IDM, yield and plant clamps are outside that filter.
Thus it attenuates temporal one-step pulses but does not remove them exactly,
does not directly fix an IDM/yield/clamp-dominant insertion response, and can
spread a negative pulse into a longer tail or delay recovery.

For example, dt=0.5 s and tau=2 s gives alpha=0.2: an isolated target pulse is
initially one fifth as large but leaves a decaying tail. This is arithmetic, not
a measured cut-in result. The four existing slow-residual CPU tests pass. The
requested 6b REGISTRATION entry near 02:52 is not present at the expected
in-workspace `runs/active/C4-V7FT-20261017/REGISTRATION.md`; no outside-workspace
copy was read, so no 6b empirical success claim can be made. Combine with the TF
loss only if measured temporal pulses support it and the response pilot does not
regress; replay carries the slow state and its gradient. Do not enable slow mode
merely because the code is compatible.

## Verification (CPU)

All commands used `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe` from the
workspace. No evaluation seeds or external data were used.

1. `-m pytest tests/test_cutin_reaction.py -q -s --basetemp="E:/agent-work/c4-v10r9k2/tmp_r9k"`
   — **33 passed**. Covers no events/two overlapping events, post-step alignment,
   recovery/cap, absence and immediate reuse, CAV neighbor handling, event masses,
   braking controls, zero-weight/empty backward, fresh graphs, causal inputs,
   feature types and chunk restoration, prefix/reset behavior, TOML-to-settings-
   to-composer, old flat hash/checkpoint loading, real optimizer reachability and
   default-off runner identity. Direct/slow replay matches fully observed rollout
   accelerations to absolute error <=1e-6 on synthetic recorded states.
2. `-m pytest tests/test_merge_context.py tests/test_merge_core.py tests/test_merge_core_v2_onpolicy.py tests/test_merge_core_v2_seqtf.py tests/test_merge_core_v2_config.py tests/unit/test_c4_slowres.py tests/unit/test_wmloss.py tests/unit/test_wmlane.py tests/unit/test_wm_discharge_check.py -q`
   — **106 passed, 2 failed**, zero skips. Failures are
   `test_round8_config_reaches_flat_and_settings` (line 62) and
   `test_compose_passes_the_r8_fields` (line 145), both asserting stale placeholder
   normalizers 1.0 instead of checked-in 395.2/2.15. The config and failing test
   file are unchanged from `6fa9d020` (`git diff 6fa9d020 --` those two paths is
   empty). These failures are **not waived** for S4. The test owner/coordinator
   must update the assertions/default-omission expectations; do not revert the
   measured normalizers to make stale tests pass.
3. `-m pytest tests/unit/config tests/unit/train/test_train_layer.py tests/unit/train/test_post_merge.py tests/unit/train/test_residual_head_rescale.py -q --basetemp="E:/agent-work/c4-v10r9k2/tmp_r9k_regression"`
   — **182 passed**, zero skips.
4. `-m ruff check src/wms/model/leader_change.py src/wms/train/losses/cutin_reaction.py src/wms/config/train.py src/wms/config/legacy_flat.py src/wms/train/settings.py src/wms/train/losses/compose.py tests/test_cutin_reaction.py`
   — passed. `git diff --check` — passed.

The round-7b legacy-flat SHA-256 remains
`db0d9376e013e22febbdd44ebd75516fc9585384c0f48192d0897b9e300ef011`.
All added defaults are in `OMIT_WHEN_DEFAULT`, and no existing loss summation
order is changed when disabled. No required focused check was skipped.

Next integration actions: install the documented coordinator hooks; repair the
two out-of-scope stale test expectations; rerun S4; provide in-workspace 6b and
round-8-inclusive S2 evidence and the real-data pilot before selecting/enabling B.
