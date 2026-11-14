# EVAL correction recovery — SOURCE DRAFT

Status: implementation and synthetic test drafts only. **Tests NOT RUN; imports,
compilation, pytest, project CLIs, model forwards, simulation, and data access
were not executed.** Acceptance remains pending the coordinator's reviewed
guard. No scientific result or predictive-quality claim is made.

Base: `25479ae`, branch `codex/evalcorr`. Scope is the evaluator/API correction
wiring, its rollout carriers, and the API checkpoint-loader integration.
No trainer, adapter, frozen config, learned
architecture, weights, summary equations, or state-token equations were edited.

## Source findings and draft repair

- `evalh.idea_a._outcome` constructed a fresh `CavMotionSummary`, then passed
  its projected features into a trained `RealCorrBundle` that already contains
  the learned projection and expects raw16. Both scorers now require the
  existing projection/head composition and pass raw16 directly. Bare heads,
  missing heads, and unsupported compositions fail closed; no fallback head
  or learned projection is constructed.
- Both scorers used presence as CAV selection and omitted t0. The shared new
  `corr/evaluation.py` uses original self-identified CAV membership, explicit
  per-boundary identities, and monotone survival. It includes t0 plus every
  executed state, with initial acceleration from original t0 kinematics.
  Forward-filling after departure makes displacement end at the last alive
  position. Reused slots and mid-window entrants do not become original CAVs.
- This follows `train/idea_a.py`'s `_cav_motion_summary` and canonical window
  construction: target-mask cumulative survival; forward-filled positions and
  channels; `is_cav(t0) & alive_state`; T+1 boundaries. Frozen feature fields
  `cav_plan_ch[...,0:2]` encode observed CAV speed/acceleration (see
  `features/idea_a.py` around 836–840), despite their historic target names.
- Encoder data follows encoder parameter dtype/device, with boolean masks;
  raw16 and final token follow correction parameter dtype/device. Inference
  is under `torch.no_grad()`. No module is relocated or recast by scoring.
- A6 ON requires original kin, mask, nonempty t0 world, and an explicit
  `[B,2,30]` counter, plus an encoder. A6 OFF retains zero token compatibility
  but corrected scoring still requires original CAV selection and identity.
  Explicit config OFF wins even if a stale encoder is attached, matching the
  trainer. Legacy bundles without a flag infer ON from encoder presence;
  actual checkpoints carry the flag. OFF never requires an A6 counter.
- API StateSnapshot reconstruction now follows `build_world`'s filtered
  `model_rows()` order and appended obstacles. Excluded junction rows no longer
  shift token or CAV positions. Obstacles are present kinematic state, not CAVs.
- For S>1 evaluator input, each sample is summarized/scored separately before
  averaging quantiles. The previous averaging of trajectories could combine
  different vehicle identities. The production `predict_quantiles` S=1 path
  keeps its aggregation behavior. A draft test covers sample-varying survival
  and identity and independently pins the two raw summaries and mean result.

## Carrier and identity contract

`Rollouts` adds optional, None-default channels:

- `t0_is_cav`: boolean `[B,K]`, captured before step/teacher mutation;
- `t0_slot_id`: integer `[B,K]`, the original world roster;
- `slot_ids`: integer `[S,T+1,B,K]`, actual `world.slot_id` at each boundary.

Legacy direct construction remains possible. Actual correction scoring refuses
missing identity channels; it never infers CAV membership from presence.
Positions/speeds/masks/IDs include t0; acceleration is `[S,T,B,K]` for executed
transitions. Original position/speed/mask/identity must match boundary zero.
Occupied original IDs must be distinct and nonnegative within each world.

The engine's default original `arange(K)` is an assigned synthetic roster of
the initial world, not evidence about external vehicle IDs. Actual future IDs
are recorded from `world.slot_id`; no identity is inferred from future slot
indices. `_insert_arrivals` assigns its monotonically incrementing counter;
both owned rollout entrypoints now initialize that counter to `max(t0 IDs)+1`
instead of zero, avoiding collisions with original roster IDs. Teacher forcing
changes kinematics, not that identity roster. Monotone presence/ID masking also
prevents revival after a departure even if an original ID reappears.

`wm_api.outcome_head` accepts `slot_ids` and the explicit `t0_*` channels.
Alternatively a StateSnapshot supplies the initial channels in build_world's
assigned roster order; **temporal IDs remain mandatory**. A partial explicit
roster is refused rather than filled from a second StateSnapshot roster.
Custom injected ID conventions require explicit original channels.

## Draft verification nodes — pending reviewed guard

The two new files intentionally keep their requested identical basenames.
They are in non-package test directories: run them in **separate guarded
pytest processes**, or have the coordinator approve importlib collection mode.
The following are exact test function nodes; parameterized cases expand under
their listed node. These are a proposed narrow verification inventory, not
commands executed or an authorization to execute.

Prefix `output/wd01/wm_n1/evalh/tests/test_correction_pipeline_recovery.py::`:

- `test_real_composition_exact_train_eval_api_parity`
- `test_hdv_and_departed_reused_slots_cannot_change_summary`
- `test_same_slot_id_after_absence_stays_dead_and_immediate_reuse_is_dead`
- `test_multiple_samples_score_coherent_identity_then_average_quantiles`
- `test_corrected_scoring_missing_channels_fails_closed`
- `test_unsupported_bare_head_and_missing_head_fail_closed`
- `test_a6_off_zeros_still_requires_explicit_cav_identity`
- `test_explicit_a6_off_ignores_stale_encoder_without_requiring_counter`
- `test_encoder_and_correction_have_independent_dtype_device_placement`

Prefix `output/wd01/wm_n1/wm_api/tests/test_correction_pipeline_recovery.py::`:

- `test_snapshot_filters_junctions_and_matches_build_world_roster`
- `test_api_hdv_motion_invariant_and_cav_motion_sensitive`
- `test_explicit_partial_channels_fail_closed_even_with_snapshot`
- `test_missing_ids_counter_or_required_encoder_fail_closed`
- `test_a6_off_allows_absent_counter_but_not_missing_identity`
- `test_rollout_member_carries_identity_after_each_step`

Focused existing nodes with changed fixtures/assertions:

- `output/wd01/wm_n1/rollout2/tests/test_rollout2_interface.py::test_engine_freezes_original_t0_channels_before_mutation`
- `output/wd01/wm_n1/evalh/tests/test_idea_a_variant.py::test_idea_a_outcome_uses_state_token`
- `output/wd01/wm_n1/evalh/tests/test_idea_a_variant.py::test_idea_a_outcome_refuses_state_token_checkpoint_with_no_encoder`
- `output/wd01/wm_n1/evalh/tests/test_idea_a_variant.py::test_idea_a_outcome_refuses_engine_bundle_cfg_on_without_encoder`
- `output/wd01/wm_n1/evalh/tests/test_idea_a_variant.py::test_idea_a_outcome_refuses_empty_t0_state`
- `output/wd01/wm_n1/evalh/tests/test_idea_a_variant.py::test_idea_a_outcome_flag_off_uses_zeros`
- `output/wd01/wm_n1/wm_api/tests/test_state_token.py::test_outcome_head_uses_state_token_not_zeros`
- `output/wd01/wm_n1/wm_api/tests/test_state_token.py::test_outcome_head_refuses_missing_state_for_state_token_checkpoint`
- `output/wd01/wm_n1/wm_api/tests/test_state_token.py::test_outcome_head_refuses_empty_state_for_state_token_checkpoint`
- `output/wd01/wm_n1/wm_api/tests/test_state_token.py::test_outcome_head_flag_off_needs_cav_identity_but_not_counter`

Existing `test_contrastfix_token_parity.py` remains unchanged: its original-t0,
counter, acceleration, length and future-independence tests target the same
token wrapper and remain part of the parent draft's pending verification.

Fixture migration: evaluator A6 tests formerly used an untrained fake head and
missing original channels; now they record tokens through real synthetic P5
composition and explicit coherent channels. API A6 tests formerly paired
unrelated trajectories with StateSnapshot and omitted counters/IDs; now they
use coherent t0 data and perturb only counters to isolate token influence.
The old OFF-with-no-state expectation is replaced by OFF-with-identity and no
counter; lack of CAV identity must fail. Adapter loader tests are unchanged
because they assert loading/restoration, not correction scoring.

## Limits and remaining integration

Synthetic test weights are fresh deterministic fixtures; no trained checkpoint
was loaded, fitted, or evaluated. CUDA coverage is conditional on availability
when authorized. Other callers/old tests that omit CAV identities, temporal IDs,
or required A6 counters will now refuse scoring and may need their own bounded
fixture/caller migration. No claim of broad suite compatibility is made.

No data, HOLDOUT/CAL/sealed/J4B/INC1 outcomes, D: path, credential/config secrets,
simulation, fitting, uploads, approval, merge, push, cleanup, deletion, or move
was performed. Only owned source and draft tests/report are proposed for the
isolated commit. Parent integrator owns any shared evidence/ledger close-out;
this work changes implementation integrity, not scientific conclusions.

## Source-only checkpoint-loader integration addendum

The parent independently identified another essential blocker after reviewing
the first draft commit `2bd8362`: `wm_api/checkpoint.py` compared the trained
composition's internal `summary_dim` to 16. The actual default trainer factory
constructs `RealCorrBundle`, whose external input is raw16 and whose trained
internal projection width is 64. The obsolete width gate rejected this normal
checkpoint even after the scorer wiring was repaired.

The loader now calls the same `corr.evaluation.trained_correction(bundle)`
validator as both scorers, translating its `ValueError` to `CheckpointError`.
It accepts the existing trained composition without redefining widths, changing
weights, or constructing a replacement projection. Bare heads and missing
projections remain unsupported. Existing strict state-dict shape/key checking
and optional file hashing remain in place.

Source audit also found that globally recognized prefixes were silently ignored
when their configured module was absent. The parent explicitly extended scope
to close this: any `hdv.*`, `arrival.*`, `corr.*`, or `state.*` payload for an
absent module now raises `CheckpointError`. This includes stray `state.*`
weights with explicit A6 OFF. Legitimate OFF checkpoints with no encoder keys
still load; ON checkpoints must contain their encoder parameters. The empty
per-module payload check now checks the module's unprefixed state dictionary,
instead of incorrectly searching those keys for their external namespace.

Normal loader source path reviewed: config is rebuilt from the payload through
`IdeaATrainConfig.from_dict`, retained as `bundle.config`; A6 True constructs
`state_enc`, False omits it. The loader casts HDV, correction and state encoder
to float64 on CPU, keeps arrival float32, loads weights strictly and sets all
modules to eval. The new inference helper follows those loaded placements.
Outcome-normalization restoration and SHA checks are unchanged. The tests
below build fresh synthetic payloads from the real factory and namespaced
state dictionaries; they do not call the trainer save path or open any existing
checkpoint, registration, corpus, or outcome file.

Additional exact DRAFT nodes, all **NOT RUN**, prefix
`output/wd01/wm_n1/wm_api/tests/test_checkpoint_recovery.py::`:

- `test_load_default_summary64_checkpoint_and_score_api` (ON and OFF)
- `test_checkpoint_strict_weight_mismatches_still_refused` (shape, missing
  projection, missing correction, missing state, extra correction key, unknown
  namespace)
- `test_loader_refuses_unsupported_composition_even_with_matching_weights`
- `test_explicit_off_checkpoint_with_stray_state_weights_refused`
- `test_checkpoint_weights_for_absent_module_refused`

The first node retains the default internal summary width 64, verifies exact
restored weights/dtypes/config, and drafts a synthetic API scoring call through
the loaded member. It reduces only HDV/arrival fixture size. These are source
proposals awaiting the reviewed guard, not executed evidence. Root source
review of the original scoring draft was reported coherent; no test acceptance
or real-checkpoint compatibility claim follows from that source review.
