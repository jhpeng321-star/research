# REVIEW — WM-N1 Idea A, piece P7 TRAIN (SECOND PASS, critical)

- **Scope:** commit `7f9351d` on `zcode/revA7g` (`zcode/wmn1p7`) —
  `train/idea_a.py`, `train/tests/test_idea_a_*.py`, bundle stage-list
  additions (`bundle/queue.py`).
- **Against:** `INTERFACES_V2.md` (P7 section + config keys + torch-2.5
  notes); amendment 3 `A3.2`/`A3.4`
  (`zcode/wd01:output/wd01/wm_n1/REGISTRATION_WM_N1_AMENDMENT_3_IDEA_A.md`);
  pre-flight gates `G4`/`G6`
  (`zcode/wd01:output/wd01/wm_n1/PREFLIGHT_GATES_IDEA_A.md`).
- **Reviewer:** GLM second pass, 2026-10-01. All paths below are relative to
  `output/wd01/wm_n1/` unless marked. Line numbers refer to the tree AFTER the
  two `review fix` changes described in "Defects found" (they only shift
  lines; no behaviour other than the fixes changed).

## Verdict per focus item

| # | focus item | verdict |
| --- | --- | --- |
| 1 | loss terms + weights vs A3.2 | PASS |
| 2 | gradient reaches HDV params in the rollout loss (not detached) | PASS |
| 3 | teacher-forced CAVs: no gradients / labels into the HDV path | PASS |
| 4 | contrast pairing: same world+t0 siblings, treatment-minus-reference | PASS |
| 5 | held-out filters (i)-(iii) exclude the right arms | PASS |
| 6 | partial-ensemble k recorded; final test refuses k<5 | PASS |
| 7 | resume determinism | **FAIL -> FIXED** |
| 8 | device/dtype handling for CUDA | PASS |
| 9 | torch 2.5.1 API compatibility | PASS |
| 10 | no CAL leak into training | **FAIL -> FIXED** |

### 1. Loss terms and weights vs amendment A3.2 — PASS

All five A3.2 losses are computed and summed with the registered weights:

- one-step HDV residual NLL, excess form (`train/idea_a.py:605-632`: target
  `target_accel - baseline_idm` on HDV slots, fixed-scale sigma from
  `HdvConfig.accel_delta_scale_mps2`, plus the residual lane cross-entropy);
- multi-step rollout loss (`train/idea_a.py:634-694`);
- arrival negative-binomial via the P4 head signature
  (`train/idea_a.py:696-700` -> `model/losses.py:74-83`);
- correction-head CRPS via the frozen `outcome_crps`
  (`train/idea_a.py:702-723` -> `model/losses.py:105-107`);
- paired-branch CONTRAST on outcomes + HDV response
  (`train/idea_a.py:732-792`).

Term gating in `loss_terms` (`train/idea_a.py:795-818`) requires exactly the
batch keys each term needs and refuses a batch that yields no term. Weights:
the five lambdas default to 1.0 (`train/idea_a.py:94-98`) and map through
`loss_weights()` (`train/idea_a.py:122-127`); the contrast sub-terms use the
frozen `lambda_c = 1` (`model/losses.py:116`, `model/constants.py:67`).
Amendment A3.2 pins no other numeric weights. G4 reachability: the residual
and rollout terms are excess/zero-reachable (constants dropped), so the gate
can drive the total to ~0 (`train/idea_a.py:292-302` cosine tail rationale).

Non-bug verified numerically: the frozen `model/losses.arrival_loss`
unsqueeze(0) makes count `[1,B,E]` against `mu [B,E]` — right-aligned
broadcast, elementwise-correct (checked for B=3 against per-item losses; no
cross-batch mixing).

### 2. Gradients reach the HDV parameters in the multi-step rollout loss — PASS

`rollout_loss` (`train/idea_a.py:634-694`): `a_hdv` is the live net output
(677-679); the HDV branch of `torch.where` keeps the graph (680-681); the D1
step is the differentiable replica of the frozen `integrate_d1`
(`train/idea_a.py:315-325` vs `rollout2/validity.py:24-30`, identical
trapezoidal rule); `a_prev = a.detach()` happens only AFTER the step loss is
accumulated (693), so per-step gradients propagate back through the HDV state
trajectory, not just the last step. `test_idea_a_losses.py:25-42` proves
non-zero finite gradients per term into its owning module (rollout -> hdv).
Arrival/correction parameters are intentionally NOT part of the rollout term —
A3.2 lists them as separate losses and they are trained by their own terms
(`loss_terms`), each with a gradient check in the same test.

### 3. Teacher-forced CAVs never receive gradients or labels into the HDV path — PASS

- CAV slots take `.detach()`ed teacher values for accel, s, v and lane
  (`train/idea_a.py:680-692`); no gradient can flow into the net through a
  CAV slot, and the step loss is masked to `step_mask & ~cav` (684).
- The teacher tensor carries only `(s, lane, v, a)` columns
  (`train/idea_a.py:57-58`); no CAV flag, action one-hot or strategy label
  exists in it. Net inputs are built by `_kin_features`
  (`train/idea_a.py:281-287`) in `hdv/config.KIN_FIELDS` order, matching the
  P9 kinematics view that strips the CAV flag and action one-hots
  (`features/idea_a.py:39-41,54-62`); the corpus path consumes
  `kinematics_view`, never raw `veh_step` (`train/idea_a.py:241-278`,
  INTERFACES mismatch 4).
- `residual_nll` masks CAV slots out of the HDV loss
  (`train/idea_a.py:622-624`); the contrast response term masks
  `~cav[a] & ~cav[b]` (`train/idea_a.py:774-778`).
- `is_cav` is used for loss-slot selection only, never as a net input
  (`train/idea_a.py:260`, `hdv/net.py:29-36` guard).

### 4. Contrast pairing: same world+t0 siblings, treatment-minus-reference — PASS

`_pairs` -> `train.dataset.contrast_batch_indices`
(`train/idea_a.py:725-730` -> `train/dataset.py:164-187`) groups by
`pair_key = "<world>@t0=<t0>"` (`train/dataset.py:11-14`, written by the
corpus, e.g. `train/tests/packsynth.py:103`), picks the all-native reference
as `b` and every other arm of the same group as `a` — so `a` is treatment and
`b` the native reference, both from the same world+t0. The outcome sub-term is
`contrast_loss(q[a], q[b], y[a], y[b])` = pinball of `q_a - q_b` vs `y_a - y_b`
(`train/idea_a.py:764-765` -> `model/losses.py:116-136`, sign convention in
its docstring), and the response sub-term is the pinball of the predicted
residual difference vs the realised residual difference with the same
orientation (`train/idea_a.py:780-787`). Tests:
`test_idea_a_losses.py:81-94` (aligned < zero-gain < opposite sign) and
`:97-117` (isolated response term matches the analytic
`treatment-minus-reference` pinball).

### 5. Held-out filters (i)-(iii) — PASS

- (i) J4->J4B: every item of a J4B world is excluded from training
  (`train/idea_a.py:339-342,394-395`); `J4B`, `J4B_*`, `J4B-*` all match, and
  `is_j4_world` excludes J4B so later J4-side logic never sees it. Eval split
  is exactly the J4B arms (`test_idea_a_heldout.py:47-55`). J4B is never in
  training.
- (ii) leave-PI-out: `is_pi_item` matches `pi`, `pi_*`, `pi-*` on arm/policy
  (`train/idea_a.py:345-351`) and `held_out_split` removes every such item
  (396-397), i.e. PI arms are removed from native-vs-PI pairs as well. The
  all-native reference arm of the pair group STAYS: the corpus model has one
  shared native reference per (world, t0) serving all sibling arms
  (`train/dataset.py:153-161`, `train/tests/packsynth.py`), so removing it
  would also delete the damping/wall pairs' reference — the amendment's
  "every PI arm (native-vs-PI pairs included) removed" is implemented as
  "every PI arm, no pairing exception", which is the only reading consistent
  with the shared-reference corpus.
- (iii) leave-highest-J4-wall-speed-out: candidates are wall arms in J4
  worlds only (J4B excluded, `train/idea_a.py:398-400`); the factor is fixed
  a priori as `max` of the factors present, with an explicit `factor=`
  override (401-409); only that factor's arms are excluded. Factor parsing
  handles `wall_v31`, `wall-27mph` and explicit keys (364-381).
- G6: `held_out_dry_run` builds all three splits, asserts disjointness and
  emits the per-test receipt with the excluded arms
  (`train/idea_a.py:479-513`); the bundle stage wires it over the packed TRAIN
  corpus (`bundle/queue.py:288-298`), gated behind `members-2-5`
  (`bundle/queue.py:43-45`). Tests: `test_idea_a_heldout.py:47-100`,
  `test_idea_a_bundle_stage.py:35-53`.

### 6. Partial ensemble recorded; final-test refusal for k<5 — PASS

`ensemble_flags(k)` records `ensemble_k`, `members`, `partial`,
`final_test_eligible` (`train/idea_a.py:177-187`); these land in the
checkpoint payload, the JSON sidecar and the fit receipt
(`train/idea_a.py:948-968,928-932`). `require_full_ensemble` refuses any
k<5 (`train/idea_a.py:190-199`) and the `idea-a-ensemble` bundle stage
applies it, recording k and refusing a partial ensemble
(`bundle/queue.py:301-321`). A missing middle member yields a conservative
refusal (k counted from the files present). Tests:
`test_idea_a_checkpoint.py:35-60`, `test_idea_a_bundle_stage.py:56-77`.

### 7. Resume determinism — FAIL, FIXED in this review

**Defect.** `save_checkpoint` writes optimizer + scheduler states
(`train/idea_a.py:982-984` old), but `resume()` only restored them when the
caller passed an optimizer or the trainer already had one. In the natural
crash-restart flow (`tr = IdeaATrainer(...); tr.resume(ckpt); tr.fit()`) no
frame exists, the saved states were silently dropped, and `_ensure_optimizer`
created a fresh Adam (zeroed moments) and a restarted cosine schedule — the
resumed member diverges from the uninterrupted run. Measured on the G4
fixture (6-step run, resume after step 3): step-5 loss **40.9636 vs 43.8003**
uninterrupted — a silently different training trajectory and wasted GPU hours.

**Fix (review fix commit).** `resume()` now stages the optimizer/scheduler
states when no frame exists (`train/idea_a.py:1036-1047`); `_ensure_optimizer`
loads the staged states right after creating the frames
(`train/idea_a.py:872-891`); `_optimizer_state`/`_scheduler_state` fall back
to the staged states so a re-save cannot lose them (`train/idea_a.py:1006-
1015`). The checkpoint additionally stores the global CPU RNG state
(`rng_state`, `train/idea_a.py:994`) and `resume` restores it (older
checkpoints fall back to seeding, `train/idea_a.py:1059-1062`); batch-order
RNG was already restored via `gen_state` (`train/idea_a.py:1057-1058`).

**Added test.** `test_idea_a_checkpoint.py::test_resume_fit_continues_
optimizer_and_schedule` — a fresh trainer resumes a 3-of-6-step checkpoint and
must reproduce the uninterrupted run's remaining losses to 1e-6 (verified to
FAIL against the pre-fix behaviour: 40.96 vs 43.80).

### 8. Device/dtype handling for CUDA — PASS

- `_to_device` moves tensors only when the device is not CPU; `meta` (list)
  passes through (`train/idea_a.py:833-838`).
- No `.cpu()` in any hot loop; the only per-step host sync is the scalar
  `float(loss.detach())` (`train/idea_a.py:908`), which is standard.
- Every newly created tensor inherits the source dtype/device: rollout
  accumulator `torch.zeros((), dtype=s.dtype, device=s.device)` (674),
  zero-token fallback (719-720, 757-759), quantile `levels` on `qa.device`
  (782-783), one-hot `arange` on `lane.device` (284-285).
- Batch-order randomness uses a CPU `torch.Generator` (`train/idea_a.py:543-
  544`), matching the INTERFACES_V2 torch-2.5 note (device-independent
  results); pair indices from `contrast_batch_indices` are CPU LongTensors,
  valid indices for CUDA tensors.
- Dtype: everything runs float32 (`arrivals.float()`, float32 targets,
  `lgamma` on float32); no float64 creep found (`wall_speed_factor` etc.
  operate on Python metadata, not tensors).

### 9. torch 2.5.1 API compatibility — PASS

Ops used are all 2.5-compatible: `Adam`, `CosineAnnealingLR(T_max, eta_min)`,
`Tensor.zero_grad(set_to_none=)`, `clip_grad_norm_`, `randperm(generator=)`,
`torch.where/clamp/stack/arange/tensor/zeros/full_like/one_hot/cat`,
`F.cross_entropy`, `torch.lgamma` (frozen losses), `Generator.get_state/set_state`,
`torch.get_rng_state/set_rng_state`. No `torch.compile`, amp/autocast,
SDPA/flex_attention or export. `torch.load(..., weights_only=True)` is passed
explicitly everywhere (`train/idea_a.py:1028`, checkpoint test :38) and the
payload holds only allowlisted types (tensors + str/int/bool/dict/list), so
2.5.1 (default `weights_only=False`) and newer defaults both load it. Local
box runs torch 2.12.1+cpu; the GPU box pins 2.5.1 (INTERFACES_V2) — no 2.6+
only API is used.

### 10. CAL leak into training — FAIL, FIXED in this review

**Defect.** `fit()` accepted any corpus-like source, including one whose
`split` is `CALIBRATION` — amendment A3.2 restricts CAL to calibration and
variant selection, and `evaluate_cal` is supposed to be its only reader
(diagnostic). A caller passing the CAL corpus would train on it.

**Fix (review fix commit).** `fit` now calls `_require_train_source`
(`train/idea_a.py:911-928,937-939`): any corpus-like source whose `split`
resolves to CALIBRATION — or any non-TRAIN label (HOLDOUT/TEST are sealed) —
raises `IdeaAConfigError` before step one; sources without a `split`
(plain window lists) pass. `evaluate_cal` remains the sole CAL reader.

**Added tests.** `test_idea_a_train_interface.py::test_fit_refuses_non_train_
split` (parametrized CALIBRATION/CAL/HOLDOUT/TEST; also asserts no checkpoint
was written) and `::test_fit_accepts_train_split_corpus`. Defence in depth
already present: `MemmapCorpus` refuses HOLDOUT/TEST/chunk-5 at open
(`train/dataset.py:59-64`), and bundle `verify-upload` refuses sealed shards
(`bundle/queue.py:161-172`).

## Bundle stage-list additions (P7 scope) — PASS

`STAGE_ORDER` appends `idea-a-ensemble`, `idea-a-held-out` after
`members-2-5` (`bundle/queue.py:43-45`); both stages are CPU-only, idempotent
via the state file (`run_stage` skip + predecessor check, `bundle/queue.py:
336-361`) and train nothing. Covered by `test_idea_a_bundle_stage.py:35-85`.

## Observations (reported, not changed)

- **O1 — shared receipt file.** `fit` writes one `TRAIN_RECEIPT.json` per
  `out_dir`; members 2-5 overwrite member 1's receipt
  (`train/idea_a.py:966-971`). Identity (`member`, `seed`) is inside the
  receipt, but per-member history is lost when members share `out_dir`.
  Suggest per-member receipt files in a follow-up (renaming now would break
  consumers/tests that assert the current name).
- **O2 — HDV lanes are static within the rollout horizon.** Only CAV lanes
  advance from the teacher (`train/idea_a.py:690-692`); HDV lane features
  stay at t0 for the whole horizon and the lane-residual head is trained by
  the one-step NLL only. Fidelity limitation (not a leak); acceptable for the
  A3.2 rollout response loss, worth flagging to P3/P8 for long-horizon lane
  outcomes.
- **O3 — refusal message tail.** `require_full_ensemble` names the tail
  members from k, not the actually missing files (e.g. member_3 absent,
  k=4 -> "train members [5] first"). Refusal itself is correct/conservative;
  the stage receipt records `found`.
- **O4 — NaN in zero-step history.** A zero-step fit records
  `mean_loss: NaN`, which `json.dumps` writes as `NaN` (non-strict JSON;
  Python `json.loads` accepts). Cosmetic.
- **O5 — corpus batches carry `baseline_idm = 0`.** The v1 pack has no
  `hdv_base` channel, so corpus-derived batches fit the FULL accel as the
  "residual" (`train/idea_a.py:261`, documented). Revisit when the P9
  featurized TRAIN pack (with `rollout_baseline_idm`) lands.
- **O6 — verified non-bug.** Frozen `model.losses.arrival_loss` `[1,B,E]` vs
  `[B,E]` broadcast is elementwise-correct (see item 1).

## Verification (this review)

```text
$ python -m pytest train/tests -q   (output/wd01/wm_n1)
68 passed in 58.14s
$ python -m ruff check train/idea_a.py train/tests/test_idea_a_checkpoint.py \
    train/tests/test_idea_a_train_interface.py
All checks passed!
```

Pre-review baseline was `62 passed in 81.89s`; the +6 are the review-fix
tests (1 resume-continuation + 4 parametrized split refusals + 1
TRAIN-accept). Defect 7 was additionally reproduced against the pre-fix
behaviour via a staged-state probe (resumed losses 45.9414/40.9636/35.2583 vs
uninterrupted 45.9414/43.8003/42.5834; fixed code matches to <1e-6).

## Summary

Two GPU-relevant defects found and fixed (resume optimizer/schedule restart;
CAL-train leak), plus global CPU RNG state now checkpointed. Everything else
reviewed matches amendment A3.2/A3.4, INTERFACES_V2 and gates G4/G6. No
SUMO, no real data, no sealed splits touched; all tests synthetic.
