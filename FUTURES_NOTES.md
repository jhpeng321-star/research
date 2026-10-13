# Sampled futures (goal D)

Decisions are evaluated on several sampled futures instead of one deterministic world-model rollout.
Everything is behind switches that default OFF; with the defaults every output is bit-identical to
`c4/int` (checked: real WM-S + policy + counterfactual rows, full `update()` metrics and post-update
policy weights, two seeds, `c4/int` source tree vs this branch).

## Module `src/wms/control/futures.py`

| piece | role |
|---|---|
| `FuturesConfig` | `n_samples`, `driver_noise_scale`, `embedding_noise_std`, `accel_noise`, `accel_noise_scale`, `accel_sigma_max`, `seed`; `active` = `n_samples > 1 or accel_noise` |
| `sample_driver_encodings(model, state, history_args, n, normal, ...)` | (a) epistemic: encodes drivers once (`model.encode_drivers`) and returns `n` perturbed copies of the IDM/MOBIL parameters (`values + z * encoder_uncertainty * scale`, capped to +/-50 % of the encoded value) and of the driver embedding (`+ z * embedding_noise_std * scale`). Factored out of the RSU check (second commit); `[n, B, K, ...]`, the RSU uses B = 1 |
| `NormalDraws` | standard normals from a caller-owned generator, cached by key: the first caller draws, all later callers (other ensemble members, RSU candidates) get the SAME numbers |
| `draw_accel_noise` | (b) aleatoric: `[n, B, T, K]` standard normals |
| `SampledFutures(cfg, seed).plan(models, state, rollout_kwargs, n_steps)` | `extras[member][sample]` = start-level `rollout` kwargs (`driver_params`, `driver_embedding`, `accel_noise`, `accel_noise_sigma_max`) |

`WmSModel.forward/step/rollout` gained optional `accel_noise [B, T, K]` (+ `accel_noise_sigma_max`):
HDV acceleration `+= z * min(exp(accel_log_sigma), sigma_max)` before the plant clamp (CAV slots are
never touched). `None` (default) takes the unchanged code path.

**CRN.** Draws come from a generator seeded by `(future_seed, update_count)` (validation: fixed
`(future_seed, "val<i>")`), independent of the global RNG (the global stream is untouched, so a run
with sampling off or on consumes the same policy noise). Per start and sample index the draws are
shared by (1) every ensemble member (member-specific encodings/sigmas stay distinct), (2) every
counterfactual variant row of that start (`expand_rows` with the variant row index), (3) every RSU
candidate (existing behaviour). Policy noise is separately shared across samples (the per-start
policy-noise draw is made at the same RNG state for every pseudo-member), so differences between
samples are WM differences only.

## Trainer: `trainer.future_samples = N` (default 1)

`DreamerConfig` fields: `future_samples`, `future_driver_scale` (1.0; 0 = no driver sampling),
`future_embedding_std` (0.05), `future_accel_noise` (False), `future_accel_scale` (1.0),
`future_accel_sigma_max` (1.0 m/s^2), `future_seed` (0). Example: `configs/c4_policy_futures.toml`.

`imagine()` rolls every member out once per sample; the (member, sample) pairs form the existing
"ensemble" axis, **member-major, E' = E*N**. Hence unchanged downstream code gives exactly the spec:
imagined returns are averaged over members x samples (`returns.mean((0, 1))`), the critic regresses
onto all E' returns, and `ensemble_penalty` (population std of the per-control-step team reward)
is computed over members x samples (so one member with N > 1 has a nonzero penalty from the
sampling alone; the penalty therefore also prices aleatoric spread - lower `uncertainty_weight` if
that is not wanted). Counterfactual rows share the samples of their start. `imagined` gains
`n_members`, `n_futures`; `update()` metrics gain `future_samples` only when sampling is on.
Validation (`validate_detail`) uses fixed futures, so selection compares like with like.
Activation checkpointing (`checkpoint_imagination`) works unchanged: the sampled tensors are
constants captured by the per-step `partial`; gradients with and without recomputation agree
(`test_real_wm_trainer_with_samples_checkpointing_equivalence`).

Caveats
* Sampled drivers are fixed per rollout (as the WM does with its own encoding), encoded at t0 on
  the padded start state; an entrant inserted at physics step 0 therefore gets the encoding of its
  empty slot instead of an in-rollout encoding (default-like parameters) - negligible.
* `accel_log_sigma` is a head of the residual net, but nothing in `src/wms` trains it (no loss uses
  it), so its value in the C4 WM is whatever the random init / drift gave. Checked on the C4 WM
  checkpoint (`wm_s_best.pt`): the log-sigma row of the final `residual_net.accel` layer has norm 0.57
  (accel-delta row 0.63; a fresh `Linear(512, 2)` has ~0.58) and bias -0.02, i.e. it is the untrained
  initialisation, so `exp(log_sigma)` is an arbitrary O(1) m/s^2. Treat `future_accel_noise` as OFF until a calibrated sigma exists;
  `future_accel_sigma_max` bounds the damage and `future_accel_scale` rescales.
* Cost is linear in N (sequential pseudo-members, no batching across samples yet).

## RSU check

`RsuChecker._sample_drivers` now calls `futures.sample_driver_encodings` (B = 1, tiled
candidate-major / sample-minor). Same generator stream, same expressions: identical values
(`test_rsu_driver_samples_unchanged_by_factoring` compares with the pre-refactor inline code).

## Tests

`tests/unit/test_c4_futures.py` (12): N=1 identical (imagination outputs, WM sees no extra kwargs,
whole-`update()` metrics); config validation; draws shared across members and bounded by the 50 %
cap; plan CRN/reproducibility/global-RNG untouched; CRN across counterfactual variants and members;
returns shape `[E*N, H, rows]` and penalty over members x samples; update with samples; real WM-S
accel noise (off = unchanged, sigma cap 0 = unchanged, CAV untouched); real-WM trainer with
checkpointing on/off gradients equal; RSU golden. Run with `OMP_NUM_THREADS=1` on a loaded box.

## Cost and checks

CPU, 1 thread, loaded shared box (22 cores busy with other jobs, so absolute times are inflated and
noisy); real WM-S architecture (random weights, hybrid), synthetic 48-slot start, batch 4, horizon 10 s,
1 member, counterfactual rows on (28 rows), activation checkpointing on (`tools/futures_cost.py`):

| future_samples | update time | peak working set |
|---|---|---|
| 1 | 80 s | 488 MB |
| 2 | 197 s | 570 MB |

Cost is linear in N by construction (N sequential rollouts of the same graph): compute ~ N x one
update's imagination; with checkpointing the peak grows ~ +80 MB per extra sample here (the saved
per-sample tensors, e.g. states / rewards / lane records, not the per-step WM activations); without
checkpointing the per-sample cost is the full rollout activation memory (CFMEM_NOTES.md: ~17x larger).
The time ratio 2.5x at N=2 is load noise on top of the expected ~2x; no 3-sample/no-checkpoint rows
were completed (box too loaded). The batch dimension could host the samples to cut Python overhead
(not done: it would change the row layout that counterfactual bookkeeping relies on).

CPU smoke on the real C4 WM (`python -m wms.control.train_cli --config configs/c4_policy_futures.toml
--wm-ckpt .../c4_wm_main/wm_s_best.pt --smoke`, `future_samples = 2`, 1 member, entries on, lane head and
counterfactual rows on): completed, 2 updates + 2 validations, finite metrics, `future_samples: 2`
logged, `ensemble_disagreement` 0.022 / 0.039 (nonzero with ONE member: the spread of the two sampled
worlds), update wall time 788 s / 2158 s on the loaded box (batch 2, horizon 4 s).

Checks (sharded, single thread): dreamer/policy2/srcwait_imagination/cfmem 57 passed,
rsu/rsu_field/import_graph/core/belief/branching 266 passed, the 12 new futures tests passed;
deploy/loop/mappo/srcwait_mappo/sumo (63 tests, slow on the loaded box): 47 passed with no failure at
commit time, the remainder was still running (they do not touch the changed code paths except through
the unchanged default `rollout`). Bit-identity of default
(`future_samples = 1`) vs the `c4/int` source tree: identical `update()` metrics and post-update
policy weights for the real WM-S + lane head + counterfactual rows, seeds 0 and 1.
