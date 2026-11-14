# V6 P2: paired-return training

The candidate `configs/c4_wm_v6_pair.toml` extends `c4_wm_v5.toml`. It keeps
the native / undirected-random data recipe, paired-batch fraction (0.25), and
existing paired weight (1). Its runtime is CPU / two threads.

## Switches under `loss.paired_diff`

- `couple_randomness = false`: opt in to common reveal selections/durations,
  t0 and rollout state noise, and sampled-lane uniforms. Keys include the seed,
  pair key (group ID fallback), stable vehicle identity, absolute unroll step,
  and draw stream. C4 episode prefixes are removed from vehicle IDs. Unknown
  identities for present vehicles are rejected. Entry renewal and candidate
  draws use separate group/entrance/identity streams, so blocking in one arm
  cannot shift another arm's future arrival draws. Default false uses the
  original RNG calls and arithmetic.
- `return_weight = 0`: enable `pair_return`, unit-delta Huber on the **cumulative**
  predicted-minus-recorded return difference. Return is negative on-road HDV
  delay: `-dt * sum(relu(1 - speed / 31.29))` over the t0 cohort and commonly
  observed horizon. CAVs and entrants are excluded. Both arms use a common
  scale: t0 HDV count times observed seconds, not either arm's later occupancy.
- `rank_weight = 0`: enable `pair_rank`. For eligible comparisons, penalize
  `relu(margin - sign(recorded_difference) * predicted_difference) / scale`.
  Both return terms average over valid nonempty arm/reference comparisons and
  are multiplied by the existing overall paired weight.
- `rank_margin_veh_s = 5`: a comparison is rank-eligible only when its absolute
  recorded cumulative effect is **strictly greater** than this vehicle-second
  margin. Ties and sub-margin effects have exactly zero ranking loss/gradient.
  The same margin is the required predicted separation.

New keys are omitted from legacy-flat configuration at their defaults. The
candidate enables coupling and uses return/rank weights `1.0` / `0.1`; these
are starting values, not experimental results.

## Truncated BPTT and diagnostics

The return/rank objectives use a no-grad full-horizon probe followed by an
RNG-replayed chunked pass. Each chunk receives the derivative evaluated at the
same whole-horizon return. This preserves bounded activation memory and the
existing truncation boundaries; it does not replace the objective with a mean
of per-chunk losses. Forward evaluation also sums returns before applying the
nonlinear losses. Each objective is logged once per batch.

Paired training records in `train_log.jsonl` include:

- `paired_unique_worlds`, from collated source-world metadata (null when absent);
- `paired_groups` and existing `paired_arms`;
- `paired_effect_bins`: negative, within-margin, and positive recorded effects,
  with the threshold in `paired_effect_margin_veh_s`;
- `paired_grad_norm`: L2 norm of the summed weighted paired gradient on model
  parameters, including propagation through the t0 driver encoder, before
  clipping (null on forward-only evaluation);
- enabled `pair_return` / `pair_rank` and the existing paired loss/value-share fields.

`tests/unit/test_v6pair.py` covers exact duplicate-arm cancellation, real chunked
rollout replay, both-arm gradients, tie handling, permutation invariance,
occupancy-independent normalization, cumulative error cancellation, identity-keyed
draws, blocking-independent entry streams, default-path parity, and persisted
training-log fields. All verification uses CPU with at most two threads.
