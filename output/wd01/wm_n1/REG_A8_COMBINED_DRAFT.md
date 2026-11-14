# REGISTRATION_WM_N1, AMENDMENT 8 (DRAFT, NOT FROZEN): combined pair-dense contrast supervision

**Status:** DRAFT, 2026-10-04, world model session (overnight integration of T2
+ T6). It freezes only on the owner's OK. Until then every code path below is
behind a default-OFF flag, and the registered A3.2 batching and loss behaviour
is unchanged byte-for-byte.

- Amends `REGISTRATION_WM_N1.md` and A1–A6. A5 is `7788c270…`; A6 is the
  state-token amendment. This draft supersedes the standalone T2 draft
  `REG_A8_PAIR_BATCHING_DRAFT.md` (kept for provenance).
- **UNCHANGED:**
  - the §4 targets, decision rule and CRPS/coverage metrics;
  - the CONTRAST loss definition (treatment − policy-native reference, same
    world + t0), its λ default (1.0) and the fail-closed support masks;
  - the chunk-5 TEST (sealed), HOLDOUT (sealed), the CAL split `bc4e32f2…`;
  - the eligible-pair support filter, the verified-identity gate, the
    owner-decision (b) outcome-supervision exclusion (base / native_base) and
    the t0-label assertion (`label_t0_rows` / `label_t0_violations`);
  - the variants, tie order, A4 tuning knobs and K_max = 192.

## A8.1 Finding that motivates it

The first CAL-tune score of checkpoint `94984cc0…` (M1FIX1, member 1;
`runs/active/WM-N1-SCORE-CALTUNE-20261004/REPORT.md`, **including the attempt-6
addendum**) reports that model A is **worse than both naive baselines at every
horizon** (PI-native paired ΔV of total delay; CRPS lower = better; 11 worlds,
PI-native only). The addendum (corrected independent inverse-CDF paired-ΔV
construction) removes the interval artifact but model A still loses on CRPS and
sign accuracy at every tau; coverage is now nominal.

| tau s | CRPS model A (addendum) | CRPS zero-effect | CRPS TRAIN-climatology | sign A | sign climatology |
| --- | --- | --- | --- | --- | --- |
| 30 | 52.4 | 5.9 | 5.3 | 0.36 | 0.55 |
| 60 | 107.9 | 32.1 | 25.0 | 0.45 | 0.91 |
| 120 | 280.9 | 152.6 | 129.1 | 0.36 | 0.64 |
| 300 | 879.8 | 767.9 | 528.2 | 0.36 | 0.73 |
| 570 | 2114.6 | 2360.4 | 1563.6 | 0.55 | 0.82 |

Two independent overnight diagnostics localise the supervision problem:

- **T1** (`E:/research-dense/tmp/overnight_20261004/T1_out/RESULT.md`): the
  paired-ΔV construction was invalid (comonotone same-level quantile
  difference; intervals inverted/too narrow) — fixed behind a default-unchanged
  `--dv-method independent`. The **in-sample TRAIN** check then shows model A is
  worse than the naive baselines on CRPS at every tau and at chance sign
  accuracy (0.42–0.51 vs in-sample climatology 0.55–0.75): the model does not
  even fit the training contrasts. This is a supervision/optimisation problem,
  not merely TRAIN→CAL generalisation.
- **T6** (`E:/research-dense/tmp/overnight_20261004/T6_out/RESULT.md`, full
  30,000-step M1FIX1 member-1 log): the outcome-contrast sub-term is present on
  only **11,488/30,000 = 38.3 %** of steps (61.7 % HDV-only, `out=0/0`); per
  step the sampler completes ~1 pair (mean 1.05). The cause is
  **label availability**: of 1,478 candidate sibling pairs only 672 (45.5 %)
  share a realised outcome tau, and the sole cause is an unlabelled treatment
  arm (806 pairs: `base` exclusion (b) 161; `fixed`/`switching` chunk2/3 588;
  chunk4 wall arms 155). When both arms are labelled they share **all five**
  taus. `response_contrast` is also the **average** of the outcome and HDV
  sub-terms, so the outcome sub-term is **halved** on the 100 % of steps where
  the HDV sub-term is present.

T2's read-only CPU measurement (`T2_out/RESULT.md`) confirms the sampling lever:
with `--pairs-per-batch 4` every one of 20 steps completes 4 eligible pairs and
none has a zero contrast sub-term, versus ~1 pair and 7/20 near-zero steps at
default.

## A8.2 Change (all default-OFF, all fail-closed checks preserved)

Three composable, independent knobs:

1. **Pair-dense batching (T2).** `--pairs-per-batch K` / config
   `pairs_per_batch: Optional[int] = None`. Every batch drawn from the eligible
   sibling-pair pool holds exactly K complete pairs; the final group may be
   short when the pool size is not a multiple of K. A shared policy-native
   reference contributes `1 + n_treatments` distinct positions (the reference
   recurs; no index is duplicated). Items in no eligible pair are still emitted
   in seeded order as trailing **coverage** batches, and a coverage batch holds
   **at most one item per `pair_key`** so it can never complete an unsupported
   sibling pair; every primary item is still seen once per epoch. `K` requires
   `2K <= batch_size` and fails closed otherwise.
2. **Outcome-required contrast pool (T6).** `--require-outcome-contrast` /
   config `contrast_require_outcome_support: bool = False`. The eligible pool
   keeps ONLY pairs sharing at least one realised outcome tau, so the pair-aware
   sampler forces an outcome-carrying pair into every batch instead of spending
   its forced pair on an HDV-only sibling. No label is fabricated: an unlabelled
   treatment is simply not forced, and a source with no outcome-supported pair
   still fails closed (empty-pool refusal, message names the required support).
3. **Outcome-contrast own weight (combined follow-up to T6 §4 proposal 3).**
   `--contrast-outcome-weight W` / config `contrast_outcome_weight:
   Optional[float] = None`. With an explicit finite `W >= 0` the present
   sub-terms combine as `W * outcome + 1.0 * hdv` instead of the historical
   `(outcome + hdv) / 2`, so the outcome sub-term is no longer halved by the
   HDV average. A lone sub-term is returned unchanged (a weight can never scale
   away the only supervised term). Default `None` reproduces the equal average
   byte-for-byte.

The pre-existing `--lambda-contrast W` / `lambda_response_contrast` (default
1.0) is exposed so the whole contrast weight can be re-balanced in the same A4
loop.

**Default unchanged:** `pairs_per_batch=None`, `contrast_require_outcome_support
=False`, `contrast_outcome_weight=None` resolve to the historical
`epoch_batches` one-completed-pair path and the historical loss exactly
(unit-tested).

**Fail-closed preserved:** the t0-label assertion, the verified-identity gate,
the owner-decision (b) exclusion receipt, the eligible-support empty-pool
refusal and the no-valid-sub-term refusal are untouched and still recorded in
`train_log.jsonl` / `TRAIN_RECEIPT.json`. The receipt additionally records
`pairs_per_batch`, `contrast_require_outcome_support` and
`contrast_outcome_weight`.

## A8.3 What does not change

- The losses, their default λ's, the target normalization, the state global
  token, the HDV/rollout/arrival/correction heads, and the ensemble design.
- The contrast definition, the eligible-support rule and the exclusion rules.
- The splits, seeds and the sealed TEST/HOLDOUT/CAL access rules.

## A8.4 Measured evidence (read-only CPU, no GPU, BEFORE freeze)

Read-only CPU TRAIN measurement, member 1, seed 101, real `pack_v3` TRAIN
(`WM_N1_TRAIN_OUTCOMES=labels_fix1_20261003T141730Z`,
`WM_N1_TRAIN_IDENTITIES=sidecars3_20261003T0005CN/identities`), K_max 192,
`num_workers=0`, `--log-every 1`; raw logs under
`E:/research-dense/tmp/overnight_20261004/T2_out/`. Terms are normalized.

| condition | pairs/step (mean; min–max) | steps with outcome support | outcome pairs/step | response_contrast mean | near-zero contrast steps |
| --- | --- | --- | --- | --- | --- |
| default (`None`) | 1.05; 1–2 | 8/20 | 0.40 | 0.0273 | 7/20 |
| `--pairs-per-batch 4` | 4.00; 4–4 | 20/20 | 2.40 | 0.1824 | 0/20 |

`label_t0_asserted` mean 28.0 (default) / 32.0 (K=4), `label_t0_violations` 0
in both; the exclusion and identity receipts were populated. T6's offline
coverage census (real item order, 15 epochs) reports the outcome-required pool
raising batches with outcome support from 1,382/4,050 (0.3412, proxy for the
default pool) to **4,050/4,050 (1.0000)**. These are mechanics measurements,
not efficacy evidence: no CAL/TEST is read and no acceptance comparison is
claimed.

## A8.5 Label coverage (PLACEHOLDER — pending T3 plan)

> **TODO (blocked on T3):** the 806 unlabelled treatment pairs cannot be fixed
> by sampling or weighting. The real lever is label coverage: `chunk1` /
> `chunk1_fix27` (451 items) and `chunk4` wall arms carry no outcome rows in
> `pack_v3`, and chunk2/3 `fixed`/`switching`-only arms have none. T3 is
> drafting the base/native_base t0-snapshot fix and the repack/relabel cost
> estimate; this section will be completed from `T3_out/` when available. No
> repack, relabel or featurization is run or authorized by this draft.

## A8.6 Evaluation and disclosure

- Every flag is OFF by default and OFF reproduces the registered batching and
  loss byte-for-byte (`train/tests/test_pair_dense_batching.py`,
  `train/tests/test_contrast_outcome_required.py`,
  `train/tests/test_pair_dense_outcome_combined.py`,
  `train/tests/test_idea_a_losses.py`).
- If frozen, a future member-1 run may use `--pairs-per-batch K`,
  `--require-outcome-contrast`, `--contrast-outcome-weight W` and/or
  `--lambda-contrast W` only as declared A4-loop knobs, with the chosen values
  and their justification recorded in the WM-N1 ledger.
- Any A8 efficacy claim requires a separate registered comparison against the
  frozen M1FIX1 checkpoint on CAL-tune, with the same metrics and baselines;
  this draft makes no such claim.
- Freeze or rejection is recorded in the WM-N1 ledger entry.
