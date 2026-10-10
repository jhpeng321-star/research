# REGISTRATION_WM_N1, AMENDMENT 8 (DRAFT, NOT FROZEN): pair-dense batching for the CONTRAST loss

**Status:** DRAFT, 2026-10-04, world model session (T2). It freezes only on the
owner's OK. Until then the code path is behind the flag `pairs_per_batch`,
default **None/OFF**, and the registered A3.2 batching behaviour is unchanged.

- Amends `REGISTRATION_WM_N1.md` and A1–A6. A5 is `7788c270…`, A6 is the
  state-token amendment.
- **UNCHANGED:**
  - the §4 targets, decision rule and CRPS/coverage metrics;
  - the CONTRAST loss definition (treatment − policy-native reference, same
    world + t0), its λ default (1.0) and the fail-closed support masks;
  - the chunk-5 TEST (sealed), HOLDOUT (sealed), the CAL split `bc4e32f2…`;
  - the eligible-pair support filter, the verified-identity gate and the
    owner-decision (b) outcome-supervision exclusion (base / native_base);
  - the variants, tie order, A4 tuning knobs and K_max = 192.

## A8.1 Finding that motivates it

The first CAL-tune score of checkpoint `94984cc0…` (M1FIX1, member 1;
`runs/active/WM-N1-SCORE-CALTUNE-20261004/REPORT.md`) reports that model A is
**worse than both naive baselines at every horizon** (PI-native paired ΔV of
total delay; CRPS lower = better; 11 worlds, PI-native only):

| tau s | CRPS model A | CRPS zero-effect | CRPS TRAIN-climatology |
| --- | --- | --- | --- |
| 30 | 40.7 | 5.9 | 5.3 |
| 60 | 93.3 | 32.1 | 25.0 |
| 120 | 268 | 153 | 129 |
| 300 | 909 | 768 | 528 |
| 570 | 2393 | 2360 | 1564 |

The report's own interpretation lists **thin contrast supervision (~1 pair per
batch; 11/40 zero-contrast steps)** as a likely contributor. A read-only CPU
measurement on the real `pack_v3` TRAIN pack (below) confirms it: with the
registered default, 20/20 steps complete only ~1 pair (mean 1.05), 12/20 have
**zero shared outcome tau** (`out=0/0`), and 7/20 steps produce a
`response_contrast` term below 1e-4. The supervised contrast signal is
therefore sparse and frequently zero, even though the training-integrity gates
(t0 labels, exclusion (b), verified identities) are all satisfied.

## A8.2 Change

A configurable **pair-dense sampler** makes each batch drawn from the eligible
sibling-pair pool hold exactly K complete pairs:

- New flag `--pairs-per-batch K` on `python -m train.idea_a` and config key
  `pairs_per_batch: Optional[int]` (default `None`).
- `train.dataset.pair_dense_batches` visits the eligible pool
  (`train.contrast_support.build_eligible_pair_pool`, the same pool the
  registered sampler already uses) in a seeded permutation and emits batches
  of exactly K pairs. A group that shares one policy-native reference
  contributes `1 + n_treatments` distinct positions (the reference recurs;
  no index is duplicated); each of the K pairs counts once through the frozen
  `contrast_batch_indices`.
- Items in no eligible pair (a group with no policy-native reference, or a
  treatment with no shared support) are still emitted, in seeded order, as
  trailing coverage batches holding at most one item per `pair_key` so no
  unsupported pair can form. Every primary item is still seen once per epoch.
- The epoch is a pure function of `(seed, epoch, pairs)`; the in-process path
  and `PairAwareBatchSampler` (DataLoader) produce byte-identical batches.
- **Fail-closed checks are preserved:** the t0-label assertion
  (`label_t0_rows` / `label_t0_violations`), the verified-identity gate and
  the owner-decision (b) exclusion receipt are untouched and still recorded in
  `train_log.jsonl` / `TRAIN_RECEIPT.json`.
- `--lambda-contrast W` (config `lambda_response_contrast`, default 1.0) is
  exposed so the contrast weight can be re-balanced in the same A4 loop; the
  default is unchanged.
- `pairs_per_batch=K` requires `2K <= batch_size` and fails closed otherwise.

**Default unchanged:** `pairs_per_batch=None` resolves to the historical
`epoch_batches` one-completed-pair-per-batch path byte-for-byte (tested).

## A8.3 What does not change

- The losses, their default λ's, the target normalization, the state global
  token, the HDV/rollout/arrival/correction heads, and the ensemble design.
- The contrast definition, the eligible-support rule and the exclusion rules.
- The splits, seeds and the sealed TEST/HOLDOUT/CAL access rules.

## A8.4 Measured evidence (read-only CPU, no GPU, BEFORE freeze)

20-step CPU TRAIN measurement, member 1, seed 101, real `pack_v3` TRAIN
(`WM_N1_TRAIN_OUTCOMES=labels_fix1_20261003T141730Z`,
`WM_N1_TRAIN_IDENTITIES=sidecars3_20261003T0005CN/identities`), K_max 192,
`num_workers=0`, `--log-every 1`; raw logs under
`E:/research-dense/tmp/overnight_20261004/T2_out/`. Terms are normalized.

| condition | pairs/step (mean; min–max) | steps with outcome support | outcome pairs/step | response_contrast mean | near-zero contrast steps |
| --- | --- | --- | --- | --- | --- |
| default (`None`) | 1.05; 1–2 | 8/20 | 0.40 | 0.0273 | 7/20 |
| `--pairs-per-batch 4` | 4.00; 4–4 | 20/20 | 2.40 | 0.1824 | 0/20 |

`label_t0_asserted` mean 28.0 (default) / 32.0 (K=4), `label_t0_violations` 0
in both; the exclusion and identity receipts were populated. The K=4 run
reached `pairs=4` on all 20 steps and never had a zero contrast sub-term,
directly removing the sparsity the CAL-tune report flagged. This is a
mechanics measurement, not efficacy evidence: no CAL/TEST is read and no
acceptance comparison is claimed here.

## A8.5 Evaluation and disclosure

- The flag is OFF by default and OFF reproduces the registered batching
  byte-for-byte (unit-tested, `train/tests/test_pair_dense_batching.py`).
- If frozen, a future member-1 run may use `--pairs-per-batch K` and/or
  `--lambda-contrast W` only as declared A4-loop knobs, with the chosen values
  and their justification recorded in the WM-N1 ledger.
- Any A8 efficacy claim requires a separate registered comparison against the
  frozen M1FIX1 checkpoint on CAL-tune, with the same metrics and baselines;
  this draft makes no such claim.
- Freeze or rejection is recorded in the WM-N1 ledger entry.
