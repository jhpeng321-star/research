# BOXFIX — two GPU-box failures on real pack_v3 (branch `zcode/boxfix` @ 144545c)

Scope: `output/wd01/wm_n1` (`train`, `bundle`, `corpus`, `evalh`, `preflight`).
No real packs were opened on the CPU box; every test uses synthetic tmp packs.
Zero SUMO, no `D:/rd-staging`, no HOLDOUT/CAL/chunk-5/J4B, K_max / pack format /
losses / registrations unchanged.

## FAILURE 2 (fixed first) — IdeaAConfigError: `['response_contrast']`

### Cause
`IdeaATrainer._iter_batches` drew `torch.randperm` batches (`batch_size=8`). The
paired CONTRAST loss (`train.dataset.contrast_batch_indices`) needs a treatment
arm **and** its all-native sibling in the *same* batch, but a random batch almost
never contains both arms of the same `world@t0` (`pair_key`) once the real pack
has hundreds of groups. `IdeaATrainer.fit` runs fail-closed
(`_require_all_terms`), so the first pair-less batch raised
`registered A3.2 loss term(s) unavailable: ['response_contrast']`.

A second defect: `train.dataset.is_reference` treated any arm literally named
`native`/`native_base` as a reference. In the real pack the `chunk1`/`chunk3_fix27`
`__native` arms carry **policy `switching`**, so they must never be references
(`chunk2`/`chunk3` has both truearms `native` policy `native` and replicated
`native` policy `switching`; `chunk4 native`/`native_base` are policy `native`).

### Fix
- `train/dataset.py`
  - `is_reference` is now **policy-driven**: reference iff `policy == 'native'`
    (arm suffix consulted only when a pack records no policy). Switching
    `__native` arms are excluded.
  - Added `pair_groups`, `contrast_pairs`, `batch_has_pair`, `ensure_pair`: a
    deterministic, seeded **pair-aware batch sampler**. Every batch is
    completed to hold ≥1 sibling pair; a batch already holding a pair is
    unchanged. The pair partner may repeat across batches; groups with no
    policy-native reference contribute no pair but their items still train the
    other losses.
- `train/idea_a.py`
  - `_iter_batches` keeps the seeded `randperm` **epoch accounting** (each item
    is still seen once per epoch) but runs every batch through `ensure_pair`
    (`train.dataset`). The pair-aware path is used exactly where cheap item
    metadata is available (`_source_item_meta`): `MemmapCorpus`, `IdeaACorpus`,
    `HeldOutTrainView`/quick-tune views. Stand-in corpora that carry no
    `pair_key` fall back to plain random batching.
  - `fit` stays fail-closed: the missing-term error is unchanged; the sampler
    guarantees the term is present on the real path.

## FAILURE 1 — `RuntimeError: the Idea A corpus timing measurement is CPU-only`

### Cause
`bundle.timing.measure_idea_a_corpus` refused any non-CPU device
(`bundle/timing.py:343`) and always built a **synthetic** pack. `bundle.queue`
called `run_timing_gate(cfg, device=..., tiny=...)` without ever passing the
converted+verified pack, so `python -m bundle ... --stop-after timing-gate
--device cuda` raised on the real GPU box.

### Fix
- `bundle/timing.py`
  - `run_timing_gate(..., pack_dir=None)` / `measure_idea_a_corpus(..., pack_dir)`
    / `loss_decrease_idea_a(..., pack_dir)`. With `pack_dir` the gate runs the
    REAL Idea A path on the requested device (CUDA allowed) through
    `IdeaACorpus` + the pair-aware sampler + all five A3.2 losses, with
    `_require_all_terms=True` (fail-closed). Records **median s/step after
    warmup**, **peak CUDA memory** (`torch.cuda.max_memory_allocated`; RSS on
    CPU), **active_loss_terms**, resolved **k_max**, and the **projected hours**
    from the registered gate config (`extrapolate`, measured at the registered
    `batch_size`). The synthetic pack remains only for `--tiny`/CPU tests.
  - A CUDA request on a host without CUDA now raises the standard
    `cuda requested but not available` (not the old CPU-only refusal).
- `bundle/queue.py`
  - `_stage_timing_gate` hands `<work_dir>/pack` (the `convert+verify` output)
    and `device` to `run_timing_gate`; `--tiny` keeps the synthetic path. The
    stage receipt records `pack_source`, `k_max` and `active_loss_terms`.

## Exact box commands

```powershell
# FAILURE 2 — real Idea A smoke (member 1) on the extracted pack_v3
python -m train.idea_a --member 1 --corpus $W/upload --out $W/smoke `
  --device cuda --k-max 192 --steps-per-epoch 50 --epochs 1

# FAILURE 1 — registered timing gate on the REAL converted+verified pack
python -m bundle --manifest $W/upload/manifest.json --work-dir $W/run `
  --approval $W/APPROVAL.json --stop-after timing-gate --device cuda
```

`$W` is the box workspace; for the queue the pack is
`$W/run/pack` (produced by the `convert+verify` stage).

## Tests / evidence

- Pair-aware sampler: `train/tests/test_dataset.py`
  (`test_contrast_pairs_exclude_switching_native_reference`,
  `test_ensure_pair_makes_every_batch_complete_and_deterministic`,
  `test_is_reference_is_policy_driven_and_excludes_switching_native`).
- Many-group main-path fit (40 groups x 8 arms = 320 items, random batches
  would be pair-less): `train/tests/test_idea_a_realbatch.py::
  test_main_path_fit_activates_every_loss_on_a_many_group_pack`; determinism +
  epoch coverage in `::test_iter_batches_pair_complete_coverage_and_deterministic`.
- Real-pack timing gate (CPU) + CUDA-request acceptance:
  `bundle/tests/test_timing.py::test_real_pack_path_reports_steps_terms_and_kmax`,
  `::test_real_pack_path_accepts_a_cuda_request`.
