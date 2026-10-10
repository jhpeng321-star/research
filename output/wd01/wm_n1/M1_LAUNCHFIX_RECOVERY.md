# M1_LAUNCHFIX_RECOVERY — production member-1 launch path (2026-10-02)

**Status:** repaired. The production entry
`python -m bundle.multibox run-box-config --config bundle/box_configs/2box/c_m1only/boxA.json --work-dir WORK`
now runs the **full registered CUDA member-1** and stops at its checkpoint, or
fails closed. No synthetic pack, no 2-step CPU "success", no CAL/eval.

Source base: `0aa3cbb` (branch `zcode/m1launchfix`). Work root:
`output/wd01/wm_n1` (the package root for `python -m bundle...`).

## What was broken

`bundle.e2e.run_config_job` (the multibox default executor, `multibox.py:166`)
silently:

- built a **synthetic** pack when `$WORK/pack` was absent;
- forced `epochs=1, steps_per_epoch=2, batch_size=4`;
- passed **no device** to `IdeaATrainer`, so it trained on **CPU**;
- wrote `members/member_1.pt` (not the handoff `members/member_1/`);
- returned success for a 2-step smoke — a real member-1 could never run.

`bundle.queue._stage_convert_verify` always called `corpus.pack.convert(upload,
pack)`. On a GPU box whose `upload`/`pack` are symlinks to the extracted
`pack_v3`, that re-packs and **overwrites the frozen target**.

## What changed

### `bundle/e2e.py` — production executor
`run_config_job(box_id, config, work_dir, *, device="cuda", cuda_available=None)`:

1. only **Idea A ensemble members** are launched; held-out, variant-N and any
   CAL/eval path raise `ProductionConfigRefusal` (no auto-launch).
2. resolves the real run root from the per-box sandbox
   (`$WORK/boxes/<box>` -> `$WORK`) so the pack and checkpoint are the handoff
   paths `$WORK/pack` and `$WORK/members/member_<k>/member_<k>.pt`.
3. builds the **full registered `IdeaATrainConfig` defaults** — 15 epochs x
   2000 steps, batch 8, `ensemble_k=5`, `state_global_token=True` (A6),
   `num_workers=None` (host-scaled), registered member seed (101 for member 1).
4. **A6 gate first**: `evalh.seals.require_frozen_amendment_6()` before any data.
5. **CUDA fail-closed**: refuses a `cuda` request when CUDA is unavailable.
6. **real pack only**: requires `$WORK/pack/{INDEX.json,manifest.json}`; reads
   the registered K_max from the pack metadata (pack_v3 -> **192**) and refuses
   an unregistered K. No synthetic fallback.
7. runs the trainer, then validates the **actual receipt** (`steps ==
   epochs*steps_per_epoch`, member, seed), the **actual device**, and the
   **checkpoint existence** before returning the recorded status.

`run_e2e_smoke` (unit-only CPU smoke / pre-flight G5) is unchanged and remains
its own explicit entry point.

### `bundle/queue.py` — prepacked v3 timing-gate sequence
`_stage_convert_verify` detects an already-extracted corpus (`INDEX.json`
schema `wm_n1_corpus_pack/2` + `manifest.json`). When the upload is prepacked it
verifies the **real `$WORK/pack` in place** with `corpus.verify.verify_pack`
(opaque-byte `SHA256SUMS` + INDEX/manifest K_max/format metadata; outcomes only
checked for presence, never interpreted) and **never** calls
`corpus.pack.convert`, so a frozen symlink target is untouched. A prepacked
upload without an extracted `$WORK/pack` is refused. The legacy raw-shard
convert path is unchanged, and `_stage_timing_gate` still measures the explicit
real `$WORK/pack` with host-scaled workers.

## Effective schedule / default identity (no real execution)

```
epochs=15 steps_per_epoch=2000 batch=8 effective_steps=30000 seed=101 a6=True num_workers=8 registered_k=[128, 192]
```

`k_max` is taken from the real pack (pack_v3 -> 192); `num_workers=8` here is
`min(8, cpu-2)` (host-scaled).

## Tests (CPU, `WD_FORBID_D=1`, mocked corpus/trainer/device)

- `bundle/tests/test_production_member_launch.py` — the CLI ->
  `run-box-config` -> `run_config` -> `run_config_job` seam: full 30000-step
  schedule, seed 101, CUDA, real `$WORK/pack`, `members/member_1` checkpoint,
  refusal without pack / without CUDA / on a 2-step schedule / on an
  unregistered K, A6 refusal before data, and no members 2-5 / held-out / N /
  CAL auto-launch.
- `bundle/tests/test_prepacked_real_gate.py` — a tiny real v3 pack verifies in
  place with no `convert`, frozen bytes unchanged (and symlink target untouched
  where symlinks exist), prepacked-without-pack refused, legacy convert intact,
  timing gate measures explicit `$WORK/pack`.

```
python -m pytest bundle/tests -q
176 passed, 1 skipped in 429.74s
```

The skip is the symlink assertion on a host without symlink privilege.

## Review repair (2026-10-02, source-only): synthetic launch-test fixtures

Independent review found `bundle/tests/test_production_member_launch.py`
importing `test_multibox._crlf_records`, whose implementation reads the BYTES
of every `MB.DEFAULT_RECORDS` entry — including the real frozen
`CAL_SPLIT_A4.json` — to build CRLF copies. Seven nodes used it and are now
repaired (this commit, source-only; tests NOT run):

- Repaired nodes (old unsafe): `test_cli_seam_runs_full_registered_member_1`,
  `test_member_1_ends_after_checkpoint_no_auto_launch`,
  `test_refuses_without_real_pack`, `test_refuses_when_cuda_absent`,
  `test_refuses_short_schedule`, `test_refuses_unregistered_pack_k`,
  `test_a6_refused_before_pack_or_cuda`. Six of them also opened the real A6
  record via `evalh.seals.require_frozen_amendment_6()` inside the (unmocked)
  `run_config_job`; the `test_multibox._approved` cross-import is gone too.
- New behavior: the file is self-contained. Eight record fixtures are written
  under pytest `tmp_path` from in-file synthetic literals; their LF-sha256 is
  computed from the literals and pinned via monkeypatch into
  `bundle.multibox.LF_SHA_*` (approval fields + validator table) and into
  `evalh.seals.AMENDMENT_6_PATH`/`LF_SHA_AMENDMENT_6` (the production A6
  gate). No real registration/amendment/CAL file is opened, read or copied,
  including for CRLF-style fixture generation (the fixtures ARE synthetic).
- Strength preserved: the production `require_config_approval` validator and
  `run_config_job` A6/CUDA/pack/K_max/schedule/checkpoint gates run unmocked
  against the fixtures; the full-schedule node adds a synthetic-only tamper
  check (one flipped fixture byte -> `ConfigApprovalMissing`). All nine nodes
  keep their original production assertions; no permissive stub bypasses the
  behavior under test. Production source untouched.

## Honest notes

- `run_config_job` refuses held-out / variant-N configs: their production
  training/eval is not authorised here (M1_READY: gpuB held-out is "not yet
  authorised"). This is a deliberate fail-closed, not a regression of the
  planner; only the requested Idea A member is launched.
- No real GPU, no real pack, no training, no CAL/HOLDOUT/J4B/chunk-5 read and
  zero drive-D access was used. `k_max`/format are read from pack metadata only;
  the frozen `CAL_SPLIT_A4.json` is never opened for its contents.
