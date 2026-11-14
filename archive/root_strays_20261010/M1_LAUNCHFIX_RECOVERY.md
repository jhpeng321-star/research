# M1_LAUNCHFIX_RECOVERY — draft commits (code-only, tests PENDING guard)

Date: 2026-10-02. Executor: GLM-5.3-Flash (model id `zhipuai-coding-plan/glm-5.3-flash`),
direct owner authorization "do anything you need to do to continue the work and
report back to me" under the CODE-ONLY resume brief. Branch: `zcode/m1launchfix`
(worktree `E:\rd-wt\m1launchfix`). CORRECTION of the earlier claim that the
prior broad bundle suite (176 passed / 1 skip) "stands": those are HISTORIC
counts for older source and are NOT current acceptance for the changed
production code. NO pytest/CLI execution happened in either draft session —
ALL tests below are DRAFT and PENDING the coordinator's reviewed guard and
explicit test nodes; nothing is accepted until they run green on this source.

## Scope honored

- Owned files only: `bundle/e2e.py`, `bundle/queue.py`, `bundle/timing.py`
  (expanded ownership), new `bundle/tests/test_timing_recovery.py`,
  existing/new `bundle/tests/test_production_member_launch.py`,
  `bundle/tests/test_prepacked_real_gate.py`, `bundle/tests/test_timing.py`
  (minimal adaptation for changed production semantics, documented below), and
  this report. `train/**`, `evalh/**`, `M1_READY.md`, approvals and box configs
  untouched. No files deleted; all dirty user edits preserved and finished.
- contrastfix2 `output/wd01/wm_n1/train/idea_a.py` read READ-ONLY via git
  (prior session c59b062; this session HEAD `25479ae` to verify the canonical
  `prepare_training_source` interface). No other worktree read/imported/executed.
- No real pack/outcome/HOLDOUT/CAL/chunk-5/J4B/INC1 reads; no
  `CAL_SPLIT_A4.json` contents; synthetic fixtures only; no approvals created;
  no GPU/training/network. No pytest and no project import/execution ran in
  either session; the only python invocations this session were two STATIC
  checks on the edited bundle files (`py_compile` + `ruff check`), which
  import and execute nothing.

## Changes (this commit)

1. `bundle/timing.py` — throughput setup now makes ONE STRICT CANONICAL
   preparation call BEFORE any batch iterator or optimizer is built:
   `trainer.prepare_training_source(source, windows=None)` (coordinator
   contrastfix contract centralizing the A6 gate, the TRAIN-only source gate,
   the eligible sibling pair pool bound on `trainer._contrast_pair_pool` with
   the fail-closed empty-pool refusal, and the TRAIN-only
   `fit_outcome_norm(source, windows=None, ...)` normalization).  The former
   inline duplicate of fit's preparation steps (and its pre-contrastfix
   raw-pool mirror fallback) is REMOVED: a trainer without the canonical
   method FAILS CLOSED (RuntimeError naming the contract) instead of silently
   measuring the old trainer — no legacy/raw fallback for the actual headline
   gate.  `_fit_preparation(trainer, source)` returns the pool for
   diagnostics, read from `trainer._contrast_pair_pool` (A's return value is
   only the fallback; the helper assumes nothing about A's return type).
   No loss floor, schedule/optimizer/math change; the measured step remains
   one real `_optim_step`.  Prior-session edits preserved: preserved workdirs
   (`_timing_workdir`, `_release_corpus_handles` — never rmtree), honest
   worker metadata (`batch_path` + actual `num_workers`), `pack_source` =
   exact supplied pack path with `source_kind` kept separate, gate artifact
   binding.  `trainer._require_all_terms = True` is still set before the
   call (what the canonical method reads for the empty-pool refusal, fit
   parity).
2. `bundle/e2e.py` — `run_config_job` receipt: `pack_source` is now the EXACT
   supplied pack path (was the literal `"real-pack"`) with `source_kind:
   "real-pack"` kept separately (owner 2.3, same convention as the timing
   gate). Prior-session edits preserved: pack_v3-only production gate, CUDA-only
   refusal, no-implicit-resume output refusal (`_refuse_existing_output`),
   A6-before-data ordering, full 15x2000/batch-8/seed-101 registered defaults.
   (Unchanged in the canonical-preparation commit.)
3. `bundle/queue.py` — prior-session edits preserved: malformed prepacked
   upload (markers present, invalid metadata) FAILS CLOSED before convert; the
   raw-shard flow only converts into a fresh, non-symlink/non-reparse pack
   target; sealed-entry refusal unchanged. (Unchanged this session.)
4. Tests (DRAFT, not executed):
   - new `bundle/tests/test_timing_recovery.py`: batching path/worker honesty,
     exact `pack_source`/`source_kind`, workdir preservation on success and
     failure, gate artifact binding, plus the NEW canonical preparation
     contracts — exactly ONE mocked `prepare_training_source(corpus,
     windows=None)` call BEFORE the batch iterator, the optimizer and every
     optim step (recorded order: prepare -> sampler -> optimizer -> optim),
     the pool/norm the canonical call bound (no inline duplicate), the
     fail-closed RuntimeError when the trainer lacks the canonical method
     (nothing measured after it, workdir preserved; no realpath bypass), and
     a refusal raised BY the canonical preparation (A's empty-pool
     `IdeaAConfigError`) propagating with the workdir preserved.
     Trainer/corpus/optimizer are mocked — no real data, `WD_FORBID_D=1`;
     the tiny fixtures use the canonical method (mocked), not inline logic.
   - `bundle/tests/test_production_member_launch.py`: receipt assertions moved
     to the exact pack path + `source_kind` (prior session; unchanged).
   - `bundle/tests/test_prepacked_real_gate.py`: prior-session fail-closed
     marker/target guard tests (unchanged this session).
   - `bundle/tests/test_timing.py`: prior-session exact-path receipt fix
     (unchanged this session).

## Pending (coordinator)

- INTEGRATION DEPENDENCY (verified 2026-10-02 by read-only `git grep` against
  contrastfix2 HEAD `25479ae`): `prepare_training_source` does NOT exist
  there yet (coordinator branch A adds it; current fit inlines the same
  steps).  Until A's merge lands, the real timing gate FAILS CLOSED with the
  documented RuntimeError — by design it never measures the pre-contrastfix
  trainer.  No train/** edit was made here.
- Reviewed guard + explicit test nodes, then run EXACTLY:
  1. `WD_FORBID_D=1 C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q output/wd01/wm_n1/bundle/tests/test_timing_recovery.py output/wd01/wm_n1/bundle/tests/test_production_member_launch.py output/wd01/wm_n1/bundle/tests/test_timing.py output/wd01/wm_n1/bundle/tests/test_prepacked_real_gate.py` (focused touched paths)
  2. `WD_FORBID_D=1 ... -m pytest -q output/wd01/wm_n1/bundle/tests/test_queue.py` (existing queue regression)
  3. optional full bundle suite ONLY if the coordinator asks (prior 176/1 skip
     is a historic count, not current acceptance).
  CPU <= 4 threads; synthetic fixtures only; timing workdirs preserved on any
  failure (raw failure receipts kept, never deleted).

## Review repair (2026-10-02, source-only, this commit)

Additive note to the draft above: independent review found
`bundle/tests/test_production_member_launch.py` pulling
`test_multibox._crlf_records`, whose source reads the bytes of all eight
`MB.DEFAULT_RECORDS` entries (including the real `CAL_SPLIT_A4.json`); seven
nodes used it (six also opened the real A6 record through the unmocked
`evalh.seals` gate). Repair (owned file only, production source untouched,
tests NOT run): the file is now self-contained — eight synthetic literal
record fixtures under pytest `tmp_path`, LF-sha256 computed from the literals
and pinned into `bundle.multibox.LF_SHA_*` and
`evalh.seals.AMENDMENT_6_PATH`/`LF_SHA_AMENDMENT_6`, so zero real frozen
records are opened or copied. All production approval/A6/CUDA/pack/K/schedule
gates stay unmocked; a synthetic-only tamper fail-closed check was added to
the full-schedule node. All nine nodes keep their original assertions; the
full-suite counts above remain HISTORIC, not acceptance for this source.

## Review repair 2 (2026-10-02, source-only, this commit)

Additive note: independent review of commit 86ebb66 found the synthetic tamper
self-check in `test_cli_seam_runs_full_registered_member_1` too narrow: it
wrote the tamper approval via `_approved(tmp_path, "member-1-tamper", records)`
but requested `config="member-1"`, so `require_config_approval` could raise
`ConfigApprovalMissing` on the config-identity mismatch and the bare
`pytest.raises` could not distinguish that from a genuine byte-tamper refusal.
Repair (owned test file only, production source untouched, no tests/imports/
data changes, tests NOT run): the tamper approval is now written under a fresh
`tmp_path/'tamper'` parent with the SAME config name `"member-1"` and bound to
the same synthetic records, so the validator clears the config check and the
only possible refusal is the production LF-sha256 re-hash of the tampered
cal_split bytes; the assertion now also pins the expected tamper message
(`cal_split.md` ... `LF-sha256 ... != ...`). Existing run/refusal assertions
are unchanged. Runtime execution remains PENDING the coordinator's reviewed
guard; no pytest ran in this session (static py_compile + ruff only).

## Constraint receipt

No git reset/clean/checkout/stash-drop/push; no merge into main/checkpoint; no
deletion; no subagents; no approvals/authorizations; no frozen-record edits;
no real data, GPU, SUMO, network or protected reads; no `CAL_SPLIT_A4.json`
content access; WD_FORBID_D noted for all pending commands.
