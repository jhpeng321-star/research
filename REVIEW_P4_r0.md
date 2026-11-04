VERDICT: PASS

## Numbered blockers

None. No required fixes identified in the P4 diff.

## Review scope and evidence

- Independent verifier: astra; round r0.
- Worktree: `E:/agent-work/c4-staging-closure`; branch: `c4/staging-closure`.
- Reviewed HEAD: `91a3d70e77a2a6b9ac5fac4936e80d9522b7cb0a`; base `c4/track-base`: `13f184b5a7e12be04aebf246ef8e4dbb81555545`.
- Read `E:/agent-work/_sol/tracks_spec.md`, `P4_READY.json`, and the complete diff against the base. The diff adds only the closure module, its CPU tests, and the readiness record.
- Checked the implementation against the production readers and relevant writers, rather than accepting the readiness claims alone.

## Correctness and rule compliance

- **Imported LS dependencies:** `tools/coopmwm/staging_closure.py:160-178` matches the label-generation layout in `replay_bank.py:415-445` and the pending-envelope, compressed-snapshot, and episode-result dependencies in `replay_bank.py:903-930`. `ls_collection.py:46-66,77-87,150-158` confirms the corresponding producer layout. Archived runner snapshot copies are not mistaken for reconstruction inputs.
- **Own snapshots:** `staging_closure.py:179-182` follows decision `state_id` to the own-bank snapshot, matching `replay_bank.py:931` and `practice_starts.py:174-181`.
- **Warm1 link targets:** `staging_closure.py:140-155` retains lexical aliases and physical targets while expanding reader-relative dependencies from the lexical location. This matches directory-link views created by `solo_seed2_links.py:63-71`. Allowed roots are derived from the resulting file parents at `staging_closure.py:202-207`, without a hard-coded source-run allowlist.
- **Compression and manifest:** `staging_closure.py:185-200` includes the RESULT marker and gzip payload, restricts the payload filename as `accel_storage.read` does, verifies the decompressed digest, and additionally checks declared size. Files and manifest records are sorted; file hashes, physical-file deduplication, and canonical manifest SHA are implemented at `staging_closure.py:61-68,134-151,202-210`. Required missing dependencies fail rather than being silently omitted.
- **Observables/no hidden truth:** the module follows structural metadata and hashes opaque snapshots; it does not unpickle snapshots or construct policy/planner inputs. Lexical paths, resolved link targets, and discovered dependencies are guarded before content reads (`staging_closure.py:28-43,104-120,140-145,190-192`). No prohibited corpus or truth file was opened during this review.
- **Look-ahead, fairness, thresholds:** the added module does not score actions, truncate tapes or horizons, select examples by arm, or introduce behavioural thresholds. Closure rules are arm-independent and preserve the complete referenced file bytes.
- **Default-off byte identity and ownership:** no existing tracked source file is modified, and no existing runtime entry point imports the new module in this diff. Existing default-off paths therefore retain their original code. The new API and integration handoff conform to P4's separately owned helper-module scope; T2C integration is outside this P4 verdict.
- **Execution constraints:** review and tests were local and CPU-only. No SSH, simulator execution, GPU calls, commits, or implementation edits were performed. Pytest used synthetic fixtures in a new worktree-local temporary directory; bytecode and pytest cache writes were disabled.

## Tests and counts

Executed from the worktree with `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`, and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`:

```text
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q tools/coopmwm/test_staging_closure_cpu.py -p no:cacheprovider --basetemp E:/agent-work/c4-staging-closure/runs/p4_astra_r0_pytest
```

The tests make substantive assertions against actual `replay_bank._receipts`, `_decision_snapshot`, and `accel_storage.read` using synthetic files. Coverage includes both LS generation layouts, own snapshots, directory-link targets, mirror mapping, deterministic manifests and mutation sensitivity, missing dependencies, digest/size failures, malformed identifiers, and rejection before prohibited file opens. The snapshot-opacity test replaces `torch.load` with a failure during closure execution.

- **33 passed, 0 failed, 0 skipped; 11.77 seconds.**
- **1 warning:** pytest reports the configured `cache_dir` as unknown because the cache provider was deliberately disabled for this review.
- `git diff --check c4/track-base`: **PASS**.
- Test scope: the complete P4 CPU test module; no live corpus or simulator validation.
