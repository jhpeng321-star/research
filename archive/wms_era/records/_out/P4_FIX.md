# P4_FIX — WM-S decision pipeline review blockers
Status: COMPLETED
Fix sha: b560d49 (branch wms/p4, parent 5d729c0)

## Fixed REVIEW.md items (wms/p4 files only)
- BLOCKER 4: `_load_release` validates immutable TRAIN provenance for EVERY
  row (row `split` == TRAIN + 64-hex `source_sha256`); any CAL/HOLDOUT/test/
  missing-provenance row fails closed (CLI exit 87).
- BLOCKER 5: `total_delay` returns the registered normalized delay in
  seconds, `sum(dt * clip(1 - v/v_free, 0, 1))` over vehicles/steps, matching
  output/wd01/wm_n1/evalh/scorer/comparator_v2g.py:289-290 (`_window_delay`);
  nonpositive free speed refused. 2 m/s deficit @ 10 m/s, 30 s -> 6 s.
- MINOR 12: decision ties use an explicit native-first rank (E2_ARMS order);
  exact ties select `native`, not the lexically first arm.

## Tests (tests/test_wms_decide.py)
- Added: test_release_loader_rejects_a_nontrain_row_inside_a_train_document,
  test_delay_score_uses_normalized_delay_seconds (6 s),
  test_decision_ties_default_to_native, loader-accept and free-speed-refusal.
- Corrected stale 60 s / 30 s expectations to normalized 6 s / 3 s.

## Verification
- pytest tests/test_wms_decide.py: 7 passed (rl_driving py3.10); ruff clean.
- CLI smoke, fixture model, no SUMO/GPU/SSH: valid TRAIN doc scored
  (pi-native +3.0 s); TRAIN doc holding a HOLDOUT row refused, exit 87.

---

# P4_FIX follow-up (2026-10-06) — BLOCKER 4 reopen: hash-to-artifact binding
Status: COMPLETED
Fix sha: 80b1592 (branch wms/p4, parent 7bc523e; review commit e695c76)

## Fixed
- BLOCKER 4 (reopened): `_validate_row_provenance` now BINDS each row's
  `source_sha256` to the referenced source artifact. It resolves the row's
  source (registered format: `state.base_run_dir` ->
  `<base_run_dir>/saved_state/STATE_MANIFEST.json`, exactly the bytes hashed
  by `_out/run_p4_e2e.py`; direct `source_file` accepted as override),
  recomputes sha256 streaming in 1 MiB chunks (memory-bounded), and fails
  closed via DecisionRefusal on: missing reference, missing source file, or
  recomputed != registered hash (case-insensitive). CLI exits 87.
- Rows without any source reference are refused ("provenance is
  unverifiable") instead of silently passing the 64-hex format check.

## Tests (tests/test_wms_decide.py)
- New: test_release_loader_rejects_wellformed_but_wrong_source_hash (64-hex
  "e"*64 vs real artifact -> refused), test_release_loader_fails_closed_on_
  missing_source_artifact, test_release_loader_refuses_rows_without_a_source_
  reference.
- Updated provenance fixtures to build real STATE_MANIFEST.json sources with
  correct hashes (uppercase hash accepted case-insensitively).

## Verification
- pytest tests/test_wms_decide.py: 15 passed (rl_driving py3.10); ruff clean
  on wms_decide.py + tests/test_wms_decide.py.
- Real TRAIN release `_out/wms_train_release.json`: all 20 rows re-verified
  against `_ext/campaign_chunk4` STATE_MANIFEST bytes (TRAIN only; no
  chunk-5/CAL/HOLDOUT access).
- CLI fail-closed smoke: tampered row (64-hex "0"*64) refused, exit 87, with
  registered vs recomputed digest in the message; no SUMO/GPU/SSH.
- Full suite: 1034 passed, 159 skipped; the 65 failures + 23 collection
  errors reproduce identically with both fix and HEAD files (missing
  `output.planner_v1` / `output.controller_verification_v21` / `wm20_common`
  artifact modules in this worktree; unrelated to wms_decide).

