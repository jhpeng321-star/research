# S-C1FLOOR verification (2026-10-31)

- Local and gpuR `python -B tools/c1floor/test_core.py`: 10 mechanics tests PASS
  (RNG-only bytes, multiple RNG streams, missing RNG refusal, seed types/ranges,
  conservative budget/duplicates, bitwise signed-zero/dtype detection, generated
  definition/truth write-only exception with prohibited reads/update modes).
- Local `python -B tools/c1floor/test_analysis.py`: 3 scientific-contract tests
  PASS (common HDVs across both draws, missing rather than zero, whole-world
  panel dimension and exact inherited CSV schema). The moving upstream schema
  added units; the test caught that change. Final scorer source is pinned in
  `C1CLOSED_analyze_source.py`; final CSV includes its units column. Initial
  unpinned exports remain preserved, not deleted.
- gpuR zero-SUMO end-to-end scenario setup PASS: original builder reached an
  intercepted launch without any SUMO process/physics step. Own generated
  truth/definition outputs were created, never read by the task.
- Real original-seed prefix/continuation reproduction PASS, all six observable
  tape arrays bitwise identical through t600 on attempts 4–6.
- Real original-seed saved-state/RNG continuation FAIL: first mismatch t300.5,
  two HDVs, positions/speeds/accelerations differ. No numerical tolerance was
  substituted and no alternative RNG branch was run. Production is STOPPED.
- `python -B tools/c1floor/verify_delivery.py`: PASS_BLOCKED_DELIVERY_NOT_FLOOR.
  Six unique precharges =9,000 sim-s; complete actuals =2,400 sim-s; 98 blank
  horizon/metric cells, exact 14-column C1CLOSED schema, report 36 lines.
- `git diff --check` PASS. Final source byte hashes match the last preflight.

This is a validated failure report, not a successful randomness-floor result.
