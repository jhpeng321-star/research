# S-RB-DECIDE verification

Source base `f0db3c82`; implementation `df4fa61e` and `16dff9ad` plus final
verification/handoff commit. All work is CPU-only, with NumPy and existing Python.

## Resume improvement — 2026-11-01 05:14 +08

The18 existing picker/credibility checks and4 snapshot checks pass. Six new
refresh tests pass (18.565s final targeted run) with the actual EVAL validator,
synthetic copied-outcome forecasts, two-world baseline/one-world matched candidate,
partial-K blocked model, full output checksums, cached immutable-score reuse,
identity/completed-task cache invalidation, handoff-history preservation, exclusive
claim admission, deadline termination and foreign-staged/branch/worktree guards.
No network or scientific inference in tests. Scratch fixtures remain preserved.

Real saved outputs collected05:04–05:05 validate WM2 complete20, FULL-v1/A2 each6,
FULL-v3-gpu3 and B2 zero complete K8 worlds. Both available and exact-three-world
portable publications pass source/output checksum validation. C2 had no JSON.
See `docs/paper_results/wmdecide/INTERIM_0505.md`; this supersedes the historical
11-task progress status below without removing its receipt.

## Resume improvement — 2026-11-01 04:50 +08

**18 CPU tests PASS, 0 skips, 5.889 s**, retaining all previous checks. New tests
verify selected-switch reverse orientation with 7/8 strict wins and a tied future,
a confidently wrong real outcome, paired climatology, unchanged choices after
real-label perturbation, zero-switch abstentions, visible interval coverage, and
explicit matched-world admission (missing/duplicate/empty/wrong-type rejection).
The actual external-validator integration also verifies explicit world filtering
and rejection before output creation. No simulation or inference was launched.

Publication verification: **18 CPU tests PASS /4.582s**, including checksum-bound
portable output, byte-exact SVG export, duplicate/empty source and overwrite
rejection, tampered-source failure before destination creation, and singleton-cell
bootstrap uncertainty rejection. The existing tests were extended, not replaced.

## Checks completed

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
python -B -m unittest tools.wmdecide.test_decision tools.wmdecide.test_credibility -v
python -B -m ruff check tools/wmdecide
git diff --check
python -B -m tools.wmdecide credibility --help
```

**15 tests PASS, 0 skips, 10.68 s. Ruff PASS; diff whitespace PASS.**

Independent checks cover CRPS by exact CDF integration, fair CRPS algebra,
continuous finite-K coverage by exhaustive rank enumeration, calibrated 70%
synthetic predictions, biased Brier score, common-noise cancellation, fixed 0.8
gate boundary, ties, rare-loss expectation, identity alignment, failed/missing
futures, unavailable labels/baselines and world-cluster replication invariance.

The integration test uses the actual trusted local evaluator's external adapter
and identity/BIC/support/ledger validator on **synthetic plumbing outputs**.
It verifies CSV/JSON/report creation, XML-valid SVG diagrams with finite points,
output manifest hashes, immutable reports, duplicate-input rejection, complete
panel enforcement, blocked partial-K reports, and the public picker CLI. Synthetic
fixtures are preserved under `E:/Temp/opencode/wmdecide-synthetic-*`.

## Existing fetched WM2 evidence

The actual credibility CLI was also run against the four fetched shard directories
under `E:/agent-work/c4-rb-eval/tools/rbeval/evidence/wm2_shard{0,1,2,3}` with
`--allow-partial` and the trusted `--eval-root E:/agent-work/c4-rb-eval`.

- Validated tasks: **11**.
- Complete K8 worlds: **0**.
- Status: **BLOCKED_NO_COMPLETE_K8_WORLD**; no factual score inferred from K<8.
- Preserved receipt/report directory:
  `E:/Temp/opencode/wmdecide-wm2-progress-58cb40f9745e47e1ac1eb43919a495f5/`.

The code is ready for complete WM2 and candidate outputs. No model efficacy or
same-state probability-calibration result is claimed from this progress check.

## Resource and access receipt

Local `nvidia-smi` reported driver unavailable; no GPU work was attempted.
CPU had 22 logical processors at 73% load; tests and scoring used one BLAS/OMP
thread. No new environment, simulator, training, downloads, raw assignment/truth
files, process signalling, deletions or pushes were used.
