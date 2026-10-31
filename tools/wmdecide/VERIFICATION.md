# S-RB-DECIDE verification

Source base `f0db3c82`; implementation `df4fa61e` and `16dff9ad` plus final
verification/handoff commit. All work is CPU-only, with NumPy and existing Python.

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
