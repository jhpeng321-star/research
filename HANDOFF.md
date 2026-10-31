# S-RB-DECIDE handoff

Branch `c4/rb-decide`, base `f0db3c82`; owner WM-PHYS registration line5683.

## Fresh-session continuation — 2026-11-01 04:44 +08

- Preserved clean HEAD `cbf042f7` and all three implementation/verification commits.
  INT already integrated `16dff9ad` and independently passed the 15 CPU tests.
- New evidence is available: EVAL's latest fetched WM2 shards now contain complete
  K8 worlds; INT retains immutable local `runs/rbint_evidence/{wm2_0400,full_v1_0400}`
  snapshots and a four-world paired comparison. Previous zero-complete-world receipt
  below is historical, not the current evidence status.
- Improved CLI adds `--worlds ID ...` for strict matched-world scoring and a third
  reliability diagram / Brier comparison for the actual selected 0.8-confidence
  switches, including reverse directions and ties. No-switch cases stay in switch
  rate / saving summaries, not calibration. `CREDIBILITY.md` now exposes coverage.
  **18 CPU tests PASS /5.889s**; next score the immutable local snapshots.
  Existing EVAL/INT/RUNA/RUNB inference and finalizers retain ownership.
- Owner 00:44 extension applies: improve until ~09:00, integrator freeze ~09:30.
  Local 04:43 resource check: 22 logical CPUs, 0% load; NVIDIA driver unavailable.
  Analysis will be CPU-only with one BLAS/OMP thread.

## READY FOR INTEGRATION

Implementation commits: `df4fa61e` (paired picker), `16dff9ad` (credibility).
Copy/merge `tools/wmdecide/` into the integrated runtime and use the CLI below.

- `tools/wmdecide/decision.py`: `Future`, `summarize`, `decide`; NumPy/CPU only.
- Shared draw IDs align all candidate actions; missing/duplicate/failed futures
  reject. Full-team expected delay, predictive spread, event probability, paired
  strict-win probability and saving distribution are returned as JSON-safe dicts.
- Fixed owner confidence `SWITCH_CONFIDENCE=0.8`; choose lowest-mean eligible
  candidate only if it also improves expected delay; otherwise keep incumbent.
- No shared model-file changes or default switches. S-C1SWITCH2 can import API.
- Credibility CLI READY: `python -B -m tools.wmdecide credibility --eval-root
  E:/agent-work/c4-rb-eval --inputs SHARD0 SHARD1 SHARD2 SHARD3 --out NEW_REPORT`.
  Run separately for WM2 and each WM-PHYS variant; output parent must exist.
  `--allow-partial` scores only complete K8 worlds, explicitly diagnostic.
- Integrator: include each output's `CREDIBILITY.md` in the morning report and
  retain adjacent `reliability_breakdown.svg`/`reliability_preference.svg` links.
  `summary.csv`/`credibility.json` carry action/cell/horizon/split breakdowns;
  `FILES.json` and `audit.json` bind inputs, source code and outputs.
- Metrics: empirical/fair CRPS for absolute and paired team delay vs same-arm/
  same-pair pooled+cell climatology; breakdown and pair-preference Brier/reliability;
  50/80/90% central and range coverage. Strict paired ties count as non-wins.
- EVAL integration reuses `read_predictions` and `real_observables.json`, preserving
  validation, K8 identity, BIC, ledger and development-only world-excluded baselines.
- CIs: 10000 pointwise stratified whole-world bootstrap resamples, seed20261031;
  saved futures fixed, no finite-K/model-error claim or simultaneous qualification.
  Existing EVAL adoption/non-regression gate remains authoritative.
- Verification: combined **15 CPU tests PASS, 0 skips (10.68s)**, independent
  CRPS/CDF/rank checks, leakage/missing-climate/tie guards, real-validator→report,
  SVG XML/coordinates, output hashes, public picker CLI and overwrite rejection.
  Ruff and `git diff --check` PASS. See `tools/wmdecide/VERIFICATION.md`.
- Fetched WM2 evidence checked with actual CLI: 11 tasks validated, **0 complete
  K8 worlds**, so **BLOCKED_NO_COMPLETE_K8_WORLD**, no partial-ensemble scores.
  Receipt: `E:/Temp/opencode/wmdecide-wm2-progress-58cb40f9745e47e1ac1eb43919a495f5/`.
- Scientific credibility scores await completed saved K8 worlds and local candidate
  shard paths. Existing EVAL/RUNA/RUNB producers retain their inference ownership.
  S-RB-INT should run the same command once per model when outputs land, then
  include each `CREDIBILITY.md` and its figures in the morning report.
- No simulation/training/remote dispatch or peer-process interaction performed.
