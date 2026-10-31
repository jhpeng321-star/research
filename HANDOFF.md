# S-RB-DECIDE handoff

Branch `c4/rb-decide`, base `f0db3c82`; owner WM-PHYS registration line5683.

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
- Verification: combined 14 CPU tests PASS, 0 skips (7.60s), including independent
  CRPS/CDF/rank checks, leakage guards and real-validator-to-report synthetic
  integration; Ruff and `git diff --check` PASS. Final packaging checks next.
- Scientific scores await complete saved K8 worlds. No efficacy result claimed.
- No simulation/training/remote dispatch or peer-process interaction performed.
