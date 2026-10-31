# S-RB-DECIDE handoff

Branch `c4/rb-decide`, base `f0db3c82`; owner WM-PHYS registration line5683.

- `tools/wmdecide/decision.py`: `Future`, `summarize`, `decide`; NumPy/CPU only.
- Shared draw IDs align all candidate actions; missing/duplicate/failed futures
  reject. Full-team expected delay, predictive spread, event probability, paired
  strict-win probability and saving distribution are returned as JSON-safe dicts.
- Fixed owner confidence `SWITCH_CONFIDENCE=0.8`; choose lowest-mean eligible
  candidate only if it also improves expected delay; otherwise keep incumbent.
- No shared model-file changes or default switches. S-C1SWITCH2 can import API.
- Verification: `python -B -m unittest tools.wmdecide.test_decision -v`: 5 PASS.
  Credibility/report adapter in progress.
- EVAL integration will reuse `read_predictions` and `real_observables.json` from
  S-RB-EVAL, preserving validation, K8 identity, BIC and development-only baselines.
- No simulation/training/remote dispatch or peer-process interaction performed.
