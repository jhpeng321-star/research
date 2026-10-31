# S-WMFIX-C — option (c) fallback prep (MAIN 12:45; branch c4/wmfixc; CPU only)
Prep only: the running pilot/verdict was not touched; no GPU/SUMO/sealed input.
Both upgrades are default-OFF, exact identity until enabled. Commits 548c20ea, 4fa0bdfd.
## What changed
1. TRAINING-ONLY smoothed admission/crossing relaxation (FEAS_BLOCKED option c).
   `entrance_gap_series` + `smooth_admission_source` (src/wms/train/merge_response.py):
   mean-field queue recursion from the recorded Q0 whose per-step service gate is
   sigmoid((gap_end - min_gap)/width) on the SAME end-of-step entrance geometry
   `LearnedAdmission.place` tests; the hard path records its tested inferred min-gaps
   only under the new `track_admission` flag. `Runtime(smooth_events=True)` swaps the
   surrogate into the differentiable source/J columns and keeps the exact integer
   ledger in the receipt; `train.py --smooth-events` selects SMOOTH_GRADIENT_COLUMNS
   (source joins upstream/crossing/road; crossing already uses `smooth_crossings`).
   OFF: no surrogate, no recording, exact identity; inference never sets the flag.
2. Finer spatial resolution of the merge-response field (REVIEW claim 1: 4 RBFs,
   sigma ~907 m, cannot resolve the 75 m merge / -379 m ramp entrance).
   `MergeResponseField(merge_bases=, ramp_bases=)` adds geometry-concentrated centres
   over the observed merge section [0, 75.44] m and ramp acceleration lane (network
   constants only, no behavioural thresholds); width = nearest-centre distance
   (reproduces the uniform sigma); shared endpoints deduplicated (4+4+4 -> 10
   centres, in 8+90; default 0/0 is bitwise the registered head). Zero-init,
   frozen-WM2 bitwise and the merge_response.* allowlist unchanged; spec/config/
   checkpoint plumbing keeps the canonical wm_s/1 reload exact.
## Tests (CPU, C:/Users/jpeng/Miniconda3/python.exe -m pytest)
- New: test_wmfix_smooth_events.py (6) incl. central finite-difference agreement of
  the surrogate on a small fixture (eps 1e-4..1e-6, float64) and hard-event
  bracketing; test_wmfix_merge_refine.py (5) incl. bitwise identity vs the
  committed registered head (5a4d64c2) and checkpoint-reload parity.
- Regression: the 11 wmfix/wmfixmerge request files + config tests: 132 passed.
## How to enable (after a NEW CPU feasibility + review round; fail-closed)
- train.py: `--smooth-events [--admission-width W]` and/or `--merge-bases 4
  --ramp-bases 4` (all default OFF). Implementation hashes changed, so existing
  feasibility/review bindings are rejected by check_gates until re-produced —
  intended. Verdict/panels/inference paths never set either flag.
## Expected cost (estimates; the new feasibility must measure them)
- Fallback 1 ON: ~3 masked-min gap series + a 600-step python queue recursion in the
  graph per rollout (~10-30% CPU rollout time; linear-in-T backward); width default
  1.0 m (numerical relaxation scale, not behavioural).
- Fallback 2 ON: head first layer 44 -> 98 inputs (~1.5k -> ~3.2k params); per-tick
  head arithmetic ~2x, but the pairwise [B,K,K] ego terms and the rollout dominate:
  ~5-15% per update; CUDA memory delta negligible.
