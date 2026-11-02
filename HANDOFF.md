# S-COOPBANDIT-LEARN -> S-COOPBANDIT handoff (2026-11-01)

## Resume check 2026-11-03: nothing outstanding

Fresh session resumed from `81a04120` (clean tree). Your worktree is closed
and verified at your end (300 states, live test complete, manifests re-checked
in your 2026-11-02 resume). All STEP 3 deliverables here are committed; the
full-panel artifacts in `docs/paper_results/coopbandit_learn_full/` are intact;
all 17 CPU tests re-ran green today (130.9 s). No new data, dispatches, or
requests were found addressed to this helper; no simulations were started.
This receipt is the only change.

## ADDENDUM 2026-11-02: your data landed and was already analyzed

Your `partial135` and full 300-state EXAM exports
(`docs/paper_results/coopbandit/{partial135/exam-partial,full/exam}` in your
worktree) were consumed through your `tools/coopbandit/helper_analysis.py`,
which hash-verifies the export, builds the documented metadata-only partial
view, and calls my `tools.coopbandit_learn.cli.analyze` at this branch's HEAD
332239c4 (code unchanged since the runs). Outputs are committed in your
worktree at `docs/paper_results/coopbandit/{partial135,full}/helper/`.

Full-panel headline (h=600, 300 states / 60 worlds): GBT OOF RMSE 5262.43
veh-s, full radius 11905.25 veh-s; linear comparator RMSE 7396.94, radius
62933.77. The documented cooperate rule fires on no OOF decision
(cooperate rate 0.000), so learned regret equals the always-O0 baseline:
588.48 veh-s = oracle headroom (helpful fraction), GBT-regret 95%
world-bootstrap CI [295.05, 961.64]. Reading: with the registered finite-
sample 95% conformal radius, no option's OOF prediction clears uncertainty;
the honest STEP 3 answer is "O0 everywhere, headroom 588 veh-s", not a
policy that cooperates. All 17 CPU tests re-verified green on 2026-11-02.

## ADDENDUM 2 2026-11-02: independent direct rerun, artifacts committed here

A second session instance re-ran the CLI directly on the same full export
(no helper mediation) and committed the artifacts in THIS worktree at
`docs/paper_results/coopbandit_learn_full/` (commit a76da843): report.json,
SUMMARY.md, importance/PDP csv+png, and the stageable `policy.json`.
Cross-check against the helper run: every statistic identical (RMSE, radii,
regret 588.48, CI [295.05, 961.64], cooperate rate 0.000); `policy.json`
differs ONLY in its `created_utc` field (tree arrays, radius, feature order
byte-identical), `report.json` only in wall-clock `seconds` and the policy
hash that inherits the timestamp. Either `policy.json` is fine for your live
test; quote the helper copies in your worktree as canonical if you prefer.

STEP 3 tooling is READY. It consumes your EXAM v1 export format directly
(`tools/coopbandit/export.py` output, verified against your `export.py` and
`learn.py` in `E:/agent-work/c4-coopbandit`); it was built and tested against a
synthetic fixture in exactly that format. The moment partial or full data
lands, run one command. I did not modify your worktree.

## Quick start

Worktree `E:/agent-work/c4-coopbandit-learn` (branch `c4/coopbandit-learn`).
Interpreter: `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe` (sklearn
1.7.2). From the repo root:

    $env:PYTHONPATH='E:\agent-work\c4-coopbandit-learn'
    & C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -X utf8 -m tools.coopbandit_learn.cli --data <path-to-exam-or-exam-partial> --out <results-dir>

Outputs in `--out`: `report.json` (CV comparison GBT vs linear baseline,
per-fold radii, policy summary with world-bootstrap CIs, per-cell breakdown),
`SUMMARY.md`, `feature_importance.csv/png`, `partial_dependence.csv/png`,
`policy.json` (stageable artifact). Partial exports run fine and are labelled
exploratory (completion-order subset, as your README already warns).

## What is implemented

- `schema.py` - EXAM v1 constants; `checked()` guard rejects `sealed/`,
  `holdout|heldout|cal_split|chunk5|j4b|inc1`, truth/type-assignment names on
  every file access; dev worlds 981080..981091 are hard-rejected at load and
  again before any policy evaluation.
- `loader.py` - reads `states.jsonl` + `labels.csv` + `group_credit.csv` +
  `worlds.json`, verifies your `MANIFEST.json` sha256s, validates: required
  columns, horizons {120,300,600}, O0 rows carry D=0, unavailable options
  carry exact no-op labels, state_id/world_id/t0 consistency, finite numeric
  features with no metadata leakage (world/cell/penetration/demand/ids never
  become features). Availability is read from labels.csv per option.
- `models.py` - your frozen hyperparameters (GBT 100x depth2, leaf 8, Huber,
  random_state 0) plus a standardized-ridge linear comparator.
- `cv.py` - world-grouped nested CV identical in structure to `learn.py`:
  GroupKFold 5 outer folds, 4 inner grouped folds, radius = finite-sample 95%
  quantile of per-world MAX signed overprediction across decision
  times/options, clamped at 0, outer-test labels never calibrate their own
  fold, full-data model calibrated by the grouped 5-fold OOF residuals.
  DOCUMENTED DIFFERENCE from `learn.py`: I fit each option only on states
  where the option is available and pool calibration residuals over available
  options; `learn.py` fits all options on all states (your unavailable options
  are measured no-ops, D=0). Choice rule, radius, folds, hyperparameters are
  identical, so results are comparable; if you prefer your pooling, say so and
  I will align exactly.
- `policy.py` - the documented rule, no hand-set threshold: argmax predicted D
  among available options, execute only if predicted D - radius > 0, else O0.
  OOF evaluation: oracle headroom (helpful fraction), learned-vs-oracle regret
  (always-O0 baseline regret equals headroom), world-bootstrap 2000 draws seed
  20261031 CIs, per-cell descriptive breakdown.
- `interpret.py` - importance + partial dependence tables and PNGs (Agg).
- `artifact.py` + `runtime.py` - the learned policy as a stageable callable:
  `policy.json` carries the full-data per-option GBTs (tree arrays), radius,
  feature order; `runtime.py` is numpy+stdlib ONLY (stageable as text through
  `tools/coopbandit/ops.py stage`, which copies only .py files) with
  `PolicyRuntime.from_json(path).choose(features_dict, available_tuple) ->
  option_id` implementing the same rule. `artifact.verify()` asserts exact
  parity with sklearn predictions (max abs diff 0.0 on fixtures; tolerance
  1e-9). This complements your `policy.pkl` for the remote live test.
- `fixture.py` - synthetic EXAM v1 generator (planted option effects, world
  random effects, geometric availability rules) used by the tests; also usable
  as a format example for S-COOPBEHAV / S-RB-INT consumers.
- `test_cpu.py` - 17 CPU-only tests on synthetic data; process spawn is
  forbidden by patch (no SUMO can start); env pinned to single-thread CPU.
  Run: `python -m pytest tools/coopbandit_learn/test_cpu.py -q` (81 s) or
  `python -m unittest tools.coopbandit_learn.test_cpu` (46 s).
- `cli.py` - everything above in one command; `--horizon` defaults to 600.

## Commit IDs (branch c4/coopbandit-learn)

- 88415195 initial EXAM v1 schema/fixture/loader (superseded format)
- 547e8f60 models + nested grouped CV + conformal radius
- 52397178 rework onto your real export format (manifest verify, dev guard)
- cd17d90f policy rule + OOF regret/headroom + bootstrap CIs
- c85ca231 importance/PDP tables + figures
- 9bb2b835 policy.json artifact + pure-numpy runtime (exact parity)
- c9f2d0a1 CLI end-to-end + 17 CPU synthetic tests

## Notes for STEP 4 and the world-model exam consumers

- The learned policy callable for the live test: copy
  `tools/coopbandit_learn/runtime.py` + `policy.json` (text) to the box; then
  `choose(features, available)` per RSU decision. Features must be exactly the
  keys recorded in `states.jsonl` (`features` dict); missing keys raise.
- For S-RB-INT: `loader.load(exam_dir)` gives the validated dataset object;
  `dataset.feature_matrix(ids)`, `dataset.labels_for(id, horizon_s)`,
  `dataset.available(id)` are the sign/rank exam interface.
- Fixture worlds use seeds 981000+ (labelled synthetic in provenance
  `generator` field); no registered seed ranges were burned - no SUMO ran.
- Nothing on hosts cpuQ/gpuR was touched; own PIDs only; no deletions; no
  pushes. My `EXECUTOR_HANDOFF.md` request (partial prefix >= 100 complete
  states, >= 7 worlds for nested CV, dev-world exclusion) is implemented and
  tested; the loader enforces >= 7 worlds implicitly via CV failures and your
  `DATA_COMPLETE.json` gate remains the full-panel authority.
