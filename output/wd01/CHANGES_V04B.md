# CHANGES_V04B — WD-01-PB-V04B implementation record

Registration: `output/wd01/REGISTRATION_PARTB_V04B.md` (LF-sha256
`e00cd98e...`). Branch `zcode/pbv04bcode` from `zcode/pbv04b2` 7203ad0.

## Added (this task: code + tests only — NO fit, NO CAL read)

New package `output/wd01/partB_v04b/` (imports partB_v0 / partB_v0data
READ-ONLY):

- `reg.py` — registration binding (the fit refuses a registration file
  whose LF-sha differs from `e00cd98e...`), tau set, 7-quantile grid,
  the fixed 6-candidate tuning grid, wall arms/factors/timing, contrast
  sources, acceptance constants (TAUS_A3, coverage band, N_BOOT 2000,
  seed 20261001), results/lock paths.
- `j4plan.py` — the J4 plan is the only split authority; 39 TRAIN / 5
  CAL asserted by name; **disclosed single-name drift**: the
  registration text lists `w8662m2700p10t300` but the plan holds
  `w8662m2600p10t300` (loading 2600, TRAIN; same cell otherwise). The
  code follows the plan and reports the drift in every log/manifest;
  any drift beyond the recorded one refuses.
- `schedule_desc.py` — the owner-required PLANNED-SCHEDULE action
  descriptor: per-segment roster action shares (19 x 30 s over 570 s,
  per-segment renormalised over known time) + the wall
  hold/speed/factor/timing/offset/cap block (has_cap=0 for the v2.1 J4
  data); canonical hash; duration strata (<=60 s / 60-240 s / whole)
  and onset bins from the ARM-vs-NATIVE differing span; first-60-only
  counts (s.12.1 item 4). "damp 60 s then native" and "damp 570 s" are
  distinct inputs (unit-tested).
- `pairs_v04b.py` — pair builder: every registered contrast (wall
  headline + aux damping/pi/fixed/switching) vs the native sibling at
  the same (world, t0); dY per (tau, head); complete horizons only;
  saved_state_id equality asserted at pair time (mismatch refuses the
  world); duplicate native rows refuse; strata + representation counts
  per pair.
- `build_v04b.py` — t0-row build. Aux/native rows come from the v0.4a
  root (read-only); wall rows are built FROM THE RAW J4 RUNS: labels
  from `v0lib.window_label` over the branch trace (pre-t0 insertion
  history merged from the shared base run), request-time wall
  descriptors via `wall_tokens_v04.WallEpisodeSet` (events <= t0 only)
  plus `branch_record.json` ALLOWED_RECORD_FIELDS (outcome block never
  bound), action tokens via `actions_v04` (schedule-authoritative),
  planned-schedule descriptor from the request fields + plan wall
  parameters (record-vs-plan drift refused). Receipt completeness and
  saved_state_id equality asserted per wall run. Post-ack root
  finalizer: COPY-only transfer of the needed v0.4a files into
  `wd01_partB_v0data_v04b`, MANIFEST + SHA256SUMS.
- `descriptors_v04b.py` — the design matrix: deployable_strict
  (demand[6]+context[8], queues zeroed) / deployable_detector
  (demand[12]+context[8]) state of the shared t0, the v0.4a per-CAV
  action counts (+ the wall count), the v0.4a policy one-hot (wall arms
  zero it; the arm-family one-hot identifies them), pi summary fixed to
  the full horizon (per-tau structure lives in the segment descriptor),
  arm-family one-hot, and the arm's 83-column planned-schedule
  descriptor. Prohibited inputs (penetration, pre_fix, speed_mode,
  world strings) never appear.
- `model_v04b.py` — DirectDiffGBM: per (tau, head, level)
  HistGradientBoosting quantile regressors on dY directly (early
  stopping off at fixed max_iter; the v0.4a hybrid machinery dropped
  per registration s.5); world-grouped GroupKFold CV over the fixed
  grid, selected by mean pinball on the delay head of the HEADLINE
  (wall-native) rows; conformal shifts from headline-row OOB residuals
  (aux rows train but never select/calibrate — v0.4a precedent);
  monotone quantiles; ridge-on-dY reference; wall-only ablation as a
  separate labelled model; all fits thread-limited (1).
- `acceptance_v04b.py` — A1 (per-tau CRPS vs zero-contrast + pooled
  10-90 % coverage in [0.75, 0.85]); A3(i) sign accuracy on
  |dY| > TRAIN median with the whole-world bootstrap 90 % CI > 0.5 and
  A3(ii) dY CRPS < zero AND < TRAIN-cell-mean, at tau in {60, 120,
  300}, pass at >= 2 of 3; ACCEPT/PARTIAL/FAIL verdict; s.12.1
  stratified tables (duration/onset x tau, per-arm, per-family).
  Distribution primitives imported from `partB_v0.acceptance_v04a`
  (identical CRPS code path, K = 40 midpoint grid).
- `cal_guard.py` — the ONE-CAL-read guard: OWNER_ACK file must name the
  registration's FULL LF-sha; `CAL_READ_ONCE_V04B.json` written before
  the first CAL row, finish-stamped after; second begin refuses
  (foreign-sha marker = loud mismatch); `assert_train_only` hard-guards
  every TRAIN-stage row frame.
- `cal_read_v04b.py` — the guarded CAL read: builds the 5 CAL worlds,
  the FROZEN fit models predict, acceptance + verdict + strata +
  aux/ablation/TRAIN-OOB sections are written; nothing is refitted.
- `pulse_hook.py` — the J-PULSE disclosed-retrain config (same family/
  grid/rule/splits; never a silent refit, never a chunk-5 selection
  input) + the registered OOF diagnostic (world-grouped 5-fold over the
  PULSE worlds; sign accuracy + CRPS vs zero/cell-mean at tau in
  {120, 300, 570} stratified by pulse length x controller;
  diagnostic-only).
- `dryrun.py`, `__main__.py`, `README.md` — entry points
  (`selfcheck | allocate | dry-run | build | fit | cal-read`) and the
  exact post-ack command sequence.
- `tests/` — 57 synthetic-data tests (see README); the real-data
  TRAIN-only dry-run ran separately (<= 4 worlds, no CAL row).

## Deferred to the post-ack run (registered outputs, not yet produced)

`partB_results/v0.4b_strict` + `v0.4b_detector` (eval_report + verify),
`REPORT_V04B.md`, `LEDGER_DRAFT_PB_V04B.json` (entry-schema close-out),
the backup of the new data root under `E:/research-dense-backups/wd01/`,
and the RESULTS_SUMMARY/validate/verify-sources pass. The allocation
entry `LEDGER_DRAFT_PB_V04B_ALLOCATION.json` is written by
`python -m partB_v04b allocate` and must be committed BEFORE the build.

## Known deviations / disclosures

1. Split-name drift `w8662m2700p10t300` (registration text) vs
   `w8662m2600p10t300` (plan) — plan followed, drift disclosed
   everywhere; owner may fix the registration at ack time.
2. The pi schedule summary is fixed to the full horizon for every
   source (v0.4a computed it per-horizon); the per-tau structure is
   carried by the s.12 segment descriptor. Disclosed here and in the
   module docstrings.
3. Open owner questions (§11.2-11.4: gating contrast granularity, A1
   gating with the small CAL, non-gating safety heads) do not change
   this code: the registered defaults are implemented and the report
   carries all reported-only sections either way.
