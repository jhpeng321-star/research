# A72 E2 — registered SECONDARY outcomes, Holm step, replay preparation (2026-11-13)

Zero new simulation. Analysis only. Convention: diff = treatment − reference; negative total delay = improvement.
Paired percentile bootstrap, 10,000 world resamples, seed 20261104, n = 96 confirmation worlds 990900–990995
(SUMO wz3to2, 2,800 veh/h, 60% CAVs). Panels not pooled. p = 0.0002 is the bootstrap floor (2/10,001).

## Registered secondary list (PROTOCOL.json `e2.secondary`, verbatim)

"F - S, every arm - U0, executed control effort, replay (<= 24 Development worlds)"

Nothing else is registered as an E2 secondary. Most-delayed-vehicles tail (CVaR90), group means and hard braking
are NOT in the E2 secondary list, so they are not computed here (see Suggestions).

## Source and reproduction

- Data: Box 4 (49.235.140.169) `/root/cv_20261113_0231/cv/out/e2/E2C_RESULTS.json`
  sha256 f698c87fbf86882deb8f0117de3a8ba5a36e95c333c2dab1c24cb160840f372a (laptop copy `box4_out/e2/` identical).
- The registered `e2_analyse.py` (sha256 2cbeb588…fbd80, identical box/laptop) already computed every registered
  E2 secondary except Holm and replay in the sealed `E2_ANALYSIS.json` (file sha256 f1dab0f6…ccb1, doc 36f65eeb…f1d).
- Re-run on Box 4 (nice 19, one process) with the unmodified `analyse()` into the NEW dir
  `cv/out/e2/secondary_20261113_131322/`: `E2_ANALYSIS_REPRO.json` is byte-identical to the sealed file
  (sha256 f1dab0f6f72827ace8d6585d24d6b8ad5b4e6d0a6185eecdad54524eb56a4ac0).
- Wrapper `e2_secondary_holm.py` (new file; sha256 dba1c34ab355b9eede27e3d0e20a573c844b301007b6b063fdb9b78c28e3a67e).
  Diff vs `e2_analyse.main`: (1) writes to the new dir instead of `out/e2/E2_ANALYSIS.json`; (2) performs the
  `--primaries` Holm step with the E1 p read from `out/e1/bank/FROZEN_BANKS.json`
  (`primary_admitted_count.permutation.p_two_sided`) and the E2 p from the recomputed F − H20. No metric changed.

## Table 1 — total delay (veh-s per episode), SECONDARY unless marked

| Contrast | Role | Mean diff | 95% CI | % diff [95% CI] | p (2-sided) | worlds lower / higher | Reading |
|---|---|---|---|---|---|---|---|
| F − H20 | PRIMARY (already reported) | +686.9 | [−205.5, +1,656.5] | +2.2% [−0.7, +5.3] | 0.142 | 48 / 48 | not separated |
| F − S | SECONDARY | +58.2 | [−886.4, +1,063.7] | +0.2% [−2.7, +3.3] | 0.903 | 50 / 46 | not separated |
| F − U0 | SECONDARY | −10,215.1 | [−12,374.6, −8,108.9] | −24.2% [−27.7, −20.3] | 0.0002 | 82 / 14 | F less delay than U0 |
| H20 − U0 | SECONDARY | −10,902.0 | [−13,044.2, −8,805.5] | −25.8% [−29.2, −22.1] | 0.0002 | 77 / 19 | H20 less delay than U0 |
| S − U0 | SECONDARY | −10,273.4 | [−12,588.1, −8,057.8] | −24.3% [−28.1, −20.3] | 0.0002 | 76 / 20 | S less delay than U0 |

Arm means (veh-s): U0 42,288.3; F 32,073.2; H20 31,386.3; S 32,015.0.

## Table 2 — executed control effort, SECONDARY (from executor receipts, 96 worlds)

Registered statistic = per-episode mean and total (descriptive; no interval is registered). U0 issues no holds.

| Arm | Activations / ep (total) | Full groups / ep | Truncated groups | Hold vehicle-s / ep (total) | Unique CAVs / ep (total) |
|---|---|---|---|---|---|
| F | 15.98 (1,534) | 15.98 | 0 | 5,429.0 (521,185) | 87.15 (8,366) |
| H20 | 16.04 (1,540) | 16.04 | 0 | 6,015.7 (577,511) | 87.51 (8,401) |
| S | 16.01 (1,537) | 16.01 | 0 | 6,067.2 (582,448) | 87.34 (8,385) |

Plain statement: activations, full groups and unique CAVs are matched within 0.4%. Executed hold vehicle-seconds
are not matched: F held 586.7 veh-s/episode fewer than H20 (−9.8%) and 638.2 fewer than S (−10.5%).

## Table 3 — Holm over the two registered primaries (PROTOCOL.json `multiplicity`, two-sided, alpha 0.05)

| Primary | Raw p | Holm step | Holm-adjusted p | Rejected at 0.05 |
|---|---|---|---|---|
| E2 F − H* (H20) total delay | 0.141986 | 1 (×2) | 0.283972 | no |
| E1 admitted count (WM − Random, exact sign-flip, 6 seeds) | 1.0 | 2 (×1, max with step 1) | 1.0 | no |

Smallest raw p (0.142) exceeds alpha/2 = 0.025, so Holm stops at step 1: neither primary is supported.
E1 source: FROZEN_BANKS.json file sha256 f27f2e5a4fbdf0f59b02352719e21a7cead58dd8746278e592d8450c2c944299,
doc c9b271b54085a5fbf3d5d0055e25d4cca0c1c9cc5c42bc7f003a3b7a985431c7. Holm doc `E2_HOLM.json`
sha256 19a32e8b937e4cd1b252384a1fff2ce99d8998fee89f6f7d229b133daeeebec0 (doc 47c5d987…756b).
Post-hoc E1 repairs (WM-ACT, FIX) are outside this family and are not included.

## Replay (registered secondary) — PREPARED, NOT RUN

Design (e2_replay.py, sha256 9990b7a2…dd60): on Development worlds 0..23 of a35_D2800_p60 (998200–998223,
block `dev`), discover runs F from t = 0 and reads only executor receipts for the first launch (t*, plan P, members),
kept if t* ≤ 600 s. Branches re-run each kept world from t = 0 as U0, F1 = P alone (one-shot, cooldown 100,000 s)
and S1 = synchronized copy of P alone. `check` requires per world: 0 launches in U0, exactly one launch in F1 and S1
at the same t* with the same members, and identical decompressed vehicle-log prefix before t*; failing worlds are
excluded and counted. `analyse` reports F1 − U0, S1 − U0 and F1 − S1 (paired bootstrap over states); F1 − S1 is
the same-state timing contrast (staggered onsets vs synchronized copy).
What it would show: whether the staggered onsets of F change total delay relative to the synchronized copy when
both start from the identical traffic state at t*, removing the state differences that widen the whole-episode
F − S interval above. At most 24 states; fewer if some worlds have no launch by 600 s.

Zero-SUMO dry run on Box 4 13:15 (`e2_replay.py plan --n-worlds 24 --dry-run`): one-shot bank checks pass for all
four families (mut_2922c89d0b31d4c8, mut_d7251d07b4f749e0, rand_3a452b1348b01083, rand_bb1cb39d47f0a524; each
F1/S1 differs only in onsets); sim_s discover 28,800, branches_max 86,400, total_max 115,200 = registered cap.
Nothing was written.

Budget: ≤ 115,200 sim-s (24 × 1,200 discover + ≤ 3 × 24 × 1,200 branches). Commands (Box 4, as root, one stage
per launch, never in a retry loop):

```bash
B=/root/cv_20261113_0231; source $B/env_cv.sh; R=$B/root; O=$B/cv/out; REG=$B/cv/REGISTRATION.json
BANK=$B/inputs/A42_BIG_BANK_D2800_p60.json; RP=$O/e2/replay
drv() { nice -n 19 "$PY" -B $B/cv/box_driver.py "$@" --registration $REG --code-root $B/src --final-root $R; }
py()  { nice -n 19 "$PY" -B "$@"; }
# zero SUMO
py $B/cv/e2_replay.py plan --bank $BANK --out $O --n-worlds 24          # -> banks/E2_ONESHOT_BANK.json, replay/REPLAY_PLAN.json
drv prepare --block dev --waves $RP/REPLAY_PLAN.json --banks-dir $O/e2/banks $B/inputs
drv dry     --block dev --waves $RP/REPLAY_PLAN.json
# charged: discover, 24 x 1,200 = 28,800 sim-s
drv run     --block dev --waves $RP/REPLAY_PLAN.json --workers 24 --chunk 24
drv collect --block dev --waves $RP/REPLAY_PLAN.json --kind e2discover --out $RP/DISCOVER_RESULTS.json
py $B/cv/e2_replay.py select --out $O --results $RP/DISCOVER_RESULTS.json   # prints n_states and sim_s (<= 86,400)
drv prepare --block dev --waves $RP/REPLAY_SELECTION.json --banks-dir $O/e2/banks
drv dry     --block dev --waves $RP/REPLAY_SELECTION.json
# charged: branches, 3 x n_states x 1,200 <= 86,400 sim-s
drv run     --block dev --waves $RP/REPLAY_SELECTION.json --workers 46 --chunk 96
drv collect --block dev --waves $RP/REPLAY_SELECTION.json --kind e2branch --selection $RP/REPLAY_SELECTION.json --out $RP/BRANCH_RESULTS.json
py $B/cv/e2_replay.py check   --out $O --results $RP/BRANCH_RESULTS.json
py $B/cv/e2_replay.py analyse --out $O --results $RP/BRANCH_RESULTS.json
```

Notes: BUILD_NOTES §6 writes `--block e2r`, but e2_replay.py tags its waves `block='dev'` and box_driver filters by
block, so `--block dev` is required (registered setting a35_D2800_p60, panel 425e8a8d…). run_cv.sh has no replay
stage; the commands above follow its `drv`/`py` wrappers. Not verified without a run: whether the E2R_U0 wave
reuses published tuning U0 episodes (budget above assumes it does not), and `drv prepare/dry` on these waves.
Workers chosen to respect current box load; adjust to the owner's allocation.

## Suggestions (NOT computed; not registered for E2)

- Most-delayed vehicles (CVaR90), five group means and hard braking per arm: E2C_RESULTS.json `compare` already
  carries tail_cvar90, group_increase and hard_brake_vs per world, so these could be reported as labelled
  post-registration descriptives.
- Paired bootstrap of per-world executed hold vehicle-seconds (F − H20, F − S) to give the effort gap an interval.

## Files

Laptop `E:/research-dense/experiments/claim_validation_20261113_0231/e2_secondary/`:
- `e2_secondary_holm.py` dba1c34ab355b9eede27e3d0e20a573c844b301007b6b063fdb9b78c28e3a67e
- `box4/E2_ANALYSIS_REPRO.json` f1dab0f6f72827ace8d6585d24d6b8ad5b4e6d0a6185eecdad54524eb56a4ac0
- `box4/E2_HOLM.json` 19a32e8b937e4cd1b252384a1fff2ce99d8998fee89f6f7d229b133daeeebec0
- `box4/run.log` 199ad68acb78794044bb3a9bc85a7451b5e43903ecf12ada7331392d2a4e8585
- `box4/e2_secondary_holm.py` dba1c34a… (identical to the laptop wrapper)

Box 4 `/root/cv_20261113_0231/cv/out/e2/secondary_20261113_131322/`: same four files, sha256 equal to the laptop
copies (checked with sha256sum on both sides).
