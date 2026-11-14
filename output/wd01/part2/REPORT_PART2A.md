# WD-01 Part 2A — Addendum report: corrected RAW overlap, re-scored RAW C2, window safety diagnostics

Scope: ADDENDUM to part-1 T2, protocol frozen in
`REGISTRATION_ADDENDUM.md` (sha256
`52e95ac3ba391a016155344d7deab43165830856c6e213b4f21ca0edb680259a`,
recorded in `results/registration_addendum_record.json` before scoring;
the addendum file itself is untouched). Part-1 files, verdicts and
registrations are not modified. **Zero simulation** — pure numpy/torch
re-run of the part-1 rollout protocol on the part-1 states caches.

Environment: Python 3.10.20, torch 2.12.1+cpu, numpy 2.2.6, device cpu.

## 0. Guards

- Checkpoint `D:/research-dense/output/worldmodel_v3_vehicle/vehicle_dynamics.pt`
  sha256 `ded83907b8d218574d3f5359c332408f17167da055574afbe29eaecb59d4a8f5`
  verified at script start AND end (identical; see
  `results/addendum_run_meta.json`).
- Addendum sha256 re-verified inside `addendum_run.py` before any scoring;
  registration record written 2026-09-25T17:26Z, before the first
  `addendum_run.py` execution on real data (mtime of the addendum:
  2026-09-25T17:17:39Z; `results/` did not exist at registration).

## 1. Commands and wall clock

```
C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe -m pytest tests/ -q        # 5 passed in 2.00s
WD_DEVICE=cpu C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe addendum_run.py
```

- pytest wall clock: 2.0 s. addendum_run wall clock: 170.3 s (first run)
  and 155.5 s (final run, after reclassifying explained deviations); the two
  runs produced IDENTICAL verdict numbers — an extra determinism check.
- Model forward: 3,568,800 vehicle-steps in 22.46 s ≈ 158.9k veh-steps/s.
- 21 holdout runs × 7 start ticks = 147 evals per (variant, H); H ∈
  {30, 60, 90, 120} are prefixes of the same 120-step rollouts.

## 2. The part-1 RAW overlap defect and the fix

Part-1 `t2_run.raw_passthrough()` sorted vehicles by CURRENT d and then
tested `d[i] <= d[lead]` along that same sorted order — false by
construction, so every RAW overlap count was 0 and part-1's recorded
"RAW satisfies C2 (zero overlaps)" was unsupported.

Proof (acceptance criterion): `tests/test_raw_overlap_fix.py` (5 tests,
all pass) imports the UNMODIFIED part-1 function via
`wd2lib.part1_t2()` and shows (a) it returns 0 on an inverted pair, (b)
it returns 0 even when fed the inverted t+1 order as input (it never
looks at t+1), while `wd2lib.corrected_passthrough()` flags the
follower on the same pair; further tests cover the equality
(`d'_follower <= d'_leader` counts), adjacency-only semantics, lane-group
independence, and a randomized cross-check against a reference
implementation.

The corrected definition (frozen in the addendum §2): for one 1 s step
t → t+1, within a lane group, a pair ADJACENT in the d-order at t whose
order is inverted at t+1 (`d_follower(t+1) <= d_leader(t+1)`) flags the
follower; rate per 1000 vehicle-steps. Reproduction of part-1's defective
counter on the re-run rollouts yields 0 at every (variant, H) cell —
part-1's stored zeros are exactly reproduced, confirming the re-run
matches part-1's rollouts and the defect, not different data.

## 3. Re-scored overlap / C2 table

| variant | H | part-1 overlap/1000 (defective) | corrected overlap/1000 | overlap events (corrected) | CONSTRAINED overlap/1000 | corrected RAW C2 |
| --- | --- | --- | --- | --- | --- | --- |
| RAW | 30 | 0.00 | 9.02 | 4026 | 15.76 | **FAIL** |
| RAW | 60 | 0.00 | 9.38 | 8369 | 12.19 | **FAIL** |
| RAW | 90 | 0.00 | 9.85 | 13177 | 10.46 | **FAIL** |
| RAW | 120 | 0.00 | 9.13 | 16292 | 8.61 | **FAIL** |

(Cap-violation vehicle-steps = 0 and max speeds are unchanged from
part-1; the corrected C2 fails solely on the overlap-rate clause,
threshold 1/1000.) CONSTRAINED C2 is unchanged from part-1 (integrator
flags; FAIL at every H, identical counts 7031/10876/13997/15356 = part-1's
stored values to the event).

Findings:

1. Part-1's "RAW satisfies C2 at every horizon" is **superseded**: the RAW
   rollout inverts an adjacent pair about once per 102–111 vehicle-steps
   and fails the registered ≤1/1000 clause by ~9× at every H.
2. At H=30 the CONSTRAINED integrator flags ~1.75× MORE overlaps than RAW
   (15.76 vs 9.02) — the no-pass-through reset fires on the same
   sub-metre order inversions RAW silently commits; by H=120 the two
   rates converge (8.61 vs 9.13). The integrator does not prevent the
   inversions that drive the rate; it repairs the state (and its resets
   inject hard braking, see §5).
3. Part-1's headline "no registered horizon passes" is unchanged in
   direction: C1 (recall) was already the binding failure, and C2 now
   fails for RAW as well under the corrected metric. The part-1
   recommendation needs no revision; its stated rationale for RAW's C2
   pass does.

## 4. Deviation check: re-run vs part-1 stored aggregates

`results/deviation_check.json` compares the re-run against part-1's
`t2_summary.json` verdicts per (variant, H).

- **Reproduced exactly** (no deviation in any of the 8 cells):
  vehicle_steps, comparable_vehicle_steps, capviol_steps (0), negspeed_steps
  (0), max_speed_mps (float-exact, tolerance 1e-9), CONSTRAINED
  overlap_events, and the RAW defective-geo reproduction (0). The rollouts
  are bit-for-bit the part-1 rollouts on this CPU build.
- **Explained deviation 1 (registered exception)**: part-1's RAW overlap
  column (0) is the defective counter's output; superseded by §3.
- **Explained deviation 2 (measured, C1 scope — reported, not
  re-registered)**: part-1 pooled its C1 event universes into dicts keyed
  by vehicle id (`pt/tt_/pb/tb_` dict comprehensions in `t2_run.py`), so a
  vehicle contributes AT MOST ONE event per window per type. Part-1's
  stored `ttc_truth/ttc_pred/brake_truth/brake_pred` are therefore
  distinct-vehicle counts, not vehicle-seconds (e.g. RAW@H30 ttc_truth
  1893 stored vs 3976 vehicle-seconds recomputed; ttc_pred 1932 vs 10054).
  The recomputed vehicle-second totals are in
  `deviation_check.json → explained_diffs`. Part-1's C1 recall/precision
  verdicts stand as registered; the direction of the bias (denominators
  undercount persistent events) is noted for any future re-registration.

## 5. Contract-level window safety diagnostics (DIAGNOSTICS ONLY — no pass/fail)

Per eval (run × t0 × H × variant): predicted vs truth over the SAME frozen
t0 vehicle set, 1 s step; truth leaders from the full truth snapshot
(part-1 truth-event convention), predicted leaders from rollout states.
Metrics: `ttc_lt3_exposure_s`, `ttc_lt1p5_exposure_s` (vehicle-seconds
with closing TTC below threshold), `hard_brake_s` (transitions with
dv < −3 m/s over 1 s), `min_ttc_s` (null when nobody is closing), and the
three `any_*` booleans. Pooled over 147 evals per (variant, H):
window-level recall/precision of the booleans; Pearson/Spearman of
exposures and min-TTC; mean bias (pred − truth); fraction of evals with
predicted exposure ≥ truth. Full tables in `results/diag_summary.json`;
per-eval rows (1176) in `results/diag_per_eval.csv`.

| variant | H | any<1.5s rec/prec | any<3s rec/prec | any-brake rec/prec | <3s bias (ge, r, ρ) | <1.5s bias (ge, r, ρ) | brake bias (ge, r, ρ) | minTTC bias (r, ρ) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAW | 30 | 0.95/0.89 | 0.95/1.00 | 0.89/1.00 | +42.6 (.87, .81, .80) | +32.1 (.95, .82, .79) | +1.0 (.56, .55, .55) | +0.81 (.88, .55) |
| RAW | 60 | 0.98/0.96 | 0.92/1.00 | 0.91/1.00 | +76.9 (.78, .71, .75) | +62.5 (.92, .71, .77) | +1.2 (.65, .56, .55) | +1.06 (.64, .56) |
| RAW | 90 | 0.96/0.96 | 0.92/1.00 | 0.89/1.00 | +108.4 (.77, .66, .74) | +93.8 (.86, .66, .73) | −3.3 (.58, .53, .48) | +1.23 (.62, .40) |
| RAW | 120 | 0.96/0.96 | 0.92/1.00 | 0.89/1.00 | +133.6 (.79, .64, .71) | +114.8 (.89, .64, .72) | −0.4 (.61, .48, .42) | +1.24 (.68, .39) |
| CONSTRAINED | 30 | 0.95/0.89 | 0.95/1.00 | 0.97/1.00 | +37.2 (.88, .74, .70) | +33.4 (.95, .76, .73) | +9.0 (.78, .41, .47) | +0.84 (.88, .43) |
| CONSTRAINED | 60 | 0.98/0.96 | 0.92/1.00 | 0.99/0.99 | +63.8 (.78, .74, .72) | +59.5 (.93, .74, .73) | +16.7 (.82, .48, .54) | +1.09 (.64, .51) |
| CONSTRAINED | 90 | 0.96/0.96 | 0.92/1.00 | 0.96/0.99 | +77.6 (.80, .74, .75) | +76.7 (.86, .74, .74) | +16.6 (.71, .51, .53) | +1.26 (.63, .42) |
| CONSTRAINED | 120 | 0.96/0.96 | 0.92/1.00 | 0.96/0.99 | +86.0 (.78, .72, .72) | +83.4 (.84, .72, .73) | +17.9 (.69, .49, .48) | +1.27 (.69, .36) |

(ge = frac pred exposure ≥ truth; r = Pearson; ρ = Spearman; bias in
vehicle-seconds per window, min-TTC bias in s. min-TTC correlations over
evals where both sides finite — n = 147 everywhere here.)

Reading (diagnostics, not verdicts):

1. **Window-level flag fidelity is high and degrades slowly**: the three
   booleans sit at 0.89–0.99 recall and 0.89–1.00 precision in BOTH
   variants at every H. The part-1 C1 failure is a co-timing failure, not
   a window-occurrence failure — the model predicts roughly WHETHER a
   window contains danger, not exactly WHEN.
2. **Exposure is systematically over-predicted**: mean bias grows with H
   (RAW +42.6 → +133.6 vehicle-s of TTC<3s at H30 → H120; 0.77–0.89 of
   evals have pred ≥ truth), with rank correlation staying ≈ 0.64–0.82.
   The rollout is "busier" than truth in near-closure states, consistent
   with the corrected overlap finding (order inversions accumulate).
3. **The CONSTRAINED integrator manufactures hard braking**: mean brake
   bias +9.0 → +17.9 s at H30 → H120 (RAW: ≈ ±3 s), and its brake
   correlations are the weakest cell in the table (r ≈ 0.41–0.51). This
   quantifies the reset cost that the corrected C2 already flags.
4. **min-TTC tracks at short horizons**: r = 0.88 at H30 for both
   variants, falling to ≈ 0.6–0.7 at H120; the model ranks a window's
   worst closing event well at H30.

## 6. Artifacts

| artifact | sha256 |
| --- | --- |
| REGISTRATION_ADDENDUM.md | `52e95ac3ba391a016155344d7deab43165830856c6e213b4f21ca0edb680259a` |
| vehicle_dynamics.pt (start = end) | `ded83907b8d218574d3f5359c332408f17167da055574afbe29eaecb59d4a8f5` |
| results/registration_addendum_record.json | `f5cded0824153a39bd06c9db898c585a803e8f2ce93518de3ee0a8a981191541` |
| results/overlap_addendum.json | `f7c2f82ef3ec7bb69c4e7a83ef635e2b4335d9dbfa336f8450f8769b3968c8bd` |
| results/overlap_addendum.csv | `9fec219cfc16f8e168be33efb737e71ce8df9543e1b8933df53a6a634c2ae7c4` |
| results/deviation_check.json | `c8a5143ceda3b81d02b56f2cb46fa1413f74f23ccb694c6be4b02a3b1511b4f2` |
| results/diag_summary.json | `ee7695ae31461efee4773245e3ca46c1b6171fb8ce5a47e39a9200d79cf214c6` |
| results/diag_per_eval.csv | `e3454047e90f33d82f95fb419e3852add7e58f168f37a754fc75311d37c623a4` |
| results/addendum_run_meta.json | `e38dc874c2c78e445c3dee015c1565b4db52267fb7228ec400cefded33c0d0c0` |

(SHA256SUMS.txt in `results/` carries the same receipts.)

## 7. Limitations

- 1 s step: TTC<1.5 s discrimination remains coarse (addendum §3; 0.5 s
  step is a Part-A deliverable).
- Frozen t0 vehicle set: truth-side exposure censors vehicles that exit or
  change lane group (they stop contributing truth-side steps); the
  predicted side keeps them for the whole window. Both sides use the same
  frozen set, per the addendum.
- Diagnostics consume per-window aggregates; they do not rehabilitate
  part-1's co-timed C1, and they carry no pass/fail.
- The corrected RAW overlap metric is gap-agnostic by frozen definition;
  pairs separated by more than GAP_MAX can still be flagged if their
  order inverts — the metric measures ordering violations, not collision
  proximity.

## 8. Git

Committed on branch `zcode/wd01`, parent `589be1a`, paths under
`output/wd01/part2/` only (forced add; `output/` is git-ignored).
Commit subject: "wd01 part2A addendum: corrected RAW overlap counter
(unit-tested), re-scored RAW C2 (FAIL all H, 9.0–9.9/1000), window-level
safety diagnostics per H × variant (diagnostics-only); registration
addendum frozen+hashed before scoring; checkpoint sha256 verified
start=end; re-run reproduces part-1 step-indexed aggregates exactly
(deviations explained in results/deviation_check.json)". The commit
itself (and `results/SHA256SUMS.txt`) is the receipt; the working tree
outside `part2/` is untouched.
