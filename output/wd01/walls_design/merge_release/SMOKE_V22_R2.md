**Status:** FROZEN 2026-10-01 (owner: "fix the test, approve J4b").

- The law is unchanged (e7fd3536). This is a POST-HOC, DISCLOSED,
  fixture-only amendment after the R1 failure.
- Fail conditions: member_left_network, a member past the merge start while
  in the wall (RC4b), a missing hold_cap/receipt, or a collision/teleport
  (RC6).
- A registered safety abort does not fail. All 3 runs execute, and ≥ 1
  merge_release is required.
- SMOKE_V22_R2.json is authoritative.

# SMOKE_V22_R2.md — declared mechanics RE-smoke fixture R2, wall law v2.2 'merge-release'

**DRAFT — pending owner decision. Nothing here is frozen or sent; the
reconstructor runs it (the proposer never runs SUMO).** Machine-readable
fixture: `SMOKE_V22_R2.json` (same directory). 2026-10-01.

## 0. Why a re-smoke (R1 failure, one paragraph)

R1 (`SMOKE_V22.json/.md`, runs root `runs/campaign_walls_v22_smoke`) failed
at run 1: the wall assembled at t=244 and was aborted at t=328 by the
frozen v2.2 rear-end defence (`non_member_rear_end_risk:ramp.29`) with the
lead at s_road 2552.2 m — **92.0 m before the merge start** — and the R1
MC3 rule "any safety_abort FAILS" + stop-on-fail ended the fixture before
any `merge_release` (runs 2–3 never launched). The abort is the safety
design working on a legal traffic state, not a law defect. Full timeline:
`SMOKE_V22_R1_DIAGNOSIS.md`. R2 re-declares the SAME budget on a NEW
write-once runs root with the WM-recommended abort-tolerant rule.

## 1. What changed vs R1

1. **Abort rule.** A registered safety abort (any frozen v2.2 reason —
   `member_frame_missing`, `member_position_unavailable`,
   `member_span_exceeded`, `non_member_rear_end_risk`, `assembly_timeout`,
   `lane_coverage_violation`, `assembly_past_release` — stripped of any
   `:<vehicle>` suffix) is RECORDED and fails NOTHING.
   `member_left_network` still FAILS its run (a member escaped control),
   and any abort reason OUTSIDE the frozen list fails its run (unregistered
   law behaviour).
2. **No stop-on-fail.** All 3 runs always run; failures are reported per
   run.
3. **Pass rule.** Per-run RC1–RC4 (below) for EACH run, plus at least ONE
   observed `merge_release` across the runs with `capped_at_merge` logged
   (RC5).
4. **Logging fix deployed.** `patches/wall_driver_v22_logging.diff`
   (behaviour-neutral, applied AFTER `patches/wall_driver.py.diff`):
   `hold_cap.a_priori_effective_hold_s` is computed AT ASSEMBLY
   (`min(H, (2644.2 − lead_s_road_at_assembly_m)/v_wall)`), not from the
   release-tick position — the R1 run's row claimed an a-priori cap of
   5.798 s while the hold ran 84 s; the assembly-position value is ≈ 89.8 s
   (see the diagnosis §4).
5. **New runs root** `runs/campaign_walls_v22_smoke_r2` (write-once; the
   frozen R1 root is untouched).

Unchanged: the law (v2.2 merge-release, release point 2644.2), the budget
(3 × 300 sim-s), the launcher (`run_wall_branch.py`, no `--hold-override-s`),
and "smoke runs are NEVER data".

## 2. Deployment (reconstructor, in a no-campaign window)

1. Verify the target files hash to the recorded PRE (v2.1) shas in
   `PATCH_SHAS.json`, then apply, from the research root (LF unified
   diffs; patched LF sha256 must equal the `post` entries):
   - `patches/common.py.diff` → `output/campaign_walls_v1/common.py`
   - `patches/wall_request.py.diff` → `output/campaign_walls_v1/wall_request.py`
   - `patches/wall_driver.py.diff` → `output/campaign_walls_v1/wall_driver.py`
   - `patches/wall_group.py.diff` → `traffic_shepherd/wall_group.py`
   - `patches/wall_driver_v22_logging.diff` → `output/campaign_walls_v1/wall_driver.py`
     (AFTER the law patch; result sha
     `ef01bfcf652eeee5439a705119acb7e6e75c4414d3ab0aa0748b0f11ca6590e3`,
     `PATCH_SHAS.json` → `logging_patch`)
   OBSERVED 2026-10-01: the live
   `E:/research-dense/output/campaign_walls_v1/wall_driver.py` diverged
   after R1 run 1 (now `ca478219…`, not the recorded pre
   `6810330d…`) — restore/verify the recorded base state BEFORE patching.
   (`sensing.py` needs NO change.)
2. Finalize the J4B v2.2 registration (write-once) at
   `output/campaign_walls_v1/results_plan/WALLS_PILOT_REGISTRATION_V22.md`
   from `REGISTRATION_WALL_V22.md` + the launch blocks of
   `REGISTRATION_J4B_DRAFT.md`. The J4 registration file stays frozen.
3. `runs/campaign_walls_v22_smoke_r2` must NOT exist (write-once).

## 3. Selection (a priori, J4 geometry only, no outcomes)

TRAIN worlds only; registered hold draw 240 s (the MERGE-START CAP — not
the draw — ends the hold); assembled in J4; **upstream rule**
`lead_s_road_request <= 2644.2 − D_min − 300 m` (`D_min = 70·v_wall`);
predicted merge-start arrival fits the 300 sim-s horizon with ≥ 100 s
slack; ranked by distance-to-merge at request DESCENDING (farthest
upstream first), distinct worlds. 13 rows survived both constraints; the
top three distinct worlds:

| run (world / arm) | p | v_wall (J4) | lead s_road @ request | D_min | passes upstream rule by | predicted capped hold | predicted arrival (rel t0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| w8661m2600p10t240 / W-mid-early | 0.1 | 22.52 m/s | 42.11 m | 1576.34 m | +725.75 m | 101.52 s | 175.52 s |
| w8697m2700p40t240 / W-mid-early | 0.4 | 14.16 m/s | 103.31 m | 990.89 m | +1250.00 m | 178.45 s | 179.45 s |
| w8700m2600p40t420 / W-mid-late | 0.4 | 21.33 m/s | 175.92 m | 1493.05 m | +675.23 m | 106.62 s | 175.62 s |

## 4. Exact launcher command lines (cwd = `E:/research-dense`)

Run the three commands SEQUENTIALLY (laptop smoke; one fresh SUMO process
per run inside the child). ALL THREE run even if an earlier run fails a
check:

```text
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe output/campaign_walls_v1/run_wall_branch.py _exec_wall_branch --run w8661m2600p10t240__w-mid-early --plan output/campaign_walls_v1/results_plan/WORLD_PLAN.json --runs-root runs/campaign_walls_v22_smoke_r2 --registration output/campaign_walls_v1/results_plan/WALLS_PILOT_REGISTRATION_V22.md --bases-root E:/research-dense/runs/campaign_stage1_chunk4 --horizon-ticks 300

C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe output/campaign_walls_v1/run_wall_branch.py _exec_wall_branch --run w8697m2700p40t240__w-mid-early --plan output/campaign_walls_v1/results_plan/WORLD_PLAN.json --runs-root runs/campaign_walls_v22_smoke_r2 --registration output/campaign_walls_v1/results_plan/WALLS_PILOT_REGISTRATION_V22.md --bases-root E:/research-dense/runs/campaign_stage1_chunk4 --horizon-ticks 300

C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe output/campaign_walls_v1/run_wall_branch.py _exec_wall_branch --run w8700m2600p40t420__w-mid-late --plan output/campaign_walls_v1/results_plan/WORLD_PLAN.json --runs-root runs/campaign_walls_v22_smoke_r2 --registration output/campaign_walls_v1/results_plan/WALLS_PILOT_REGISTRATION_V22.md --bases-root E:/research-dense/runs/campaign_stage1_chunk4 --horizon-ticks 300
```

## 5. Checks (field-by-field in `SMOKE_V22_R2.json`)

- **RC1** law version is v2.2 everywhere it is logged (every run).
- **RC2** ZERO `member_left_network` (every run); other registered aborts
  recorded only.
- **RC3** exactly one `hold_cap` row (requested 240 s, cap point 2644.2) +
  the full receipt contract (`complete`, one `loadState`, `smoke_cap`,
  600 physics rows) — per run.
- **RC4** no release or abort PAST the merge start: a `merge_release` sits
  in `[2644.2, 2644.2 + v_wall)` (one control tick of travel); any other
  terminal outcome has the lead `<= 2644.2` — per run.
- **RC5** at least ONE run ends `merge_released` with `capped_at_merge`
  true and no member on the merge/exit edge before the release tick.

## 6. Pass rule

ALL of RC1–RC4 for EACH of the three runs AND RC5 across the runs.
Registered safety aborts never fail anything; a per-run FAIL does not stop
the remaining runs. The J4B launch gate (registration + package +
dependency digests) binds only on a recorded PASS.
