# WD-01 — pause / resume state (PAUSE received 2026-09-26 17:38 UTC)

Owner restarts the PC to attach the RTX 3060. Nothing new is dispatched.

## Integrated on branch zcode/wd01 (worktree D:/rd-wt/wd01, HEAD d81c249)

| part | status |
| --- | --- |
| p1 | identity PASS; drift study; WD-01-DEV-1 deviation record (LEDGER_DRAFT_WD01_DEVIATIONS.json) |
| p2A–2C | overlap fix, window diagnostics, 0.5 s native dataset (26 roots) |
| predict module | contract v1.1 + v2 (PREDICT_CONTRACT_v1.md §1–§9 incl. §8.14/§8.15; PREDICT_CONTRACT_v2.md sha256 db475371…0775); safety_veto_v2 aligned to PL-01, with flags `native_collision_vetoes` (default True) and `window_must_not_increase` (default False), both pending the owner; 259 tests |
| p4 | delay library (contract §4.1) and a synthetic-only Part B skeleton |
| p3c–p3k | native Part A: structured SUMO emulator (IDM + fitted dawdle + prior vType + LC hazard + merge dynamics, engine kind `idm_merge`); holdout block; WD01_DATA_ROOT override |
| design | DESIGN_PARTS_A_B.md, DESIGN_PARTB_V2.md (§2.2 owner decisions, G-1), NOTE_PARTB_VS_DREAMER.md; gate v2 + amendment 1 in part3f/ |

Key validation numbers (p3k, H=30 primary): physics-delay bias −0.02 veh·s per
window; CRPS 38.8 vs L0 MAE 293. Gate v2 FAIL on (b) LC +30%, (c) coverage 13%,
and (d) marginal (per-horizon reading).

## In flight at pause

- **p3l** (ZCode d37600f4, worktree D:/rd-wt/wd01f, branch zcode/wd01-p3l):
  - TASK A done: the follower-deceleration term binds after a fix; b_accept 3.0 is best, at the grid edge.
  - TASK B done: per-member lc_scale 0.0005–0.003 and dawdle_scale 0.8–1.5.
  - TASK C (TRAIN-internal spread calibration) and TASK D (validation ladder L0/L4/L5/L5c, gate v2 at H=30/60) are running.
  - Hard time limit about 19:08 UTC. Left to finish.
  - **Resume:** if it finished, review REPORT_P3L.md at code level, then merge zcode/wd01-p3l into zcode/wd01. If it was interrupted, re-dispatch TASKs C–D in a fresh worktree, reusing results/fit_p3l.json.
- The veto follow-up (17aee4a5) is done and merged (d81c249).

## Next steps after the restart (in design order)

1. Set up the CUDA torch environment. Check `WD_DEVICE` resolution on the RTX 3060.
2. Finish native Part A gate v2 (p3l results; spread/LC).
3. Part A action conditioning: controller emulators (A2) for damping, PI and wall, validated on chunk-1 campaign branches.
4. Part B round 0: V0 native value (TD(λ), τ-conditioned) on the GPU, using chunk-1 native bases/branches. Confirm with the reconstructor that runs reach t0 + 600 s.
5. Holdout second use (DEV-1 option a): waits for the owner's review. The one-command registration prep is not done yet.

## Owner decisions pending

- `native_collision_vetoes` (main discussion recommends option A: a native collision does not veto a safe plan).
- `window_must_not_increase` (main discussion recommends True).
- Holdout second use (WD-01-DEV-1).

## Housekeeping

- Worktrees: wd01 (integration), wd01b/c/d/e/g/h (merged; can be removed after the owner's OK — never with --force), and wd01f (p3l in flight).
- Rule: each ZCode task goes to a FRESH worktree, because a workspace reused its executor thread and degraded quality.
- Backups: D:/research-dense-backups/wd01/{preexisting_20260925, part1, part3, part3d}, each with SHA256SUMS.

## Chunk-1 data facts (reconstructor, 2026-09-27)

- Runs: D:/research-dense/runs/campaign_stage1_chunk1/<world>__<arm>/. Plan:
  D:/research-dense/output/campaign_stage1_chunk1/results_plan_wm44/WORLD_PLAN.json
  (authoritative). Contract: output/campaign_stage1_chunk1/DATA_CONTRACT.md.
- Completion: use WORLD_PLAN + RUN_RECEIPT.json (exit_status complete): 547/560.
  INDEX.json covers bases only. Ignore *_failed_branches_attempt1.
- Bases cover [0, t0+600]; every branch covers (t0, t0+600]. physics_steps.jsonl
  logs every vehicle at 0.5 s. The t0 row is in
  saved_state/STATE_MANIFEST.json physics_row_at_t0.
- Branch CAV internals:
  - dispatcher_rows.jsonl: per-CAV action, plan_id, damping_target_speed_mps,
    pi_running_target_mps;
  - suffix_commands.jsonl: issued_target_speed_mps, owner, fallback_reason;
  - switch_schedule.json.
  There are NO walls in chunk 1. BASE runs do not log internals.
- Split by root in WORLD_PLAN worlds[].split: TRAIN 49 / CALIBRATION 7 / HOLDOUT 14.
- ALL of chunk 1 is PRE-FIX: CAV speedMode 31 at every penetration; collisions
  concentrate at 40% penetration (output/campaign_stage1_chunk1_launch/COLLISION_TALLY_FINAL.txt).
  Future data records cav_speed_mode = 27.
