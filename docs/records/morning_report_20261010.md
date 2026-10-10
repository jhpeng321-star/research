# Morning report — 2026-10-10 (written 08:40; updated as training/scoring finish)

## 1. Round-3 training (gpuA)
- **Finished 08:56 (rc=0).** Results copied to `runs/active/WM-S-R3-20261010/box/gpuA/wms_train_r3_final.tar` (sha256 `939608fb…`, matches the box) and extracted next to it. **gpuA is shut down** (ssh no longer logs in) — please confirm "stopped" in the AutoDL console.
- Run `WM-S-R3-20261010`, `W=/root/autodl-tmp/wms_train_r3/r3_main_a2`.
- First launch (`r3_main_a1`) was rejected by the trial-run gate (false alarm: the two new terms start near zero, so they "grew" >10x). The gate was fixed (commit 21219a1) and the run relaunched once. Details: `runs/active/WM-S-R3-20261010/REGISTRATION.md`.
- Recipe: C1 recipe + pack_v3 (+294 windows) + soft residual bound (4 m/s²) + residual-size penalty + lane-change-rate loss + best-long-horizon checkpoint + 60 s driver history + braking losses (no hard cap). Merge calibration ON (C1 had it off) — recorded deviation.
- Switching pack (only existed on the boxes): copied to `runs/active/WM-S-R3-20261010/box/gpuA/wmn1_switching_train` — 39/39 files sha256-verified.
- **First result (quick, 20 unseen-wall states, standard set):** round 3 is **about level with C1, not better yet**.

  | horizon | round 3 | C1 | paired diff (neg = R3 better) |
  |---|---|---|---|
  | 10 s | 1.89 m | 2.29 m | −0.23 [−0.44, 0.13] |
  | 30 s | 5.51 m | 5.08 m | +0.38 [−1.46, 1.36] |
  | 60 s | 14.80 m | 11.96 m | **+1.71 [0.55, 5.44]** (worse) |
  | 120 s | 110.8 m | 111.8 m | +10.4 [−23.9, 48.3] (no clear difference) |

  What worked: the learned correction stays tame over 120 s (avg 1.6 vs C1's 7.5 m/s²) and the 120 s lane-change drift halved (4.5% vs 8.5%). What didn't: round 3 now drives **too slowly** (speed bias −1.2 m/s vs −0.3) — the braking losses over-correct — and lane changes in the first 60 s are too frequent (2.1% vs real 1.2%).

- **Final verdict (full sets + real-history set): round 3 does NOT beat C1 → C1 stays the current best.**
  - Full standard sets (67 + 53 wall states): 10 s slightly better (−0.14 m), 30 s equal, 60 s 13.4 vs 8.7 m (+1.5, CI crosses 0), 120 s 132 vs 100 m (+39, borderline worse); 2 blow-ups vs 0. The best-long-horizon checkpoint is a bit better (0 blow-ups, 120 s +24) but still not better than C1.
  - Real-60 s-history set (20 states): 60 s 28.3 vs 16.8 m (+4.4 [0.2, 16.7], worse), 3/19 blow-ups vs 0 → **the 60 s history encoder did not help** in this form.
  - Keep: soft bound + residual penalty (tame 120 s corrections). Suspected harms: braking losses (over-braking), history encoder, too-frequent early lane changes.
  - **Suggested next round (one model):** C1 recipe + pack_v3 + soft bound + residual penalty + stronger lane-rate loss + best checkpoint; **no braking losses, no history encoder**. Recorded in `docs/WM_CHANGE_LOG.md`. Reports: `E:/rd-wt/ws-r3-score2/_out/r3_score/{r3_quick_h60,r3_full_h60,r3_quick_h60real}/ROUND_SCORE.md`.

## 2. Clean rewrite (one modular package `src/wms/`)
- Branch `wms/clean-int` (worktree `E:/rd-wt/ws-clean-int`). Layout: `src/wms/{core,config,cli,physics,model,data,train,eval,legacy}`, one TOML config per run (`configs/r3.toml`, `c1fast.toml`, ...), `python -m wms train|preflight|eval|score|fetch --config ...`.
- Proven equal to the old code: unit + golden **518 passed** (re-run by me), equivalence **76 passed** (incl. C1's 11.9576 m on the 20-world sample), golden reference outputs match (windows 911/151 exact, forwards 1e-6, losses 1e-5, rollouts 1e-5). Real-data trial run works end to end.
- Old WM-S code, 15 stray root files, 231 old `_out` records and 11 dead tests moved to `archive/` with PATH_MAPs (nothing deleted).
- Remaining gaps: a few placeholder modules (eval walls/decide helpers, closed-loop CAV mode); planner/sim tests that already failed before (missing `output/planner_v1`, MIG-04 deletions); `tools/holdout_guard.py` name clash with the planner's guard → plan renames the new one to `tools/data_guard.py`.
- Full details: `E:/rd-wt/ws-clean-int/docs/INTEGRATION_REPORT.md`, design `docs/ARCHITECTURE_V4.md`.

## 3. Making it your single clean work tree (needs you; ~10 min)
Exact steps, pause list and rollback: `tmp/morning/SWITCH_OVER.md`. In short: pause other sessions → commit the current dirty tree to `checkpoint-pre-clean-20261010` (never `git add -A`: `sealed/` is untracked and not ignored) → resolve 9 real conflicts (8 shared records + one test) → switch to the clean line. I will walk you through it.

## 4. Data on Google Drive
- Uploaded + verified overnight: ~249 GB of source (outputs, shards_v2/v3, campaigns chunk2/chunk4/walls-pilot/truearms, research-github, ...), plus earlier uploads.
- Still uploading: `runs/active` (62 GB → ~10 GB compressed) and `runs/campaign_stage1_chunk3` (72 GB → ~4 GB) — ETA ~09:30. Worktree backups: **all 211 finished worktrees verified on Drive** (72 as compressed archives, round-trip checked).
- Measured: ~8 MB/s total on the network adapter; Drive per-stream 1–7 MB/s. Drive API quota (shared rclone client_id) caused rate-limit errors — creating your own Google API client_id for rclone would remove that risk.
- Tiny probe files can be deleted: `gdrive:research-archive/_dataops_probe/`.

## 5. Clean-up (you run these; I am not allowed to delete)
Tools in `tmp/morning/` (regenerated 08:31 from the latest receipts). Every item is re-checked against its Drive copy (size + md5) right before removal; junction-safe; journalled.

Stage 1 — remove Drive-verified items. Report-only run at 09:3x: **571 eligible items; would free ~112 GiB on D: and ~122 GiB on E:** (includes the old worktrees, now all backed up). `-Execute` asks you to type DELETE first:
```
powershell -NoProfile -ExecutionPolicy Bypass -File E:/research-dense/tmp/morning/MORNING_CLEANUP.ps1 -Stage 1
powershell -NoProfile -ExecutionPolicy Bypass -File E:/research-dense/tmp/morning/MORNING_CLEANUP.ps1 -Stage 1 -Execute
```
Stage 2 — copy the keep-local data (packs, sidecars, eval states, checkpoints, switching pack; ~74 GiB) into `E:/research-dense/data_cache/` with sha256 checks (needs Stage 1 first for space); Stage 2b removes the old copies only after verification. Stage 3 prints the switch-over git steps. Run Stage 1 again after the remaining uploads finish to clear the rest. Full numbers: `tmp/morning/CLEANUP_SUMMARY.md`.

What stays outside `E:/research-dense` after all stages: only items still uploading and owner decisions (below).

## 6. Decisions for you
1. Sealed / CAL / HOLDOUT / chunk-5 / INC1 data (`E:/research-dense/sealed`, ~62 GiB): upload as closed, hashed, never-opened archives to a restricted Drive folder? (recommended)
2. Local cache size: ~74 GiB of hot data (pack_v3 49, dspeed 12, damping 5, switching 7, states/sidecars/checkpoints) — OK?
3. Optional clean-up flags (`-RemoveAroundProtected`, `-AllowSensitiveDup`, ...) — explained in `CLEANUP_SUMMARY.md`.
4. Create the private GitHub repo (you create it and the credentials; I prepare the push).
5. `tools/holdout_guard.py` collision → rename the new one to `tools/data_guard.py`?

## 7. Findings overnight (in `docs/WM_CHANGE_LOG.md`)
- The test states can only have ~5 s of history (runs start at their origin); a second test set with real 60 s history was built (120 + 116 wall states, 38 controls).
- C1's transformer barely attends to the car ahead (6%); 60% goes to adjacent lanes, 42% to cars behind. Candidate future change: a role-structured neighbour network.
- C1 also drifts over 120 s (lane-change rate 8.5% vs 1.6% real) — the round-3 additions target exactly this.
