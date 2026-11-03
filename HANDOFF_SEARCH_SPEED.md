# HANDOFF: search speed-up / learned search (branch c4/plan-proposer, 2026-11-02 ~22:45 +08)

These are the rules for all work here:
- CPU only: no SUMO, no GPU.
- TRAIN worlds 986000-986071 only.
- Never open sealed/holdout/heldout/cal_split/chunk5/j4b/inc1 paths.
- Remote hosts are read-only (ssh `find`, `tar -c`, `cat` only).
- Do not touch E:/agent-work/c4-coopnative.
- No git reset, clean or stash drop. Never kill processes by image name.

Activation of any speed-up is owner-approved only once it passes its registered check, and MAIN does the switch.

## Status of each track
- **Plan proposer (`tools/coopmwm/plan_proposer.py`; prereg `docs/paper_results/PLAN_PROPOSER_PREREG.md`):** FAIL.
  - The warm start is worse than plain_alt: -0.28 [-0.37, -0.22]. Widening or shrinking it gives -0.20 to -0.23.
  - It is too narrow (sample spread x0.55 of the plain distribution) but has not collapsed (`LEARNED_SEARCH_DIVERSITY.json`, `LEARNED_SEARCH_TRACKA.json`).
- **Learned screen (`learned_screen.py`; prereg `LEARNED_SCREEN_PREREG.md`, frozen at commit 70d4812a):** FAIL.
  - Held-out: q_LS 0.19 vs plain_alt 0.98. On real WM labels the top-16 is worse than a random 16.
  - TRAIN-only CV (`learned_screen_cv.py`): Spearman 0.13-0.16 for ridge, gradient-boosted trees and the net. A learned pre-filter is no better than random dropping. Dropped.
  - **Noise ceiling:** among finalists, K4 screen vs K8 final Spearman is 0.44, and top-1 agreement is 0.51.
- **Budget-reduced plain, v1 (`BUDGET_SEARCH_PREREG.md`, `budget_search_eval.py`):**
  - Quick read: **it2 (2 iterations, full population) is best at 1.43x** (WM futures x0.70, about 490 vs 698 s per search). Held-out q8 is +0.24 [-0.005, 0.75]; dev is -0.02 [-0.27, 0.40].
  - **pop50 and pop50_it2 are worse** on dev: q8 -0.67 and -0.29, both with CIs below 0. pop50 also loses elite clusters.
  - Held-out had 27 searches / 11 worlds, which is NOT ENOUGH DATA (40 needed).
  - The v1 normalisation is unstable, so v1 margins are too wide. This is why v2 exists.
- **v2 registration (`BUDGET_SEARCH_PREREG_V2.md`, commit 29bb908e): RUN and TOPPED UP to the registered
  minimum (2026-11-03; report `SEARCH_SPEED_REPORT.md` §1, ranked `SEARCH_SPEED_RESULT.json`).**
  - What it registers:
    - configs pop50, it2, pop50_it2, K2, K2_pop50, it2_K2;
    - quality in pool-sd units (`unit='pool_sd'`);
    - margin = plain vs plain_alt seed noise;
    - primary metric = best of the final top-8.
  - Noise alpha = 0.5, chosen by the registered dev-only rule (closest simulated screen-vs-final Spearman among
    finalists to observed 0.44; 0.35→0.506, 0.5→0.457, 0.7→0.353; `_plan_proposer_cache/ls/calib.json`).
  - **Result (56 evals / 17 held worlds — ENOUGH DATA): K2 PASSES (rank 1, 1.81x wall), it2 PASSES (rank 2,
    1.43x); it2_K2 FAILS the q8 CI gate by 0.001 at 2.48x; every pop50 variant FAILS. Activation is
    owner-approved only (MAIN).**
- **State triage (v2 B; `docs/paper_results/TRIAGE_SCORE.json`):** TOPPED UP 2026-11-03: dev 153 searches refit;
  held 60/18 ENOUGH DATA — top-50% keeps 0.695 [0.350, 0.882] of the saving (1.26x per GPU-second), top-33% keeps
  0.533 (1.43x). Registered PASS rule still FAILS (CI-lo 0.350 < 0.5) — ordering only.
- **Per-episode warm start (v2 C):** coded + tested (commit 237cbd82); refreshed 2026-11-03: 103 raw request pairs
  but still only 6 usable full-pool pairs < 20 gate → NOT EVALUATED (exploratory: held 4, dev 2). Needs the
  collector to log full-pool receipts on BOTH sides of a pair.
- **Previous session's final commits (before freeze) already contained a provisional n=27/n=26 read; the numbers
  above supersede them.**
- **Watcher** (`tools/coopmwm/learned_search_watch.py`, PID 35992, log `E:/agent-work/_plan_proposer_cache/watch.log`):
  - It does an hourly and per-B-update refresh of the proposer checks. It writes `docs/paper_results/LEARNED_SEARCH_ACTIVATION_READY.json` on a PASS, and exits at 2026-11-03 10:00 +08.
  - It runs pre-noise-ceiling code. Restart it for the new report fields and the B-REPLAY bank discovery (`BANK_STATUS.json`).
  - It is my own process: stop it by PID only.

## Data and cache
- Local read-only copies live in `E:/agent-work/_plan_proposer_cache/`:
  - `cpuQ/home/ubuntu/coopmwm20261102_warm1/{requests,responses}`: responses embed `result.proposer_receipt` full pools.
  - `gpuA|gpuR/root/*/proposer_receipts`.
  - Checkpoints and `ls/` working files.
- Load with `learned_search_report.load_cache(cache)`, which keeps TRAIN worlds only and dedupes by search id. There are about 111 full pools from 32 worlds.
- **Fixed held-out worlds:** `learned_screen_cv.HELD72` (18 worlds). Develop only on the others.
- The collector, B-REPLAY bank and LS-COLLECT outputs: `learned_search_watch.merged_sources` finds `LS_COLLECT_REGISTRATION.json` and `BANK_STATUS.json` on cpuQ read-only.

## Commands (run from E:/agent-work/c4-plan-proposer with `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`)
- **Tests:** `python -m pytest -q tools/coopmwm/test_plan_proposer_cpu.py tools/coopmwm/test_learned_search_cpu.py tools/coopmwm/test_learned_screen_cpu.py tools/coopmwm/test_mppi_search_cpu.py`. All passed (49 + 3) as of the last code changes.
- **Refresh the data (read-only fetch):**
  ```
  python -c "from tools.coopmwm import learned_search_report as lr, learned_search_watch as lw; s,_=lw.merged_sources('E:/agent-work/_plan_proposer_cache'); print(lr.collect('E:/agent-work/_plan_proposer_cache', s))"
  ```
- **v2 budget held-out:** use `E:/agent-work/_plan_proposer_cache/ls/budget_run.py held` as the template. Change its call to `be.evaluate_configs(sel, configs=be.CONFIGS_V2, alpha=<calibrated>, unit='pool_sd')`. Output goes to `ls/budget_held.json`.
- **Triage:** `be.evaluate_triage(dev, held)`, with data from `be.triage_data(cache)`, split by `learned_screen_cv.HELD72`.
- **Learned-screen check:** `E:/agent-work/_plan_proposer_cache/ls/run_ls.py`. The proposer report is `python -m tools.coopmwm.learned_search_report`.
- **Watcher:** `python -m tools.coopmwm.learned_search_watch --poll-s 600 --max-hours 30`.
