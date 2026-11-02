# S-BDRANK fresh-session verification — 2026-11-02

- Recovery began with `git status --short --branch` and `git log --oneline -10`:
  branch `c4/bdrank`, clean worktree, completed analysis commit `7655c0db`.
- Read the existing report, evidence index, registration, verification and round-2
  availability receipts before considering further work. Read the required
  WMBREAKDOWN report/method, WMPHYS_MORNING, WMTEACH and COOPPLAN contracts.
- Ran only `python -B tools/bdrank/verify.py`; PASS. It checked 4,500 recorded
  rows, 27 paired-regression/fixed-effect equivalences, 900 complete K8 aggregates,
  17,750 pair/tie classifications, 7,348 concordance summaries, 300 decisions,
  development-only selection, clustered-bootstrap invariance, option quality,
  all 173 input hashes and the single shared registration line.
- Post-verification Git status remained clean: all saved report/CSV/manifest/
  verification bytes were unchanged. The primary report remains 55 lines.
- Metadata-only peer recheck: round-2 `CURRENT_JOBS.md` records the authoritative
  23:27 recovery at 31/216 new world jobs and an explicit zero-admission hold.
  `COLLECT_20261102T035947+0800_gpuA.json` still lists missing jobs. The full
  registered comparison remains incomplete; no partial-panel efficacy was read.
- All stopped items and GPU-minute estimates remain as recorded in
  [BDRANK.md](BDRANK.md) and [blocked_items.csv](bdrank/blocked_items.csv).
- No analysis rerun, new SUMO, training, inference, dispatch, process signals,
  peer-worktree writes, additional registration line, deletion or push occurred.
- This receipt is the only resume change; the completed analysis is preserved.

## Fresh-session verification — 2026-11-03

- Began with Git status/log and these saved receipts: clean `c4/bdrank` at
  `60e6e205`; completed analysis remains `7655c0db`. Re-read all four required
  source reports, the registered detector method and the BDRANK analysis contract.
- Re-ran only `python -B tools/bdrank/verify.py`: PASS with the same check counts
  above, including all 173 input hashes and exactly one shared registration line.
  Post-check Git status was clean; all report, CSV and verification bytes match.
- Read only `c4-rb2-int/docs/rb2int/CURRENT_JOBS.md` for current availability.
  Its authoritative Nov2 10:20 +08 receipt supersedes the earlier recovery:
  31/216 new jobs collected, 185 unadmitted, 12 matched complete-K8 v5/v4 worlds;
  collector and publisher exited incomplete, zero active INT evaluations, explicit
  hold with no resume release. The full registered comparison remains incomplete.
  No partial-panel efficacy or v5 prediction cache was opened or scored.
- Preserve the original report's frozen availability snapshot; current metadata
  is recorded here. All STOP items and inference estimates remain unchanged.
- No analysis rerun, SUMO, training, inference, dispatch, process signals, peer
  writes, extra registration, deletion or push. Only this receipt was updated.
