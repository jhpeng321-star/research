# LEARNED SEARCH REPORT (plan proposer; informational, never auto-activated)

Generated 2026-11-01T20:11:39+00:00 by `python -m tools.coopmwm.learned_search_report` (CPU only; no SUMO/GPU).
**NOT ACTIVATED: requires registered minimum + offline PASS + explicit owner approval via MAIN.**

## Data (read-only copies; TRAIN worlds 986000-986071 only)
- Collection: 4 remote dirs, 10 new files; errors: none.
- Request/response pairs 3; usable B search receipts 2 (skipped {'search without scored perturbations (rejected or unavailable)': 1}; non-TRAIN dropped 0); SUMO entries 0.
- Searches 2 (2 with elites) from 2 worlds; full-pool receipts 0 (curve needs >= 48).

## Training / offline check
- not run: need >= 24 B searches from >= 4 TRAIN worlds.

## Caveats
- Surrogate replays are noise-free fits of each search's own WM screen pool; they estimate search efficiency, not SUMO benefit.
- Top-k-only service receipts carry no screened pool; the evaluations curve needs `RecordingEvaluator` logging (`<queue_root>/proposer_receipts/<state_id>.json`).
- The script never activates the proposer and never writes to remote hosts.
