# Stage 2 pause/resume state (2026-09-05)

All compute is PAUSED at user request. Nothing is lost: the runner skips
completed per-seed JSONs, so resuming is the same command.

## Status at pause

| item | status |
| --- | --- |
| Stage 2 machinery | **COMPLETE and tested** — `stage2_generation_contract.py`, `run_stage2_generation.py` (dynamic capture at t_on / t_on−60 / 150, unclipped lane geometry, onset-triggered WON_H300/H450/H650), `analyze_stage2_generation.py` (6-item descriptive report + summary.json with SHA-256 sidecar), registration in `idm/contract.py` (`stage2_generation`, seeds 8001-8240 p20 / 8241-8360 p30, excluded from prospective). Tests `test_stage2_generation_contract.py` (4) + full suite 1030 passed before compute. |
| p20 generation | **PAUSED at 32/240** (seeds 8001-8032 complete, contiguous). Resume command below. |
| p30 generation | NOT STARTED (120 seeds; run after p20; report p20 first per task). |
| Analyzer | Written and compile-checked; run after p20 for the interim report (`--strata p20`), then both strata at the end. |

## Resume commands (from D:\research-dense, rl_driving env)

1. Resume p20 (skips the 32 completed seeds):
   `python -m experiments.idm.run_stage2_generation --strate p20 --workers 8`
2. When p20 completes: run
   `python -m experiments.idm.analyze_stage2_generation --strata p20`
   and report the interim results; then start p30:
   `python -m experiments.idm.run_stage2_generation --strate p30 --workers 8`
3. Final: `python -m experiments.idm.analyze_stage2_generation --strata p20 p30`,
   fill `docs/STAGE2_GENERATION_RESULTS.md`, commit.

Pace note: ~17-18 seeds/hour at 8 workers (each seed = D + up to 3 wall
replays with full snapshot capture), so the remaining 208 p20 seeds ≈ 12 h
and p30 ≈ 6 h.

Seeds 4601-4660 and 5001-5060 remain untouched.
