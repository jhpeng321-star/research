# Stage 1b sequencing — pause/resume state (2026-09-04)

All compute is PAUSED. Nothing is lost; every stage is resumable from the
per-seed JSON checkpoints (the runners skip completed seed files).

## Status at pause

| part | status |
| --- | --- |
| Part 1 (p20 stratum) | **COMPLETE** — 240/240 seeds, 960 branches, integrity 0 failures, 12/12 repeats. `runs/probe/stage1b_generation_p20_s7001_7240/`. Doc `docs/SUSTAINED_WALL_V1_STAGE1B_GENERATION_RESULTS.md` (report restricted to p20; p30 deferred by order). |
| Part 3 (v1 gate fit) | **COMPLETE** — `runs/probe/stage1b_generation/gate_fit_summary.json`; doc `docs/SUSTAINED_WALL_V1_GATE_V1_FIT_RESULTS.md` (headline: L1-L5 beat L0 at lambda 0 with CI; no rung beats L0 robustly at lambda 2; timing selection not better than fixed W250). |
| Part 2 (Stage 1c) | **COMPLETE (2026-09-05)** — all four cells 30/30 (120/120 seeds, 600 branches, 0 integrity failures, 12/12 D-repeat identities). Report: `docs/SUSTAINED_WALL_V1_STAGE1C_RELEASE_RESULTS.md`; summary `runs/probe/stage1c_release/summary.json` via `experiments/idm/analyze_stage1c_release.py`. |

## Resume commands (from D:\research-dense, same env as always)

1. Resume m2400 (skips the 16 completed seeds):
   `python -m experiments.idm.run_stage1c_release --mainline-vph 2400 --workers 8`
2. Then m2600:
   `python -m experiments.idm.run_stage1c_release --mainline-vph 2600 --workers 8`
3. Then m2700:
   `python -m experiments.idm.run_stage1c_release --mainline-vph 2700 --workers 8`
4. After all four cells: run the Part 2 report per the task spec
   (release timing R1/R2, axes vs D, oracle-recovery fraction, safety
   flags, flow), write
   `docs/SUSTAINED_WALL_V1_STAGE1C_RELEASE_RESULTS.md` and commit.

Artifact dirs: `runs/probe/stage1c_release_m{2200,2400,2600,2700}_p20_s*/`
(manifest + per-seed JSON with arms D/W_ORACLE/W_R1/W_R2/W_HOLD, rule-rule
inputs at 1 Hz, integrity, r2 thresholds, repeats every 10th seed).

Operational notes for the resume session: (a) the deterministic SUMO
launcher (`experiments/idm/deterministic_sumo.py`) is installed by
import in the 1c runner — no action needed; (b) long background tasks die
unpredictably on this machine — relaunch the same command and let the
skip-cache resume; the p20/noop-style runs survived when not
sibling-polled; (c) do not run the p30 stratum — explicitly deferred.

## Commits

- `5238b21` — p20 stratum complete + Part B report + gate-fit results.
- (this pause checkpoint will be committed next as `stage1b pause checkpoint`).
