# S-COOPBANDIT executor handoff (resumed 2026-10-31 23:56)

Collection has not yet dispatched; existing implementation preserved and being verified.
Exact committed design: `E:/agent-work/c4-coopbandit/docs/paper_results/COOPBANDIT_DESIGN.md`.
Export implementation: `E:/agent-work/c4-coopbandit/tools/coopbandit/export.py`.
EXAM uses `states.jsonl` with state_id, world_id, t0, features (numeric dict), observed_vehicles,
prefix_sha256; `labels.csv` long format state_id/world_id/cell/t0/setting/option/horizon_s/
available/D/D_road/D_source/J/collisions/teleports/ttc_below_3_vs/hard_brake_vs/override_share.
Options O0..O4, horizons 120/300/600. Group credit exported separately.
Please support an explicitly PARTIAL prefix of >=100 complete states, world-grouped
nested CV, and final full panel without fitting on any dev world 981080..981091.
The owner's registered confidence rule is nested 5-outer/4-inner grouped CV;
95% finite-sample quantile of per-world maximum signed overprediction over all
decision times/options, O0 unless predicted D minus radius is positive.
We already have minimal tooling in tools/coopbandit/learn.py; your requested
linear comparator/figures and CLI remain useful. Do not modify the executor worktree.
Please leave your HANDOFF.md with invocation/API and commit IDs when ready.
