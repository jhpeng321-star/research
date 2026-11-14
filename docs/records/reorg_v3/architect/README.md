# reorg3 architect

REORG_PLAN_V3 — one-folder reorganisation plan for research data on `E:`
(planning only; no moves/deletes/uploads executed).

| file | description |
|---|---|
| `REORG_PLAN_V3.md` | target tree, decisions, buckets, space math, rollback, phases |
| `MOVE_MAP.json` | 83 moves old→new (size/method/compat/order) + drive-only + delete lists |
| `tools/build_move_map.py` | reproducible generator for `MOVE_MAP.json` |
| `scripts/phase2_moves.ps1` | dry-run-default move executor (journal before each mutation, refuses live worktrees/open handles) |
| `scripts/drive_offload.sh` | `rclone copy` + `rclone check --one-way` per item, writes a verification manifest |
| `scripts/DELETE_AFTER_VERIFY.ps1` | owner-run deletion of Drive-verified items only; report-only default |

Read-only inputs live in the sibling `s1_E/`, `s2_D/`, `s3_C_G/`, `s4_refs/`.
