# WP-I end-to-end evidence (2026-10-10, branch `wms/clean-int`)

Small receipts of the real-data CPU runs described in `docs/INTEGRATION_REPORT.md` section 4. Checkpoints (`*.pt`) are not committed.

| file | what |
|---|---|
| `preflight_r3.json` | `python -m wms preflight --config configs/r3.toml`: refused (exit 2) only because the six switching-pack names are not on this machine |
| `preflight_smoke_local.json` | same recipe without the switching mix (`configs/smoke_local.toml`): passed |
| `smoke_train_stdout.log` | stdout/stderr of `python -m wms train --config configs/smoke_local.toml --mode run --out runs/active/int_smoke_local2` (911 train / 151 corpus-val window counts, final receipt JSON) |
| `smoke_train_log.jsonl`, `smoke_val_log.jsonl`, `smoke_TRAIN_RECEIPT.json`, `smoke_physics.json` | the run's per-step records (all enabled loss terms nonzero and finite), validation records, receipt and physics block |
| `smoke_first_attempt_full_val_aborted.log` | first attempt with the full 151-window validation list; stopped by the integrator after two stages (validation of 151 windows at up to 120 s is hours on CPU), replaced by the 6-window list of `scripts/make_smoke_lists.py` |
| `score_smoke_vs_c1_result.json`, `score_smoke_vs_c1_summary.json` | `python -m wms score --config configs/score_smoke_local.toml`: smoke checkpoint vs C1FAST on `wall_same`, `first_per_world:20`, 60 s; C1FAST median HDV \|ds\| 11.9576 m |
