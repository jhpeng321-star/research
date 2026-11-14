# FEAT1 v4 status — resume after the 09:28 BrokenProcessPool crash

Branch `zcode/feat1run4` (b1d9598 + crash-robustness commit); workspace
`E:/rd-wt/feat1run4`; cwd `output/wd01/wm_n1`; interpreter
`C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`; `WD_FORBID_D=1`.

## Crash post-mortem (2026-10-01 ~09:28 local)

The 12-worker run (launcher 14980 / featurizer 32340, logs_v3) died with
`concurrent.futures.process.BrokenProcessPool` after 1825/2463 shards. The
captured traceback (`logs_v3/featurize.log`) shows the chain: a worker
returned/raised a `wall_tokens_v04`-module exception; the PARENT failed to
unpickle that result (`ModuleNotFoundError: No module named
'wall_tokens_v04'` inside the executor result reader) because only workers
put the wm_n1 root on `sys.path` — one such result marked the whole pool
broken. Fix (this branch): `featurize_all()` now calls
`ensure_wm_n1_on_path()` in the parent before batching, and `_run_batch`
prints flushed per-run `[start]`/`[end]`/`[error]` lines so a crashing run
is identifiable (previously per-run lines printed only after ALL batches).
Corpus suite 72 passed (`WD_FORBID_D=1`).

## Quarantine v3 (before relaunch)

`D:/rd-staging/wmn1_feat/quarantine_v3/20261001T013855Z/QUARANTINE.json`:
1825 arm dirs scanned, every v1 `DONE.json` output sha AND every
`IDEA_A_DONE.json` array sha re-verified → 1824 ok, 1 moved (never deleted,
never overwritten):

- `chunk4/w8659m2700p10t420/base` — complete v1 shard but no
  IDEA_A_DONE.json (Idea-A-incomplete; J4 wall arm).

## Byte-identity (before relaunch)

3 already-finished TRAIN runs re-featurized with `--idea-fast` into
E: `_tmp_byteid` (removed after) and sha-compared against the live
`shards_v2` (read-only): **BYTE-IDENTITY: 3/3 runs ok, 61 sha comparisons**
(switching1 `w8443m2600p10t480`, native `w8447m2700p10t240`, base
`w8439m2700p10t300` — all 12 v1 arrays + slot_vid + meta + outcomes +
5 Idea A arrays + input physics equal). Full stdout:
`logs_v4/byte_identity_resume_20261001.txt`.

## Live run (v4, resumable, skip-by-sha, `--force` never used)

- launched_utc: 2026-10-01T01:47:58Z (local 09:47)
- launcher PID 6996 (`logs_v4/launcher_pid.txt`), featurizer PID 8240
  (`logs_v4/featurize_pid.txt`), detached via `logs_v4/run_featurize.cmd`
- command: `python -m corpus featurize-all --inventory
  E:/rd-wt/feat1run4/output/wd01/wm_n1/inventory/inventory.parquet
  --data-root E:/research-dense --out D:/rd-staging/wmn1_feat/shards_v2
  --workers 10 --max-workers-override 10 --min-free-ram-gb 8 --idea-fast
  --allow-d-output-root D:/rd-staging/wmn1_feat`
  (override 10 recorded in ALLOCATION.json; `idea_fast: true`,
  `force: false`)
- prelaunch gates (`logs_v4/prelaunch_resources.json`): free RAM
  14.73 GB (>= 8), free virtual 34.57 GB (>= 10), D: free 173.76 GB (>= 40),
  22 logical cores - 10 workers
- progress at T+5 min: ~90 skips = ~1285 runs/h through the done region
  (resume still re-builds each run in memory by design; DONE shards verified
  by quarantine v3 are re-checked by sha, no overwrite); remaining fresh
  runs: 639 (2463 - 1824 kept + 1 quarantined)
- log: `logs_v4/featurize.log` with per-run `[start]`/`[end]` lines

Crash protocol (pre-agreed): if it dies again, the runs with `[start]` but
no `[end]` were in flight; each is re-run alone with `--workers 1`; a run
that reproducibly kills its worker is EXCLUDED explicitly (listed with the
error) and the rest continue.

## Non-featurized runs (13, unchanged)

`inventory/non_featurized_traincal.json`: 12 chunk1 runs
(`is_complete=False`) + 1 walls_pilot_chunk4 smoke (`is_smoke=True`).

## Next

Poll ~every 10 min. On completion: pack -> `D:/rd-staging/wmn1_feat/pack_v2`,
`corpus verify`, `corpus.idea_a verify`, `corpus manifest`, then
`FEAT1_REPORT.md` + `FEAT1_REPORT.json` (runs by batch/split/policy, ticks,
bytes, wall time, J4 relabel counts, disclosure chain, manifest sha).
