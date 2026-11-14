STATUS: COMPLETE - C1 leave-PI-out trainer patch implemented, tested, dry-run verified, committed and archived.

# C1 WM-S leave-PI-out trainer (branch wms/p2, relaunch)

Code commit: 8401277 (branch wms/p2). Prior kept work (T4/T4_FIX, commit cf78c6f) was
reused unchanged; only the LPO trainer patch, its tests and this report are new.

## Archive
git archive --format=tar --output=_out/code_wms_8401277.tar HEAD output/wd01/wm_n1
SHA-256 7c643b7e68ec96a188ceeba3c6a9fdfdc7a46d2f9ad31a725f0d75e823cfb64a
size 9,216,000 bytes; 695 entries (contains wms/cli.py, wms/train.py, wms/wms_chain.sh,
wms/tests/test_wms_lpo.py). Sidecar: _out/code_wms_8401277.tar.sha256.

## Implemented (LPO_PLAN "Trainer changes", items 1-2 + requested support)
1. wms/cli.py `--exclude-window-list PATH` (repeatable) -> `load_exclusion_keys`
   (accepts `{"keys": [...]}`, `{"windows": [{"key": ...}]}` or a bare key list),
   `TrajectoryDataset.from_real_packs(exclude_keys=...)` filters corpus._v1 items by
   pack run key AFTER IdeaACorpus split checks and BEFORE identity/position validation,
   refuses absent keys (TrainDataRefusal), logs kept counts via
   `_kept_summary` (`train kept windows total K/N | <pack> kept/total ...`).
2. `--window-list PATH` -> second TRAIN dataset (`select_windows`, `(pack, key)` scoped
   by pack basename) used only as monitoring: `WmSTrainer.fit(val_dataset=...)` runs a
   forward-only `evaluate` per stage, writes `val_log.jsonl`, records
   `val_monitoring` with `early_stopping=false`, `checkpoint_selection=false`; the
   fixed step budget is never changed.
   run key = `<batch>/<world>__<arm>` (batch, never chunk alone; chunk fallback).
3. `--stage-steps` validated as one positive count per `--unroll-s` stage (refused with
   a clear message on a length mismatch).
4. `--gpu-share` mode note: defaults `--num-workers` to 4 and records a `gpu_share`
   receipt block ("up to 3 trainings on one 48 GB GPU").
5. wms_chain.sh passes `--exclude-window-list $LPO/pi_excluded_runs.json` and
   `--window-list $LPO/windows_val.json` in run(); aborts if either list is missing.
   smoke400 gate and plan_steps are unchanged.

Inputs used (TRAIN, READ-ONLY): pi_excluded_runs.json (1564 keys =
1494 PI runs + 70 unrecoverable chunk1_fix27) and windows_val.json (189 windows).
No CAL, CAL_SPLIT_A4, HOLDOUT, chunk-5, J4B, INC1 or sealed source is opened; only the
two LPO JSON metadata files are read.

## Tests
- New: wms/tests/test_wms_lpo.py (14 tests): run_key, all three exclusion-list shapes
  + bad shape, window-list pack/key requirement, keep/refuse/empty/all-excluded filter
  behaviour, pack-name scoping, both-selected-and-excluded refusal, parser repeatable
  flags, stage-steps length mismatch refusal, gpu-share num-workers default.
- Targeted: pytest output/wd01/wm_n1/wms/tests/test_wms_lpo.py test_wms.py -q -> 25 passed.
- Suite: python -m pytest output/wd01/wm_n1/wms -q -> 4 failed, 25 passed. The 4 are the
  pre-existing test_physics_torch.py failures (physics_torch is off the trainer path and
  was not edited; repair belongs to branch wms/torchphys).
- ruff check output/wd01/wm_n1/wms -> All checks passed.

## Real-data CPU dry run (dry-run, exclusion + val list applied)
cwd output/wd01/wm_n1; OMP/MKL threads 4; device cpu; batch 4; unroll 10 s; 2 steps.
Packs pack_v3 + pack_dspeed + pack_damping; WM_N1_TRAIN_OUTCOMES / WM_N1_TRAIN_IDENTITIES
set as in T4.md. Logs: _out/wms_dry5.stdout.log / .stderr.log; receipt
_out/wms_dry5/TRAIN_RECEIPT.json. Result: status passed, optimizer_steps 2.
- TRAIN kept: total 1496/3060 | pack_damping 263/263, pack_dspeed 644/644,
  pack_v3 589/2153; exclude_keys_requested 1564, excluded_records 1564.
- VAL kept: 189/3060 | pack_v3 73, pack_dspeed 84, pack_damping 32 (list validated; the
  dry-run command does not train so per-stage val loss runs only under `fit`).
- Count reconciliation with the brief's "1496/2153, 644, 263": 1496 is the TOTAL kept
  (3060 corpus - 1564 excluded); 2153/644/263 are the per-pack corpus sizes, so the
  pack_v3 kept/total is 589/2153, while pack_dspeed and pack_damping keep all their
  windows (644, 263). No discrepancy.
- updates: loss 2.943 / 1.638, both finite=true, valid_targets=80, unroll_steps=20;
  gate_g0=0.0 and monotone. 10 s profile peak working set 7.9 GB (horizon 20 skipped the
  30 s profile).

## Memory
Packs are opened with np.load(..., mmap_mode="r", allow_pickle=False); the corpus is
consumed one window at a time and no whole pack / all-runs array is materialised (dataset
build adds ~0.3 GB). The observed peak is model compute (HdvResidualNet [B,K,K,64] at
K=192), not data loading, and is unrelated to the earlier ~50 GB crash. All runs were a
single CPU process with 4 threads; no SUMO, no GPU/SSH.

## Not done / handoff
- GPU timing and memory are still unmeasured; no GPU run was started here.
- The 4 pre-existing physics_torch failures remain (wms/torchphys).
- Next for C1 evidence: launch the LPO WM-S run on GPU via wms_chain.sh with the two
  lists, then score on pi_eval_states (registered CAL-tune PI eval is a separate sealed
  step) vs WM-S-TRAIN-20261005 / keep-speed / pure-physics baselines via wms_eval.
