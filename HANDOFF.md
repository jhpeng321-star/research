# S-RB-EVAL handoff — WM2 complete, corrected baseline READY

Branch `c4/rb-eval`. Preserve all running candidate jobs and immutable sources.
Detailed prior receipts: `docs/paper_results/RBEVAL_HANDOFF_ARCHIVE_COMPLETE.md`.

## Baseline and current ownership
- WM2 **160/160 tasks,20 K8 worlds,800 arm-futures COMPLETE**. Original gpuR
  receipt `/root/rbeval20261031/FINALIZED.json`; original report
  `/root/rbeval20261031/reports/wm2_complete_1793478613086371110` retained.
- **Use corrected baseline:** local `tools/rbeval/evidence/reports/wm2_corrected_63cf5d83/`.
  Immutable manifest: `docs/paper_results/RBEVAL_WM2_CORRECTED_VERIFIED.json`.
  Gate **NOT_QUALIFIED**, diagnostic pilot; no powered qualification claim.
- Original bootstrap CIs retracted: unscored-peer composition and constant-draw
  roundoff. Revision `scored-only-singleton-roundoff-v2` uses scored worlds only,
  rejects singleton-only world uncertainty and roundoff-only variation. All
  point forecasts/Wilson/ranks/BIC retained; development12 singleton CIs now NA.
  Details: `docs/paper_results/RBEVAL_UNCERTAINTY_CORRECTION.md`.
- Verified160 task hashes,15 original report hashes,7,200 BIC decisions; corrected
  point-data audit70,488 CSV rows/max cross-platform numeric delta1.0914e-11.
  Verification:11 scorer+4 core+1 completion+2 transfer tests PASS.
- Standby **753102** remains bounded through11:30+08, preserving/verifying the
  original report. No replacement inference/finalizer/watcher launched.
- RUNA owns A2/C2; RUNB B2; INT FULL-v1/v3/v4 and WM2-HDV60. Last EVAL snapshot:
  A2=67,B2=45,FULL-v1=73,FULL-v3=62,FULL-v4=19 tasks (incomplete,not rankings).
  `python -B -m tools.rbeval.ops candidate-status` checks both hosts read-only.

## Exact future-builder-capable candidate CLI
Source4015b641 **invocation shape only**; existing RUNA claim owns A2 dispatch:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/autodl-tmp/c4/speedhead20261030/pythondeps nice -n 19 /root/miniconda3/bin/python -B -u /root/rbeval20261031/phys_code_4015b6412cdf/tools/rbint/evaluate_A.py --out /root/rbeval20261031/NEW_A2_shard0 --runtime-root /root/rbeval20261031/phys_code_4015b6412cdf --wm /root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt --wm-config /root/rbeval20261031/phys_code_4015b6412cdf/configs/wm_phys.toml --model-name WM-PHYS-A2 --shard 0 --shards 3 --memory-gib 5
```

B/C use `evaluate_B.py`/`evaluate_C.py`. Immutable launch specs:
`E:/agent-work/rb-shared/evalclaims/WM-PHYS-{A2,C2}.runa.json`, `WM-PHYS-B2.runb.json`.
Newer FULL sources/claims are in INT's handoff. Preserve active runtimes and claims.

API: `run_wm.main(model_loader=load_candidate,future_builder=candidate_future,
identity_sources=[entrypoint,rebuild,recipe])`. Loader returns `([model],receipt)`;
builder receives `(model,batch,initial,future,seed,frozen)` as keywords, returns
`(kwargs,instrumented_entries,receipt)`. It owns hidden-settings/arrival/entrant
draws, bypassing WM2 futures. Scorer verifies builder/construction identity and
finite/conserved due+initial_backlog=admitted+final_pending ledger. README has schema.

## Corrected scoring / verification / remaining gaps
Stage corrections only in NEW analysis directories; retain each frozen original:

```bash
python -B -m tools.rbeval.score --inputs SHARD0 SHARD1 SHARD2 --out NEW_REPORT
```

Use EVAL source63cf5d83+; add `--allow-partial` only for explicit progress reports
(only complete K8 worlds score). External S-SUMOCORE: `tools.rbeval.import_external`;
its fixed20world/fivearm/K8 schema cannot silently pool different SUMOCORE panels.
`python -B -m tools.rbeval.verify_wm2 --evidence tools/rbeval/evidence --corrected-report tools/rbeval/evidence/reports/wm2_corrected_63cf5d83 --out NEW_VERIFICATION.json`
verifies original+corrected chains without scoring/inference. `ops fetch-reports`
retrieves original remote receipts. `docs/paper_results/RBEVAL_DESIGN.md` and central
registration5697 retain the pre-scoring freeze. Small cells, missing speed support
and one factual continuation per world limit qualification; no training tonight.
Shared coordination: `E:/agent-work/rb-shared/EVAL_WM2_COMPLETE_AND_CI_CORRECTION.md`.
