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
  Fresh-session verification: all21 core/scorer/completion/support/transfer tests
  PASS/18.993s; no scoring or inference repeated.
- Standby **753102** completed its bounded support through11:30+08; final log
  verifies all160 tasks, frozen sources and original report hashes. Workers and
  original finalizer have exited. Resume17:07+08 launched no replacement jobs.
- Read-only resume snapshot: A2=160,FULL-v4=160 tasks COMPLETE; B2=157
  (worker exit1,no retry); C2=102 (INT-owned cutoff). FULL-v1/v3 retain81/87.
  INT's final publication owns candidate comparisons and C2 ownership transfer:
  `E:/agent-work/c4-rb-int/docs/rbint/FINAL_DELIVERY.md`. No variant redispatch.
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

## Fresh-session reconciliation — 2026-11-01 17:07+08
- Clean HEADdf24b624 retained; owner registration/program, frozen design,
  WMBREAKDOWN gate and RUNA hook request reread. The request is already resolved.
- `ops status`: WM2 four40-task shards COMPLETE, finalizer complete; support's
  last11:30 receipt REPORT_READY/report_verified=true. No EVAL process remains.
- `ops candidate-status`: A2/FULL-v4 complete; preserve B2 FAILED and C2 cutoff,
  plus all superseded sources. Both GPUs currently busy with other work.
- Requested support cutoff and INT11:45 delivery have passed. Final comparison:
  `E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md`; source manifest
  `E:/agent-work/c4-rb-int/docs/rbint/noon/20261101T113555+0800/PUBLICATION.json`.
- Reverified original+corrected evidence at commit43d67b24:
  `docs/paper_results/RBEVAL_WM2_RESUME_20261101T1710_VERIFIED.json` (160 task
  hashes,15 original report files,7,200 BIC decisions,70,488 point-data rows).
- Closure receipt: `docs/paper_results/RBEVAL_RESUME_CLOSED_20261101.md`;
  shared notice: `E:/agent-work/rb-shared/EVAL_RESUME_CLOSED_20261101.md`.

## Subsequent resume — 2026-11-01 21:31+08
- Clean HEAD5a08962d preserved. Registration5683/5697, fixed design, rationale,
  WMBREAKDOWN gate, RUNA request and INT final-delivery receipt reconciled.
- Read-only `ops status` confirms all four WM2 shards40/40 and original
  `FINALIZED.json`; no matching EVAL process was listed. Existing support log
  ends at its11:30 cutoff with REPORT_READY/report_verified=true.
- Read-only `ops candidate-status` confirms A2/FULL-v4=160, B2=157/FAILED,
  C2=102. INT's final delivery already accounts for these retained partial runs.
  gpuA/gpuR currently99%/98% utilized; no replacement work is needed.
- Corrected local `gate.json` remains NOT_QUALIFIED with revision
  `scored-only-singleton-roundoff-v2`. Prior verification/test receipts above
  remain the evidence; this resume did not regenerate forecasts, scores or tests.
- The requested support deadline has passed and no unresolved EVAL request was
  found. Exact future-builder CLI/API and canonical corrected report remain above.

## Fresh-session reconciliation — 2026-11-02 15:48+08
- Clean HEADa663469d preserved; registration5683/5697, frozen design, rationale,
  WMBREAKDOWN gate, RUNA request and INT final-delivery receipt reread.
- Read-only `ops status`: four40/40 WM2 shards; finalizer complete; support ended
  at11:30 with REPORT_READY/report_verified=true; no matching EVAL process listed.
- Read-only `ops candidate-status`: A2/FULL-v4=160 complete, B2=157/FAILED,
  C2=102; both GPUs busy. These match INT's delivered terminal accounting.
- Corrected gate remains NOT_QUALIFIED/scored-only-singleton-roundoff-v2.
  Exact CLI/API above remains delivered. New RB2 infrastructure has its own INT
  registration/ownership; the original overnight request has no pending EVAL step.
- Receipt: `docs/paper_results/RBEVAL_RESUME_20261102.md`.

## Fresh-session reconciliation — 2026-11-02
- Resumed clean HEADa663469d; read history/status and task receipts first.
  Reconciled owner registration5683/5697/5753, required rationale, the frozen
  design, WMBREAKDOWN gate, RUNA request and INT's final-delivery receipt.
- Read-only `python -B -m tools.rbeval.ops status` again confirms four40-task
  WM2 shards COMPLETE and original FINALIZED.json; no matching EVAL process
  was listed. Support ends at11:30+08 with REPORT_READY/report_verified=true.
- Read-only `python -B -m tools.rbeval.ops candidate-status` confirms A2 and
  FULL-v4=160 complete, B2=157/FAILED, C2=102; INT retains their final accounting.
  Both GPUs are busy (gpuA98%, gpuR82–89%); no dispatch or process action taken.
- Confirmed the future-builder API exists in run_wm.py and the canonical local
  corrected gate remains NOT_QUALIFIED, scored-only-singleton-roundoff-v2.
  Existing verification/test receipts remain applicable; no forecasts, scoring
  or completed tests were repeated. This resume changes documentation only.

## Fresh-session reconciliation — 2026-11-03 01:35+08
- Clean HEADf402d36f preserved; original registration5683/5697/5753, rationale,
  frozen design, WMBREAKDOWN gate, RUNA request and INT final delivery reconciled.
- Read-only `ops status`: all four WM2 shards40/40; original finalizer FINALIZED;
  support ended at11:30 with REPORT_READY/report_verified=true. No matching EVAL
  process was listed. `ops candidate-status`: A2/FULL-v4=160, B2=157/FAILED,
  C2=102, matching the delivered accounting. Both GPUs currently busy.
- Confirmed future-builder callback and source-identity binding in run_wm.py;
  corrected gate remains NOT_QUALIFIED/scored-only-singleton-roundoff-v2.
- Later central registration6072 assigns GPUs to S-COOPMWM; original overnight
  EVAL work is complete. No inference/scoring/test repeat or process action.
- Receipt: `docs/paper_results/RBEVAL_RESUME_20261103.md`.
