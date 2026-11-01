# Bounded read-only scoring continuation

Inference ownership is already fixed: final supervisors473053/gpuA and924719/gpuR.
The `finish.py` continuation **never launches inference**. Do not create a second
dispatcher or repeat an original/candidate rollout to obtain a score.

Prepared and verified:

- Additional50 recorded60s target extractor926841 EXITED/PASS,250episodes;
  transferred SHA6cd2254e2ad7596e4c9cdc4a510e4fc367baf36b3108cdf00a1afd35dbaaed34.
  Original100targets reused, giving350unique observable target episodes.
- Scoring-only data `docs/rb2int/scoring_data_v1/`:450five-arm factual episodes
  (32FIT +58final worlds); all4050region/horizon BIC decisions rechecked against
  frozen detector. Target/real source hashes bound. No parameter fit/model input.
- Original reserved WM2/FULL-v4 local caches and original WM2 prefix-matched60s
  outputs located and source-validated. Initial approved-run registry passes.
- Read-only incremental snapshotter transfers only complete K8 world jobs and
  validates hashes before retaining them. Already-collected complete worlds are
  checked/reused, not retransferred or reforecast.
- Fixed source/report/gate/credibility/endpoint comparators prepared. Full58 gate
  includes singleton diagnostics; paired primary54 gives scoped replicated-cell
  CIs. Reserved8/all58 sensitivities keep singleton intervals NA.
- Source-bound strict0.8decision/50/80/90/range definitions reused. Credibility
  intervals resample whole-world ratios and retain scored zero-selection worlds;
  no empty-selected-resample filtering or single-world confidence claims.

The finite collector polls every20min, commits only its named collection receipts,
then, once216worldjobs complete, scores all four models, assembles WMPHYS_V5.md,
publishes hash-bound summaries and commits the delivery. Full raw bin/forecast
evidence remains in `runs` with an immutable full manifest. It stops on changed
analysis source, foreign staging, incomplete transfer or deadline; no scientific
retry. A partial panel cannot be labeled a complete final delivery.

Latest collection/credibility/report/source-guard gate16CPU PASS/1.68s; target
extraction/protocol gate13PASS. Do not edit watched analysis scripts or the progress
report after collector launch without first inspecting its exact owned process
and STOP/COMPLETE receipts; source edits are preserved and cause a safe stop.

Before starting, inspect local PROCESS/CIM receipts for an existing collector.
The exclusive `runs/rb2int_finish_v1/DISPATCH_ONCE.json` prevents a duplicate.

## Actual continuation ownership

V1 exited after its successful collection at CRLF-byte whitespace validation;
see COLLECTOR_V1_RECOVERY.md. V2 is LIVE local9660/create_filetime134379946028423282,
exclusive `runs/rb2int_finish_v2`. First poll16:16:43:7/216worldjobs collected and
committed. Same verified cache, no inference replay. Inspect V2 PROCESS and
STOP/COMPLETE before creating any further continuation.
