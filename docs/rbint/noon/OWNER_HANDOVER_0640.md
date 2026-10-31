# Owner handover, saved-analysis recovery — 2026-11-01

The original v3 collector PID21808 completed all06:32:33 snapshot/scoring outputs
then stopped at its HEAD-change guard BEFORE staging or publishing mixed edits.
Its immutable `runs/rbint_noon_v3/STOP.json` reports
`HEAD changed during analysis: preserve foreign work`,automatic_retry=false,
remote_inference_dispatched=false. No worker was signalled.

All completed outputs under `20261101T063233+0800` are retained. The publication
receipt remains unchanged. `noon_report` renders these existing results into the
52-line report; it additionally displays matched absolute CRPS,rates and600s
delay bias. It does not score again or read new forecasts. Its8CPU report tests
PASS after handling missing optional historical rate metadata explicitly.

After this transaction commits, one v4 collector may resume from this committed
publication using the diagnosed v3 STOP receipt. Source-specific complete-K8
cache recovery is retained. Only new complete ensembles get new scores.
The new cooperative owner-edit latch is required for later engineering edits.

C2 exclusive claim now belongs to INT on gpuR. Original4015b641 source hashes
and smoke were verified before launch. Canary own842915/start4211291759/
bootd19e6898-efd0-4288-a901-d63b21f11163 started06:38:50; still live/0tasks at
the06:40 audit. A2/B2/FULL jobs remain untouched. No C2 efficacy is claimed.
