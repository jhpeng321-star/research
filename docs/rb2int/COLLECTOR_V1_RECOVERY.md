# Collector-only diagnosed recovery (no inference replay)

The first local finisher46500 successfully fetched/verified completed K8 worlds,
then stopped at `git diff --cached --check`: Windows-generated CRLF JSON receipts
were preserved as exact bytes, but Git interpreted CR as trailing whitespace.
`runs/rb2int_finish_v1/{START_ONCE,STOP,PROCESS}.json` and watch.log are retained.
Process audit confirms46500 exited; no owned/peer inference process was stopped.

Scoped .gitattributes now recognizes CR-at-EOL for hash-bound INT JSON/CSV and
final report artifacts while preserving raw bytes. The same diff/check still
runs; no hook/check is skipped. Regression3PASS/0.51s. Original staged collection
receipts are retained and committed with this correction.

The next finisher uses a NEW exclusive `runs/rb2int_finish_v2` receipt root and
the SAME verified `runs/rb2int_final_cache`. Already-collected world/task bytes
are reused and checked, never downloaded/reforecast again. Neither remote final
supervisor or its model/source/recipe/queue changes. Do not restart V1.
