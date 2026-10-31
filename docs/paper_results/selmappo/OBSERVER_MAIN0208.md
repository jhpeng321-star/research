# Read-only observer and final checkpoint publication

`tools/selmappo/watch.py` is a local observer, separate from immutable training sources.
It observes only build9bb7869d8f590818 and never dispatches or signals any process.
One exclusive `WATCH_MAIN0208_ONCE.json` and exclusive log prevent duplicate launches.

Started once as localPID3668, verified exact Windows process command/creation time
and successful18:18:40UTC observer log. Lock binds build/source and is committed.
Subsequent read-only18:19:31UTC check verifies2genuineupdates,46new complete/6open,
0newfailures,90,000charged (incl43,200history), actual72,000..84,600bounds.

- Every300s, use the existing read-only progress collector; retain immutable JSON
  snapshots and maintain the concise `docs/paper_results/SELMAPPO.md` status.
- Stop observing a dead supervisor; no automatic training retry/replay.
- Publish final only after the completed final169 status/checkpoint is stable.
- Before publication, cpuQ verifies all staged source hashes, all169 update rows,
  exact23+23+167x46counts, all7728 unused-world START/END identities and900actual,
  exact6,998,400charged with14,400unstarted reservation, and no new failure log.
- Load the genuine checkpoint on CPU through FrozenFleetSelector, require finite
  model parameters and final169/protocol/library match; bind the optimizer config,
  source hashes, design hash, training-log/ledger hashes, and accounting disclosure.
- Write immutable cpuQ `training/FROZEN_FINAL.json` and local
  `docs/paper_results/selmappo/FROZEN_FINAL_MAIN0208.json`; supply a host-labelled
  final-spec receipt to `E:/agent-work/c4-c1switch2/SELMAPPO_FINAL.json`.
  C1SWITCH2 owns transfer/binding and evaluation physics; observer performs none.

Verification: RuffPASS; `watch once` successfully collected live first-update receipts
and generated the status report. A real read-only final-publication probe against the
incomplete run refused at the169-update gate, before freeze/peer writes or checkpoint
loading. Forty CPU/scientific-runtime tests and real frozen integration already passed.

First actual training update:23episodes/690frames/690fleet samples,150.014wall-s,
496,755sim-s/hour including resource-priority waiting. This is a training-throughput
measurement, not an efficacy claim; do not change checkpoint choice based on it.
