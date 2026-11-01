# Saved-analysis child exit — explicit recovery, no inference retry

Collector v6 PID23680 stopped at08:31:38. Its FULL-v4 comparison child38748
returned1 after about8s,with an empty stdout/stderr log and no comparison
artifact. Original PROCESS/EXIT/log/STOP and all snapshot bytes remain under
`noon/20261101T082618+0800` and `runs/rbint_noon_v6`.

The A2/B2/C2 comparison,distribution and credibility components already
completed successfully in that round. They must be reused,not rescored.
Unchanged FULL-v3 reports remain cached.08interim stays published and valid.

Read-only diagnostics found no Application error event at08:31–08:32. At the
later audit Windows had about12GiB free physical and33GiB free virtual memory;
this does not prove memory availability at failure. Root cause remains unknown.
All remote inference continues under its original ownership/deadlines.

One explicit CPU-only recovery of UNFINISHED components is authorized here.
Progress logging,unbuffered output and Python faulthandler are enabled so any
repeat failure identifies a phase/trace. No successful component or forecast
is repeated. Distinct `_recovered` output/log names preserve failed artifacts;
exclusive claim prevents an automatic second recovery. No GPU/SUMO/training.

`finish_saved_round` verifies snapshot transfer exit receipts/fingerprints,
reuses each successful component or identical-ensemble prior report,then
executes only absent changed-ensemble components. It publishes a recovery
manifest only after all components finish. No new forecast fetch or dispatch.
Comparator semantics unchanged;21CPU recovery/comparator/noon/pause tests PASS
in2.02s. Commit this tool/design before the explicit analysis continuation.
