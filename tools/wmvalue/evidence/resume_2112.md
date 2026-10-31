# Fresh-session recovery and MAIN21:12 execution order

- Recovered clean c4/wmvalue at1260d681; read log, shared OWNER/amendments,
  original LAUNCH/dispatch receipts and live claims before any dispatch.
- At21:13,46claims/4completed/1SSH-error; original12CPU workers still live.
  s969432_t300_k0 log contains only START and remote connection closure,
  but remote PID251940 remains alive. Its authorized retry is NOT consumed;
  never retry while the original worker remains alive or has an output.
- Verified local scheduler PID28924 command/start19:38:20 and assistant27688
  command/start20:05:46 against receipts. Stopped these exact OWN scheduler PIDs
  only to prevent additional long-A tasks under MAIN21:12. No remote worker or
  peer PID signalled. Preserve all logs/claims/results, including orphaned SSH.
- Both gpuR and gpuA GPUs99% with peer compute PIDs at recovery: no GPU admission.
- Next execution must collect original completed outputs first, exclude all live
  original task keys from new work, and run only missing90s prefixes with the
  original full900 stochastic plans. Existing certified long results supply
  their own60/90 checkpoints. Frozen V/D/E and bootstrap remain unchanged.
- Evaluate all200states/400pairs once for B/C vs D/E. If neither short method
  passes BOTH sign and MAE lower-CI gates against BOTH D/E, publish FAIL without
  requiring A. Otherwise complete A and apply the original full gate.
