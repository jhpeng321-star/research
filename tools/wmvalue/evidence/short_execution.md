# MAIN21:12 short-prefix implementation gate

- Frozen runtime/restart/reconstruct/core and fitted V/D/E unchanged.
- worker --short executes90s only, using the existing runtime's full-long padding,
  stochastic plans and CRN; both60/90 checkpoints emitted. Never labels it A.
- Actual frozen-WM CPU audit gpuA short_v1,35.207s synthetic-only: all five arms,
  exact60/90 checkpoint dictionaries identical for90vs120s, futures0and3.
  Runtime SHA20f7f23d442978d60d4ea92aebfc47c50b28aa586c3c444f216c4dbea0b46568.
  Full receipt: _out/wmvalue_short_prefix_audit_v1.json; remote immutable
  /root/wmvalue20261031/short_prefix_audit_v1.json. No real future/labels in audit.
-39CPUtests PASS. New tests prove complete-K8 composition parity, reject missing
  cells/unregistered worker/runtime changes, and match every B/C/D/E metric/CI
  and D/E contrast EXACTLY against the original full evaluator's same10000draws.
  A necessary-condition success is NEEDS_A, never PASS; equality fails the gate.
- short_dispatch inventories only original receipt PIDs and owned output paths,
  harvests original completed trajectories, excludes live or uncertain-start keys,
  and does not repeat any existing short claim. Old unstarted reservations may
  be serviced after verified old scheduler stop. No automatic scientific retries.
- gpuA<=4/gpuR<=8 CPU lanes count original live workers against their caps;
  nice19/thread1 and cgroup admission. Both GPUs require no peer compute PIDs;
  host-specific ownership avoids cross-host PID collisions.
- Immutable short_v1 snapshots deployed to both hosts before dispatch. No new
  environment, dependency install, SUMO, refit, peer signals or policy training.
- Only complete320futures/200cells can invoke once-only frozen B/C/D/E test.
  FAIL writes<=60line WMVALUE.md, evidence, one verdict registration and commit.
  NEEDS_A preserves the short result and requires matched A before full PASS.
