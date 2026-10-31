# Completed-output collection recovery (no scientific retry)

- s969436_t420_k5 worker297663 completed28.986s; DONE SHA
  8256a501ae7a5dca71ed8b4d35c3be10ef72ec56816da270c1ba480c58e4908d.
  SCP returned1 and left partial local bytes. Scientific inference succeeded.
- Preserve the partial file in short_v1/tasks, all original logs/claims, and the
  failed transfer. Re-fetch identical remote bytes to recovered_tasks/*.copy1.json;
  immutable *.receipt.json binds DONE hash, original partial hash and actual path.
  No overwrites/deletions, no new WM inference, no consumed scientific retry.
- task_path resolves only verified task-root recovery copies and checks their
  hashes; finalizer/auditor use that path. No prediction or bootstrap changes.
  Future SCP nonzero exits with complete bytes are accepted ONLY on DONE-hash
  equality; partial transfers use the same preserved-copy collection route.
- Exact local paused scheduler43404 stopped after command ownership check;
  its remaining remote workers kept alive. Resume inventory collects inherited
  claims, excludes live keys, and counts them against host/device admission caps.
  Metadata/collection SSH faults wait without dispatching a duplicate worker.
-40CPUtests+ruffPASS; added meaningful partial-preservation/tamper-rejection test.
  All181available short outputs audited:133runtime sources identical to original
  long outputs except registered worker driver, same clocks/prefixes/WM/controller.
  Evidence short_transfer_recovery_binding.json. No reserved label/model scoring.
- Worker short_v1 and frozen V/D/E unchanged. finalize_short change is ONLY
  hash-bound transfer-path resolution. Scientific predictions/test rules frozen.
