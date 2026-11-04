# Engineering data repair — registered before restoration

Restore exactly five missing LS-COLLECT pending envelopes needed by the frozen
B-u6 lesson list, 655,103 bytes total. The read-only audit traced all five to
the same certified archive batch, `archive1450_v3/ARCHIVE_MOVE_batch0028.json`;
its SHA is `1400f869e34d6c5e69531befc2fc51cc9e706cae64643066240155f1a705dd3e`.
The full original envelopes remain in gpuR's original archive.

Exact source/destination paths, byte counts and SHA256 values are pinned in
`U6_PENDING_RESTORE_PLAN_1205.json`, SHA
`edf83459c0d65d2472b6e777e7d80ec5a89a5ddf78e1294205b727285a30ec55`.

Before installing: re-verify original archive proof and remote bytes, relay only
the five envelopes, verify local bytes, refuse unequal existing targets, and
write an additive protection binding consumed by the existing archive guard.
Keep archive originals. No process signals, new simulation, inference, budget
change or diagnostic replay. Existing B-u6 branch jobs remain owned by the live
runner. The original incomplete lesson receipt is preserved.

After restoration: rerun only the read-only input audit, including snapshot and
episode-result existence; report any additional missing input before action.
Future scheduled diagnostics may consume the repaired inputs normally.
