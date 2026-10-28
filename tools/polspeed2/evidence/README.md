# S-POLSPEED2 measured evidence

Exported by `tools/polspeed2/finalize_evidence.py` from the task-owned gpuA
output `/root/autodl-tmp/c4/polspeed2_20261028`.

`REPORT.json` includes every metric difference, per-worker allocator peaks,
and bitwise/absolute/relative comparisons of the updated actor and Adam state.
`reference181.json` is the last update-181 row of the original resolved
candidate-A dry run; `FGA181.json` is the last update-181 row of S-POLSPEED FGA.
The last row matters: the trainer writes first, then the CLI appends its
independent rescore. Timing/memory keys alone are excluded from metric parity.

Per-run configs, environment/argv receipts, update logs, GPU memory samples,
worker traces, a post-run host-memory snapshot and STOP receipts are included. Large transport snapshots,
per-future gradient tensors and policy checkpoints remain on gpuA.
See `POLSPEED2.md` for the recommendation and the six-row rounding caveat.
