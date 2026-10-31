# COOPBANDIT receipts and exported data

`partial135/` is the first recovered immutable EXAM snapshot (135 decision
states; completion-order subset). It is exploratory, not the final mixture.
The remote source is cpuQ:
`/home/ubuntu/coopbandit20261031_aggregate_v3/snapshots/partial135/`.

The registered learner is in `tools/coopbandit/learn.py`. Supplementary
S-COOPBANDIT-LEARN diagnostics use its available-option-only fitting and
calibration variant; that variant does not replace the registered live policy.
All joins/splits must group by world_id; metadata are not model inputs.

Options/rerun runner for S-COOPPLAN: `tools/coopbandit/protocol.py`,
`backend.py`, `runner.py`, and the frozen `tools/leverprobe` contracts.
The exact label sign is D = J(MAPPO) - J(option), positive saves delay.
Consumers must preserve the unchanged original safety layer, full-rerun
bitwise prefix gate, option availability and joint-context solo targets.
