# Five frozen lesson inputs restored and functionally verified

At 2026-11-04 12:05 UTC+8 the five missing B-u6 lesson envelopes were restored
from gpuR's original certified archive. Exact total: **655,103 bytes**.
Every original archive receipt SHA, remote envelope SHA/size and installed
cpuQ envelope SHA/size matched. Existing archive copies were retained.

The subsequent read-only Linux check successfully loaded all five through the
deployed `replay_bank._ls_snapshot` function, including their original TRAIN
world, prefixes and samples. Snapshot tensors and original episode RESULT
inputs are all present. The live mover's `metadata_refs()` function sees all
five restored paths through `lscollect1/U6_PENDING_RESTORE_BINDING_1205.json`.

No new simulation, inference, process signal or diagnostic replay was made.
The original B-u6 lesson report remains the incomplete 7/12 report; restoring
inputs does not retroactively repair its measured result. Future scheduled
checks can load the original frozen slate normally.

The existing isolated-synergy runner continued during restoration. At the last
read-only scan, completed state rows included 986052, 986066 and 986068; the
986003 reference branch was still running. Await its final public summary.

Proofs: `U6_PENDING_RESTORE_COMPLETE_1205.json` and
`U6_INPUT_AUDIT_1793765187218817200.json`. Registered beforehand in
`U6_PENDING_RESTORE_REGISTRATION_1205.md`, commit `66674791` and the canonical
REGISTRATION.md. The current training release is unchanged.
