# Endpoint target-reader representation repair

First CPUQ extraction, source6ea49b1e, failed BEFORE writing its output at
registered969420/ramp.31. Recorded s/v/a and start hashes match exactly; source
start lane is−1 with junction=True, while the converted causal carry is ramp2.
The reader was incorrectly expecting a carried lane directly in start.state.

Repair: infer the source carry from the SAME hash-bound pre-t0 history, just as
the factual converted carry is inferred from its causal prefix. No tolerance,
nearest-vehicle substitution, future conditioning or output/forecast changes.
Parallel-junction and recycled-ID regression tests included;7CPU tests PASS.
Original failed immutable staging remains; corrected extraction uses a new path.
