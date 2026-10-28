# S-POLSPEED2 — ordered parallel future execution

Branch: `c4/polspeed2`, based on `c4/polspeed` `f6dea57b`.
Owned remote output: `/root/autodl-tmp/c4/polspeed2_20261028`.

Initial gpuA inspection, 2026-10-28 23:09:03 server time: RTX 4090 D,
49,140 MiB total, 1 MiB used, no compute processes. Aggregate cap: 0.60
(29,484 MiB including the coordinator and worker CUDA contexts).

Work in progress. Timings and exactness claims will be recorded only after
measurement. The six-row algebra preserves the mathematical objective; its
floating-point and telemetry parity must be measured separately from replay
parallelism with unchanged row geometry.
