# S-WMFIX-ES — derivative-free (ES) calibration receipt — 2026-10-31

Owner 12:50: CPU route to a working world model before the deadline; gradient
path unreliable (dense hard-event breakpoints, FD/analytic 30-350x) while the
teacher ceiling is large (300 s paired MAE 2185->526). Hosts: cpuQ <= 8 cores
(shared; S-WMDATA owns up to 24) + gpuA host CPUs (20-CPU quota, GPU jobs
untouched). No SUMO; sealed/holdout paths never opened.

## Status (updated as steps land)
- [running] task 1 CPU throughput probe (one TRAIN group, K8, 120+300 s, entry
  path, mix ON, torch threads=1).
- [pending] task 2 antithetic-ES/CMA-ES calibration on the 12 FIT groups
  (TRAIN_PREPARED.json 7bbd41f3), SAME paired objective (road/crossing paired
  terms; source reported separately), frozen-bitwise everything else.
- [pending] task 3 verdict A/B/C1 for the best ES-WM candidate.
- WMDATA.md not present yet at 13:0x; wmdata_fit worlds not yet available.

## Pre-registered ES protocol (before any run)
- Fitness: deterministic (CRN r1==r0 exact), so elitism needs no resampling.
- Parameterisation: (a) merge_response head (1,473, zero init) via antithetic
  ES, or (b) <= 20 existing physical merge/admission/discharge scalars
  (identified in code; values optimised, never hand-set) via CMA-ES; choose by
  measured CPU budget, prefer (b) only if (a) exceeds it.
- Rollout cells pinned to one host each for the whole run (numeric identity).
- Immutable per-generation checkpoints + GEN receipts; paired train J MAE per
  generation recorded (ES curve); stop after 3 consecutive non-improving
  generations; no validation feedback at any point.
