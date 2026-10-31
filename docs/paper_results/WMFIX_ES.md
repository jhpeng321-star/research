# S-WMFIX-ES — derivative-free (ES) calibration receipt — 2026-10-31

Owner 12:50: CPU route to a working world model before the deadline; gradient
path unreliable (dense hard-event breakpoints, FD/analytic 30-350x) while the
teacher ceiling is large (300 s paired MAE 2185->526). Hosts: cpuQ <= 8 cores
(shared; S-WMDATA owns up to 24) + gpuA host CPUs (20-CPU quota, GPU jobs
untouched). No SUMO; sealed/holdout paths never opened.

## Status (updated as steps land)
- [done] task 1 CPU throughput probe (one TRAIN group, K8 = 2 arms x {120,300} s,
  entry path, mix ON, torch threads=1); see numbers below.
- [running] task 2: first ES run (gpuA PID 175512, out `es_run`, 12-FIT-only
  corpus, groups 0-5) launched 13:32 by the frozen session. SUPERSEDED by MAIN
  13:38 (combined corpus required): to be stopped and relaunched as
  `es_run_combined` (12 FIT + wmdata_fit); artifacts kept.
- [pending] task 3 verdict A/B/C1 for the best ES-WM candidate.

## Task 1: CPU throughput (TRAIN group, K8, threads=1 per process)
- gpuA host CPU (quota 20): group 1920.0 s = 32.0 min; h300 cell 812-825 s;
  h120 cell 141-142 s; max RSS 1775 MB/probe (PROBE_gpuA_group0.json).
- cpuQ (32-core VM): group 1766.6 s = 29.4 min; h300 cell 752-757 s; h120 cell
  128.7-128.8 s; max RSS 1372 MB (PROBE_cpuQ_group0.json). Live sweep cell
  (g000 grp4 arm0 h300) 474.7 s under the running load.
- Aggregate: cpuQ N=30 -> ~30 groups / 29.4 min; gpuA N=18 -> 18 groups /
  32.0 min; combined 48 procs ~= one 52-group panel in ~52/48*30.6 ~ 33 min,
  i.e. the same order as one GPU panel (28.5 min/panel), so ES generations
  (8 cands x 20 groups = 160 h300 cells) cost ~160/46*650s ~ 38-55 min/gen.

## Combined corpus (MAIN addendum 13:38; pre-registered before relaunch)
- Cherry-pick S-WMDATA 9b6f8054 (role wmdata_fit, seeds 969400-969439, arms
  ctrl0/mappo/pi/circles); prepare.py over cpuQ views -> TRAIN_PREPARED_WMDATA.
- Registry doc (prepare_combined.py) binds FIT 7bbd41f3 + the wmdata doc
  digests; fitness scales/anchor_scales/crossing_width stay the FIT doc's
  (the SAME pre-registered objective normalisation extended to more groups).
- Fitness groups: FIT 0-11 + wmdata groups {0,5,10,15,20,25,30,35} of the
  doc's sorted pair order (every 5th of 40; budget-capped, seeds recorded in
  RUN.json). Contrast arm: ctrl0 (arm_index 1, sorted-arm order; FIT groups
  use their single registered arm, arm_index 0, deduped to one cell/group).
- Host pinning (whole run): gpuA 18 workers <- FIT groups 0-7; cpuQ 28 workers
  <- FIT 8-11 + all wmdata fitness groups (gpuA needs no wmdata files).
- ES: option (b) 7 physical scalars, popsize 8, sigma 0.10, lr 0.25,
  seed 140014, horizon 300 s, max 12600 s, stop after 3 stale generations.

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
