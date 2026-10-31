# S-WMFIX-ES — CPU calibration receipt — 2026-10-31

## Resume / resources
- Clean resume at 89d7e29b; frozen-session artifacts retained; no prior ES jobs live.
- Combined smoke had failed on missing CTRL0 feedback; no generation completed.
- MAIN 13:38 supersedes sharing limits: cpuQ <=30 workers; gpuA <=18 low-priority
  CPU workers (20-CPU quota). GPU processes and S-WMDATA processes untouched.
- gpuA ES code is isolated in `wmfixmerge20261031/es_cpu_v3`; cpuQ owns
  `/home/ubuntu/wmfix_es`. No SUMO, prohibited inputs, deletions or push.

## Throughput — genuine K8, entry path/mix ON, threads=1
- Each host: 8 sticky-future processes, each two paired arms x {120,300} s.
- cpuQ: 20319.6 CPU-worker-s/K8 group; 8-way wall ~2571.9 s;
  peak RSS 1250-1916 MB/process (measured).
- gpuA CPUs: 17462.2 CPU-worker-s/group; wall ~2226.9 s;
  peak RSS 1588-2510 MB/process (measured).
- Extrapolated 12-FIT-group K8 panel: cpuQ N=30 135.5 min; gpuA N=18 194.0 min;
  combined 79.8 min (2.80x requested GPU reference 28.5 min).
  Linear scaling estimate, not a measured full parallel-panel run.
- Prior probe was K1; its ~33 min/52-world estimate was invalid and withdrawn.
  GPU pilot code is also K1; 28.5 min is not measured K8 GPU throughput.

## Full-corpus ES protocol — fixed before first combined generation
- Verified combined registry: original FIT doc SHA 7bbd41f3 + wmdata d3ad5638;
  all 12 FIT worlds x 2 arms and all 40 wmdata_fit worlds x 4 arms = 184 contrasts.
- Both corpora use original FIT scales/anchor scales/crossing width. Fitness is
  SAME normalized paired upstream/road/crossing MSE + absolute road/crossing
  anchors; average arms within each world, then worlds equally. J/source reported.
- Frozen policies: CTRL0 64f70af4 with residual zeroed; speed-only MAPPO d9042cc5;
  PI/CIRCLES published controller code. Exact pre-t0 pairing and request forcing.
- Budget amendment: calibrate at 120 s (existing training-stage horizon), K1 CRN,
  full corpus every candidate/generation; no world/arm subset. 300 s and K8 remain
  independent diagnostics/verdict, never tuning feedback.
- Option (b): six EXISTING PhysicsConfig scalars (accel_min/max_mps2,
  lane_physical_margin_m, mainline_courtesy_zone_m, free_road_gap_m,
  speed_limit_mps), identified in es.py; incumbent WM2 values, optimised, no
  hand-set behavioural replacement. Entry scalar and all WM2 tensors frozen.
- Antithetic ES: population 2, sigma .10, lr .01, seed 140014, <=8 generations,
  <=12600 s; stop after 3 stale generations. Selection: minimum TRAIN objective.
- Host pinning: FIT on gpuA; wmdata on cpuQ. Reserve cores for live K8 probes;
  every rollout checks frozen WM2 tensors; immutable population/GEN/checkpoint
  per generation, canonical reload; no validation feedback.

## ES curve / verdict / recommendation
- Calibration: RUNNING (gpuA driver PID 195604); `es_cpu_v3/es_run_full`, cpuQ `es_run_full`.
- Curve (generation winner by objective, TRAIN paired J MAE veh-s): pending generation 0.
- Source contrast MAE reported separately: pending.
- Scoring-ready selected checkpoint: pending; 20 CPUQ + 3 receipt checks PASS.
- Verdict INCOMPLETE / NOT ADOPTABLE; unchanged A vs WM2 AND entry, B, C1/accept.
- Missing registered inputs: accept panel, C1 cohort/runner, candidate quartet completion.
- MAIN: bind C1/accept inputs and allocate A/B GPU slot after checkpoint readiness.
- Recommendation: retain WM2 until registered independent A/B/C1/accept PASS;
  120 s TRAIN calibration alone does not establish 300/600 s response quality.
