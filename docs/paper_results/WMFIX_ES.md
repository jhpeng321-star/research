# S-WMFIX-ES — CPU calibration receipt — 2026-10-31

## Resume / resources
- Final resume at 29faed51; prior work retained; existing jobs reconciled before action.
- Earlier missing-CTRL0-feedback smoke failure was fixed before full-corpus ES.
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
- Antithetic ES: population 2, sigma .10, lr .01, seed 140014; original cap
  <=8 generations/12600 s, patience 3; MAIN 17:53 supersedes: STOP after g2.
- Host pinning: FIT on gpuA; wmdata on cpuQ. Reserve cores for live K8 probes;
  every rollout checks frozen WM2 tensors; immutable population/GEN/checkpoint
  per generation, canonical reload; no validation feedback.

## ES curve / verdict / recommendation
- STOPPED per MAIN 17:53; g0/g1/g2 complete (1980.4/3025.3/2810.9 s).
- Generation-winner TRAIN paired J MAE veh-s: g0=237.69; g1=243.87; g2=233.78.
- Selection fitness: g0=0.368865; g1=0.338086; g2=0.353828 (no best improvement).
- Source contrast MAE separately: g0=26.62; g1=23.67; g2=26.08.
- Minimum TRAIN objective selects g1/c1; paired J MAE 243.87 (+2.60% vs g0).
- Immutable selected artifact: gpuA `es_cpu_v3/es_run_full/generation_001.pt`;
  SHA256 `4683f0b7e6dbd9a03e5405e88e1114206d44d4e9440b342ff61c7e9f02d99056`.
- g3 had already started before resume; partial cells retained, excluded from selection.
  Driver 195604, finalizer 199849 and 30 remaining cpuQ workers stopped by owned PIDs.
- Both hosts: zero active owned ES processes; CPU allocations released; collectors absent.
- Verdict NOT RUN / NOT ADOPTABLE. No A/B dispatch; prior GPU-slot request withdrawn.
  A vs WM2 AND entry, B SPEEDHEAD, C1/accept rules unchanged; no PASS claimed.
- Recommendation: STOP generic physics-scalar sweeps and retain WM2: fitness/J diverge.
- Receipts: `wmfix_es_receipts/{GEN_002,*STOP_MAIN_1753,FINAL_STOP_*}.json`;
  final: `wmfix_es_receipts/WMFIX_ES_FINAL.json`; registration: root `REGISTRATION.md`.
- Resume 2026-11-03 +08: no ES jobs/collectors; selected SHA reverified; cpuQ run directory absent. Audit: `wmfix_es_receipts/RESUME_STOP_VERIFIED_20261103.json`.
