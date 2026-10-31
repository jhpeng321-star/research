# S-WMFIX-ES — CPU calibration receipt — 2026-10-31

## Resume / resources
- Clean start at 89d7e29b; frozen-session artifacts retained; no ES jobs running.
- Combined smoke had failed on missing CTRL0 feedback; no generation completed.
- MAIN 13:38 supersedes sharing limits: cpuQ <=30 workers; gpuA <=18 low-priority
  CPU workers (20-CPU quota). GPU processes and S-WMDATA processes untouched.
- gpuA ES code is isolated in `wmfixmerge20261031/es_cpu_v3`; cpuQ owns
  `/home/ubuntu/wmfix_es`. No SUMO, prohibited inputs, deletions or push.

## Throughput — correction and genuine K8 measurement
- Earlier `es/PROBE_*_group0.json` claims of K8 were wrong: Runtime used K1.
- Valid K1 measurements: gpuA 1920.0 s/group, max RSS 1775 MB; cpuQ 1766.6 s,
  max RSS 1372 MB; two paired arms x {120,300} s, entry path/mix ON, threads=1.
- Genuine K8 now dispatched: 8 distinct sticky-future processes per host,
  each running that same four-cell workload from one common eight-future plan.
- cpuQ PIDs 3030648-3030655; gpuA 193235-193242. `probe_k8/DISPATCH.json`,
  per-future PID/log/PROBE receipts. Completion/timings still pending.
- Prior ~33 min combined 52-world panel estimate omitted K8 and heterogeneous
  arm counts; withdrawn. K8 CPU extrapolation is ~8x K1 CPU work, not free.
- Requested GPU comparison reference: ~28.5 min/panel; its pilot code is K1,
  so it is not evidence of measured K8 GPU throughput.

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
- ES curve: no complete combined generation yet; earlier partial FIT cells are
  excluded. All four wmdata arms smoke PASS (31.2-31.9 s); 20 unit tests PASS.
- Verdict: INCOMPLETE, not adoptable. A vs WM2 AND entry, B SPEEDHEAD eight
  worlds, C1 and canonical accept.py remain the unchanged mandatory gates.
- MAIN GPU-slot request: reserve A/B evaluation after selected ES-WM checkpoint;
  corrected K8 CPU cost cannot support the prior optimistic deadline estimate.
- Recommendation: finish full-corpus CPU ES, report measured curve and long-
  horizon transfer honestly; adopt only if the registered A/B/C1/accept verdict passes.
