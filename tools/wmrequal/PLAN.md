# S-WMREQUAL — owner approval 2026-10-30 16:15

Base: c4/wmrequal, 9613f956; WM2 SHA256
b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436.
Scoring only: no optimizer, training, SUMO or push. gpuA only, fraction <= .90.

1. Bind permitted existing validation/pairs300/calibration panels and PANEL120
   (<=120 seconds), runtime/controller/checkpoint hashes; port cbb7920b's mix.
2. Recompute all-fleet road + source waiting, alongside t0-HDV road, at
   60/90/120/180/300 seconds. Save eight coupled draws, median paired gain,
   q10 paired gain and the training median>2sigma AND q10>0 decision.
3. Source labels reproduce 51e32bf8:replay_complete.py:departures/build_record:
   independent damping_requests_v1:<seed>:<stream> exponential RNG, rounded
   departures, due count minus latest recorded inserted ordinal +1. Never
   reconstruct CAV/driver assignment. Retain the original half-second rectangle
   integration, explicitly distinguish boundary/continuous-ledger sensitivity.
4. Recompute null-perturbation sigma only if complete permitted N1 tapes exist.
   Deterministic replicate noise=0 is distinct from perturbation sensitivity;
   unavailable perturbation sigma is unavailable, never the old metric's sigma.
5. Disjoint road bins plus source give additive error decomposition. Discharge
   association/queue conservation is diagnostic, not a causal capacity estimate.
   Real crossings use continuous recorded identity bracketing; no inferred exits.
6. CCT-1 per panel; CCT-2 pools only pairs300+validation, 2000 world draws seed 0,
   >=20 confident gains and >=.80 correct, H_G largest pass in 120/180/300.
   Calibration and PANEL120 remain informational. No authority publication until
   training-objective sigma and contract are both measured.

Forbidden paths/seeds follow the owner prompt. Remote code is staged only in
owned /root/autodl-tmp/c4/wmrequal20261030; record child PID/start token.
Commits after setup, scoring, decomposition, final <=90-line WMREQUAL report.
