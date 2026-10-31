# S-WMFIX-PILOT FAIL — pre-registered gate, 2026-10-31 13:37 +08

Decision rule (MAIN REVIEW DECISION 10:40, amended per 11:52): PASS required the
best of steps 10-30 to improve BOTH horizons by > 2x frozen-rerun SD (= 0 by
determinism, r1==r0), with finite gradients and frozen-bitwise retained.

**VERDICT: FAIL — the 120 s horizon never improves; 300 s improves early then
diverges by step 30. No curriculum training was started (pilot_continue
returned exit 2 right after writing PILOT.json; FINAL.json absent by design).**

## Panel results (gpuA-native decision inputs; 12x2x{120,300} s TRAIN panel paired-J MAE)
Receipts sha256: PANEL_step10 d90bf90b..., step20 79c6ec8c..., step30 ab597eef...,
PILOT.json 61a9059a... (all in gpuA train/, copied hash-verified to local).

| step | 120 s MAE | d120 vs 283.288 | 300 s MAE | d300 vs 2185.236 |
|-----:|----------:|----------------:|----------:|-----------------:|
| 0 (baseline r0/r1, SD 0) | 283.288 | - | 2185.236 | - |
| 10 | 300.457 | +17.17 | 1954.140 | -231.10 |
| 20 | 286.205 | +2.92 | 2012.755 | -172.48 |
| 30 | 312.138 | +28.85 | 2501.051 | +315.82 |

best-of-steps: 120 s 286.205 (improvement -2.92 <= 0 -> FAIL), 300 s 1954.140
(+231.10). finite_gradients true, frozen_bitwise true, completed_steps 30/30,
cap_stop false, elapsed 5291 s.

Cross-check: gpuR step10 (a1854e15..., cross-hardware guard measured at step 0:
+0.20/+1.39 veh-s) 300.502/1959.673 — same shape, ~0.15-0.3% hardware offset;
decision correctly used gpuA-native panels only (WMFIX_PANEL_RELAY.md 12:40-12:47).

## Diagnosis (what the numbers say)
1. The head's gradient path is REAL but horizon-asymmetric: at step 10 it moves
   300 s by -231 veh-s while 120 s moves +17 the WRONG way. The stage-0 training
   horizon is 60 s; the training loss fell 0.0593 -> 0.0010 (train objective,
   60 s rollouts) yet the 120 s paired panel never improves -> short-horizon
   fitting does not transfer; the 120 s response is dominated by structure the
   60 s rollout objective does not see (consistent with FEASIBILITY: one-step
   300 s -497 / 120 s +25; teacher ceiling exists only when teacher-forced).
2. Unstable across steps: step 30 is worse at BOTH horizons than baseline
   (300 s +316). The Adam trajectory at lr 3e-4 overshoots the panel objective
   even while the training loss keeps falling -> the panel objective and the 60 s
   training objective diverge within 30 steps (overfit to 12 FIT groups at 60 s).
3. Frozen-bitwise and finite gradients held throughout -> the failure is the
   OBJECTIVE/HORIZON mismatch, not the mechanics.

## Consequences per the pre-registered protocol
- No <= 6 h curriculum training; no verdict panels (sharded panel-A/quartet
  dispatch stays staged and unused until a passing pilot exists).
- Next: option (c) design (training-only smoothed admission) per the 10:40
  decision, coordinated with the already-running fallback tasks S-WMFIX-C
  (c4/wmfixc: smoothed admission/crossing surrogate + finer spatial resolution)
  and S-WMFIX-ES (c4/wmfixes: derivative-free ES calibration of the same
  head/params on the same 12 FIT groups — sidesteps the hard-event gradient
  problem that this pilot just re-demonstrated at the panel level).
- GPU state after FAIL: gpuA + gpuR free of S-WMFIXMERGE processes (verified:
  PID 160005 exited, no compute apps; gpuR watcher exited 13:10:15 "ALL PANELS
  DONE" after skip markers).
