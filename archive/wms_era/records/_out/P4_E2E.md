# P4 E2E dry run — TRAIN decision pipeline
Status: COMPLETED (CPU inference; no SUMO/GPU/SSH).

`_out/run_p4_e2e.py` streams 20 TRAIN states (HORIZON_ERROR meta.states)
recorded SUMO runs from `_ext/campaign_chunk4` (READ-ONLY) into
`_out/wms_train_release.json`, then runs `python -m wms_decide
--wms-eval-model <spec>` to score each plugin's predicted states.
Delay = sum(dt * clip(1 - v/v_free, 0, 1)) seconds; both arms share the
same start state, seed and random draws. Recorded truth uses the same
normalized delay on the recorded native/PI runs.

| plugin | tau_s | sign_agr | rank_corr | regret_s | always_native | improve_s |
| --- | --- | --- | --- | --- | --- | --- |
| keep_speed | 30.0 | 0.250 | n/a | 57.7 | 57.7 | 0.0 |
| keep_speed | 60.0 | 0.250 | n/a | 353.4 | 353.4 | 0.0 |
| keep_speed | 120.0 | 0.100 | n/a | 1359.8 | 1359.8 | 0.0 |
| keep_speed | 300.0 | 0.050 | n/a | 9542.8 | 9542.8 | 0.0 |
| pure_physics | 30.0 | 0.450 | 0.270 | 45.3 | 57.7 | 12.4 |
| pure_physics | 60.0 | 0.400 | -0.123 | 370.3 | 353.4 | -16.9 |
| pure_physics | 120.0 | 0.350 | 0.134 | 1210.1 | 1359.8 | 149.8 |
| pure_physics | 300.0 | 0.150 | -0.272 | 8661.9 | 9542.8 | 880.9 |

sign_agr = agreement of sign(predicted PI-native) with recorded;
rank_corr = Spearman over the 20 states; regret is the recorded delay of
the arm chosen from predicted differences; improve = always_native - regret
(negative means always-native was better). Predictions are the physics /
keep-speed baselines, not the trained WM-S checkpoint (P4 gate is the
paired-difference decision, validated here end-to-end).
Artifacts: `_out/p4_<plugin>/DECISIONS.json` (per-state + `decision_summary`).
