STATUS: COMPLETE

BLOCKER 6 (wm_s_predictor.py normalize_states + window_label_delay): closed-loop
WM-S plugin predictions now fail closed. normalize_states refuses a single-state
trajectory, non-finite/shape-misaligned arrays, a first sample that does not
reproduce the live state (t0 equality on position/speed/lane), and a
non-monotonic or mis-anchored time axis. window_label_delay refuses a prediction
spanning fewer than the requested horizon (ceil(horizon/dt)+1 samples) instead of
returning 0.0, so the planner can no longer prefer an invalid candidate.

Evidence: test_closed_loop_predictor_rejects_a_single_state_prediction, plus
short-horizon and misaligned-t0 coverage in
output/wd01/wm_n1/closedloop/tests/test_wm_s_predictor.py.
Closedloop suite: 30 passed (threads 2), ruff clean. No other REVIEW.md item
applies to wms/cl files. No SUMO/CAL/HOLDOUT/sealed data touched.

sha: 124e43b284c6c4b37a74f2ed3c05207ace8aeda9

---

FOLLOW-UP (2026-10-06, commit 827de36): BLOCKER 6 residual closed. The first
patch still scored only the present/live slot intersection at t0 and accepted
any strictly-increasing (possibly nonuniform) time axis while integrating with
a fixed dt_s. normalize_states now fails closed when any live on-lane slot is
absent from the plug-in output at t0 (no silent intersection), and requires the
time axis to be the uniform DT_S grid. window_label_delay additionally requires
its grid spacing to equal dt_s and integrates lost-time with the actual sample
deltas, so scoring matches the prediction's real intervals.

Evidence (output/wd01/wm_n1/closedloop/tests/test_wm_s_predictor.py):
- test_closed_loop_predictor_rejects_missing_live_slot_at_t0
- test_closed_loop_predictor_rejects_nonuniform_time_grid
- test_closed_loop_predictor_accepts_explicit_uniform_dt_grid
- test_window_label_delay_integrates_actual_deltas_and_refuses_nonuniform
Closedloop suite: 34 passed (OMP/MKL threads 4); ruff clean on both changed
files. No SUMO/CAL/HOLDOUT/sealed data touched; no process exceeded 6 GB.
sha: 827de364966baf23163441051d3e7a86b1ab85e7
