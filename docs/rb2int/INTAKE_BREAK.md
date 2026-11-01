# BREAK engineering intake — snapshot116b2147

Default-OFF source corrections reviewed and integrated:

- `physics.sumo_cf_lane_speed_limits=true`: applies current network lane limit
  to inferred reference desired speed once in SUMO IDM, without overwriting the
  persistent driver setting. SUMO IDM131–133 source rationale retained. Ramp
  approach22.22m/s was already handled in insertion/merge, but absent in CF.
- `variants.FULL.match_cav_execution_settings=true`: same registered motion-fitted
  CAV vector in initial/lane/admission settings and the unchanged executor.
  HDV draws, controller, plant and latent RNG streams stay as registered.
- Explicit development world/future subsets supported by the shared runner.
  Conflict with extended-panel hooks resolved with one selector: final execution
  cannot shrink its approved roster or relabel a subset as complete K8.

Peer development evidence only: source queue accounts for84% of600s bias;
this is accounting, not causal attribution. Mainline-LC shutdown and merge-advice
shutdown both worsen road delay and do not explain source bias. They are rejected
attribution interventions, not selectable v5 recipes. The two source-correction
traffic screens are pending; no improvement is claimed from source parity alone.

Worker explicitly excludes reserved8/WMDATA2 and retains failed/truncated archives
and operationally retired own canaries. INT did not restart or signal these jobs.

INT CPU command:

`python -B -m pytest -q tests/test_rb2break.py tests/test_rb2break_execution_settings.py tests/test_rb2break_lane_limits.py tests/test_rb2int_task_roster.py tests/test_rb2int_runtime_isolation.py tests/test_rbint_departure.py tests/test_rbint_motion_posterior.py tests/test_rb2int_lane_cal_composition.py`

**22 PASS / 0 skips / 6.36s.** Shared-recipe auto-merge retains both LANE/CAL
conditioning and calibrated wrapper preservation. HANDOFF/source-runner conflicts
resolved explicitly. Full smoke/candidate freeze/LP intake remain pending.
