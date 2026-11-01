# Original gpuR per-CAV construction failure (retained)

- Root `/root/c1switch2_run_edcdbebb738ef1c6`, owned supervisor973470.
- Failed first world969139, zero controlled decisions,300sim-s charged.
- No CLOSE receipt: do not label closed actual as300; historical reservation retained.
- Error: `AttributeError: 'CommandOwnership' object has no attribute 'execute_residual'`.
- Diagnostic receipt `SELECTOR_PROGRESS_cav_mappo_gpuR_1793535921.json`.
- Frozen training attaches the hook even when it does not already exist.
- Correction records presence/absence, installs the training mixed hook and restores
  exact original hook presence and acceleration conversion when switching to fleet.
- Regression constructs actual peer grouped execution against both hook ABIs,
  performs WALL start/stop and validates exact fleet handback;37targetedCPU PASS.
- Original gpuR scope terminal. No failed-world retry, refund or source overwrite.
- Original cpuQ stage remains wholly unstarted and may be replaced by a source-bound
  CPU repair after proving no claims/jobs/ledger/STOP; all old bytes retained.
- Both GPUs remain HELD; any prospective CPU-only observer cannot start GPU work.
