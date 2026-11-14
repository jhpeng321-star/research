# IDM Decentralized Training Contamination Report

Date: 2026-08-28

Status: quarantined diagnostic artifact; not scientific evidence.

## Affected directory

`runs/rllib/idm_decentralized_activation_smoke_v1`

The directory is preserved unchanged.  Its metrics and checkpoints must not be
used for checkpoint selection, learning curves, controller claims, or warm
starts.

## Cause

The original resumable trainer remained alive after its monitoring call was
detached.  A later process audit used an overly narrow command-line pattern
(`research-dense*train.py`) whose token order did not match the real command
line (`train.py ... D:\research-dense ...`).  The false negative led to a
second `train.py --resume` launch against the same output directory.

Both parents created independent Ray sessions and wrote the same CSV, JSONL,
training-state, TensorBoard, and checkpoint paths.  The second parent was
stopped first; the original parent and its complete Ray/SUMO child tree were
then stopped through its retained terminal session.  A subsequent CIM audit
found zero matching `train.py`, `raylet`, `gcs_server`, or IDM-transition SUMO
processes.

## Evidence of concurrent writes

- `metrics.csv` contains two different rows for iteration 12 / 24,000 steps.
- `metrics.csv` contains two different rows for iteration 13 / 26,000 steps.
- `checkpoints/iter_12` was created at 04:03:03, while every checkpoint file
  was modified at 04:03:33.
- `checkpoints/iter_13` was created at 04:03:50, while every checkpoint file
  was modified at 04:04:23.

This establishes that at least iterations 12 and 13 were overwritten by a
different training lineage.  Checkpoints after the collision are also
scientifically contaminated because their ancestry and shared output metadata
cannot be certified.

## Corrective controls

- Fresh training uses a new output directory and does not use `--resume`.
- `train.py` now takes an exclusive per-run process lock before starting Ray.
- The lock uses a Windows-safe process-handle liveness query.
- Fresh runs save a frozen `step_0` checkpoint and a complete command manifest.
- The replacement smoke is capped at 10,000 environment steps.
- Per-iteration telemetry records genuine eligible decisions, action
  probabilities and counts, option dose/completions/aborts, reward components,
  learner diagnostics, and traffic safety/service endpoints.

The contaminated directory remains useful only for diagnosing the duplicate-
writer failure and the earlier near-zero aggregate entropy.  It does not count
toward verification of a trained controller.
