# Round 9 objective amendment — supplied section (2) rules

This is a transcription of the rules supplied in the execution brief, not a
verified copy of the external advice file. The original is outside this worktree
and was not read under the workspace-only access restriction.

- Set `loss.merge_core.weight = 0` and retain `metrics = true` in
  `configs/c4_wm_v10round9.toml`.
- Exclude the recorded-matched on-policy NLL from the loss and backward when its
  weight is zero. Never multiply an infinite NLL by zero. Retain events,
  zero-mass counts, and scored seconds under explicitly diagnostic keys.
- Log `merge_pivot_on_objective_disabled = 1` for the disabled objective.
- Keep TF normalization exactly `(0.6 / 395.2) * L_TF`, without renormalizing
  its weight. Preserve the existing per-car definition of `L_TF`.
- Preserve F2 survivor conditioning and the full pathwise adjoint.
- Use this same TF+F2 objective in the exact fitter, stage-2 fitter, and audit.
- Preserve support failures, comparison tolerances, and non-finite TF terms.
