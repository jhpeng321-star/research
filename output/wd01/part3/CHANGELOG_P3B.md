# CHANGELOG_P3B - fixes after the reviewer stop of the p3 run

Stop reason: validation rollouts broken (val speed RMSE 13.16 m/s at H=60
after phase 1; phase-2 multi-step loss ~2e6 and degrading validation).
Scope: diagnose on VALIDATION ONLY, fix, retrain (weights/run2/), then the
single registered holdout pass under REGISTRATION_P3_v2.md.

Run-1 weights preserved: sha256 recorded in
results/run1_weights_sha256.txt (originals) and
results/run1_weights_sha256_after_move.txt (copies in
weights/run1_stopped/).

Diagnostics D1-D5 and fixes are appended below as they complete.
