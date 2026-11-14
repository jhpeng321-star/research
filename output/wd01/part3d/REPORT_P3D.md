# WD-01 Part 3D — holdout block + structured SUMO-emulating Part A (PARTIAL)

**Status: PARTIAL — validation-only session; holdout NOT accessed.**

The 5 holdout roots were never loaded, scored or tuned on in p3d (gate
evidence: results/holdout_untouched.json — the p3b deviation artifacts
remain byte-identical, mtimes 2026-09-25T21:19Z).  Per the brief's
stop rule, the holdout was not evaluated because the structured rollout
was not completed within the session budget; the diagnosis and the
remaining work are documented below.

## A. Physical holdout block — COMPLETE

partA_data/holdout_block.py + hooks in loader.py::load_root and
part3/p3lib.py::root_arrays (registration_sha256 passthrough).  A
holdout root loads only with registration_sha256 = H where H is the
sha256 of an existing REGISTRATION file AND a record JSON in the
holdout-records registry recorded H before the access (the recorded
file's current hash must still match).  Train/validation roots are
unaffected.

Tests: partA_data/tests/test_holdout_block.py (5 tests, all pass):
no-hash raises; wrong-hash raises; correct SYNTHETIC registration
passes; tampered registration fails; train/validation roots
unaffected; loader.load_root raises for a holdout root without a
hash.  See HOLDOUT_BLOCK.md.

## B1. SUMO IDM emulator — COMPLETE, with sigma finding

sumo_emu.py implements SUMO MSCFModel_IDM (delta=4, single-step default
update, v' = clip(v + a*0.5, 0, 1.2*limit), s' = s + v'*0.5 — the
MEASURED form; free driving = interaction dropped beyond 250 m;
per-segment network limits: main 31.29, accel ramp 22.22, merge 31.29).

One-step check on TRAIN roots with TRUE vTypes (residual = logged
v[t+1] - deterministic IDM v[t+1]):

| vType | regime | n | MAE | mean | sd |
| --- | --- | --- | --- | --- | --- |
| aggressive | following | 4692 | 0.405 | -0.404 | 0.388 |
| aggressive | free | 2025 | 0.751 | -0.751 | 0.899 |
| normal | following | 9012 | 0.213 | -0.212 | 0.260 |
| normal | free | 4224 | 0.244 | -0.244 | 0.574 |
| cautious | following | 5462 | 0.094 | -0.082 | 0.415 |
| cautious | free | 4014 | 0.225 | -0.225 | 0.464 |

Sigma finding: SUMO's IDM is deterministic in this build (vType sigma
does not enter the IDM equation); residuals are small (following MAE
0.09-0.41 m/s) with a systematic negative mean and vType ordering that
tracks speedFactor.  Modeled as per-(vType, regime) Gaussian noise
N(mean, sd) with CRN streams (results/b1_sigma.json).

## B2. vType posterior — trained, calibrated on validation

3-class MLP on the 11-feature 10 s history summary; trained on the 17
train roots (4303 samples at t0 in {240,360,480}).  Validation (4
roots, 989 vehicle-steps): log loss 0.7931, accuracy 0.6946 vs
prior-accuracy 0.7027 — the history signal barely exceeds the prior.
Documented as a limitation of the current history summary features.

## B3. Lane-change hazard — NOT COMPLETED

Design (per brief): per-direction logistic hazard on target/own-lane
gaps+speeds, merge-zone position, vType sample; class-balanced training
on declared LC events; reliability calibration on validation;
stochastic firing with member CRN; safety gap check; legality 0<->1 and
2->1 in the merge zone only.  models.Hazard and the feature builder
exist (models.py); the training/eval integration was not completed in
this session — documented as the main open item.

## B4. Ensemble rollout — NOT COMPLETED

p3d_rollout.py is an incomplete draft (structured rollout skeleton with
births/noise/collisions but the LC-hazard application and eval glue
missing).  The p3c finding stands: member-averaged metrics are the
valid ensemble form; the position-averaged 'ens_mean' is invalid and
was dropped.

## Validation gate — NOT EVALUATED

Because the structured rollout was not completed, the p3d gate (beat
B2 4.97 m/s at H=60; collisions <= 1/1000; LC within +-20%; delay
coverage >= 50%) could not be evaluated.  The p3c reference numbers
stand: B1 9.52, B2 4.97, B3 4.58, p3c residual model 6.88 (collisions
13/1000) at H=60 on validation.

## Receipts

- vehicle_dynamics.pt sha256 ded83907...d4a8f5 unchanged (start/end).
- p2C MANIFEST.json sha256 2439004d...c82dd27 unchanged.
- Weights: part3d/weights/posterior.pt, part3d/weights/hazard.pt
  (sha256 in results/p3d_weights_manifest.json); backups in
  D:/research-dense-backups/wd01/part3d/ with SHA256SUMS verified.
- pytest: all part3 + partA_data tests pass (13 + 5 + others).
- Git: committed on zcode/wd01, part3d + partA_data holdout-block paths
  only.
