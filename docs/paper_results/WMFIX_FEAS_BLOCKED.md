# WMFIX real TRAIN feasibility — BLOCKED (fail-closed) + gradient diagnosis

FEASIBILITY.json sha256 `65771d0290d75df530110cfe91f3ac65f09424ed2b5318db94c35bee0627f7b9`
(gpuA, cuda, 2026-10-31 ~10:45, 30.5 min panel seconds 1829.7). No gate was changed;
no training was launched. This file is evidence for MAIN's launch decision.

## Sub-gate results
- PASS: source_clock_match (contract f5548777..., residual_remains true);
  frozen_bitwise true; component gradient norms finite J 57.1 / crossing 2.07 /
  source 0.0 (hard service has no source gradient, by design); admission ablation
  sane (ledger-only vs inferred differ only through admission); source contrast
  error exactly [0,0]; teacher ceiling at BOTH horizons: 120 s 283.3->169.7,
  300 s 2185.2->525.6 (substituting recorded upstream — enormous learnable room).
- FAIL (hence BLOCKED, launch_ready false):
  1. finite_difference_ok false: analytic 0.0317 vs central FD 1.1835 at eps 1e-3.
  2. one-Adam-step panel: 300 s improves 2185.2->1687.8 (-497, noise 0.0) but
     120 s worsens 283.3->308.4; gate requires improvement at BOTH horizons.
     Probe loss worsens 0.0593->0.0733 (anchors included).

## Gradient diagnosis (probe-level re-analysis, group 0, 60 s; /tmp/diag_grad.py)
eps sweep on the max-grad element of merge_response.net.2.weight
(analytic 0.0199 in this process; feasibility process measured 0.0317 at its own
max element — backward on GPU is itself nondeterministic at the element level):

| eps | FD slope | ratio to analytic |
|---|---|---|
| 1e-3 | +0.927 | 46.6 |
| 1e-4 | +0.646 | 32.4 |
| 1e-5 | -1.009 | -50.7 |
| 1e-6 | +7.041 | 353.8 (float32 loss floor: f diff ~1.4e-8) |

Interpretation: the free-running rollout has DENSE hard-event breakpoints
(sumo-like admission/crossing decisions flip under ~1e-5-scale weight changes);
every measurable eps is event-noise dominated, and below ~1e-6 the FD is float
noise. The FD-agreement gate is therefore UNSATISFIABLE-AS-WRITTEN for this
objective class, independent of any bug: autograd reports only the smooth part.
5-Adam-step probe (same group/horizon): pairedJ 26.45 -> 34.0 -> 17.3 -> 17.8 ->
15.1 -> 18.2; loss 0.0593 -> 0.0310 — noisy but real descent signal at 60 s,
consistent with the 300 s panel improving and the teacher ceiling.

## Options for MAIN/owner (not executed here)
- (a) No head training: the entry path alone fixes source (300 s 584->82.5) but
  leaves upstream untouched and worsens paired J vs WM2 (unmasked cancellation) —
  not adoptable; today's fix would be "none".
- (b) AMEND the feasibility gate with owner approval: replace FD-agreement
  (unmeasurable for hard-event rollouts) with a pre-registered k-step panel
  descent pilot: ~20-50 Adam steps on the 12 FIT groups must reduce the full
  12x2x{120,300}s panel paired-J MAE beyond frozen-rerun noise at BOTH horizons,
  THEN launch the 6 h run. Cost: ~30-60 min GPU pilot + re-review. This keeps
  every other safeguard (frozen-bitwise, telemetry, immutable checkpoints).
- (c) Soften discrete paths for TRAINING only (differentiable admission
  relaxation, flag OFF at inference; identity-bracketed crossing supervision
  already exists): code change + CPU tests + new feasibility; ~2-4 h before any
  GPU pilot; largest change, needs its own review round.
Recommendation: (b) — smallest honest step that can still meet the owner's
evening target if the pilot descends; (c) only if (b)'s pilot fails to descend.

## Boundary compliance
Probe-level diagnostics only (eps sweep + 5 single-group steps, cuda, <3 GB);
no GPU training run, no gate/FEASIBILITY edits, no SUMO, no sealed inputs.
