# PILOTDIAG — why exploratory pilots drift worse than CTRL0 (S-PILOTDIAG, 2026-10-31)

CPU repros on saved pilot checkpoints (tools/pilotdiag/repro_dense.py, evidence_pilotdiag/*.json);
no new GPU training. Facts unchanged from CAPREL/ESLONG/GATEMEAS/LONGGRAD/DEADZONE2/WMREQUAL3.

## Q1. Where the dense 60/90 s gradient explodes
- Per-row/per-step instrumentation (futures=8, non-deterministic replay): adjoints stay finite
  and O(1) — state-cotangent max 0.53 (abs193/s1/h91, peak step 3), 0.044 (abs192/s1/h61);
  no inf/NaN, no emergency-floor or segment-boundary discontinuity found.
- The explosion is in the CREDIT, not the VJP transport. At 91 s one row per start dominates:
  the kind-3 harm row (coefficient 18.8): s1 (w963625) +317.8 veh-s (loss +0.130) vs
  s3 (w963904) -290.4 veh-s (loss -0.120). Same s1 start at 61 s: only -43.9 veh-s.
  The +/-300 veh-s tail enters between 61 s and 91 s — the WM's chaotic edge (WMREQUAL3:
  60/90 s mostly right, beyond unreliable) — and flips SIGN across starts.
- abs192 -> abs193 (policy leaves zero; residual min -0.362 -> -0.486 m/s^2) activates the
  harm row on part of the panel: gpuA dense grad 57 -> 215,287; gpuR B-L 29,898. gpuR u195
  decomposition: residual_head 29,897.69 of 29,897.74 total (encoders 5-44). Grad clip 100
  then rescales the whole step: every update is a max-size Adam(5e-5) step in a start-dependent,
  sign-flipping direction — a random walk away from CTRL0, and dJ worsens monotonically.

## Q2. Real worsening, not rescore noise
- Trend is monotone on both runs across >=3 updates: B dense dJ +8.6 -> +13.8, 300 s +60 -> +439;
  B-L 300 s +126 -> +405 -> +578 (single-start pre-update K8 telemetry is noisy per number,
  but sign and trend are consistent, and independent CPU starts reproduce them).
- The policy really brakes: executed residual (vs CTRL0 executed accel) is 100% braking where
  nonzero (mean_delta -0.049, fraction_nonzero 0.47-0.52, gpuR u195; deepening abs192->193 in
  CPU repros), ES workers reach min executed accel -5.4..-8.0 m/s^2, action min -5.09.
- Conclusion: harmful net braking is real; the rescore just under-reports its magnitude.

## Q3. Fix (implemented, default OFF, CPU-tested)
- Option (b) `[trainer] update_acceptance_panel` (policy14_cli: requires --gate-panel):
  deterministic CRN rescore of the frozen gate panel before/after every update; actor+Adam
  rolled back unless mean panel J improves monotonically. No behavioural thresholds.
- Option (d) `[trainer] imagination_step_cotangent_normalize` (dreamer/longgrad/parallel_
  futures): unit-normalise the accumulated state cotangent at each controller-input boundary
  during future-tape VJPs; raw pre-normalisation adjoint kept in telemetry. Bounded transport,
  cheap; preconditioning, not the exact derivative.
- Rejected: (a) antithetic dense doubles cost and keeps ES-level noise in the 60/90 s term;
  (c) smaller lr/trust region only slows the same sign-flip random walk.
- Tests: tests/unit/test_pilotdiag_controls.py 7 PASS (acceptance rollback, normalisation,
  default-OFF). Config: configs/c4_policy14_candB_pilot_rel2.toml (both guards ON).

## Relaunch
- Config: `configs/c4_policy14_candB_pilot_rel2.toml` (= c4_policy14_candB.toml +
  update_acceptance_panel + imagination_step_cotangent_normalize), fresh zero start.
- Core command (owner-dispatched exploratory wrapper as in tools/candB/PILOT.md):
  `python -B -u -m wms.control.policy14_cli --config configs/c4_policy14_candB_pilot_rel2.toml
   --wm-ckpt <b8797242...> --wm-config <f95f7c4c...> --gate-panel <frozen panel 3eaa69a1...>
   --horizon-gate <be0297ab...> --exploratory-owner-<ts>` — same frozen inputs as the B-L run
  (input_sha256 in evidence_pilotdiag/gpuR_candBL_updates.jsonl u195).
- Watch: acceptance_panel_J_before/after + acceptance_mean_J_delta per update; dense grad norm
  should drop O(100x) with (d); if (b) rolls back >2/3 of updates, dense credit itself is noise
  at 90 s and the horizon should return to 60 s dense / ES-only 300 s.

## Evidence
- evidence_pilotdiag/abs19{2,3}_s{1,3}_f8_h{3,61,91}*.json (per-row credit, per-row/per-step
  cotangent + residual stats); gpuA_retry_updates.jsonl (u192 grad 57); gpuR_candBL_updates.jsonl
  (u191-195: dJ trend, grad decomposition 29,898, executed residual, action stats).
