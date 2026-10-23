# BASE_FOLLOWER_DIAG - why the base WM over-predicts the lane-1 target follower's acceleration before a merge

S-BASEDIAG, 2026-10-24 03:00-07:00. Diagnosis only: no training, no SUMO, no model-code change. Local CPU (8 workers), TRAIN
windows only (`c4-v10r10b/tmp_r10b/roots` -> `c4-v10r9fix/data/r9_train`, 192 episodes, v5_native + genrand_plain). No sealed /
holdout / eval-seed path, no `hetdata_truth`, no vType file or LCSTATE log header was opened (only the LCSTATE.md aggregates and
the population medians already in `src/wms/train/losses/driver_prior.py`). Checkpoints: round 8 stage300
(`c4-v10r9fix/data/r9_train/wm_s_stage300.pt`, sha256 ae094e59..., the base rounds 9/10/10b/11 share) and round 10 final
(`c4-v10r10/tmp_diag/r10/wm_s.pt`, the copy S-FIXBDIAG pulled from gpuR `runs/c4_wm_v10_round10`; read-only, nothing copied from
gpuR by me). Scripts: `tools/diag_base_*.py` (branch `c4/v10r11`). Raw outputs: `tmp_basediag/` (not committed).

## 0. Answer first

**Root cause (measured).** The +1.3..1.5 m/s^2 is produced by the **IDM core**, not by the residual, the gate, a courtesy remnant
or the plant clamp, and its larger part is **not merge-specific**:

1. **Generic: the IDM core's driver settings cannot reach the data's population (about half of the bias, ~+0.6 of +1.34 in
   the last 3 s).** The v10 line still uses the v5 `legacy` encoder: encoded = legacy default (a 2.49, b 4.2) + shrink x
   tanh(.) x bound (0.8 / 1.5). The floor is a >= 1.78, b >= 2.87, and the encoder sits **on** that floor for most drivers
   (a p1 = p10 = 1.78, median 1.87; b p1 = p10 = 2.87; the p90 values 2.49 / 4.2 are drivers without history). SUMO HDVs follow
   textbook IDM (population a 1.2, b 2.0, T 1.2, s0 2.0, the medians already in `driver_prior.py`): far from the merge
   (s < -400 m), slow lane-1 HDVs whose leader pulls away accelerate +0.90 / +1.10 / +1.03 / +0.84 m/s^2 at bumper gaps
   3-5 / 5-8 / 8-12 / 12-20 m. Textbook IDM gives +0.88 / +1.07 / +1.14 / +1.18; the WM's IDM gives +1.25 / +1.91 / +2.20 / +2.36.
   Start-up from a queue is therefore about 2x too fast everywhere, and the jammed merge zone is where followers sit at
   v ~ 1 m/s with an opening gap (median 12 m) in front of them. Replacing only a, b by 1.2 / 2.0 (residual unchanged) moves the
   last-3-s bias from +1.34 to +0.72 and the matched-control slow-start bias from +1.52 to +0.78. This is the issue V5_DIAGNOSIS Q3
   found on 10-15 (fixed in v6 by `driver_param_mode = "physical"`); the v10 line descends from v5m, so it never got that fix.
2. **The learned residual cannot correct it (and adds one-step noise).** The attention residual head is dead: its pre-gate output
   is -1.5 (tanh bound) on 100 % of rows at every k (p1 = p50 = p99 = -1.5). The saturating gate has g_max ~ 0.39, so that head is a
   constant -0.36..-0.58 offset. The temporal MLP is state-dependent but has no merge or neighbour context. Net residual in the last
   3 s: -0.10. One-step MAE on all lane-1 HDV rows: model 0.98; IDM core alone with a = 1.2, b = 2.0 (encoded T, s0, v0) and no
   residual 0.43 (v > 15 m/s: 0.80 vs 0.36; after the merge: 1.07 vs 0.32).
3. **Merge-specific: the SUMO target follower "holds" (does not accelerate); the WM has no such term (the remaining ~+0.75..0.85).**
   With SUMO's own textbook IDM the target follower in the last 3 s should accelerate +0.84; recorded is 0.00. "Do not accelerate"
   (min(own IDM, 0)) fits it: bias -0.12, MAE 0.48 (own IDM +0.84 / 1.05). The ramp car as a full IDM leader at its current
   position over-brakes by -5.0, because it is almost alongside (median bumper gap 1.5 m, 34 % overlapping). The least-squares
   share of that virtual-leader deceleration SUMO followers realise is 0.09-0.30 (by gap). So the ramp car is **not** a leader of the
   SUMO follower before the change. LC2013's cooperative speed request (the follower is in a cooperative state 66 % of wait
   car-seconds, LCSTATE 7.3) shows up as "hold speed". The R9_YIELD -0.4 excess and the 54 % active-yield merges are this term.
   A zone-wide part exists too: unpartnered merge-zone followers with an opening gap reach only +0.5..0.7 vs textbook +0.85..1.2.

**Recommended fix (proposed), ranked.**
1. Make the IDM settings reachable: `model.driver_param_mode = "physical"` (V6ENC: implemented and tested; full range a 0.3-4.0,
   b 0.8-6.0; learned shared centre; prediction error only, no labels). Retrain the encoder head and centre jointly with a
   re-initialised residual: zero the dead attention head's output layer and do not keep its constant offset. The residual co-adapted to
   the floor: with a = 1.2 and the old residual, post-merge rows go from +0.07 to -0.48.
2. Then train the merge-specific hold as a learned term on the calibrated base. The R11 contrast courtesy head is the right vehicle:
   with an unbiased base, matched controls carry ~0 generic error, so contrast = absolute. A virtual leader is not recommended.
3. Fix the teacher-forced replay's visibility mask (section 6): `_recorded` drops junction-transit vehicles (loss_valid) from the
   replay STATE, so followers see phantom gaps near s ~ 61-75 m.

**Expected effect (proposed, not measured in rollouts).** Queues in the merge zone discharge more slowly. Merge-zone delay should
rise toward SUMO (now 0.48x), lane-1 followers should stop filling the ramp gap, so ramp delay should fall toward SUMO (now 2.1x),
and stuck ramp cars (R9_STUCK bistability) should become rarer. Controller ranking: the HDV reaction to a gap is currently ~2x too
eager, so any controller that opens gaps is mis-scored; the direction of the ranking change cannot be predicted offline.

**Offline test before any GPU run (section 8).** A teacher-forced CPU fit of only the physical-mode encoder (head + centre),
residual removed, on TRAIN windows. Pass if: recovered a median 0.9-1.5; far-upstream slow-start bias < 0.2; last-3-s
target-follower bias <= +0.9 (generic part removed); all-row one-step MAE <= 0.5. Then a 60-120 s rollout check of merge-zone
delay, ramp delay and stuck cars against round 10b on TRAIN states.

## 1. Data and method
(see sections below)
