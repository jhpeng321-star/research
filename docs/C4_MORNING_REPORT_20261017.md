# C4 morning report — 2026-10-17 (overnight v7 world model + policy fixes)

Registration: runs/active/C4-V7FT-20261017/REGISTRATION.md. Diagnosis that motivated the work:
docs/C4_DIAGNOSIS_20261016.md.

## TL;DR
- **v7 world model** (trained from scratch with merge calibration and the loop's policy episodes) fixes what it
  targeted: the merge/discharge bias and how controllers change downstream traffic. Its overall calibration is much
  better. But it is **worse than v5m on unseen walls at 120 s** (+11 m, blow-ups 24 vs 12).
- **Policy fixes** (CAV-only TTC relative to native, scaled Lagrangian, alpha floor, SUMO lane changes,
  merge-discharge reward):
  - On v5m: clearly helpful early. u162 reached +22.0 s/veh vs native, the approach is no longer timid, and lane
    commands are 0.
  - On v7: the native-relative TTC constraint is never met in imagination, so λ escalates (0.6 → 6+). The safety
    term dominates again and the CAVs crawl in SUMO (+44 at u200, +34 at u300).
- **The main remaining problem is unchanged:** the CAVs still do not accelerate after the merge (+0.05 m/s² vs native
  +0.5) in any run.

## 1. World model v7 vs v5m (pre-set criteria)

| Criterion | Result | Verdict |
|---|---|---|
| 1 merge errors at least halved | Discharge error −3.8 → −1.1 veh per 120 s (≈ −70 %). Merge-zone speed at 60 s 0.98 m/s closer; downstream 0.7–1.0 m/s closer. | mostly PASS |
| 2 unseen walls no worse | Median HDV error 10/30/60/120 s: v5m 4.07/15.06/31.43/78.76, v7 4.08/16.13/34.66/93.97. Paired v7−v5m at 120 s +11.2 [+2.4, +25.2]. Blow-ups 12 → 24 of 139. | **FAIL** at 120 s |
| 3 controller ranking closer to SUMO | Spearman on delay is unchanged (0.94 both). The controller-effect error is smaller for v7 on downstream speed (significant, 60 and 120 s) and on delay over 60 s (strict mode). The absolute native delay bias falls from +112 to +20 veh-s. | partial PASS |
| 4 policy-traffic prediction | After-loop policy delay effect error −4.9 (v7) vs −11.9 (v5m), not significant. Downstream slowdown −1.39 vs SUMO −1.56 (v5m −0.47), from the interim checkpoint. | partial PASS |

Training validation at step 3000 (v7, which includes the harder policy windows): 5.0/38.8/96.7/160.5 m. v5 final:
4.7/38.1/96.7/162.

Files: runs/active/C4-V7FT-20261017/scores/ (walls_h60, merge_check_{sumo,none}, ranking_{sumo,none}).

## 2. Policy tests in SUMO

Change in system time per vehicle vs native at 40 % CAVs, demand 2400, 600 s, 12 paired training seeds each
(C4_LOOP_TRAIN block):

| Run | Checkpoint | Δ vs native | CAV speed 500–800 m | Post-merge CAV accel |
|---|---|---|---|---|
| Native | – | 0 | 27.4 m/s | +0.50 |
| Fixes on v5m | u100 | +38.9 | 12.1 | +0.08 |
| Fixes on v5m | u162 | +22.0 | 16.2 | +0.04 |
| Fixes on v7 | u200 | +44.1 | 9.6 | +0.08 |
| Fixes on v7 | u300 | +33.9 | 13.2 | +0.05 |
| Fixes on v7 | u600 (final) | +18.4 | 16.7 | +0.08 |

For reference, the old loop policy before the fixes was at +11.5 (pre-loop) and +4.2 (after 4 loop rounds) on the
final evaluation grid. Those are different seeds and penetrations, so the numbers are not directly comparable.

## 3. Diagnosis: what is still going wrong

1. **The CAVs never learn to accelerate after the merge.**
   - In every run the post-merge acceleration is about 0 (native +0.5).
   - The zero-acceleration initialisation means "hold speed", and nothing in training pushes hard enough to leave it.
   - The actor gradient on v5m is about 0.5, the same weak-signal problem as before.
2. **The TTC constraint on v7 is unsatisfiable.**
   - The policy's CAV-pair TTC cost stays about 2 above the native counterfactual, so λ grows without bound and
     dominates again (actor grad 2–3 from the safety term).
   - Likely cause: the cost counts pairs where a CAV is the LEADER. A slow CAV makes human followers close in, so
     braking does not reduce the cost, while the direct gradient on CAV-as-follower pairs still says "slow down".
   - Fixes: count only CAV-as-follower pairs, cap λ, or make the constraint per-CAV against its own native twin.
3. **v7 long-horizon stability regressed** (blow-ups doubled). The merge loss or the extra policy data may destabilise
   120 s rollouts.
4. **Entrance blocking is still invisible within 30 s of imagination.** No change tonight; the critic still sees only
   imagined returns.
5. **Lane control:** now handled by SUMO (0 policy lane commands). In imagination CAVs stay in lane apart from the
   forced merge, so imagination and SUMO differ for CAV lane changes.

## 4. Options for the next version (owner decides)
- **A. Policy-side first, keep v5m.**
  - Fix the TTC constraint (CAV-as-follower only, λ cap).
  - Add a mild "reach desired speed when the road ahead is free" incentive, or start from a learned speed-keeping
    prior (not imitation; e.g. a positive mean-acceleration init toward the speed limit).
  - Re-test on v5m. This is the cheapest option and v5m is the better long-horizon WM.
- **B. Combine the WMs.** Fine-tune v5m with the merge loss at a lower weight and without the stability regression
  (v7ft-style, ~3 h). Accept only if walls do not get worse.
- **C. Real-outcome critic or backlog.** Train the critic's entrance/backlog term on the loop's SUMO episodes, so the
  entrance harm is visible to the policy.
- **D. Then** run the Dreamer loop on the vGPU-48GB machine with the chosen WM and the fixed policy settings.

Recommended order: A (≈ 3 h, 1 GPU) → D. B and C in parallel if a second GPU is available.

## Addendum: final u600 check (12:40)
- Seeds 969984–969995, 12 paired: **+18.4 s/veh** vs native.
- The policy is still improving: +44 → +34 → +18 over u200 → u600, despite λ rising to ≈ 8.
- CAVs reach 16.7 m/s downstream (native 27.2), but the post-merge acceleration is still only +0.08 (native +0.44).
- Not converged after 600 updates.

## Follow-ups started 12:35 (owner: "full diagnosis of the world model; I want a better world model")
- WM diagnosis (local CPU): runs/active/C4-V7FT-20261017/wm_diag/WM_DIAGNOSIS_v7.md.
- v7ft on gpuR (fine-tune from v5m with the same fixes, ~3 h). This isolates "from scratch vs fine-tune" as the
  cause of the 120 s regression.
