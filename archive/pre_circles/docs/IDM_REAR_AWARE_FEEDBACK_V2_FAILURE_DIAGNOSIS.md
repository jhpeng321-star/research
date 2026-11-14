# Rear-Aware Feedback Pulse V2 — Failure Diagnosis

**Date:** 2026-08-30
**Status:** diagnosis-only; no new SUMO treatment branch, no controller change,
no seed opened. All quantities below come from the frozen pilot artifacts:
`runs/probe/idm_rear_aware_feedback_v2_p15_m2000_l3_six_seed_v1`
(branches, discoveries, summary) and the frozen P30 source
`runs/eval/idm_paired_option_penetration_grid_l3_v1/idm_paired_option_grid_p15_m2000_l3_v1`.
**Interpretation rule from the diagnostic charge:** preserve the frozen v2
classification of all five cases (4132 stays constraint-failed/"harmful" under
the frozen protocol; its positive physical effect is retained as mechanism
evidence only).

## 1. Per-seed composite and phase decomposition

Sign convention: `u(t)` is the frozen safety composite accumulated over the
first `t` control steps (positive = benefit). The paired phase response uses
the treatment-minus-control spatial trace; the phase intervals are
probe = steps 1–5, release = steps 6–15 (10-step projected release), and
post-release = steps 16–60. The corridor reference scales are
`B_ref = 0.1343 s^-1`, `E_ref = 9.236 m2/s4`.

### 1.1 Seed 4123 (P30 helpful +0.484; v2 helpful +0.1179, dose 5/5)

| step | phase | dSevere | dEnergy (m2/s4) | u(t) |
|------|-------|---------|-----------------|------|
| 5 | probe | 0 | +0.053 | −0.0000 |
| 10 | release | 0 | +0.670 | −0.0006 |
| 15 | release | 0 | +0.733 | −0.0007 |
| 20 | post-release | 0 | +2.421 | −0.0022 |
| 30 | post-release | 0 | +11.512 | −0.0104 |
| 35 | post-release | +1 | +68.683 | −0.1240 |
| 40 | post-release | +1 | +24.327 | −0.0840 |
| 45 | post-release | +1 | −2.311 | −0.0599 |
| 47 | post-release | +0 | −17.9 | (sign flip) |
| 50 | post-release | −1 | −55.865 | +0.1124 |
| 55 | post-release | −1 | −61.448 | +0.1175 |
| 60 | post-release | −1 | −61.908 | **+0.1179** |

Interpretation: 4123 was *unfavorable* from step 32 to step 46 (worst
−0.124 at step 35) and only became beneficial after step 47. The final sign
was therefore established ~30 s after the probe and ~30 s after the native-IDM
handoff, in the late post-handoff window. The probe and release themselves
were neutral.

### 1.2 Seed 4127 (P30 helpful +0.321; v2 harmful −0.4059, dose 5/5, release reason disturbance_resolved at age 5)

| step | phase | dSevere | dEnergy (m2/s4) | u(t) |
|------|-------|---------|-----------------|------|
| 5 | probe | 0 | +0.028 | −0.0000 |
| 10 | release | 0 | +0.198 | −0.0002 |
| 12 | release | 0 | −1.029 | +0.0009 |
| 16 | post-release | 0 | −0.880 | +0.0008 |
| 25 | post-release | 0 | −0.428 | +0.0004 |
| 30 | post-release | −1 | −16.102 | **+0.0766** |
| 35 | post-release | 0 | +10.970 | −0.0099 |
| 40 | post-release | 0 | −1.288 | +0.0012 |
| 45 | post-release | +1 | +5.270 | −0.0668 |
| 50 | post-release | +2 | −14.410 | −0.1111 |
| 55 | post-release | +3 | +13.874 | −0.1986 |
| 60 | post-release | +5 | +106.077 | **−0.4059** |

Answers to the phase-resolved questions:

- **First treatment-versus-control divergence:** step 3–5 in the frame-level
  energy trace (|dE| ≥ 0.01 m2/s4 per frame), i.e., inside the probe, but
  two orders of magnitude below the eventual bursts.
- **Active-probe interval:** steps 1–5, all five commands realized (deltas
  −0.102 … −0.757 m/s2, projection survival ≈ 1.0); u(5) = −0.0000.
- **Projected-release interval:** steps 6–15; u ≤ 0.0002. 4127 was actually
  mildly favorable at steps 12–16 (dE −0.88 … −1.37).
- **Exact native-IDM handoff:** completed (phase `native_handoff`,
  setSpeed(-1), released_to_idm true).
- **Post-handoff interval:** steps 16–60. The final harmful sign was
  established at step 35 (u crosses below −0.02) and became irrevocably
  negative at step 45 (u = −0.0668); from step 45 onward monotone to −0.4059.
- **Which region/vehicle generated the added braking/energy:** vehicle-level
  safety events (a < −3.0 m/s2 or TTC < 1.5 s) show the treatment branch had
  +5 severe events vs control (25 vs 20) and +3 low-TTC (5 vs 2), all in the
  **ramp/merge lanes** (ramp.69 ×5, ramp.70 ×2, ramp.71 ×2, ramp.72 ×4 on
  lanes `1487606179_0/1` and `1487606178_0`) plus one second-order upstream
  follower `mainline.304` (`upstream_follower_2`). None was on the controlled
  CAV or its immediate follower (mainline.302: follower accel never below
  −0.20 m/s2, TTC > 27 s throughout).
- **Disturbance statistics before/after release (reconstructed lane-level
  1 km window, observation):** lane `1191554636_0` pooled mean/std: mean
  ≈ 23.5 m/s (low, slow regime) rising monotonically to ≈ 25.2 by step 22;
  std declines monotonically 2.35 → 2.08 (step 3) → 1.96 (step 4) → 1.97 →
  1.94 …, i.e., the std arm of the detector **genuinely and monotonically
  cleared** around the release at step 5. There is **no threshold flicker**:
  no oscillation of std across the 2.0 boundary in the ten steps before or
  after release. (The p70–p30 deficit arm is not reconstructible from stored
  frames; the detector itself — which saw the full state — reported
  `disturbance_resolved` one step before the probe-completion gate.)
- **Native/front/requested/rear-floor/executed accelerations during
  probe (observation):** identical in shape to 4123 (see §1.3); rear floor
  feasible, no projection conflict, no severe follower braking.

### 1.3 4123 vs 4127: the identical-command puzzle

The predecision candidate previews and realized commands are nearly the same:

| quantity | 4123 | 4127 |
|---|---|---|
| lane / geometric rank | `1191554636_1` / 1 | `1191554636_0` / 3 |
| runway / predicted dose | 725 m / 20.5 s | 443 m / 17.2 s |
| disturbance location | 64.6 m from merge | 298.0 m from merge |
| front gap / front TTC | 183 m / 42.9 s | 133.8 m / 61.6 s |
| P50 reference | 31.32 m/s | 23.45 m/s |
| p50 projected acceleration (preview) | −0.533 m/s2 | −0.516 m/s2 |
| realized probe deltas | −0.103…−0.760 m/s2 | −0.102…−0.757 m/s2 |
| immediate-follower response | closing −0.30→+3.51 (a→−0.36) | closing +0.46→+3.76 (a→−0.20) |
| severe-braking EMA / trend at decision | 0.54 / +0.135 | 0.135 / −0.029 |
| wave-energy level / trend at decision | 5.15 / +2.29 | 8.40 / +1.28 |
| u at probe end / release end / step 30 | 0.000 / −0.001 / −0.010 | 0.000 / −0.000 / **+0.077** |

Conclusion from the data: at the decision instant and through the whole
probe/release interval, **no recorded causal signal separated the two cases**
— the commanded sequences differ by < 0.01 m/s2 per step, both were neutral
during probe and release, and 4127 was *more favorable* than 4123 at step 30.
The outcome divergence was decided in the late post-handoff interaction of
the wave with the ramp/merge (steps 35–60 for 4127; 32–46 unfavorable then
positive for 4123). The two seeds are **not distinguishable using information
available before the harmful response** in any recorded treatment-local or
predecision variable.

Inference (explicitly separated from observation): the harm mechanism in 4127
is wave-timing sensitivity in the merge zone; the 5-step probe shifted the
wave such that ~30–60 s later the ramp/merge interaction generated extra
severe braking (+5 events) that the rear-protection/identity design neither
sees nor can influence. There is no evidence that the perturbed vehicle was
the controlled CAV, its follower, or any vehicle within the influence region
at the time of the command.

### 1.4 Seed 4129 (P30 helpful +0.075; v2 neutral +0.0071, dose 12 — the only hold case)

Probe (1–5) neutral; hold (6–12) neutral (dE ≤ 0.004); released at age 12 on
`disturbance_resolved`; post-release ±0.014 and final +0.0071. This case
shows the conditional hold worked mechanically and yielded nothing
measurable, which is exactly why longer holding is not a supported repair:
4129 already held and stayed neutral.

### 1.5 Seed 4130 (P30 harmful −0.0789; v2 neutral −0.0030, dose 5/5)

**Authority failure — root cause found.** All five probe commands show:

- front zone = **free** every step (leader gap 64.7–67.2 m, leader speed
  26.45–27.0 m/s, reference 28.68–28.77 m/s);
- `desired` feedback acceleration ≈ (v_ref − v)/dt = **positive**
  (+0.0002 … +0.0116);
- but the hard projection caps at the native follow speed
  (`idm_follow_speed` reason), so the projected feedback endpoint equals the
  projected native endpoint {−0.0046, +0.0000, +0.0002, +0.0065, +0.0116}
  — realized deltas 0.000–0.012 m/s2, below the 0.05 meaningful threshold
  (`insufficient_command_authority` at age 5);
- rear floor feasible throughout (follower decel margin ≈ 4.3 m/s2, no TTC
  pressure).

**Was it predictable at the decision instant? Yes.** The frozen predecision
preview already recorded for the decision: p50 terminal-speed preview
`desired_acceleration = +0.4149` and `projected_acceleration = −0.0046` with
projection reason `['idm_follow_speed']` — the feedback endpoint had no
separation from native at the instant of decision, because the target CAV was
free-flowing behind a leader that was itself below (and later near) the
reference with a ~65 m gap. The authority absence was therefore observable
before any command.

### 1.6 Seed 4132 (P30 neutral −0.0037; v2 harmful per frozen constraint, physical +0.0786, dose 4/5)

- **Why the follower changed:** an exogenous same-lane ordering change.
  `mainline.468` becomes the nearest upstream follower at step 3 in **both the
  treatment and the abstain (control) branches identically** (upstream
  follower sequences are identical: [473,475,477,476,478] → [468,473,475,477,476]
  at step 3, closing ≈ −5.9 m/s for 468 entering from another lane / reordering
  near the merge). The change was not caused by the treatment; the target
  commands had zero delta at steps 1–4.
- **Was the release safe/smooth?** Yes — the frozen identity rule released at
  age 4 (baseline mainline.473 latched at step 1; 468 recorded at step 4),
  the projected release ran, `native_handoff` with setSpeed(-1) completed, and
  the immediate follower TTC stayed large.
- **When did the +0.079 emerge?** u first |·| > 0.02 at step 23 (dSevere −1
  at step 25, dEnergy −15.9 at step 25), i.e., **only after** the identity
  change (step 4), after release completion (~step 15), and from a single
  fewer severe-braking event in a late corridor window. The pulse's own
  commands contributed nothing (zero realized deltas steps 1–4); the
  measurable effect came from the release-path command difference and its
  30+ s delayed wave interaction. The rule was not weakened; 4132 keeps its
  frozen constraint-failed classification and its positive effect is retained
  as mechanism evidence only.

## 2. Candidate v2.1 mechanisms — evaluation (not implementation)

Ranked by physical justification (never by retrospective seed score):

1. **A. Pre-activation realized-authority admission** — require the
   first-step projected feedback-vs-native command separation to exceed a
   minimum (e.g., 0.05 m/s2) before allowing the probe.
   - Justification: directly addresses a measured defect (4130: zone=free,
     zero separation; definitively predictable from the frozen preview).
   - Predicted per case: 4123 retain (separation 0.103); 4127 retain (0.102);
     4129 retain (0.100); 4130 **abstain** (0.000, becomes neutral/no-activation
     instead of an empty probe); 4132 **abstain** (0.000 — its zero-delta probe
     would be rejected before the identity change).
   - Predicted gate effect: removes 2 ineffective activations but does **not**
     reduce 4127's harm and leaves only 1 helpful seed — gate 1 (fewer
     harmful than P30: 1 vs 1, tie) and gate 2 (≥2 benefit seeds) still fail.
   - Risk: the threshold is calibrated on five development cases (must be
     versioned if used); it is an abstention rule, not a benefit source; it
     should never be described as creating benefit.
2. **B. Disturbance-resolution persistence/hysteresis** — **not supported**.
   4127's reconstructed lane std declines monotonically (2.35 → 1.96 → 1.94)
    without any flicker across the detector's 2.0 boundary; the resolution was
   genuine at the std level and the detector's own verdict was consistent.
   Persistence would only delay a legitimate release and cannot affect the
   post-handoff harm (which is not tied to when the release happened — 4129
   held and stayed neutral).
3. **C. Release-law adjustment** — **not supported**. No harmful divergence
   exists during the projected-release interval (u ≤ 0.0002; 4127 mildly
   favorable at steps 12–16); no acceleration discontinuity or taper anomaly;
   the harm begins at steps 35–60, well after handoff.
4. **D. Probe termination** — **not supported**. Harm is not visible during
   the probe (u(5) = 0.0000 on both 4123 and 4127, and 4127 was favorable at
   step 30). Any termination signal that would stop 4127 would also stop 4123:
   the two probes are indistinguishable (Δa < 0.01 m/s2 per step).
5. **E. Candidate admission (predecision variable)** — **no causal
   justification**. Predecision differences exist (runway 725 vs 443 m,
   disturbance position 64.6 vs 298 m, lane, severe-EMA, front gap) but with
   n = 2 cases any discriminating rule is post-hoc, no recorded variable is
   mechanistically linked to the late merge-zone harm, and the two commands
   are identical. Rejecting 4127 on any of these would be tuning on seed
   outcomes — prohibited.

## 3. Recommendation

**Declare the closed-loop bilateral pulse a mechanism no-go for this
single-CAV, 20 s, probe–hold–release design.** The governing decision rule is
satisfied: there is no causal signal before or during the treatment that
distinguishes the harmful realization (4127) from the helpful one (4123), and
no treatment-local change can address the 4127 harm, because the harm arises
in the ramp/merge-zone wave interaction 30–60 s after handoff — outside the
treatment's influence region and posterior to the release. Mechanism A
(realized-authority admission) is physically justified, but it is an
abstention improvement only: it would make 4130 and 4132 abstain and cannot
pass either primary efficacy gate. Longer holding is explicitly unsupported
(4129). This is not a safety failure and not a classification artifact; it is
a genuine authority/stability boundary of the decentralized actuator at
2000 veh/h / 15% penetration with this cohort.

## 4. Next program (honest fallback)

Continue toward the trained damping-controller goal with:

1. retain fixed P30 **only** in the supported 5–10% penetration operating
   domain (recorded: 25.9% held coverage, +0.0434 conditional reward, 7
   helpful / 3 harmful / 5 neutral);
2. conservative abstention outside that domain;
3. verify whether causal local observations improve activation value within
   that limited domain (grouped seed splits, AUROC ≥ 0.70 / balanced accuracy
   ≥ 0.65 / 20% coverage gates);
4. audit the paired binary option reward for the limited-domain
   abstain-versus-P30 problem;
5. train an abstain-versus-P30 contextual-bandit/PPO policy only if the
   observation and expected-value gates pass;
6. prospectively compare against never-activate, always-activate and the
   simple 5–10% rule; freeze only a bounded-harm improved checkpoint.

## 5. Clarification: +3 diagnostic low-TTC events vs zero `low_ttc_exposure_step_delta` in seed 4127

The two numbers are produced by **different estimators and cannot be compared
as the same measurement**:

- Frozen gate metric (`low_ttc_exposure_step_delta`, used by the effect
  constraint): per step, `env._corridor_ttc_count()` pairs consecutive
  same-lane vehicles on the four corridor edges by snapshot lane position and
  computes `ttc = (p_lane(leader) − p_lane(follower) − 5.0 m) / (v_fol − v_lead)`
  with a strict closing deadband (`closing > 0.01 m/s`) and threshold 1.5 s;
  `low_ttc_exposure_steps` counts **steps with at least one such pair**. In
  seed 4127 this estimator found **zero pairs below threshold in both the
  abstain and treatment branches** (`low_ttc_exposure_steps = 0`,
  `ttc_count_sum = 0.0` for both arms) — hence a delta of exactly 0. It also
  counts a step once regardless of how many pairs it contains.
- Diagnostic `safety_events` counter: per-vehicle-per-step events from the
  SUMO `VAR_LEADER` bumper-to-bumper gap (`ttc = leader_gap / closing`, no
  5 m fixed-length correction, no closing deadband), same 1.5 s threshold. In
  seed 4127 it recorded 2 events in the abstain branch (steps 39, 41,
  `ramp.69`) and **5 events in the treatment branch** (steps 38–41, 50:
  `ramp.69` ×4, `ramp.71` ×1).

The +3 event difference is therefore fully compatible with the frozen gate:
the additional treatment events involve ramp vehicles whose pair-based,
5 m-corrected corridor estimate stayed at or above 1.5 s on every step of both
branches, and the step-level exposure boolean did not change (both branches
were 0). The diagnostic events are retained as mechanism evidence (the harm
window is real — it arose from ramp/merge braking), while the frozen gate's
conservative pair estimator governs the constraint. No gate value for 4127 is
changed by this clarification, and the frozen v2 classification is untouched.

