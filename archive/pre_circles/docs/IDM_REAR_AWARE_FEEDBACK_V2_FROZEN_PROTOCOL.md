# Frozen Rear-Aware Closed-Loop Bilateral Pulse — Version 2 (Development Pilot Protocol)

**Status:** frozen development mechanism pilot protocol for the user-authorized
closed-loop bilateral pulse, superseding the treatment-proposal status of
`IDM_REAR_AWARE_FEEDBACK_TREATMENT_PROPOSAL.md` (preserved as provenance).
`docs/CLOSED_LOOP_BILATERAL_PULSE_HANDOFF.md` remains the authoritative transfer
record.
**Date:** 2026-08-30
**Workspace:** `D:\research-dense`

This protocol records the exact treatment, cohort, gates and exclusions that
were frozen before opening any SUMO mechanism-pilot efficacy result. It
authorizes implementation, focused/full tests and one six-seed development
mechanism pilot. It does **not** authorize RL training, predictor data
generation, the moving wall, or any efficacy claim.

## 1. Treatment schema and identity

- Action set: `activation_rear_aware_feedback_pulse_v2` (`idm/decentralized_damping.py`).
- Pure contract: `RearAwareFeedbackContract(schema="idm_rear_aware_feedback_pulse/2")`
  (`idm/rear_aware_feedback.py`), explicit `REAR_AWARE_FEEDBACK_V2`.
- Release contract: `idm_p30_projected_release/2` (smoothstep
  `3x^2-2x^3`, `PROJECTED_RELEASE_V2`), 10-step budget, rear margin 5 m.
- Verification invariant: every v2 runtime path receives `REAR_AWARE_FEEDBACK_V2`
  explicitly. Engine selection is `IDMResidualActionEngine._rear_aware_feedback_contract()`;
  runner selection is `_feedback_contract_for_action_set()`. No path may fall
  back to v1 silently; the v1/v2 contracts and treatment fingerprints are
  distinct (`feedback_contract.schema` is part of the runner's treatment
  identity and protocol hash).
- v1 (`activation_rear_aware_feedback_pulse_v1`, schema `/1`) remains available
  only as historical diagnostic reference and is deliberately not the pilot.

## 2. Front-feedback equation

Per control step for the single selected CAV (all measurements from the
pre-action snapshot; no postdecision input):

```text
v_ref = lowpass_20s(rate_limited_0.3_mps2(clip(lane_p50(disturbance), 8, v_allowed)))
x_i = base_gap_i + v_closing^2 / (2 b_i) + tau_i * v_ego      (i = 1,2,3)
      base_gaps  = (4.5, 5.25, 6.0) m
      b_i        = (1.5, 1.0, 0.5) m/s2
      tau_i      = (0.4, 1.2, 1.8) s
      v_closing  = min(v_lead - v_ego, 0)

v_feedback =
      0                                  if gap_lead <= x1        (stop zone)
      clip(v_lead, 0, v_ref) * (g-x1)/(x2-x1)   if x1 < gap_lead <= x2
      clip(v_lead,0,v_ref) + (v_ref - clip(v_lead,0,v_ref))*(g-x2)/(x3-x2)
                                                if x2 < gap_lead <= x3
      v_ref                              if gap_lead > x3          (free)

a_feedback_raw = (v_feedback - v_ego) / dt
a_native        = native-IDM proxy acceleration (SUMO follow-speed proxy, speedMode 31)
```

P50 is a **reference** endpoint (the closed-loop FollowerStopper-shaped law),
never a mandatory terminal speed. Open gaps return toward native IDM; close
leader gaps may request stronger but still jerk/projection-limited slowing.

Command blend by probe/hold strength `s` (see §4):

```text
a_cmd = a_native_projected + s * (a_feedback_projected - a_native_projected)
```

both endpoints projected through the hard safety layer first
(`projected_blend_acceleration`), and the blended result is re-projected with
the rear floor before `setSpeed(target, speedMode 31)`.

## 3. Rear-protection acceleration floor

Pure solver `solve_rear_speed_floor` (`idm/projected_release.py`) with
`RearFloorInput` from the pre-action follower state. The floor for the next
speed `v_next` is the least speed satisfying all of:

```text
1. TTC floor:      v_next >= v_follower - gap_follower / rear_ttc_floor(4 s)
2. Secure gap:     predicted_gap - secure_gap_follower - 5.0 m >= 0
                   (SUMO getFollowSpeed/getSecureGap of the physical follower,
                   using ego decel and the follower's predicted response)
3. Follower margin: predicted_follower_v >= v_follower - follower_decel * dt
```

One-step kinematics assume constant mean speeds; the floor is found by 32
bisection iterations over `[comfortable lower bound, comfortable upper bound]`.
If no feasible speed exists inside the comfortable interval, the treatment is
infeasible (`rear_floor_infeasible`) and releases toward native IDM.

Hard safety layer (projected **last**, always authoritative): comfortable
accel `[-1.0, +0.5] m/s²`, jerk `0.5 m/s³`, IDM follow speed, front TTC floor
3 s, rear TTC floor 4 s, rear secure-gap floor, SUMO speedMode 31. Front
collision avoidance dominates a front/rear conflict; a damping request may
never override rear protection, and conflicts resolve to release, never to
stronger braking.

## 4. State machine (1 controller step = 1.0 s = two 0.5 s SUMO ticks)

Age is 1-based command age. Maximum active duration 20 steps.

```text
probe (ages 1..5):  strength = (0.2, 0.4, 0.6, 0.8, 1.0)
                    baseline latched at age 1 from the pre-action response
hold  (ages 6..20): strength = 1.0, only while all conditions pass
```

Hard checks (any age, including probe step 1):
- `rear_floor_infeasible` — rear floor has no feasible solution;
- `projection_conflict` — projection interval conflict (front vs rear);
- `severe_follower_braking` — follower acceleration < -3.0 m/s²;
- `follower_identity_changed` — see §5.

Soft checks (v2: evaluated only after the five-step probe completes,
age >= 6):
- `rear_gap_compression` — gap loss > 3.5 m vs baseline, only when risk
  relevant (gap <= 20 m or rear TTC <= 6 s);
- `rear_closing_deterioration` — closing-speed increase > 1.0 m/s vs baseline,
  only when risk relevant;
- `disturbance_worsening` — lane disturbance speed-std increase > 1.0 m/s;
- `disturbance_resolved` — disturbance detection no longer present.

Probe-completion gates (evaluated at age == 5): projection survival count >=
0.8 × 5; >= 2 meaningful command deltas (|Δa| >= 0.05 m/s² vs native proxy);
and at least one improving causal response: disturbance std reduction >= 0.10
m/s, rear closing reduction >= 0.25 m/s, or local acceleration energy
(ego²+follower²) reduction >= 0.05 m²/s⁴; otherwise
`insufficient_projection_survival`, `insufficient_command_authority` or
`probe_no_improving_response`.

Member lifecycle releases (outside the feedback evaluator): lane change,
departure/target loss or safety-admission loss (`lane_changed`, `road_changed`,
`missing_sample`, admission reasons) and `option_completed` at 20 steps.

Release to native IDM: projected release (smoothstep, 10 steps from the last
projected command toward the exact native-IDM command reconstructed from
`getSpeedWithoutTraCI`, tolerances 0.5 m/s and 0.25 m/s²), then `setSpeed(-1)`
hands back full authority. No uncontrolled interval: the release path commands
every step until handoff and the treatment lane is cooldown-blocked.

## 5. Follower-identity rule

- `FeedbackResponse` records `follower_id` of the immediate follower at each
  active step; the identity is logged per step and is part of the causal
  preamble to every rear-gap/closing/braking measurement.
- The identity is latched with the baseline at activation (age 1).
- During probe or hold, follower appearance (None -> id), disappearance
  (id -> None) or replacement (id1 -> id2, id1 != id2) causes `follower_identity_changed`
  release to native IDM. The code never silently rebases to the new follower,
  and never compares rear measurements across different physical followers.
- An identity change is reported as an explicit release reason; it is not
  counted as a hard safety abort and does not by itself violate the
  safety-service constraints.

## 6. Cohort (frozen — do not change)

- Development mechanism cohort, p15/m2000/l3: CAV penetration 0.15, mainline
  demand 2000 veh/h, `risk_onset` landmark, decision index 3,
  max 3 landmarks per cell, runway ranking, candidate count 1.
- Seeds: **4123, 4127, 4129, 4130, 4131, 4132**.
- Decision source (frozen P30 matched source):
  `runs/eval/idm_paired_option_penetration_grid_l3_v1/idm_paired_option_grid_p15_m2000_l3_v1`.
- Seed 4131 has no decision in the source: it is reported as no-decision and
  not replaced by a different seed. The source P30 profile for the five
  matched seeds is: helpful 4123, 4127, 4129; harmful 4130; neutral 4132.
- Outcome horizon: safety 60 controller steps = 60 s; service horizon 60 steps.
  Reward references:
  `runs/eval/idm_decentralized_lower_reward_refs_v1.json` (matching the
  source grid byte-for-byte).
- Arms: abstain, abstain-replay, v2 feedback pulse at the frozen decision
  step/target (candidate slot 1). The fixed-P30 arm is read from the source
  artifact; the comparison joins only rows with exact seed, decision step,
  target and predecision identity.

Runner command (frozen):

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
$env:PYTHONPATH = "D:\research-dense;$env:PYTHONPATH"
Set-Location D:\research-dense

python tools/run_idm_decentralized_authority_verification.py `
  --output runs/probe/idm_rear_aware_feedback_v2_p15_m2000_l3_six_seed_v1 `
  --seeds 4123 4127 4129 4130 4131 4132 `
  --demand 2000 `
  --workers 4 `
  --action-set activation_rear_aware_feedback_pulse_v2 `
  --option-steps 20 `
  --safety-horizon-steps 60 `
  --cav-penetration 0.15 `
  --decision-index 3 `
  --maximum-landmarks-per-cell 3 `
  --reward-refs runs/eval/idm_decentralized_lower_reward_refs_v1.json `
  --candidate-ranking runway `
  --candidate-count 1 `
  --landmark risk_onset `
  --decision-source runs/eval/idm_paired_option_penetration_grid_l3_v1/idm_paired_option_grid_p15_m2000_l3_v1 `
  --response-trace-contract idm_phase_response/1 `
  --diagnostic-spatial-history
```

Expected runtime: 6 discoveries + 15 branches (5 cases × 3 arms); measured
source-grid median branch runtime was ~18.5 s, so the full pilot should finish
in well under 30 minutes with 4 workers. The runner freezes its own
`protocol_hash` in `manifest.json`; that hash and the exact code state are
recorded with the results.

## 7. Diagnostics per activation

Selected vehicle, lane, runway, rank; requested/realized probe/hold/release
steps; command delta vs native IDM per active/release step; front target and
zone; rear floor and feasibility; projection reasons/survival; follower id,
gap, closing speed, acceleration, TTC; causal disturbance presence and
speed-std; release reason and phase transitions; service/safety metrics on the
identical paired horizon; post-release response through 60 steps.

## 8. Integrity gates (before efficacy interpretation)

1. exact abstain-versus-abstain replay identity;
2. exact predecision branch identity (all branches share the discovery
   decision fingerprint);
3. zero treatment commands in both control arms;
4. feedback commands only on the declared target CAV;
5. no hard safety-projection bypass;
6. zero added collisions and teleports;
7. complete outcome horizons for all branches;
8. every feasible case receives the full five-step probe unless an explicit
   hard-safety or follower-identity release occurred (each such early release
   reported with its reason);
9. phase traces and command deltas exceed replay/numerical noise;
10. release returns to exact native IDM without an uncontrolled interval;
11. follower identity logged throughout (per-step responses and release rows);
12. v2 and fixed-P30 rows share exact seed, decision step, target and
    predecision identity before effect comparison;
13. every abort/release has an explicit reason.

## 9. Development efficacy gates (mechanism-development only)

1. fewer harmful qualified activations than matched fixed P30;
2. benefit beyond the 0.02 replay deadband in at least two independent seeds;
3. at least two nonbeneficial (neutral or harmful) seeds, preserving honest
   outcome diversity;
4. no added low-TTC exposure;
5. no added immediate-follower severe-braking failure;
6. no concealed arrival, discharge, delay or service loss;
7. improvement not produced only by trivial treatment doses;
8. active and post-release effects both included (60-step horizon).

This is not final paper verification; it need not win all six seeds, and every
unfavorable result must be reported.

## 10. Seed exclusions and prohibited claims

Seeds 4107--4132 (including 4121--4132) are development only. None may be used
for prospective claims; a fresh untouched set is reserved. No seed may be
replaced by a more favorable one.

Prohibited until prospective evidence exists: claiming the bilateral pulse
works or generalizes; claiming a trained IDM damping checkpoint exists;
claiming calibrated probabilities, demonstrated collapse prevention, a
validated moving wall, or a working escalation predictor. Prohibited actions:
training another predictor, generating paper-scale data, long PPO/MAPPO runs,
ordinary per-step MAPPO with the rejected variable-lifetime reward, discarding
harmful/neutral/constraint-failed cases, or changing the treatment definition
after viewing pilot outcomes without creating and documenting a new version.
