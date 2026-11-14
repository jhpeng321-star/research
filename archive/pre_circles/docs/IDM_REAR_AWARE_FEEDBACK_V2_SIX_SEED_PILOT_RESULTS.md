# Rear-Aware Feedback Pulse V2 — Six-Seed Matched Mechanism Pilot Results

**Date:** 2026-08-30
**Status:** development mechanism pilot **complete**; integrity gates passed;
primary development efficacy gates **failed**; no training authorized; no seed
added; no treatment change made after viewing outcomes.
**Protocol:** `docs/IDM_REAR_AWARE_FEEDBACK_V2_FROZEN_PROTOCOL.md`
**Runner protocol hash:** `5938f7197ae9ca9c00fcbcfc5647ae95f29974b9667d0bbe9f8abf5f5e915aa8`
(recorded in
`runs/probe/idm_rear_aware_feedback_v2_p15_m2000_l3_six_seed_v1/manifest.json`)
**Code state:** HEAD `170ab28` plus the dirty worktree; `env/idm_residual.py`
contains the v2 pulse (feedback contract `idm_rear_aware_feedback_pulse/2`,
release contract `idm_p30_projected_release/2`), follower-identity rule, and
diagnostics.
**Environment:** SUMO 1.27.1, `rl_driving` conda env, Python 3.14, 4 workers.
**Tests:** focused suite 76 passed; full suite **741 passed, 0 failed**
(prior to the pilot; the pilot did not run with a failing relevant test).
**Seed 4131:** no valid decision in the frozen source; reported as no-decision
and excluded (not replaced).

## 1. Command actually executed

```powershell
python tools/run_idm_decentralized_authority_verification.py `
  --output runs/probe/idm_rear_aware_feedback_v2_p15_m2000_l3_six_seed_v1 `
  --seeds 4123 4127 4129 4130 4131 4132 --demand 2000 --workers 4 `
  --action-set activation_rear_aware_feedback_pulse_v2 --option-steps 20 `
  --safety-horizon-steps 60 --cav-penetration 0.15 --decision-index 3 `
  --maximum-landmarks-per-cell 3 `
  --reward-refs runs/eval/idm_decentralized_lower_reward_refs_v1.json `
  --candidate-ranking runway --candidate-count 1 --landmark risk_onset `
  --decision-source runs/eval/idm_paired_option_penetration_grid_l3_v1/idm_paired_option_grid_p15_m2000_l3_v1 `
  --response-trace-contract idm_phase_response/1 --diagnostic-spatial-history
```

5/5 matched decisions were rediscovered at the frozen step/target; 15/15
branches completed; exit code 0. Analysis join:
`tools/analyze_idm_rear_aware_feedback_v2.py` →
`runs/probe/idm_rear_aware_feedback_v2_p15_m2000_l3_six_seed_v1/p30_v2_joined_report.json`
(5 unit tests; joined 5 exact-identity cases, excluded 0).

## 2. Per-seed matched results (P30 arm from the frozen source artifact)

All composites are the frozen safety composite (reference-scaled; positive =
benefit). Both arms use the identical case/decision/target: raw decision
context identity, nonclassic observation identity and treatment-difference
expectation all pass for every seed (verified per seed in the joined report).

| seed | P30 class | P30 comp | P30 dose | P30 constr | v2 class | v2 comp | v2 dose | v2 constr | v2 early-release reason |
|------|-----------|----------|----------|-----------|----------|---------|---------|-----------|-------------------------|
| 4123 | helpful | +0.4840 | 20 | pass | **helpful** | +0.1179 | 5 | pass | probe_no_improving_response (age 5) |
| 4127 | helpful | +0.3213 | 20 | pass | **harmful** | **−0.4059** | 5 | pass | disturbance_resolved (age 5) |
| 4129 | helpful | +0.0753 | 20 | pass | neutral | +0.0071 | 12 | pass | disturbance_resolved (age 12) |
| 4130 | harmful | −0.0789 | 20 | fail (discharge −2) | neutral | −0.0030 | 5 | pass | insufficient_command_authority (age 5) |
| 4132 | neutral | −0.0037 | 20 | pass | harmful (constraint) | **+0.0786** | 4 | fail (dose 4/5) | **follower_identity_changed (age 4)** |

Notes:

- 4123: full 5-step probe, projection survival 1.0, realized command deltas
  −0.10..−0.76 m/s²; benefit +0.118 (< P30's +0.484) but beyond the 0.02
  deadband. Released at probe-end because no improving response was detected.
- 4127: identical probe profile (deltas −0.10..−0.76, survival ~1.0,
  follower constant, no severe braking), released at age 5 on
  `disturbance_resolved`; outcome **harmful −0.406** — the largest v2 harm and
  the seed where P30 was most similar in direction (P30 +0.321). Post-release
  corridor response worsened; this is a metastable realization where a 5-step
  damping probe shifted wave timing adversely.
- 4129: the only case that entered the conditional **hold** (ages 6–12;
  deltas decayed and turned positive toward native as gaps opened), then
  released on `disturbance_resolved`; net effect neutral (+0.007). Mechanism
  worked, value did not emerge.
- 4130: **no delivered authority** — realized deltas 0.00–0.012 m/s², i.e.
  below the 0.05 meaningful threshold, so `insufficient_command_authority` at
  age 5. The candidate was admitted but the FollowerStopper-shaped law tracked
  the native command; the pulse effectively abstained. Outcome neutral
  (−0.003) where P30 was harmful.
- 4132: three zero-delta commands, then the immediate follower changed at
  step 4 (mainline.473 → mainline.468); the frozen identity rule released
  smoothly to native IDM (`follower_identity_changed`), causing dose 4/5 and a
  constraint-failed ("harmful" per frozen contract) classification even
  though the physical composite was **positive +0.0786** (delay delta ≈ 0,
  arrival/discharge noninferior).

## 3. Integrity gates (frozen protocol §8)

| gate | result |
|------|--------|
| 1 abstain-vs-abstain replay exact | PASS (replay effect 0.0) |
| 2 exact predecision identity | PASS (all_branch_decision_identity, all_discovery_identity true) |
| 3 zero commands in control arms | PASS (control_command_count 0, 2 branches/seed) |
| 4 feedback commands target-only | PASS (unexpected_command_count 0) |
| 5 no projection bypass | PASS (survival ~1.0 where authority existed; no treatment safety aborts) |
| 6 zero added collisions/teleports | PASS |
| 7 complete horizons | PASS (60/60 steps each branch) |
| 8 full 5-step probe | 4/5 feasible seeds received the full probe; 4132 released at 4/5 for the frozen follower-identity rule (explicit authorized early release — protocol §8.8). The literal handoff wording ("at least five of six") is not met because only five seeds are feasible, but the frozen v2 protocol condition is met. |
| 9 phase traces/deltas exceed replay noise | PASS (realized deltas up to 0.76 m/s²; 60-step phase response recorded) |
| 10 release to exact native IDM | PASS (native_handoff phase with setSpeed(-1), released_to_idm true, all five branches) |
| 11 follower identity logged | PASS (per-step follower_id in command rows; identity change at 4132 step 4 logged) |
| 12 P30 and v2 exact identity | PASS (seed+step+target exact; raw context/observation identity true) |
| 13 explicit reasons for all releases | PASS (probe_no_improving_response, disturbance_resolved ×2, insufficient_command_authority, follower_identity_changed) |

## 4. Development efficacy gates (frozen protocol §9)

| gate | result |
|------|--------|
| 1 fewer harmful qualified activations than P30 | **FAIL** — v2 2 (4127, 4132 per frozen classification) vs P30 1 (4130) |
| 2 benefit beyond 0.02 deadband in ≥2 seeds | **FAIL** — 1 seed (4123 +0.118) |
| 3 ≥2 nonbeneficial seeds | PASS (4127, 4129, 4130, 4132) |
| 4 no added low-TTC | PASS (Δ low-TTC exposure 0 in all matched cases) |
| 5 no added follower severe braking | PASS (no severe_follower_braking release; min follower accel −0.25 m/s²) |
| 6 no concealed service loss | PASS (delay/arrival/discharge deltas within budget in all cases) |
| 7 improvement not from trivial dose | PASS (helpful case 4123 delivered full 5-step probe) |
| 8 active + post-release effects included | PASS (60-step horizon; post-release present) |

## 5. Failure classification (protocol/spec §9)

Mechanics and safety mostly held (projection survival, rear floor, follower
identity, smooth release). The dominant failures are:

1. **A-class (mechanics/authority) — seed 4130:** candidate admitted but the
   closed-loop front law delivered <0.012 m/s² realized delta; the pulse
   effectively abstained. Diagnose the front-target blending at this landing
   state before rerunning.
2. **D-class (dose/safety pass, inconsistent effect) — seed 4127:** the same
   probe that helped 4123 harmed 4127 (−0.406) after a `disturbance_resolved`
   release; post-release response worsened. Metastability-driven and not
   attributable to a resolved hard-safety event.

Follow the protocol: no new seeds, no RL, no reward change, and at most two
clearly versioned mechanism iterations before a formal go/no-go. The v2 plan
is not promoted to the broader development screen (§10), observation
sufficiency (§11), reward audit (§12), or training (§13).

## 6. What was not done (prohibited-boundary confirmation)

No RL/PPO/MAPPO run; no predictor data; no moving-wall run; no prospective
seed used; no thresholds changed after viewing results; no case discarded
(4132's failure is reported, not hidden); the runner's `training_authorized`
flag is explicitly False for this action set.
