# Frozen P30 Phase/Spatial Mechanism Diagnostic

Status: frozen before the multi-case replay. This is a development-only
mechanism audit. It does not authorize RL, predictor training, a moving wall,
new seeds, or paper-scale data.

## Question

For the unchanged P30 speed-ceiling treatment, determine whether helpful and
harmful final outcomes originate during the active command, the five-step
diagnostic release window, or the later post-release response. Attribute
braking/TTC events to the target, its same-lane follower chain, or other
corridor traffic. Measure whether the fixed spatial slow-front detector is
available often enough to support a later treatment design.

The phase labels are diagnostic. In particular, `release` does not imply that
P30 currently implements the proposed rear-aware projected release law.

## Frozen sources and cases

No new seed is introduced. Seeds 4123, 4127, 4129, 4130, 4131 and 4132 were
already inspected and remain permanently excluded from prospective evidence.

The exact source directories are:

- recurrence 1: `runs/eval/idm_p30_p50_m2000_l1_p30_development_v1`, manifest
  SHA-256 `2f037400c6efbb2c252ceea9cef949b3bf4f4870ba059716ca7d8ace7bb2d240`,
  protocol `073275f7db72321048de8ed5f99f5a62eb1582b773f87c0a4e5c952ccfc49328`;
- recurrence 2: `runs/eval/idm_p30_p50_m2000_l2_p30_development_v1`, manifest
  SHA-256 `e302640ea44cf50c0e08d82103829859b9bc9c0967eed9b64ede261255b65f63`,
  protocol `ddcd25dd198980a9594061a8fb03a04229df335e682b5f76a757aa1018d5c843`;
- recurrence 3: `runs/eval/idm_p30_p50_m2000_l3_p30_development_v1`, manifest
  SHA-256 `419fc2c01d60bc291c81aa9a5c128e00b6ac08d9721e754c658c83b1cf2ebb91`,
  protocol `8b367b5592f03075e4732bb85ec8de823f45545bdf18815f89ff12bb51153c9a`.

The 15 available cases are:

- recurrence 1: 4123/t61, 4127/t76, 4129/t354, 4130/t55, 4131/t382,
  4132/t49;
- recurrence 2: 4123/t146, 4127/t200, 4129/t436, 4130/t192 and 4132/t291;
- recurrence 3: 4127/t289, 4129/t528, 4130/t331 and 4132/t574.

The frozen prior labels are six helpful, four harmful and five neutral. Missing
source recurrences remain missing and must not be replaced by a different
decision.

## Replay identity and treatment

The treatment remains `activation_speed_ceiling_v1`, P30, candidate slot 1,
runway ranking, 20 requested command steps and a 60-step outcome horizon at
2000 veh/h mainline plus the frozen 400 veh/h ramp and 10% CAV penetration.

The new matched-source purpose is `p30_phase_spatial_replay`. It is accepted
only with legacy observations, non-discovery P30 treatment, the
`idm_phase_response/1` response trace and `idm_spatial_rsu_history/1` spatial
history. It may request a subset of a frozen manifest but never a seed absent
from that manifest. Each replay must exactly match the source decision,
physical-state, treatment and non-classic-observation fingerprints.

Each case contains native-IDM abstain, identical abstain replay and unchanged
P30. Control branches issue zero commands; P30 may command only the declared
target. Predecision spatial histories must be identical across all branches.

## Frozen analysis

`tools/analyze_idm_phase_spatial_diagnostic.py` emits
`idm_phase_spatial_mechanism_audit/1`. Raw cumulative deltas are treatment minus
control. Reference-normalized safety scores are control minus treatment, so
positive is beneficial.

For every case it records:

- active, release and post-release endpoints;
- the first post-active transition from helpful to non-helpful and harmful;
- whether an active helpful response becomes a harmful final outcome;
- step-aligned event-count deltas by treatment phase and relation to target;
- valid slow-front detections and causal velocity-fit availability;
- dose, identity, safety, TTC, delay and discharge fields.

No thresholds, detector settings, labels or cases may be changed after viewing
the combined output. The fixed slow-front detector is descriptive only; it is
not promoted into an actor, reward or treatment in this slice.

## Commands

Run the following for `n=1`, `2`, and `3`, using the matching source/output:

```powershell
$seeds = 4123,4127,4129,4130,4131,4132
$py = "C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe"

& $py tools/run_idm_decentralized_authority_verification.py `
  --output "runs/probe/idm_p30_phase_spatial_m2000_l${n}_development_v1" `
  --seeds $seeds --demand 2000 --workers 4 `
  --action-set activation_speed_ceiling_v1 `
  --option-steps 20 --safety-horizon-steps 60 --cav-penetration 0.10 `
  --decision-index $n --maximum-landmarks-per-cell 3 `
  --reward-refs runs/eval/idm_decentralized_lower_reward_refs_v1.json `
  --candidate-ranking runway --candidate-count 1 --landmark risk_onset `
  --decision-source "runs/eval/idm_p30_p50_m2000_l${n}_p30_development_v1" `
  --response-trace-contract idm_phase_response/1 `
  --diagnostic-spatial-history
```

Then aggregate:

```powershell
& $py tools/analyze_idm_phase_spatial_diagnostic.py `
  --runs `
    runs/probe/idm_p30_phase_spatial_m2000_l1_development_v1 `
    runs/probe/idm_p30_phase_spatial_m2000_l2_development_v1 `
    runs/probe/idm_p30_phase_spatial_m2000_l3_development_v1 `
  --output runs/probe/idm_p30_phase_spatial_m2000_analysis_v1
```

## Decision after the audit

- Harm developing during active control requires a new target/feedback law or
  admission rule.
- Harm developing mainly during release requires a projected, jerk-limited
  return to native IDM.
- Delayed follower-chain harm requires treatment-local follower-chain feedback
  and a reward/evaluation horizon that includes the rebound.
- If the fixed slow-front signal is rarely available, it cannot be a required
  actor or conditional-hold input.

Only one separately named rear-aware treatment may be frozen after this audit.
P30 remains unchanged as a comparator.
