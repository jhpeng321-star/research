# Modified FollowerStopper-P30 Treatment-Naive Validation

Status: frozen on 2026-08-29 before running the remaining treatment branches.
Development validation only; no fresh prospective seeds are consumed.

## Why this is a separate gate

The recurrence-1 seed-4129/t354 mechanism smoke was viewed while designing the
screen. It therefore cannot validate the treatment. Its original strict gate
did not pass: although the safety composite changed from -0.1382 for P30 to
+0.0802, the fixed upstream cohort minimum TTC decreased from 17.89 s to
13.80 s, the immediate-follower minimum TTC decreased from 20.48 s to 15.75 s,
and the immediate follower's minimum acceleration changed from -0.463 to
-0.666 m/s^2. All values remained outside the hard-risk region, the new arm
removed two stable-cohort severe-braking exposures, and the bounded-safety
diagnostic passed. The case is retained as a viewed design artifact and is
excluded from every validation statistic.

The treatment code and parameters are unchanged from
`IDM_MODIFIED_FOLLOWERSTOPPER_P30_DEVELOPMENT_PROTOCOL.md`. No threshold,
headway, reference, duration, target, or release parameter may be tuned after
the design smoke.

## Validation cases

The 14 treatment-naive cases are:

- recurrence 1: seeds 4123, 4127, 4130, 4131 and 4132;
- recurrence 2: seeds 4123, 4127, 4129, 4130 and 4132;
- recurrence 3: seeds 4127, 4129, 4130 and 4132.

All seeds were previously inspected under P30 and remain development seeds,
but none of these exact cases has been run under the new treatment. The frozen
P30 phase/spatial replays are the matched comparator. Each new branch includes
native IDM, exact native replay and one declared target treatment.

## Frozen gates

Identity and mechanics must satisfy:

1. exactly 14 cases and exclusion of `m2000.seed4129.t354`;
2. identical demand, option, dose, penetration, recurrence, ranking, horizon,
   landmark and reward-reference fields;
3. exact raw predecision context, non-classic observation and target identity;
4. the frozen feedback contract and diagnostics on every treatment command;
5. 60 completed steps, at least 16 of 20 commands, exact abstain replay, no
   unexpected commands, and no treatment safety abort.

Hard safety and service must satisfy:

1. zero added collisions or teleports and the runner's low-TTC, delay,
   arrivals and discharge bounds in every case;
2. zero infeasible projection intervals;
3. decision-cohort and immediate-follower minimum TTC at least 4.0 s;
4. decision-cohort and immediate-follower acceleration never below
   -4.5 m/s^2; and
5. aggregate decision-cohort and immediate-follower exposure below
   -3.0 m/s^2 no greater than frozen P30.

Efficacy/headroom must satisfy all of:

1. fewer harmful cases than frozen P30;
2. treatment tenth-percentile effect no lower than P30;
3. mean effect no more than 0.02 below P30;
4. helpful outcomes in at least three independent seeds;
5. nonbeneficial outcomes in at least three independent seeds;
6. at least two projected-command differences of 0.05 m/s^2 or greater in at
   least 30% of cases; and
7. effects beyond the replay deadband in at least 30% of cases.

The analyzer is
`tools/analyze_idm_modified_followerstopper_screen.py`. The gate may authorize
only the next observation/reward audit. It never directly authorizes RL.

## Commands

Use the common command below with the indicated recurrence-specific seeds and
paths:

```powershell
$py = "C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe"
& $py tools/run_idm_decentralized_authority_verification.py `
  --output $output --seeds $seeds --demand 2000 --workers 4 `
  --action-set activation_modified_followerstopper_p30_ceiling_v1 `
  --option-steps 20 --safety-horizon-steps 60 --cav-penetration 0.10 `
  --decision-index $n --maximum-landmarks-per-cell 3 `
  --reward-refs runs/eval/idm_decentralized_lower_reward_refs_v1.json `
  --candidate-ranking runway --candidate-count 1 --landmark risk_onset `
  --decision-source $source `
  --response-trace-contract idm_phase_response/1 `
  --diagnostic-spatial-history
```

Values:

```text
n=1 seeds=4123,4127,4130,4131,4132
source=runs/eval/idm_p30_p50_m2000_l1_p30_development_v1
output=runs/probe/idm_modified_followerstopper_p30_m2000_l1_validation_v1

n=2 seeds=4123,4127,4129,4130,4132
source=runs/eval/idm_p30_p50_m2000_l2_p30_development_v1
output=runs/probe/idm_modified_followerstopper_p30_m2000_l2_validation_v1

n=3 seeds=4127,4129,4130,4132
source=runs/eval/idm_p30_p50_m2000_l3_p30_development_v1
output=runs/probe/idm_modified_followerstopper_p30_m2000_l3_validation_v1
```

Then run:

```powershell
& $py tools/analyze_idm_modified_followerstopper_screen.py `
  --p30-runs `
    runs/probe/idm_p30_phase_spatial_m2000_l1_development_v1 `
    runs/probe/idm_p30_phase_spatial_m2000_l2_development_v1 `
    runs/probe/idm_p30_phase_spatial_m2000_l3_development_v1 `
  --candidate-runs `
    runs/probe/idm_modified_followerstopper_p30_m2000_l1_validation_v1 `
    runs/probe/idm_modified_followerstopper_p30_m2000_l2_validation_v1 `
    runs/probe/idm_modified_followerstopper_p30_m2000_l3_validation_v1 `
  --design-p30 runs/probe/idm_p30_phase_spatial_m2000_l1_development_v1 `
  --design-candidate `
    runs/probe/idm_modified_followerstopper_p30_m2000_l1_seed4129_v1 `
  --output `
    runs/probe/idm_modified_followerstopper_p30_m2000_validation_analysis_v1
```
