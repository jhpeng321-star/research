# Modified FollowerStopper-Shaped P30 Development Protocol

Status: frozen before SUMO execution on 2026-08-29. Development evidence only.
This protocol does not authorize RL, predictor training, wall experiments,
paper-scale data, calibration, collapse-prevention claims, or IDM
generalization.

## Mechanism question

The frozen P30 controller produces six helpful, five neutral, and four harmful
outcomes across 15 already inspected cases. Two harmful cases are helpful at
the active endpoint and reverse after release. Corrected aligned experiments
show that extending P30 to 29 or 30 commands, and combining 30 commands with a
rear-aware projected release, does not remove the delayed harm. That
duration/release family is closed.

This screen tests one different hypothesis: whether causal front phase-space
feedback can shape the active P30 ceiling without changing admission, target,
duration, or the P30 downstream reference.

## Frozen treatment

Action set:
`activation_modified_followerstopper_p30_ceiling_v1`.

The treatment preserves the frozen P30 decision step, lane, target CAV,
runway ranking, secure-gap admission, terminal P30 rear-feasibility check,
filtered P30 reference, 20-command option, contention, hard projection and
legacy controlled release. It replaces the active command from step 1.

For ego speed `v`, leader speed `vl`, bumper gap `s`, and filtered P30
reference `U`:

```text
dv       = vl - v
closing  = min(dv, 0)
q        = min(max(vl, 0), U)
b[k]     = omega[k] + closing^2 / (2 alpha[k]) + h[k] v
omega    = (4.5, 5.25, 6.0) m
alpha    = (1.5, 1.0, 0.5) m/s^2
h        = (0.4, 1.2, 1.8) s
```

The desired speed is zero below `b1`, interpolates from zero to `q` over
`[b1,b2]`, interpolates from `q` to `U` over `[b2,b3]`, and equals `U` above
`b3`. With no leader it equals `U`. The active command is:

```text
a_ceiling = (v_FS - v) / dt
a_desired = min(a_analytical_P30, a_ceiling)
```

The unchanged projection then applies comfort, jerk, SUMO follow-speed,
front-TTC, rear-TTC and allowed-speed bounds. Any infeasible projection
interval is a hard treatment failure: front avoidance remains authoritative,
the member aborts into controlled legacy release, and its lane is blocked
during handoff. This is a rear-constrained local smoothing probe, not
canonical FollowerStopper, JAD, a safety proof, or a trained controller.

No efficacy-conditioned probe/hold rule is allowed. In the frozen cases,
step-3 and step-5 local/corridor responses do not distinguish helpful from
harmful final outcomes.

## Sources and seed scope

The switching geometry is from Stern et al. (2018),
<https://arxiv.org/abs/1705.01693>. The speed-dependent headway terms are from
Chou et al. (2021), <https://arxiv.org/abs/2112.14345>. The parameters are a
literature-seeded treatment definition, not tuned on this repository.

Only the previously inspected seeds 4123, 4127, 4129, 4130, 4131 and 4132 may
be used. They remain excluded from later prospective evidence. The frozen P30
sources are the three `runs/eval/idm_p30_p50_m2000_l{1,2,3}_p30_development_v1`
directories listed in `IDM_P30_PHASE_SPATIAL_DIAGNOSTIC_PROTOCOL.md`.

## Staged gate

Stage A is only recurrence-1 seed 4129 at decision step 354, a known P30
helpful-active-to-harmful-final reversal. It must satisfy all of the following
before any wider treatment run:

1. exact replay, predecision raw-context, non-classic observation, target,
   lane and decision-step identity;
2. 60 completed postdecision steps, at least 16 of 20 treatment commands, no
   unexpected commands, and no treatment safety abort;
3. zero collisions, teleports and projection-interval conflicts, with the
   frozen TTC and service constraints passing;
4. at least two active steps whose projected acceleration differs from frozen
   P30 by at least 0.05 m/s^2;
5. a safety-composite effect above the replay deadband versus native IDM and
   at least one deadband better than frozen P30; and
6. no worse follower minimum TTC, follower peak braking, ego jerk, or merge
   service than frozen P30.

If Stage A fails, the arm is rejected without parameter tuning or more seeds.
If it passes, Stage B replays all 15 frozen recurrence cases. Stage B requires
all integrity/safety gates, fewer harmful cases than P30, no lower tenth
percentile effect, at least three helpful and three nonbeneficial independent
seeds, and meaningful command/effect diversity. Only a passing Stage B may
authorize observation/reward re-audit; it still does not authorize RL by
itself.

## Stage-A command

```powershell
$py = "C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe"
& $py tools/run_idm_decentralized_authority_verification.py `
  --output runs/probe/idm_modified_followerstopper_p30_m2000_l1_seed4129_v1 `
  --seeds 4129 --demand 2000 --workers 1 `
  --action-set activation_modified_followerstopper_p30_ceiling_v1 `
  --option-steps 20 --safety-horizon-steps 60 --cav-penetration 0.10 `
  --decision-index 1 --maximum-landmarks-per-cell 3 `
  --reward-refs runs/eval/idm_decentralized_lower_reward_refs_v1.json `
  --candidate-ranking runway --candidate-count 1 --landmark risk_onset `
  --decision-source runs/eval/idm_p30_p50_m2000_l1_p30_development_v1 `
  --response-trace-contract idm_phase_response/1 `
  --diagnostic-spatial-history
```
