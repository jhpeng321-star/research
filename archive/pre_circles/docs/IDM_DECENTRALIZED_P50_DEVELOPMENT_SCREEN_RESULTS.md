# IDM Decentralized P30-versus-P50 Development Screen Results

Status: complete development diagnosis; P50 rejected by its frozen retention
criterion. This is not independent efficacy evidence. Seeds 4121--4132 have
been inspected and are permanently excluded from prospective evaluation.

## Decision

The P50 treatment is mechanically valid but is not a publishable damping
controller and is not authorized as an RL or predictor action. The
first-risk-onset smoke passed replay, branch-identity, dose, target-integrity,
safety, service and command-distinctness gates. The predeclared extension to
the already opened second and third onsets produced 24 common qualified paired
cases. P50 tied P30 at four harmful cases (16.67%) rather than strictly reducing
the harmful fraction, so the frozen retention gate failed.

No 1900- or 2100-veh/h expansion, percentile search, reward revision, MAPPO
training, escalation-predictor generation, or moving-wall experiment was run.

## Frozen comparison

- Demand: 2000 veh/h mainline and 400 veh/h ramp.
- Penetration: 10% scattered CAVs.
- Development seeds: 4121--4132.
- Events: first three persistent `risk_onset` recurrences when present.
- Control and replay: exact native IDM abstention.
- Treatments: the same candidate and decision state, with only terminal
  reference changed from P30 to P50.
- Requested treatment: 20 one-second commands; minimum adequate dose: 16.
- Outcome horizon: 60 seconds.
- Harm deadband: the frozen 0.02 reference-unit threshold.

P50 branch generation was locked to the P30 decision source and exact P30
candidate. This correction was necessary because an independent P50 discovery
can admit a gentler treatment earlier or rank a different candidate, which
would no longer isolate terminal reference. Such independently discovered and
partially matched outputs are retained only as invalid diagnostic artifacts.

## Integrity result

The first-onset matched screen had 11 common qualified cases and passed every
predeclared integrity item:

- exact discovery, control-versus-replay and cross-treatment physical
  predecision identity;
- the same declared target vehicle;
- complete 60-second branch horizons and zero control commands;
- no commands to a vehicle other than the target;
- zero collisions and teleports;
- 91.7% of feasible P50 activations received at least 16 of 20 commands;
- 100% passed the frozen safety/service constraints when dose was excluded;
- 36.36% of cases had P30/P50 command traces differing by at least 0.05 m/s2
  on at least 30% of shared command steps.

Later recurrences were compared only where the recorded P30 recurrence existed.
Ten P30 second onsets and eight P30 third onsets existed; the paired common
qualified intersections contained seven and six cases, respectively. Their
per-run `decisions_found_for_all_requested_seeds` flag is therefore false by
construction, while exact physical predecision and target identity hold for
every included pair. Missing recurrences were not imputed or replaced.

Across the final 24 common qualified cases, nine (37.5%) met the case-level
command-distinctness threshold. Four had no differing command step, and the
median fraction of differing shared steps was 20%. All four P50 harmful cases
received the full 20-command dose and completed normally rather than ending in
a safety abort.

## Comparative outcome

| Metric | P30 | P50 | Frozen P50 requirement |
| --- | ---: | ---: | --- |
| Common qualified cases | 24 | 24 | nonzero |
| Harmful cases | 4 | 4 | strictly fewer than P30 |
| Harmful fraction | 16.67% | 16.67% | strictly lower than P30: **fail** |
| Mean effect | 0.01924 | 0.00715 | no more than 0.02 below P30: pass |
| Tenth-percentile effect | -0.10736 | -0.07690 | higher than P30: pass |
| Helpful independent development seeds | 5 | 3 | at least 3: pass |
| Nonbeneficial development seeds | 11 | 11 | at least 3: pass |
| Effects beyond replay noise | 45.83% | 33.33% | at least 30%: pass |

The improved P50 tenth percentile is useful mechanism evidence, but it does not
override the predeclared primary requirement. The harmful-rate tie means the
less severe terminal ceiling did not reliably solve the bounded-harm problem.

## Failure-mechanism diagnosis

### 1. Excessive target-speed differential

Not sufficient as the sole explanation. P50 improved the pooled tenth
percentile, which is consistent with reduced severity, but harmful P50 cases
covered first-command terminal drops from approximately 0.06 to 4.77 m/s.
Helpful P50 cases covered approximately 0.38 to 5.10 m/s. The strong overlap
means a smaller fixed percentile alone cannot separate helpful from harmful
states.

### 2. Secondary shock imposed on followers

Not the dominant measured mechanism at the immediate follower. None of the
four harmful P50 cases added an immediate-follower severe-braking or low-TTC
event. Immediate-follower acceleration-energy and integrated-negative-
acceleration changes were mixed, with modest deterioration in two of four
cases. A wave propagating beyond the nearest follower remains plausible, but
the current nearest-follower diagnostic does not prove it.

### 3. Fixed duration or late release

Plausible but unresolved. Every harmful P50 treatment delivered all 20 commands
and ended with normal option completion. This rules out insufficient dose and
safety-abort truncation. Current aggregate metrics do not localize whether harm
appeared during hold or after release, so phase-resolved or deliberately
truncated treatment would be required to test this mechanism.

### 4. Candidate-placement and shock-location mismatch

Plausible. Helpful and harmful groups had almost identical mean available
runway (about 680 m) and predicted dose (about 23 s), so the current closest
runway-ranked geometry does not discriminate them. One second-onset case
changed from helpful under P30 to harmful under P50 despite exact decision and
target matching. Selection should therefore consider estimated shock
interception time and rear headroom, not runway alone.

### 5. Lane-change or service loss

Rejected as the primary explanation for the retained paired cases. Harmful P50
cases still passed the frozen arrival, discharge, delay, TTC, collision and
teleport constraints; their candidate did not indicate an imminent desired or
available lane change at selection. The harmful label was driven mainly by
corridor severe-braking and wave terms, not queue or delay cost.

### 6. Safety-projection saturation

An important authority limitation, but not the only source of harm. Only nine
of 24 cases were mechanically distinct at the predeclared case threshold, four
had identical projected command traces, and the median distinct-step fraction
was 20%. Front-vehicle safe speed, comfort and jerk projections frequently
collapsed P30 and P50 into nearly the same executed action. However, harmful
outcomes also occurred in mechanically distinct cases, so relaxing the shield
would not be a justified remedy.

### 7. Inherently low effective authority

This is the central diagnosis. Sixteen of 24 P50 outcomes were neutral and only
eight exceeded replay noise. A fixed terminal percentile changes too few
executed commands and does not adapt to heterogeneous traffic response. The
treatment has real authority in some states, but that authority is sparse and
is not consistently aligned with corridor benefit.

## Smallest scientifically defensible replacement

The next treatment should be a rear-aware feedback pulse with an explicit
slow-in, conditional-hold, fast-out lifecycle. This is a new scientific
treatment definition and requires approval before implementation.

1. Keep native IDM as the exact abstention control and safety fallback.
2. Rank candidates by predicted shock-interception time, usable runway and rear
   headroom, rather than closest runway alone.
3. Replace the fixed terminal-speed ceiling with a FollowerStopper/Jam-
   Absorption-Driving-style feedback command using front gap and relative speed,
   while applying a rear-gap/closing-speed constraint so the selected CAV does
   not create an unsafe follower demand.
4. Ramp into a short 3--5 second probe. Continue holding only when measured
   downstream wave/severe-braking response improves without rear or service
   deterioration; otherwise release early.
5. Cap total treatment at 20 seconds and return smoothly but promptly to native
   IDM. Log phase-specific response so probe, hold and release effects can be
   separated.

The smallest next experiment, after freezing this replacement, is a six-seed
matched mechanism pilot on already inspected development seeds. It should
verify exact replay/target identity, rear admissibility, phase-specific dose,
mechanical authority, zero added hard-safety events, and both helpful and
nonbeneficial outcomes before any broader data or RL work. It must compare
against both native IDM and frozen P30. More terminal percentiles should not be
tested.

## Reproducible artifacts

- First onset P30:
  `runs/eval/idm_p30_p50_m2000_l1_p30_development_v1`
- First onset matched P50:
  `runs/eval/idm_p30_p50_m2000_l1_p50_matched_development_v2`
- First onset analysis:
  `runs/eval/idm_p30_p50_m2000_l1_matched_analysis_v3`
- Second onset P30 / matched P50 / analysis:
  `runs/eval/idm_p30_p50_m2000_l2_p30_development_v1`,
  `runs/eval/idm_p30_p50_m2000_l2_p50_matched_development_v1`, and
  `runs/eval/idm_p30_p50_m2000_l2_matched_analysis_v2`
- Third onset P30 / matched P50 / analysis:
  `runs/eval/idm_p30_p50_m2000_l3_p30_development_v1`,
  `runs/eval/idm_p30_p50_m2000_l3_p50_matched_development_v2`, and
  `runs/eval/idm_p30_p50_m2000_l3_matched_analysis_v3`
- Final pooled decision:
  `runs/eval/idm_p30_p50_m2000_pooled_analysis_v1/summary.json`

The invalid independently discovered P50 first-onset artifact and incomplete
third-onset matching attempt remain preserved as diagnostics and are excluded
from all values above.

## Reproducible commands

Run these from `D:\research-dense` in the configured `rl_driving` environment.
The seed list and all scientific arguments are explicit. Replace `{n}` with
`1`, `2`, then `3`, and use the matching output paths shown below.

```powershell
python tools/run_idm_decentralized_authority_verification.py `
  --output runs/eval/idm_p30_p50_m2000_l{n}_p30_development_v1 `
  --seeds 4121 4122 4123 4124 4125 4126 4127 4128 4129 4130 4131 4132 `
  --demand 2000 --workers 4 `
  --action-set activation_speed_ceiling_v1 `
  --option-steps 20 --safety-horizon-steps 60 --cav-penetration 0.10 `
  --decision-index {n} --maximum-landmarks-per-cell 3 `
  --reward-refs runs/eval/idm_decentralized_lower_reward_refs_v1.json `
  --candidate-ranking runway --candidate-count 1 --landmark risk_onset
```

For P50, use the corresponding P30 directory as the frozen decision source.
The executed P50 output suffix is `v2` for recurrences 1 and 3 and `v1` for
recurrence 2.

The exact substitution map is: recurrence 1 uses `{n}=1`, `{version}=2`,
`{analysis_version}=3`; recurrence 2 uses `2`, `1`, `2`; recurrence 3 uses
`3`, `2`, `3`.

```powershell
python tools/run_idm_decentralized_authority_verification.py `
  --output runs/eval/idm_p30_p50_m2000_l{n}_p50_matched_development_v{version} `
  --seeds 4121 4122 4123 4124 4125 4126 4127 4128 4129 4130 4131 4132 `
  --demand 2000 --workers 4 `
  --action-set activation_speed_ceiling_p50_v1 `
  --option-steps 20 --safety-horizon-steps 60 --cav-penetration 0.10 `
  --decision-index {n} --maximum-landmarks-per-cell 3 `
  --reward-refs runs/eval/idm_decentralized_lower_reward_refs_v1.json `
  --candidate-ranking runway --candidate-count 1 --landmark risk_onset `
  --decision-source runs/eval/idm_p30_p50_m2000_l{n}_p30_development_v1
```

Analyze each recurrence and then pool the three summaries in recurrence order:

```powershell
python tools/analyze_idm_p30_p50_development_screen.py `
  --p30 runs/eval/idm_p30_p50_m2000_l{n}_p30_development_v1 `
  --p50 runs/eval/idm_p30_p50_m2000_l{n}_p50_matched_development_v{version} `
  --output runs/eval/idm_p30_p50_m2000_l{n}_matched_analysis_v{analysis_version}

python tools/pool_idm_p30_p50_development_screen.py `
  --analyses `
    runs/eval/idm_p30_p50_m2000_l1_matched_analysis_v3/summary.json `
    runs/eval/idm_p30_p50_m2000_l2_matched_analysis_v2/summary.json `
    runs/eval/idm_p30_p50_m2000_l3_matched_analysis_v3/summary.json `
  --output runs/eval/idm_p30_p50_m2000_pooled_analysis_v1
```

## Claim boundary

This screen supports only the statement that a matched P50 terminal-reference
treatment was implemented with valid branch identity and adequate dose, and
that it failed to reduce the harmful fraction relative to P30 in inspected
development cases. It does not establish a working trained damping controller,
observation sufficiency, calibrated activation, prospective efficacy,
generalization, collapse prevention, or readiness for moving-wall escalation.

## Software verification

The focused P50, matched-replay and pooled-decision slice passes 40 tests. The
complete repository suite passes 602 tests. Both were run with the project
interpreter at
`C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe`. Conda activation
metadata on this host currently does not recognize the environment directory,
so direct interpreter invocation is the reproducible test command:

```powershell
C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe -m pytest -q
```
