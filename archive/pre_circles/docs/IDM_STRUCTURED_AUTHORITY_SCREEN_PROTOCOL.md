# IDM structured-action authority screen

Status: frozen development-only action-space screen. It does not authorize PPO
or opening validation seeds.

## Scientific control and treatments

The shared control is exact analytical damping. A duplicate nominal branch
measures replay noise. Every treatment retains the deterministic absorber
selector, five-second action transition, analytical low-level command, existing
front/rear projection, acceleration and jerk limits, and smooth release.

- Gain: `a_desired = alpha_filtered * a_analytical`; values 0.50, 0.75,
  1.00, 1.25 and 1.50. Nominal is 1.00.
- Reference offset: the analytical reference is first computed and filtered as
  before, then `delta_v_filtered` is added and clipped to the frozen reference
  floor and SUMO allowed speed before evaluating the same analytical equation.
  Values are -2, -1, 0, +1 and +2 m/s. Nominal is 0.
- Discrete modes are aliases, not duplicate simulations: mild=0.75, nominal=1,
  strong=1.25 gain.

Arbitrary absorber choice is excluded. The existing selector does not yet
expose a fully safety-admissible ranked candidate set, and changing it requires
a separate scientific freeze.

## Cohort and horizons

The instrumentation smoke uses demand 2200 and development seeds 4107 and
4109. It evaluates wave and severe-braking onset, at most one of each per seed.
The pulse is 20 controller seconds, the safety/wave horizon is 40 seconds, and
the service horizon is 60 seconds.

Only after the smoke passes may the initial screen use:

- demand 2200: seeds 4107--4112;
- demand 2700: seeds 4113--4116.

Seeds 4101--4106 remain excluded reference-construction seeds. Seeds 4201 and
above remain unopened.

## Resume and artifacts

The coordinator writes an immutable manifest/protocol hash, atomic landmark and
branch JSON, compressed per-branch NPZ traces, progress, and summary. Rerunning
the identical command skips complete artifacts. The scheduling unit is
`(seed, demand, landmark, action_family, action_value)`. Three workers are
frozen for the smoke.

## Smoke command

```powershell
python tools/run_idm_structured_authority_screen.py `
  --output runs/eval/idm_structured_authority_smoke_v1 `
  --cell 2200:4107,4109 `
  --landmarks wave_onset severe_onset `
  --gain-values 0.75 1.0 1.25 `
  --reference-offsets-mps -1.0 0.0 1.0 `
  --pulse-steps 20 --safety-horizon-steps 40 `
  --service-horizon-steps 60 --max-predecision-steps 1200 `
  --workers 3
```

## Full initial-screen command

```powershell
python tools/run_idm_structured_authority_screen.py `
  --output runs/eval/idm_structured_authority_screen_v1 `
  --cell 2200:4107,4108,4109,4110,4111,4112 `
  --cell 2700:4113,4114,4115,4116 `
  --landmarks wave_onset severe_onset `
  --gain-values 0.50 0.75 1.00 1.25 1.50 `
  --reference-offsets-mps -2.0 -1.0 0.0 1.0 2.0 `
  --pulse-steps 20 --safety-horizon-steps 40 `
  --service-horizon-steps 60 --max-predecision-steps 1200 `
  --workers 3
```

## Frozen gates

Integrity requires exact predecision/nominal replay identity, causal one-second
timestamps, correct follower/lane association, no action leakage, complete
horizons, no collision/teleport, and at least 16/20 action-role steps.

A family advances only if at least 30% of eligible effects exceed the symmetric
replay deadband, helpful and nonbeneficial outcomes each span three development
seeds, two values win in different states, median paired projection survival is
at least 50%, execution dose and effect magnitude have Spearman rho at least
0.40, effects reach follower/influence/lane, low-TTC excess is at most two
steps, delay regression is at most 1 s/vehicle, and arrivals/discharge lose no
more than one vehicle. No criterion may be waived after outcomes are inspected.
