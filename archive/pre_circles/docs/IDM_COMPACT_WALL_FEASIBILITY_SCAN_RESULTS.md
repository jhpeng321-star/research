# IDM compact-wall treatment-blind feasibility results

Status: **passed at 30% CAV penetration**.

Date: 2026-08-30

The frozen scan evaluated 60 analytical-damping-only trajectories and 240
predecision windows at IDM demand 2700 veh/h mainline plus 400 veh/h ramp.
No wall was applied and no treatment outcome or benefit label was read.

The predeclared rule selected the lowest penetration with at least 8 admitted
seeds, 12 admitted seed-window cases, and coverage of at least two decision
steps.

## Results

- 10%: 1 admitted window across 1 seed and 1 decision step -- fail.
- 15%: 3 admitted windows across 3 seeds and 2 decision steps -- fail.
- 20%: 7 admitted windows across 6 seeds and 4 decision steps -- fail.
- 25%: 7 admitted windows across 6 seeds and 4 decision steps -- fail.
- 30%: 12 admitted windows across 9 seeds and all 4 decision steps -- pass.

The selected 30% cohort has admitted opportunities in seeds 4301, 4302, 4304,
4306, 4307, 4308, 4309, 4311, and 4312. The first two predeclared repeat cases
are seed 4301 at step 660 and seed 4302 at step 620.

At 30%, the remaining 36 rejected windows were primarily rejected for compact
geometry rather than CAV absence: 22 exceeded the 50 m formation span, 8
exceeded the activation speed-spread limit, 3 lacked projected kinematic
headroom, 1 exceeded projected trajectory span, 1 left an occupied lane
uncovered, and 1 failed the 20 m front-gap guard.

## Interpretation

This establishes placement availability only. It does not establish realized
treatment dose, runtime safety, treatment benefit, collapse prevention, or
predictor performance. It authorizes the paired p30 treatment-dose/headroom
pilot and nothing beyond that pilot.

Machine-readable artifacts:

- `runs/probe/idm_compact_wall_feasibility_p10_p30_s4301_4312/analysis.json`
- analysis SHA-256:
  `3b0baa9ad6b0ecaed0e1cfecd99b1c0c503d02b680a91402a0b3b39ea7475ad4`

