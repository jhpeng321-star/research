# IDM compact-wall penetration feasibility scan protocol

Status: **frozen implementation; not executed**.

Date: 2026-08-30

## Scientific question

At the already-frozen high-loading IDM condition, what is the **lowest CAV
penetration** at which the implemented compact moving wall is available often
enough to justify a later paired treatment pilot?

This is a placement-feasibility question. CAV penetration changes the chance
that a compact cross-lane cohort exists; it is not a traffic-density variable
and this scan does not estimate a critical density.

The scan is treatment blind. Every trajectory uses frozen analytical
`disturbance_p30` damping only. A temporary wall is never applied. The scan
contains no outcome, reward, benefit, collapse-prevention, service, or
treatment-effect label.

## Frozen grid

- IDM mainline demand: 2700 veh/h.
- Ramp demand: 400 veh/h.
- Seeded Bernoulli CAV penetrations, checked in ascending order: 10%, 15%, 20%,
  25%, 30%.
- Frozen v5.2 wall-pilot seeds: 4301--4312 (12 seeds).
- Predeclared controller steps: 620, 660, 680, 700.
- Total grid: 60 damping-only trajectories and 240 predecision seed-window
  checks.

Seeds 4301--4312 are the separately assigned v5.2 wall cohort. This scan opens
only their predecision damping trajectories and never inspects a treatment
outcome. They remain outside controller acceptance, predictor validation, and
final testing. The scan runner fails closed on any different seed set. It does
not touch 4201--4240, 4313--4320, 4601--4660, or 5001--5060.

## Frozen wall feasibility definition

The scan imports the separately versioned v5.2 overlay and its exact profile
fingerprint without changing it:

- `compact_moving` geometry;
- trajectory-coherent cohort selection;
- complete coverage of every occupied recruitment-band lane;
- one selected CAV per required occupied lane;
- maximum initial and predicted longitudinal span of 50 m;
- 20 s requested assemble--hold pulse;
- the implemented kinematic headroom, activation speed-spread, front/rear TTC,
  and acceleration checks;
- 20 m front and rear activation-gap floors;
- the existing adaptive-servo parameters.

The v5.2 profile accepts only the canonical
`build_frozen_idm_analytical_policy` factory. Lambdas, wrappers, residual
controllers, legacy checkpoints, and implicit fallbacks fail closed.

For this binary admission contract, a formation is geometrically feasible only
if its predecision trajectory projection supports the full 20 s pulse. The
recorded projected-dose fraction is therefore 1.0 for a complete feasible
projection and 0.0 otherwise. It is not realized treatment dose and says
nothing about runtime completion after a wall is activated.

A seed-window is admitted only when all four predecision conditions hold:

1. compact trajectory geometry is feasible and stays within 50 m;
2. selected members cover 100% of occupied recruitment-band lanes;
3. projected duration is at least 80% of the requested 20 s;
4. the frozen activation-safety check passes.

The analyzer independently reconstructs admission from those fields. A stored
`admitted` value that disagrees causes an integrity failure.

## Predeclared penetration-selection rule

Starting at 10% and moving upward, select the first penetration with:

- at least 8 of 12 independent seeds having one or more admitted windows;
- at least 12 admitted seed-window cases in total; and
- admitted cases at two or more distinct decision steps.

All 60 declared cells and all 240 windows must be present before selection.
The analyzer rejects files containing treatment-outcome fields and does not
fall back to a higher penetration when a grid artifact is missing.

If no penetration passes, compact full occupied-lane coverage is not available
often enough on this grid. Do not relax lane coverage, inspect treatment
benefits, add seeds, or train an escalation predictor to manufacture a pass.
Instead, make an explicit scientific decision about the treatment geometry.

If a penetration passes, that result authorizes only a later treatment-dose
pilot at the selected penetration. It does not establish realized dose,
safety after activation, treatment benefit, predictor performance, or a paper
claim.

## Commands (not yet executed)

The scan runner creates only damping trajectories and predecision diagnostics:

```powershell
python experiments/idm/scan_idm_compact_wall_feasibility.py `
  --workers 4 `
  --output runs/probe/idm_compact_wall_feasibility_p10_p30_s4301_4312
```

After all 60 cells are complete, the treatment-blind analyzer is:

```powershell
python tools/analyze_idm_compact_wall_feasibility.py `
  --scan-root runs/probe/idm_compact_wall_feasibility_p10_p30_s4301_4312 `
  --output runs/probe/idm_compact_wall_feasibility_p10_p30_s4301_4312/analysis.json
```

Expected wall-clock time is approximately 1--3 hours with four local workers,
depending on SUMO and IDM-controller throughput. The analyzer should finish in
seconds. The scan is resumable; existing cell files are skipped unless
`--force` is deliberately supplied.

## Implementation files

- `experiments/idm/scan_idm_compact_wall_feasibility.py`: frozen, damping-only
  grid runner and predecision diagnostic writer.
- `tools/analyze_idm_compact_wall_feasibility.py`: fail-closed grid validation,
  treatment-outcome rejection, admission reconstruction, and lowest-passing
  penetration selection.
- `tests/test_idm_compact_wall_feasibility.py`: contract, contamination,
  full-lane, projected-dose, safety, grid-completeness, and selection tests.
