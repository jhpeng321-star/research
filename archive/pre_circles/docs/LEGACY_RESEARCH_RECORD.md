# Consolidated legacy research record

This document preserves the scientific information carried by the removed
one-off probes, phase launchers, root logs, and duplicate `archive/` tree. Exact
tables, caveats, and run paths remain in `RESEARCH_OVERVIEW.md` and the detailed
historical reports indexed by `INDEX.md`. Git history preserves formerly tracked
scripts when exact procedural reconstruction is ever required.

## Why the older pipeline was retired

The earlier program used a Krauss-trained discrete damping controller, a scalar
metastable-health trigger, and several hand-built or learned wall supervisors.
It established useful actuator and failure-boundary evidence, but it did not
provide a valid counterfactual benefit predictor for an IDM-native paper:

- the damping controller was not designed or trained for IDM wave physics;
- the original composite gate label mixed wall feasibility with benefit;
- 97 of 98 engaged treatments in an older audit terminated early, commonly
  from uncovered occupied lanes, and averaged roughly four seconds rather than
  the requested 30 seconds;
- across 300 older cases, sustained-collapse prevention occurred only once;
- engaged-only benefit prediction was approximately chance;
- development seeds and treatment variants had already been repeatedly inspected.

These findings motivated the current sequence: build and accept an IDM-native
damping checkpoint, freeze it, verify a damping-preserving wall treatment, map
an actionable transition band, then train a two-stage predictor.

## Retained mechanistic findings

1. Isolated-CAV damping can be too weak in HDV-dominated overcritical traffic.
   A coordinated cross-lane moving bottleneck is a materially different actuator.
2. Damping and wall commands must not compete. The treatment arm must use the
   same damping controller everywhere except the explicitly overridden wall
   members and must return those members smoothly to damping on release.
3. Ramp metering, VSL, and binary gating did not rescue the tested mainline-
   overcritical cases; several incurred large delay penalties.
4. Structural capacity loss from a stalled vehicle could not be controlled away;
   upstream control relocated the queue rather than removing the shortage.
5. Wall headway/dose, lane coverage, geometry, penetration, and terminal speed
   matter more than simply allowing a stronger instantaneous braking command.
6. Krauss under-produced the service cost of collapse relative to the tested IDM
   scenarios. Safety, severe braking, oscillation, discharge, and bounded service
   harm therefore require separate reporting rather than a delay-only win label.

## Historical quantitative results (not current-paper validation)

The legacy campaign reported strong collapse reduction for some sustained wall
configurations, including Krauss critical-density cases and an IDM demand ladder.
It also reported IDM delay reductions at high demand for some wall supervisors.
Those values are useful motivation but cannot validate the current paper because
the physics, baseline controller, treatment lifecycle, geometry, seed exposure,
and data schema differ. Consult `RESEARCH_OVERVIEW.md` before quoting any value;
its sample sizes and caveats are part of every result.

Important boundaries retained from that record:

- Krauss wall protection generally traded increased delay for lower collapse.
- IDM protection degraded with demand and penetration changed the frontier.
- several damping variants under IDM were null or harmful.
- old wall-plus-damping stacking was antagonistic.
- penetration changes the simulated population and is not a pure dose variable.
- evidence came from one merge geometry and a limited demand family.

## Removed implementation classes

The cleanup removes duplicate or obsolete root/archive copies of:

- checkpoint-specific `_eval_*` scripts;
- one-off `_probe_*`, `_autopsy_*`, `_analyze_*`, and `_prep_*` scripts;
- `run_phase*` and shell/batch orchestration for the retired campaign;
- old supervisor-training launchers and progress/watchdog files;
- root training/probe logs and completion sentinels;
- temporary render files and local virtual environments.

Maintained scientific mechanisms remain in `env/`, `policies/`,
`counterfactual/`, `evaluation.py`, `evaluate_policy.py`, and the current
IDM/counterfactual drivers. The complete test suite is the deletion guard.

## Seed exclusions inherited by current work

- Wall/treatment development: at minimum seeds 401–408.
- IDM reward/reference and readiness development: seeds 4101–4112.
- Any seed named in a development, sensitivity, timing, or pilot document must
  be added to the exclusion registry before prospective evaluation.

The current protocol must maintain a machine-readable exclusion set before a
paper-scale experiment. This record is a lower bound, not a complete substitute
for scanning every development artifact.
