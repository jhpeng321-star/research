# Estimand reset — 2026-09-02

## Status

Owner-directed reset of the treatment and label definitions. This supersedes
the v5.x treatment treadmill. It does not reinterpret any prior no-go: those
results stand exactly for the treatments they tested.

## Diagnosis being acted on

The reset acts on four measured problems:

1. The v5.4 compact 20 s wall (10 s assembly, 5 s HOLD, 5 s taper, two CAVs,
   <=50 m span) has no systematic paired effect. Across the 199 engaged
   development cases, delay effect p10/p90 = −1337/+1557 s with 53% positive;
   193/199 cases have no sustained congestion in either arm; hard-brake
   excess median +2 against a budget of ~4.
2. The v5.5 switch from the merge-edge health collapse label (h < 0.15 for
   60 s) to the premerge queue label (>=5 vehicles <15 m/s and mean <=15 m/s
   for 60 s) removed the 2200–2400 veh/h metastable band from the development
   cells (health: 3/6 and 6/6 events; physical: 0/6 and 0/6).
3. The Gate-B safety label is a small-count difference with budget
   max(2, 10%) over the full network and horizon, never calibrated against a
   perturbation-noise floor; and Gate-B excludes delay and collapse
   prevention from "benefit".
4. Legacy evidence (`docs/RESEARCH_OVERVIEW.md` section 4) shows a sustained
   health-gated wall reduced IDM collapse at 2200 veh/h from 0.86 to 0.48
   (rule) and 0.08 (learned), n = 50. That effect is the premise; it has not
   been tested under the current paired-replay infrastructure.

## Decisions

1. Treatment under test: a sustained, health-gated moving wall with
   membership by band passage (legacy actuator semantics), engaged at a
   decision landmark and released on health recovery or a maximum duration.
   Defined precisely by the Stage 1a effect-existence pilot
   (`experiments/idm/run_sustained_wall_v1_stage1a.py`), which executes it.
2. Both collapse labels (health and physical) are recorded and reported;
   neither is dropped.
3. Safety is reported as paired effect sizes against a measured
   perturbation-noise floor, not as a pass/fail count budget.
4. Delay (s/veh) and collapse incidence/duration are primary paired outcomes;
   discharge, hard-brake, low-TTC are reported alongside.
5. No gate is applied on fewer than 20 seeds. Gates are effect-size
   confidence intervals, not conjunctive pass/fail counts.
6. Decentralized damping is demoted from prerequisite to ablation arm.
7. The v5.8 multiwall skeleton is halted.

## What is retained unchanged

Twin-replay identity machinery, Gate F formation screen, the 10 s shadow,
leakage discipline, seed registry, and the development / validation / final
partitions.
