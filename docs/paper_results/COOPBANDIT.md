# Cooperation bandit — first partial results, 2026-11-01

Some fixed group moves save substantial delay in exact SUMO twins, but the
registered observable-state learner has not yet learned when to execute them
confidently. This is an exploratory 135/300-state completion-order subset.

## First milestone
- 135 states, 28 TRAIN worlds, 10/12 cells represented; remaining labels running.
- Every included state has all four options and 120/300/600 s exact labels.
- 16/135 states (11.85%) have an available helpful option at 600 s.
- Equal-world helpful fraction: 12.5%, bootstrap 95% CI [6.43%, 19.64%].
- Equal-world oracle saving: 284.87 veh-s/state, 95% CI [122.91, 484.80].
- Registered nested-CV policy: zero interventions, zero saving; oracle regret
  284.87 veh-s/state with the same CI. Always-MAPPO fallback is active.
- Its full-data CV radius is 11,465.27 veh-s; it is learned from world-max
  signed overprediction, not a hand-set traffic threshold.
- Available/helpful counts: O1 28/12, O2 14/2, O3 12/5, O4 28/11.
- Best observed D600: O1 6,360.02; O2 1,521.60; O3 2,273.65;
  O4 6,315.78 veh-s. These maxima are selected examples, not policy gains.
- Seven available O2/O3 solo-credit cases are complete. None has positive D_G;
  C_G ranges from -3,912.18 to +11,303.43. Positive interaction alone does
  not mean the group move beats MAPPO.
- No collisions or teleports in the exported 600 s outcomes. Detailed TTC,
  hard-braking and override shares are in labels.csv; no safety superiority claim.

## Companion analysis and artifacts
S-COOPBANDIT-LEARN GBT and linear diagnostics also select O0 in every OOF
state. Their available-option-only fitting/calibration differs from the registered
all-state learner; GBT radius 9,262.55, RMSE 4,247.30 veh-s (exploratory).
Feature-importance/PDP CSVs and PNGs are in `coopbandit/partial135/helper/`.
Model importances describe fits; no stable beneficial traffic rule is established.
The helper input is a documented metadata-only view of the verified EXAM;
the original labels/features/manifest are preserved unchanged.

EXAM: `coopbandit/partial135/exam-partial/README.md`, `states.jsonl`,
`labels.csv`, `group_credit.csv`, `worlds.json`, provenance and SHA manifest.
Registered CV outputs: `coopbandit/partial135/learned/`.
World IDs are grouping metadata; no true share/demand/driver truth is a feature.
All dev worlds 981080–981091 are excluded from fitting and calibration.
Option/runner handoff: `../../HANDOFF_COOPBANDIT.md` from this directory.

## Execution and remaining work
Recovered existing cpuQ/gpuA jobs; admitted gpuR only after its full 1140 s
bitwise compatibility proof. Preserved all old STOPs, attempts and charges.
Frozen MAPPO169/E3/SUMO1.27.1/noise/original safety layer unchanged.
Conservative final data reservation 1,497,840; live reservation 99,360;
combined 1,597,200 <= 1,600,000 sim-s. Live test is not yet dispatched.
Final 300-state analysis, fresh-dev paired J600/J900 test and executed-event
cooperation evidence remain pending. No learned cooperation success is claimed.
