# Cooperation bandit — complete results, 2026-11-01

Cooperation can help in exact SUMO twins, including one genuinely beneficial
group interaction. The registered observable-state learner did not learn to
select it confidently: it always fell back to MAPPO, including on fresh dev.

## Exact TRAIN exam and learning
- Complete: 300 states / 60 worlds / all 12 demand×share cells; 4,500 labels
  at 120/300/600 s, and 240 O2/O3 joint/solo-credit rows. No dev data used for fitting.
- High-demand .2/.4 cells have 180 states; 20 satisfy the prospective dense rule.
- Helpful available option in 45/300 states = 15%, world-bootstrap 95% CI [10%,20.33%].
  Only 81 states have an eligible two-CAV group. The mixture is intentionally oversampled.
- Oracle saving 588.48 veh-s/state, CI [295.05,961.64]. Registered nested-CV saving 0;
  regret 588.48 with the same CI; 0 selected options. Full-fit uncertainty radius 9612.20.
- Confidence uses nested world-grouped signed overprediction, a 95% quantile of
  each world's worst error across options/times; choose positive lower bound or O0.
  This is CV-derived uncertainty, not a behavioural threshold or live guarantee.
- Available/helpful: O1 metering 81/30, O2 wall 34/9, O3 ramp gap 43/14, O4 gap 81/31.
  Mean available-option effects are harmful: -1176/-1153/-1736/-1022 veh-s.
  Best observed effects: +14992/+9917/+6338/+17154 veh-s; selected TRAIN maxima.
- Primary model importance emphasizes second-member rear gap/position/speed;
  O3 also emphasizes first-member position and receiving-lane speed. PDP plots
  describe model associations; no reliably beneficial observable rule was learned.
- S-COOPBANDIT-LEARN available-only GBT/linear diagnostics also select O0 everywhere;
  diagnostic GBT RMSE 5262.43/radius 11905.25. They do not replace the registered fit.

## Measured useful interaction
Of 80 measured O2/O3 state-option credit cases, 12 are available; one has D_G>0
and C_G>0: s981057_t420/O3, 3 s headway held 60 s. Two almost-stopped receiving-lane
CAVs near the merge face a slow queue and observed ramp vehicles. Joint saving
+755.39 veh-s; solos -5203.75/-1481.94; C_G=+7441.08. This is a selected TRAIN
example, not learned deployment success. See `coopbandit/full/COOPERATIVE_CASE.md`.
Positive C_G alone is insufficient: available-case C_G spans -5211.84..+11303.43.

## Fixed fresh-dev live test
- All 12 worlds 981080–981091 complete; RSU 15 decisions/world every 60 s.
  There were 44/180 group-eligible decisions; policy executed 0 options.
- Learned/MAPPO clocks match BITWISE. J600/J900 paired differences 0 veh-s,
  both 95% bootstrap CIs [0,0]. Mean J600 61006.69, J900 106159.67 veh-s in both arms.
- Restricted baseline-t300 oracle: all 48 options unavailable/no-op, saving 0.
  It had no intervention opportunity and is not a closed-loop oracle bound.
- Executed-event C_G is NA: no events, so no credit twins or attributed cooperation.
- No collisions/teleports. Both arms' mean TTC<3 s exposure 424.25/597.33 veh-s
  and hard-braking 12.46/18.08 veh-s at 600/900 s; option override share 0.
- COOPBEHAV finds 185 descriptive cross-lane-hold candidates in EACH arm,
  zero yield/gap/wave events; no new attributable cooperative behaviour.
  TRAIN safety/audit counters are exported; original safety layer is unchanged.

## Artifacts, provenance and budget
EXAM `coopbandit/full/exam/`; schema `coopbandit/EXAM_SCHEMA.md`; world-fold OOF
rows `coopbandit/full/learned/`; primary CSVs/figures `coopbandit/full/registered-summary/`;
helper diagnostics `coopbandit/full/helper/`; live results `coopbandit/live/report/`.
World IDs/cells are metadata, never predictors. Original byte SHA manifests verify.
Options/traces/planner handoff: `../../HANDOFF_COOPBANDIT.md`. First 135-state
partial analysis remains preserved in `coopbandit/partial135/` (published before full).
All three hosts used nice19 CPUs, frozen E3/MAPPO169/SUMO1.27.1/noise; bitwise
cross-host reducer gates passed without tolerance or safety changes. CPU 20/20 PASS.
Final actual 1,567,326 sim-s = data 1,495,326 + live 72,000; conservative charged 1,569,840
<=1,600,000. Event-credit reserve unused; all failed charges/STOPs retained; no automatic retries.
