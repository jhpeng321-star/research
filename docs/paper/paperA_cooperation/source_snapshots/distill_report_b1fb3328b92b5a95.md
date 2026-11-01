# S-DISTILL — observable-only exact-teacher student
COMPLETE 2026-11-01: frozen student tested LIVE on24freshDEVworlds; **does not establish a win over MAPPO**.

## Data and registered roles
Existing EXAM300states/60TRAINworlds, all4500D120/D300/D600 labels verified by original byte hashes.
All300feature vectors and option masks exactly match the causal COOPBANDIT builder.
All720COOPPLAN decisions/60worlds extracted; these are explicitly never-fit DEV in their registrations.
Thus they are post-freeze audits only; no fitting/selection on them or previous/live DEV worlds.
Predictors: sensed lane count/speed/queue/density, CAV telemetry, member position/speed/gaps, clock and setting.
No world/seed/cell/demand/true-share/driver features; no new SUMO for training.

## Frozen student and honest offline results
Available-only per-option squared-loss GBT and scaled16x8tanhMLP; direct-choice GBT comparator.
Outer5/inner4world-grouped CV selects model and confidence0/.5/.65/.8/.9 by equal-world exact saving.
Rank by predicted mean D600; O0 if maximum<=0 or selected confidence fails; no lower-ranked fallback.
Final full-TRAIN selection: MLP/confidence0.9; selection-CV saving+18.312veh-s/state,5moves (selection-biased).
Honest nested-CV saving **−16.497 [−49.493,+0.003] veh-s/state**,2/300moves (O1=1/O4=1).
Teacher agreement84.67%; helpful-teacher agreement0/45; oracle+588.478[296.569,966.148].
Direct classifier:85.00%agreement,26moves, saving−82.552[−323.319,145.751].
Unconstrained mean-only OOF:GBT46moves/−140.174;MLP62moves/−286.792 (not independent selection estimates).
Confidence0.9 was CV-chosen, not a hand-set pre-gate; negative CV does not prevent the approved live test.
All four final MLP fits reached the300iteration limit; finite frozen predictions, disclosed rather than retuned.
PolicySHA256`0d0743c84582329efb775b95807ad677336c005dcda02a97f0b9a3e973aff144`.

## Post-freeze planner DEV audit
H120:432states/36worlds,6student moves,87.27%teacher agreement;212selected effects measured/220missing.
H300:288states/24worlds,3student moves,94.44%agreement;30selected effects measured/258missing.
O0-only shortcuts and quota-pruned candidates lack lookahead labels; never impute these as measured effects.
These horizons differ from trainedD600; incomplete-label conditional saving is not a full-panel policy estimate.

## Live test
FreshDEV984000..984023/all12cells, all3hostexposure+dualguards+CPU gates PASS; mechanics BITWISE PASS.
All24paired1200s episodes/288decisions COMPLETE; saving=MAPPO−student, full-team on-road+source delay.
**J600 saving−96.482[−575.232,+288.473]; J900−54.565[−294.222,+188.886] veh-s/world.**
95%percentile CIs:10000whole-world resamples/RNG20261101; negative means higher student delay.
Six executed groups in5worlds:O1=1/O2wall=1/O3=1/O4=3;282O0; **ZERO budget-pruned decisions**.
Thus the credit5850/world resource ceiling did not change this live policy's proposed actions.
All6groups have same-prefix H120 O0/joint/BOTHsolo reruns;24/24BITWISEprefixes PASS.
MeanC_G−35.890[−59.837,+6.218]veh-s/action,2/6positive;35/10000zero-event bootstrap draws excluded/reported.
Both positiveC_G cases have negativeD_G; **0/6groups have BOTH D_G>0 and C_G>0**.
MAPPO/student collisions0/0,teleports0/0,TTC<3 exposure13516.5/13580.5veh-s,hard-braking452.5/438.5veh-s.
Student400member requests/181clamped;19all-O0worlds exactly match MAPPO clocks/finalprefixes.
Actual=charged**72960sim-s**=57600paired+14160credit+1200extra mechanics,<=199200upper<=200000cap.
All49episode/mechanics+24credit jobs closed; no STOP/incomplete/outstanding physics or automatic retry armed.

## Which observables trigger moves?
FinalMLP TRAIN permutation sensitivity (descriptive):O1 member1lane/lane1CAVcount/member1speed;
O2 lane1count/density/member1speed;O3 lane1speed/rampspeed/member0gap;O4 memberlanes/lane1density.
Held-world O1 trigger:stopped lane1pair,speeds0.91/1.00m/s,gaps2.94/3.15m;exactD600−4949.30.
Held-world O4 trigger:mixed-lane free-flow pair28.55/30.01m/s;exactD600+0.25;no O2/O3outerCVmoves.

## Reproducibility
Design:`DISTILL_DESIGN.md`; code:`tools/distill/`; immutable evidence:`distill_v1/`.
Run:`py -3.13 -B -m tools.distill.train <new-output-directory>` (exclusive-create; never overwrite).
CPU:local31PASS; frozen gpuA8PASS/0skips/BOTH24guards; original-vs-numpy1020context parityPASS;RuffPASS.
Evidence includes world folds/OOF, all model-confidence scores, full teacher DEV records/hashes,
TRAIN feature importance/prediction sensitivity and per-option observable trigger summaries.
TRAIN prediction sensitivity is descriptive, not causal feature attribution.
Live design/evidence:`DISTILL_LIVE_DESIGN.md`/`distill_live/`; independent report:`python -B -m tools.distill.report --verify`.
