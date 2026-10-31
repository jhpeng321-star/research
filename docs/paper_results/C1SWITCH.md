# S-C1SWITCH — controller-switching experiment

## Status
**PARTIAL — registered statistics on 5 all-arm-complete worlds, per MAIN01:28.**
The 16-world panel remains incomplete after the GPU-budget stop. No further SUMO/GPU is authorised; the analysis below uses retained receipts only and is not a full-panel efficacy verdict.
Terminal stop receipt: inference stopped/timed out; SUMO stays paused
GPU terminal cause: combined owner GPU budget exhausted

## Frozen scientific contract
- WM2 b8797242+mix (entries OFF), full 60 s anonymous observable history; frozen BB-traj afaaa32c. Original v7 anonymous-seed adapter/controller plant retained.
- Ten decisions every 60 s after 300 s native warmup; library native/CTRL0/MAPPO169/PI/CIRCLES; K8 median predicted on-road fleet delay, incumbent-retaining 1e-6 tie rule.
- Evaluation is true fleet + pending-source team delay. Predicted entries/source delay are unsupported; this objective mismatch is disclosed. BB-id cannot execute feedback controllers.
- Main WM/BB/five fixed arms retain all 16 paired TRAIN worlds969112–969127. Random retains first6 only969112–969117; last10 omitted prospectively for interruption accounting.
- Exact counterfactuals rerun from t0 with the episode controller schedule; three pre-registered shared decision indices/world; all five candidates at60/90s, matching live prefix before intervention. Evaluation branch scheduling cannot feed picks.
- Exposure and limitations: see C1SWITCH_DESIGN.md/C1SWITCH_WINNERS.md and frozen C1-EXPOSURE labels. No model selection/tuning used switching outcomes.

## Retained evidence and compute
- Final charged = closed actual: 383,190 sim-s. Registered complete-recovery projection499680; owner cap500000.
- Retained 107 fully evaluated episodes, 107 complete episode-physics runs, 315 completed full-rerun branches. Complete physics without branches is retained and never rerun.
- All47010 Step0 sim-s,12450 superseded original WM canary sim-s and17820 abandoned prior-prefix sim-s remain charged. Any terminal incomplete shared-v10 physics also remains charged.
- Actual cumulative GPU allocation wall: 14,382.225540s (3.995063h), including loading, parity validation, idle and interrupted allocations. Conservative admission debit: 14,396.231360s (3.998953h); owner cap14400s. Exact phase totals: tools/c1switch/evidence/GPU_ACTUAL_SEGMENTS.json.
- Every original v4–v7 source, STOP, schedule and ledger remains preserved. Initial 10 completed episodes +231 branches were imported once; subsequent complete physics/forecasts reused exactly.

## Engineering gates and interruption history
- Initial paired canary, both original-draw GPU parity gates and anonymous BB-reuse validation passed. Corrected WM-only canary replaced the original scientifically superseded schedule.
- First main interruption hit the old BB3600s internal cap; owner MAIN23:25 authorised shared GPU budget and random-tail trim.
- gpuA recovery CPU gates passed but full60s rawK8 bitwise parity failed both models before physics;57.231276GPU-s retained. No numerical cause or efficacy conclusion was inferred.
- Original gpuR rawK8 bitwise parity passed both models. A subsequent BB200s internal partition expired before owner-cap exhaustion; this operational error caused another stop and6720 abandoned sim-s.
- Final shared_v10 uses one process/one4117s combined alarm for both unchanged models, no per-model subcaps. CPU gates49cpuQ/50gpuR passed, including shared-budget alarm/debit and completed-physics retention tests.
- Own PIDs only; all terminal owned jobs closed. Durable atomic requests, hashes, observable-history cache equality, committed-pick reproduction and live-prefix gates remain enforced.

## Retained-data QA
PASS: 107 complete episodes, 315 unique full-rerun branches, and identical available-arm warmup fingerprints on all 16 worlds. Registered clocks, schedules, branch/live prefixes, standalone/embedded results, finite metrics and charged/actual durations match. Retained episodes/branches have zero collisions and teleports. This QA preceded the separately authorised PARTIAL analysis below.

## Panel coverage and stopping
The full 16-world estimand is **not evaluated**. MAIN01:28 explicitly authorises the registered estimators on the intersection where all arms are complete, labelled PARTIAL. Availability depends on interrupted scheduling/model requests, so these conditional subset results cannot represent the planned E3 population.
- Complete episode physics: WM 5/16; BB 16/16; random 6/6; each fixed controller16/16.
- Committed switching picks: WM 56/160; BB 160/160. Completed counterfactuals: 315/480.
Terminal shared-v10 abandoned physics: 960 sim-s (retained, no refund). Standard full-rerun completion projection retaining every successful run: 500,640 sim-s. MAIN01:28 directs no further spend; the task exits after this analysis.

## PARTIAL cohort and registered statistics
- **n=5 worlds, all eight arms:** WM-S, BB-traj, random, native, CTRL0, MAPPO169, PI, CIRCLES. Exactly 40 complete episodes and 150 five-way full-rerun branches enter this analysis; 15 measured decisions/model, each evaluated at 60 and 90 s.
- Exact worlds/cells (penetration, demand veh/h): 969112 (.05,1900), 969113 (.05,2400), 969114 (.05,2700), 969115 (.1,1900), 969116 (.1,2400). No .2/.4 worlds are included. Other completed BB/fixed/random episodes and incomplete WM schedules are excluded from these paired contrasts.
- Selection is the intersection of final `RESULT.json` receipts for all eight episode arms, requiring all three registered WM/BB five-controller branch sets. No outcome-based filtering or replacement worlds. Input byte SHA256 hashes and availability for all 16 worlds are in `c1switch/partial_results.json`.
- Equal-world means over three pre-registered decisions; 2,000 paired world-cluster resamples, numpy seed 0, percentile 95% CIs. Best fixed is selected by mean fixed-episode J600 **within this subset and reselected inside each bootstrap draw**. Two exploratory primary contrasts, no multiplicity adjustment.
- All differences below are **WM minus comparator**; negative favours WM. Regret uses true full-team branch delay, not predicted road-only cost. WM/BB decision states differ on-policy, paired by world and clock.

### Primary: 60 s decision regret (veh-s)
- Mean regret: WM **22.373**, BB **17.271**.
- **WM minus BB: +5.102; 95% CI [-9.389, +16.746].**
- **WM minus best fixed: +8.232; 95% CI [+1.692, +18.477].** MAPPO169 is the subset's mean-J600 best fixed. Comparator costs are measured at the WM decision states, not subtracted from fixed-episode trajectories.
- All-five fixed-controller regret contrasts: native +8.364 [-3.071,+19.799]; CTRL0 +15.306 [+5.872,+22.567]; MAPPO169 +8.232 [-1.170,+18.866]; PI +9.202 [+1.708,+17.842]; CIRCLES -12.845 [-23.052,-2.595]. The MAPPO-only CI differs from the best-fixed CI because the latter reselects controller identity in each draw.
- Interpretation restricted to this subset: WM does not show lower decision regret than BB or the selected best fixed. The best-fixed contrast favours the comparator under this conditional bootstrap; this is not a population-level model verdict.

### Secondary: episode full-team J600 (veh-s)
- Mean J600, same five worlds for every arm: WM 56,607.662; BB 57,319.158; random 58,027.691; native 57,319.557; CTRL0 58,363.990; MAPPO169 55,963.792; PI 56,603.384; CIRCLES 58,100.360.
- Paired WM differences, 95% CIs: native -711.896 [-3360.162,+1752.619]; CTRL0 -1756.328 [-5278.066,+673.982]; MAPPO169 +643.870 [-1529.908,+2599.148]; PI +4.278 [-2975.479,+2039.833]; CIRCLES -1492.698 [-2546.967,-438.430]; random -1420.029 [-3692.548,+881.701].
- Versus each world's best-fixed-in-hindsight: WM +2055.787 [+1275.442,+2776.377]; BB +2767.283 [+1084.615,+4665.681]; random +3475.816 [+846.077,+6105.555]. This per-world minimum is distinct from the primary's population-best-fixed selection.
- All registered BB/random-versus-each-fixed episode contrasts and intervals are retained in `c1switch/partial_results.json`; all use the identical five-world cohort.

### Direction, rank, switching and safety
- 60 s: WM Spearman mean **0.3221** (15 states), direction vs native **38/60 =63.33%**, 0 tied/excluded contrasts; BB Spearman **0.0846** (15 states), direction **28/54 =51.85%**, 6 tied/excluded. Direction counts span the four nonnative controllers; signs within 1e-6 are ties, and a tie in either prediction or truth is excluded.
- 90 s: WM Spearman **0.1087**, direction **31/60 =51.67%**; BB Spearman **0.1433**, direction **30/54 =55.56%**. Both Spearman means use 15 states; excluded direction counts remain 0/6. Ranks use exact midranks. These are descriptive counts, not independent-observation confidence tests.
- Mean controller changes/episode, including initial native-to-controller handover: WM **6.6**, BB **8.2**, random **8.4**; native 0, each nonnative fixed arm 1.
- **Zero collisions and teleports in all 40 analysed episodes and their 150 evaluation branches.** Episode TTC<3 vehicle-seconds / hard-braking vehicle-seconds, totals over five worlds: WM 2123.5/62.5; BB 2261/79; random 2268/69.5; native 2127.5/84.5; CTRL0 2263.5/57.5; MAPPO169 2126.5/62; PI 2128.5/80; CIRCLES 2336.5/76.5. Hard braking is the backend's integrated braking-vehicle-seconds metric, not an event count.

### Per-cell breakdown and uncertainty
Each observed cell has **n=1**; empirical within-cell bootstrap CIs are degenerate point intervals, not population uncertainty bounds. Full arm J600, regret means, counts and intervals are in the JSON and world CSV. Mean 60 s regret WM / BB (veh-s):
- (.05,1900): 0.069 / 17.595.
- (.05,2400): 28.562 / 9.019.
- (.05,2700): 29.254 / 10.888.
- (.1,1900): 26.694 / 17.605.
- (.1,2400): 27.284 / 31.248.
Absent cells are not imputed. The subset is small, low-penetration and scheduler/budget-selected; the empirical world-bootstrap intervals quantify only resampling of these five worlds. Road-only prediction versus full-team evaluation, controller-exposure differences, approximate native plant and single frozen training checkpoint remain limitations.

## Analysis artifacts and verification
- `c1switch/partial_results.json`: full PARTIAL statistics, input hashes, world/arm coverage, all primary/secondary/safety/cell results.
- `c1switch/partial_worlds.csv`: five paired world rows; `c1switch/partial_decisions.csv`: 60 rows (five worlds x two models x three decisions x two clocks).
- Reproduce the receipt-only analysis with `python -B -m tools.c1switch.partial`. The historical `finish report` command predates MAIN01:28 and must not overwrite this amended report.
- Analysis commit `08de06cc`; 12 CPU tests PASS (`test_c1switch_analysis.py`, `test_c1switch_partial.py`, `test_c1switch_finish.py`), including independent bootstrap reconstruction, exclusion on a missing arm, corrupt-branch/accounting/warmup rejection, and preservation of the full-panel gate. Targeted Ruff PASS.
- **New SUMO 0 sim-s; new GPU 0 seconds.** Final retained compute totals above are unchanged. MAIN01:28 is analysis-only; no jobs, polling loop or continuation is armed.

## Existing paper figure
Recorded WMDATA/WMDATA2 fixed-controller winner-by-condition figure remains c1switch/recorded_winner_by_condition.png (and PDF), with 90 recorded worlds and zero new SUMO. All five fixed controllers win somewhere; this is not evidence that WM switching works.

## Receipts
tools/c1switch/evidence/V10_COLLECTION.json, V10_PROGRESS.json, V10_PLAN.json, V10_STAGE.json, V10_CPU.json and shared_v10_panel/ contain the terminal accounting and retained scientific receipts. Historical V7_INTERRUPTION_AUDIT.json/V9_FINAL_AUDIT.json preserve preceding interruptions.
