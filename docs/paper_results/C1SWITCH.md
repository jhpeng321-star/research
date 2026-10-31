# S-C1SWITCH — controller-switching experiment

## Status
**Incomplete panel: operational/budget stop; no full-panel efficacy verdict.**
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
PASS: 107 complete episodes, 315 unique full-rerun branches, and identical available-arm warmup fingerprints on all 16 worlds. Registered clocks, schedules, branch/live prefixes, standalone/embedded results, finite metrics and charged/actual durations match. Retained episodes/branches have zero collisions and teleports. No efficacy statistics were computed.

## Efficacy analysis gate
The registered complete-panel bootstrap, WM-versus-BB/best-fixed primary contrasts, and paired episode comparisons are **not evaluated**. Availability depends on the interrupted scheduler/model requests; treating the retained subset as a paired16-world experiment would be invalid.
- Complete episode physics: WM 5/16; BB 16/16; random 6/6; each fixed controller16/16.
- Committed switching picks: WM 56/160; BB 160/160. Completed counterfactuals: 315/480.
Terminal shared-v10 abandoned physics: 960 sim-s (retained, no refund). Standard full-rerun completion projection retaining every successful run: 500,640 sim-s. Further compute requires explicit MAIN continuation/budget authority; none is assumed.

## Existing paper figure
Recorded WMDATA/WMDATA2 fixed-controller winner-by-condition figure remains c1switch/recorded_winner_by_condition.png (and PDF), with 90 recorded worlds and zero new SUMO. All five fixed controllers win somewhere; this is not evidence that WM switching works.

## Receipts
tools/c1switch/evidence/V10_COLLECTION.json, V10_PROGRESS.json, V10_PLAN.json, V10_STAGE.json, V10_CPU.json and shared_v10_panel/ contain the terminal accounting and retained scientific receipts. Historical V7_INTERRUPTION_AUDIT.json/V9_FINAL_AUDIT.json preserve preceding interruptions.
