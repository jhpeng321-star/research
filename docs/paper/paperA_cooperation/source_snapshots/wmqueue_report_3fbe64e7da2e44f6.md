# S-WMQUEUE — conservative queue/service falsifier (2026-10-31)

**KILL this registered pilot.** It fails paired-effect MAE/sign against state-only; no coupling adoption.
Pre-fit design/registration: `c5ed4c79`; implementation `a00a09b8`; pre-TEST-extraction fit freeze `279df385`.
FIT: 82 worlds (969400–969431 + 969550–969599), 460 episodes, 27,600 transitions; TEST: 969432–969439 only.
One fixed 64×64 MLP, 100 epochs; 10-s bins; 9 lane/region cells, 2 source queues, one flux ledger.
State-dependent joint service, speeds/gaps/discharge persistence; bounded supply/storage/service; no jam/speed trigger or driver types.
Free-run uses observable t0/history, causal request-rate estimate and t0 issued-action summary only; no future speed/service/action input.
All FIT road ledgers close exactly; rollout mass residual <=7.96e-13 vehicles. Four invariant/detector checks pass.
Primary: t0=300, fleet+source objective J, all unordered contrasts. CIs: 2,000 whole-world draws, seed 20261031.

## Paired J: MAE (veh-s) / sign accuracy / Spearman
Eight TEST worlds, four cached-WM arms, 48 contrasts/horizon; point estimates (all CIs in linked CSVs):
- 120 s: queue 276/.438/−.047; state-only 239/.625/.334; arm-mean 220/.688/.452.
- 300 s: queue 1362/.313/−.314; state-only 932/.625/.691; arm-mean 930/.646/.585.
- 600 s: queue 5296/.229/−.539; state-only 3642/.667/.631; arm-mean 3227/.708/.571.
Five-arm panel including native: queue MAE/sign 257/.475, 1437/.338, 5053/.325 at 120/300/600; full baselines in metrics.csv.
WM2 comparison uses IDENTITY-bound b8797242+mix K8 means, matching start hashes, arms and objective clocks.
Six complete worlds 969432–969437, 36 contrasts/horizon; 969438/969439 had only 5/1 original futures, excluded:
- 120 s: queue 232/.472/.137 versus raw WM 387/.583/.494.
- 300 s: queue 1252/.389/−.130 versus raw WM 2271/.750/.471.
- 600 s: queue 5557/.278/−.500 versus raw WM 5766/.528/.323.

## Registered improvement gates, point [95% CI]
Positive means better: MAE = baseline−queue; sign = queue−baseline. None of 12 lower bounds is >0.
- State-only, 120 s: MAE −36.7 [−71.5,0.9]; sign −.188 [−.354,.063].
- State-only, 300 s: MAE −429.5 [−645.2,−213.3]; sign −.313 [−.563,−.104].
- State-only, 600 s: MAE −1654.7 [−2555.6,−865.1]; sign −.438 [−.688,−.188].
- Raw WM, matched six worlds, 120 s: MAE +155.2 [−163.4,641.3]; sign −.111 [−.444,.250].
- Raw WM, matched six worlds, 300 s: MAE +1019.0 [−417.1,3418.0]; sign −.361 [−.611,−.111].
- Raw WM, matched six worlds, 600 s: MAE +208.5 [−2372.1,3075.7]; sign −.250 [−.361,−.111].
Incomplete raw coverage independently blocks PASS; the significant state-only deficits already decide KILL.

## Breakdown and components
LRDIAG BIC step+trend detector reused exactly (AST/source hashes retained), 60-s prefix, no speed cutoff; own TEST population:
- 120 s: observed/queue occurrence .400/.525; Brier .325 [.050,.625]; matched onset MAE 25.8 s [2.9,55.0], n=12.
- 300 s: observed/queue occurrence .825/.850; Brier .275 [.075,.525]; matched onset MAE 125.4 s [68.1,182.7], n=28.
- 600 s: observed/queue occurrence 1.000/.875; Brier .125 [.000,.325]; matched onset MAE 193.4 s [134.8,257.7], n=35.
State-only Brier .219/.188/.012 and onset MAE 28.4/42.2/125.0 s; arm-mean Brier .268/.151/.003 at 120/300/600.
Raw cached files lack regional speed trajectories: raw BIC occurrence/timing comparison unavailable, not inferred from J.
Per-arm road/source/J MAE, five arms (veh-s): queue 401/1183/1254, 1424/7534/7459, 4262/34529/32111 at 120/300/600.
Source-queue drift is substantial; exact conservation alone does not establish correct admissions or controller effects.
Authoritative cumulative-team-clock sensitivity preserves every paired sign and the adverse ordering (separate CSV).

## Later starts, cost and evidence
t0=360: road-only paired MAE/sign 231/.700 (120 s), 1414/.425 (300 s); t0=480: 489/.738, 2188/.600.
Later source queues are causal posteriors, not observed labels; later fleet+source truth unavailable; 600-s later labels right-censored.
All 360 per-arm/start/horizon predictions are in [WMQUEUE.csv](WMQUEUE.csv), with unavailable labels blank, never zero.
Single-thread wall: remote aggregation 223 s; fitting 24 s; final scoring 19 s. 0 GPU, no new WM inference, no SUMO or process signals.
Local relay cache contained conflicting unbound copies; scoring-only fix excludes them and snapshots the original producer's existing files.
Frozen model/source hashes unchanged; no TEST tuning/refitting. Designed-controller exposure is adaptation, not zero-shot evidence.
Protocol note: the initial schema probe parsed global PROVENANCE.json (all-world clocks), returning a FIT example only; never used for fitting/selection.
[Design](WMQUEUE_DESIGN.md); [reproduce](../../tools/wmqueue/README.md); [gate/CIs/receipts](../../tools/wmqueue/out/).
This falsifies this fixed offline prototype, not every possible queue tail; no micro-tail coupling plan is activated.
