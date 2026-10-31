# S-C1SWITCH — prospective Paper-B design (2026-10-31)
1. Question: can observable-belief, 60 s world-model controller selection beat frozen BB-traj and fixed controllers in real SUMO?
2. Approval: REGISTRATION owner 17:14; HARD caps 500,000 actual sim-s and 14,400 GPU-seconds (4 GPU-hours); no retraining.
3. Library/order: native, CTRL0 (zero.py 24095a10; C1 64f70af4), speed-only MAPPO iter169 d9042cc5, PI external-U, CIRCLES damping.
4. All library arms use the frozen E3 executor b3c031ad, native lanes, textbook population and identical command bounds/safety modes.
5. Nine paired arms: WM-S, frozen BB-traj afaaa32c, uniform random (seed 0 keyed by world/decision), greedy SUMO-oracle, and all five fixed controllers.
6. BB-id cannot execute feedback controllers and is excluded/disclosed; BB-traj receives the identical belief, controller plant and entry contract.
7. Every arm: 300 s native warmup; control t=300..900; ten decisions at 300+60j, j=0..9; switch all current/entering CAVs.
8. WM/BB: five feedback rollouts, K=8 paired futures/controller, horizon60; minimize median fleet+source delay (veh-s).
9. Tie: costs within 1e-6 veh-s of minimum; retain incumbent if tied, else first in fixed library order; incumbent initially native.
10. On change release_all(live_ids) through existing CommandOwnership, clear controller-specific state; same controller retains state; no extra simulation tick.
11. Predictor input is a strict allowlist: CAV self reports, RSU detections/neighbour receipts, timestamps, coverage, observed request/entry counters only.
12. No env.state/global truth is serialized to inference: no HDV type/settings, true penetration/demand, request schedule, unseen vehicles, or future presence.
13. Unknown space/source mass remains uncertain; entry/queue estimates use observable counters and population priors, never simulator request ledger.
14. Exact truth metrics use BranchTracker/RequestLedger, separately from predictor input; road integral sum .5*max(0,1-v/31.29) plus pending-source waiting.
15. Predicted team delay must include sampled current fleet, exits, sampled arrivals/admission and uninserted-source wait with the same units/clock.
16. C1 eligibility requires an executable source/entry adapter; road-only C1 prediction is ineligible. WM2+learned mix uses sampled entry/queue ledger, not true schedules.
17. Model rule frozen now: use C1PIDIAG original fixed PI+wall panel, genuine longest available history, inferred settings, original components, teacher-forced60.
18. Among C1 struct87ed02e0 and WM2b8797242+mix with verified team+entry support, choose lowest equal-world common-HDV absolute endpoint error; tie -> C1.
19. Require both cohorts and identical scored starts per candidate; unavailable diagnosis/entry support -> BLOCK before switching outcomes. No tuning on Step0 or main outcomes.
20. Publish MODEL_SELECTION.json with diagnosis/checkpoint/mix/adapter hashes and scores BEFORE any WM/BB/random/oracle switching episode.
21. Exposure: C1_GAPS.md C1-EXPOSURE: C1+BB native/trajectory/damping seen; PI/MAPPO absent. WM2 native/random/own-policy seen, PI/MAPPO/CIRCLES absent in audited corpus.
22. Direct command channels are seen; incomplete initialization-lineage exclusion remains disclosed; new-world exclusion audited independently of controller exposure.
23. Fresh proposed TRAIN block969100..969135: Step0 first12 (one per E3 cell); main last24; exact guards+registration interval+safe filename audits before allocation.
24. Cells ordered penetration(.05,.1,.2,.4) outer, demand(1900,2400,2700) inner; Step0 seed969100+i -> cell i.
25. Main969112..969135: all12 cells once, then four high cells(.2/.4 x2400/2700) three extra times each; high cells n4, others n1.
26. Disjoint from all registered experiment blocks (umbrella C4_LOOP_TRAIN excluded), WMDATA/WMDATA2, SUMOCF, calibration/validation; reject conflicts, never silently replace.
27. Step0a: recorded WMDATA40 and WMDATA2 50 worlds, common five arms only, cumulative120/300/600, per-world winners/ties/margins and cell win shares; zero new SUMO.
28. Step0b: 12 native trunks, ten states each, five90s branches/state, exact60/90 metrics; save native branch at60 and reuse it as next native trunk.
29. Step0b budget: warmups12*300=3,600; branches12*10*5*90=54,000; native repeat at first decision/world12*90=1,080.
30. Native restore canary on first Step0 world: live90+restored90 plus warmup300=480; prefix will be rerun once for the registered panel; no implicit free prefix.
31. Step0 TOTAL59,160 <=60,000 sim-s; repeated native branches test replay noise, not an independent stochastic divergence floor.
32. Gate: same unique controller >=95% of all120 states at60 AND90, or <5% states with noise-resolved non-dominant wins -> STOP main, report headroom absent/unresolved.
33. Resolve margins using max(1e-6,2*maximum same-state native-repeat absolute delay difference); bitwise canary failure -> STOP immediately, no claimed oracle/noise floor.
34. Step0 report WINNERS.md+CSV+winner-by-condition PNG/PDF; describe observed near-merge density/speed/queue/CAV count and observed CAV fraction, never true penetration as policy input.
35. Main WM:24*300+24*10*5*90=115,200 sim-s, includes all exact-measurement branches; BB-traj same115,200.
36. Main oracle:24*300+24*10*5*60=79,200; take true minimum60 (greedy clairvoyant reference, not a proven episode-optimal upper bound).
37. Main random:24*900=21,600; fixed native21,600; CTRL0 21,600; MAPPO21,600; PI21,600; CIRCLES21,600.
38. Main TOTAL439,200; Step0+main498,360; unallocated cap reserve1,640 sim-s requires prospective amendment before use, no automatic retries.
39. Budget relies on selected60s branch state becoming trunk, with Python ownership/controller/belief/metric state restored exactly; separately advancing trunk would exceed cap and is forbidden.
40. WM/BB branches run to90, retaining all five60s snapshots; oracle to60. Measurement branches never feed predictor inputs or controller selection except oracle.
41. GPU cap WM10,800s (3h), BB3,600s(1h), including loading/canary; Step0/fixed/random/oracle0; stop before cap, no partial panel claims.
42. Hosts: gpuR CPUs only, one nice19 single-thread worker, CUDA hidden; leave priority to SUMOCF/COOPMEASURE. Inference gpuA/gpuR only after free-memory admission.
43. SUMO pauses at each decision; atomic hash-bound file request/result handshake; separate GPU worker, finite(5,8) cost matrix, matching request/checkpoint/adapter hashes, timeout -> STOP.
44. Primary: per-decision60s regret (chosen true cost minus min five), WM minus BB and minus best-fixed controller on that arm's same decision states.
45. Best fixed primary: controller minimizing equal-world J600 over five fixed episodes; recompute identity inside each bootstrap draw; also report all-five contrasts.
46. Pair WM and BB by world/decision index (different on-policy states disclosed); fixed comparison evaluated by branching at the switching state, not by subtracting unrelated trajectories.
47. Statistics: equal-world mean of ten regrets; 2,000 paired world-cluster bootstrap draws, numpy seed0, percentile95%CI; exploratory two primary contrasts, multiplicity disclosed.
48. Secondary: paired episode J600 vs each fixed and per-world best-fixed-in-hindsight; oracle/random;60/90 direction vs native (ties separate), Spearman midranks.
49. Safety: collisions/teleports, TTC<3 vehicle-seconds, hard braking; switch count incl/native entries; per-cell means, counts/CIs; single-training checkpoint uncertainty disclosed.
50. Guards: never seeds964000..964199/962xxx/977xxx, forbidden paths/truth/assignment inputs; no new venvs/deletions/push; own PIDs only.
51. Mechanical stop: restore mismatch, duplicate IDs/requests, controller lane call, nonfinite output, missing source contract, collision/teleport/drop, ledger/cap/hash/pairing error.
52. Durable exclusive dispatch lock, fsynced precharge before every prefix/branch, actual half-second accounting incl failures; existing START without terminal result blocks redispatch.
53. Resume: inspect task receipts/PIDs; reuse finished analyses/jobs without recomputation; register changes before compute; commits after design/build/analysis/canary/results.
