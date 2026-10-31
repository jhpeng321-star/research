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
23. TRAIN block969100..969135: reserve/exclude first12 for Step0, main last24; existing exact guards+registry/corpus/safe-name audits passed; no new allocation.
24. Cells ordered penetration(.05,.1,.2,.4) outer, demand(1900,2400,2700) inner; Step0 seed969100+i -> cell i.
25. Main969112..969135: all12 cells once, then four high cells(.2/.4 x2400/2700) three extra times each; high cells n4, others n1; dispatch BLOCKED by budget feasibility.
26. Disjoint from all registered experiment blocks (umbrella C4_LOOP_TRAIN excluded), WMDATA/WMDATA2, SUMOCF, calibration/validation; reject conflicts, never silently replace.
27. Step0a: recorded WMDATA40 and WMDATA2 50 worlds, common five arms only, cumulative120/300/600, per-world winners/ties/margins and cell win shares; zero new SUMO.
28. MAIN17:53 supersedes ALL saved-state continuations: each alternative starts a fresh SUMO at t0, same seed, replayed controller schedule to the decision, then alternative; never loadState.
29. Step0b v2: eight unused TRAIN seeds969101/102/103/105/107/108/110/111; native trunks; decision times300/480; five90s branches plus one native90 repeat per state.
30. Step0 retained cost930 (480old canary+450failed attempt); new five-controller full reruns8*5*((300+90)+(480+90))=38,400; native repeats8*((300+90)+(480+90))=7,680.
31. Step0 cumulative47,010 <=60,000; all warmups/trunks/repeats/failures counted; GPU0. Exclusive new version/attempt, legacy STOP/ledger/source retained, imported930 debit, no refunds/retries.
32. First native pair proves bitwise complete warmup+trunk tape, observable belief, counters/metrics and native90 continuation BEFORE any nonnative continuation; then match every candidate prefix before select.
33. Gate: same unique controller >=95% of16states at60 AND90, or <5% states with noise-resolved non-dominant wins -> STOP main; margin threshold max(1e-6,2*max native-repeat delay difference).
34. Step0 report WINNERS.md+CSV+winner-by-condition PNG/PDF; describe observed near-merge density/speed/queue/CAV count and observed CAV fraction, never true penetration as policy input.
35. Prospective main measurement subset: sorted random.Random(world_seed).sample(range(10),3), fixed now and shared WM/BB; outcomes cannot change subset. All ten policy decisions retained.
36. Full-rerun candidate budgets: WM21,600+235,800=257,400; BB257,400; oracle24*5*sum(360+60j,j=0..9)=756,000; oracle resumes live trunk after selection, no continuation reuse.
37. Main random21,600; fixed native/CTRL0/MAPPO/PI/CIRCLES21,600 EACH. Main total1,400,400; with Step0=1,447,410 >500,000: NOT AUTHORISED; no main dispatch.
38. Even optimal live-trunk-as-one-oracle-candidate costs24*(900+4*sum(360+60j))=626,400 oracle-only >entire cap; measurement subsampling cannot solve this conflict.
39. MAIN must reduce oracle decision/world count or increase cap and re-register exact per-arm plan BEFORE main; do not silently relabel a sparse oracle as the ten-decision arm.
40. Exact60/90 costs are increments from the replayed decision state; measurement outcomes never enter WM/BB input/pick. Any prefix/repeat bitwise mismatch -> durable STOP, no oracle/headroom claim.
41. GPU cap WM10,800s (3h), BB3,600s(1h), including loading/canary; Step0/fixed/random/oracle0; stop before cap, no partial panel claims.
42. Hosts: gpuR CPUs only, one nice19 single-thread worker, CUDA hidden; leave priority to SUMOCF/COOPMEASURE. Inference gpuA/gpuR only after free-memory admission.
43. SUMO pauses at each decision; atomic hash-bound file request/result handshake; separate GPU worker, finite(5,8) cost matrix, matching request/checkpoint/adapter hashes, timeout -> STOP.
44. Primary: per-decision60s regret (chosen true cost minus min five), WM minus BB and minus best-fixed controller on that arm's same decision states.
45. Best fixed primary: controller minimizing equal-world J600 over five fixed episodes; recompute identity inside each bootstrap draw; also report all-five contrasts.
46. Pair WM and BB by world/decision index (different on-policy states disclosed); fixed comparison evaluated by branching at the switching state, not by subtracting unrelated trajectories.
47. Statistics: equal-world mean over three registered measured decisions; 2,000 paired world-cluster bootstrap draws, numpy seed0, percentile95%CI; two exploratory primary contrasts, multiplicity disclosed.
48. Secondary: paired episode J600 vs each fixed and per-world best-fixed-in-hindsight; oracle/random;60/90 direction vs native (ties separate), Spearman midranks.
49. Safety: collisions/teleports, TTC<3 vehicle-seconds, hard braking; switch count incl/native entries; per-cell means, counts/CIs; single-training checkpoint uncertainty disclosed.
50. Guards: never seeds964000..964199/962xxx/977xxx, forbidden paths/truth/assignment inputs; no new venvs/deletions/push; own PIDs only.
51. Mechanical stop: full-rerun prefix/repeat mismatch, duplicate IDs/requests, controller lane call, nonfinite output, missing source contract, collision/teleport/drop, ledger/cap/hash/pairing error.
52. Durable exclusive dispatch lock, fsynced precharge before every prefix/branch, actual half-second accounting incl failures; existing START without terminal result blocks redispatch.
53. Resume: inspect task receipts/PIDs; reuse finished analyses/jobs without recomputation; register changes before compute; commits after design/build/analysis/canary/results.
