# S-C1SWITCH — prospective Paper-B design (2026-10-31)
1. Question: can observable-belief, 60 s world-model controller selection beat frozen BB-traj and fixed controllers in real SUMO?
2. Approval: REGISTRATION owner 17:14; HARD caps 500,000 actual sim-s and 14,400 GPU-seconds (4 GPU-hours); no retraining.
3. Library/order: native, CTRL0 (zero.py 24095a10; C1 64f70af4), speed-only MAPPO iter169 d9042cc5, PI external-U, CIRCLES damping.
4. All library arms use the frozen E3 executor b3c031ad, native lanes, textbook population and identical command bounds/safety modes.
5. MAIN18:33: eight paired arms: WM-S, frozen BB-traj afaaa32c, uniform random (seed 0 keyed by world/decision), and all five fixed controllers; no oracle episode arm.
6. BB-id cannot execute feedback controllers and is excluded/disclosed; BB-traj receives the identical observable belief and feedback controller plant; per-decision oracle is best of five branches.
7. Every arm: 300 s native warmup; control t=300..900; ten decisions at 300+60j, j=0..9; switch all current/entering CAVs.
8. MAIN18:33: WM/BB five feedback rollouts, K=8 paired futures/controller, horizon60; minimize median predicted ON-ROAD fleet delay (veh-s), both models same objective.
9. Tie: costs within 1e-6 veh-s of minimum; retain incumbent if tied, else first in fixed library order; incumbent initially native.
10. On change release_all(live_ids) through existing CommandOwnership, clear controller-specific state; same controller retains state; no extra simulation tick.
11. Predictor input is a strict allowlist: CAV self reports, anonymous RSU/neighbour receipts, timestamps, coverage, observed request/entry counters only; remove HDV simulator IDs BEFORE RPC and stochastic-key hashing.
12. No env.state/global truth is serialized to inference: no HDV type/settings, true penetration/demand, request schedule, unseen vehicles, or future presence.
13. Unknown space/source mass remains uncertain; entry/queue estimates use observable counters and population priors, never simulator request ledger.
14. Exact truth metrics use BranchTracker/RequestLedger, separately from predictor input; road integral sum .5*max(0,1-v/31.29) plus pending-source waiting.
15. Evaluation remains true FULL team delay (fleet+source); predictions have no source/entrant adapter and this objective mismatch is disclosed, never represented as predicted team delay.
16. Both prediction arms integrate on-road delay over modeled present vehicles, respecting predicted exits; no true future arrivals/presence or request ledger supplied to either model.
17. MAIN18:33 fixes WM2 b8797242+learned mix, full60s observable history, inferred settings, before any switching outcomes; supersedes the earlier team-support eligibility rule.
18. C1PIDIAG diagnosis: observable history helps WM2 on unseen PI; original C1/C1FAST is seen-world; no model selection or tuning using Step0/main outcomes.
19. BB frozen afaaa32c uses the same 60s history available to its adapter; architecture limitations disclosed; no black-box retraining.
20. Publish MODEL_SELECTION.json with diagnosis/checkpoint/mix/adapter hashes and MAIN rule BEFORE any switching episode; executable feedback/observable-only inference gate still required.
21. Exposure: C1_GAPS.md C1-EXPOSURE: C1+BB native/trajectory/damping seen; PI/MAPPO absent. WM2 native/random/own-policy seen, PI/MAPPO/CIRCLES absent in audited corpus.
22. Direct command channels are seen; incomplete initialization-lineage exclusion remains disclosed; new-world exclusion audited independently of controller exposure.
23. TRAIN block969100..969135 previously audited/reserved: exclude first12 for Step0, main969112..969127 only;969128..969135 unused, no new allocation.
24. Cells ordered penetration(.05,.1,.2,.4) outer, demand(1900,2400,2700) inner; Step0 seed969100+i -> cell i.
25. Main969112..969127: all12 cells once, then four high cells(.2/.4 x2400/2700) once extra; high cells n2, others n1. Same worlds/subsets across arms.
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
36. Exact per-model main:16*900=14,400 episode + sum_world sum_sampled_j 5*(300+60j+90)=158,100 branches =172,500 sim-s EACH WM/BB; no main native repeats.
37. Main random14,400; fixed native/CTRL0/MAPPO/PI/CIRCLES14,400 EACH; oracle episode0. Main431,400 + cumulativeStep0 47,010 =478,410 sim-s; unallocated reserve21,590.
38. All five main candidates must match LIVE episode fingerprint before alternative ticks; Step0 repeat gates establish native reproducibility. Main failures counted with no automatic retries.
39. MAIN18:33 authorizes cpuQ Step0 now and this16-world protocol; revised design SHA+exact per-arm registration required before dispatch; main also requires Step0 headroom and validated inference.
40. Exact60/90 costs are increments from the replayed decision state; measurement outcomes never enter WM/BB input/pick. Any prefix/repeat bitwise mismatch -> durable STOP, no oracle/headroom claim.
41. GPU cap WM10,800s (3h), BB3,600s(1h), including loading/canary; Step0/fixed/random/oracle0; stop before cap, no partial panel claims.
42. Hosts MAIN18:33: cpuQ CPUs <=28 workers, nice19/single-thread/CUDA hidden (initial Step0 one worker); leave gpuR CPUs to SUMOCF. Inference gpuA/gpuR after free-memory admission.
43. SUMO pauses at each decision; atomic hash-bound file request/result handshake; separate GPU worker, finite(5,8) cost matrix, matching request/checkpoint/adapter hashes, timeout -> STOP.
44. Primary: per-decision60s regret (chosen true cost minus min five), WM minus BB and minus best-fixed controller on that arm's same decision states.
45. Best fixed primary: controller minimizing equal-world J600 over five fixed episodes; recompute identity inside each bootstrap draw; also report all-five contrasts.
46. Pair WM and BB by world/decision index (different on-policy states disclosed); fixed comparison evaluated by branching at the switching state, not by subtracting unrelated trajectories.
47. Statistics: equal-world mean over three registered measured decisions; 2,000 paired world-cluster bootstrap draws, numpy seed0, percentile95%CI; two exploratory primary contrasts, multiplicity disclosed.
48. Secondary: paired episode J600 vs each fixed and per-world best-fixed-in-hindsight; random;60/90 direction vs native (ties separate), Spearman midranks; per-decision oracle only.
49. Safety: collisions/teleports, TTC<3 vehicle-seconds, hard braking; switch count incl/native entries; per-cell means, counts/CIs; single-training checkpoint uncertainty disclosed.
50. Guards: never seeds964000..964199/962xxx/977xxx, forbidden paths/truth/assignment inputs; no new venvs/deletions/push; own PIDs only.
51. Mechanical stop: full-rerun prefix/repeat mismatch, duplicate IDs/requests, controller lane call, nonfinite output, invalid observable inference contract, collision/teleport/drop, ledger/cap/hash/pairing error.
52. Durable exclusive dispatch lock, fsynced precharge before every prefix/branch, actual half-second accounting incl failures; existing START without terminal result blocks redispatch.
53. Resume: inspect task receipts/PIDs; reuse finished analyses/jobs without recomputation; register changes before compute; commits after design/build/analysis/canary/results.
54. MAIN19:12 integration: Step0 PASS47,010; real WM2/BB CPU gate and cross-host hash-bound relay; one-world canary is retained main969112 WM+BB episodes/registered measurements, no extra sim-s.
55. Causal PI raw recurrence/MAPPO command memory uses self IDs only; reset on change, preserve incumbent; shared controller code+gap6 ramp approximates SUMO lanes. PI/CIRCLES retain WM-supplied native safety envelope when present, inferred-head fallback otherwise (BB); no exact/unified native-plant forecast claim.
56. Entries OFF: WM2 learned mix configured but dormant; K8 shares controller-paired draws where supported, BB repeats deterministic forecasts where no stochastic head exists. GPU services2.5GiB allocator EACH (5GiB combined), admission>=7GiB free; cpuQ14episode workers+<=14branch children, nice19; schedule BB first to honor its tighter cumulative wall cap.
57. v4 real canary COMPLETE:71910cumulative charged=actual,2episodes/30prefix gates/60metric clocks PASS,0collisions/teleports. v5 exact speedups CPU sample parity PASS; full60s original-draw GPU parity required. Retain every sim-s; GPU prior debitceil(elapsed)+2 =WM1902/BB1342s, remaining8898/2258s; no refunds/repeated episodes.
58. End-to-end ID audit: v4 tensors were anonymous but its CRN key hashed detector IDs. v5 hashes only anonymous observable payload; CPU whole-prediction ID noninterference required. Revalidate all20canary picks under corrected draws before retention: unchanged schedules -> update forecast metadata with old receipts preserved; ANY changed pick -> STOP main. Original-draw GPU parity is diagnostic only, never a live pick seed override.
