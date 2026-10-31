# S-SELMAPPO — fleet PPO controller picker, protocol v2 (prospective)
1. Authority: MAIN23:46 supersedes the original two-variant/60s design; fleet PPO ONLY here; per-CAV MAPPO belongs to S-SELMAPPO2.
2. MAIN23:42 owner advance approval stands for the exact registered budget below; register SHA and budget before any SUMO; no approval polling is required after registration.
3. Match S-C1SWITCH2: native warmup300s + control600s, exactly30 choices at t=300+20j,j=0..29; hold each choice20s, terminate900s; no extra decisions/dwell rules.
4. Library/order: native, CTRL0, speed-MAPPO, PI external-U, CIRCLES damping, WALL; one RSU categorical action applies to all current and entering CAVs.
5. CTRL0 zero.py SHA24095a1092eafb4c894804aa8142478a802eaea046142ec777e5f24f49f73596/policy64f70af4; speed-MAPPO iter169 d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca.
6. Reuse frozen E3 b3c031ad plant and C1SWITCH wrappers; WALL must come from S-WALLSW's start/stop adapter, bound by source SHA and a published HANDOFF; never substitute another law.
7. Hand-over releases outgoing ownership and resets outgoing controller memory only on a change; incumbent state persists; no added physics tick; WALL assembly/release stays inside WALL.
8. Share tools/selmappo/{environment,observations,protocol}.py and HANDOFF.md with S-SELMAPPO2; common implementation is not a second training allocation here.
9. Actor/critic receive only allowlisted CAV self reports, detected neighbours, anonymous RSU merge/ramp receipts, coverage/age and observed entrance counters; never hidden types/parameters/true penetration/demand/global truth.
10. Features: permutation-invariant telemetry summaries, normalized clock, incumbent6-way one-hot; IDs route messages only, never numerical features or stochastic keys; missing channels have masks.
11. Missing entrance sensing stays masked, never replaced by RequestLedger/backlog truth; training reward/metrics are a separate channel, inaccessible to the observation builder.
12. Reward each20s = negative (on-road speed-deficit vehicle-seconds + source-wait vehicle-seconds), all physics contributions included; no safety/throughput/fairness shaping.
13. Gamma=1 and GAE lambda=1 per20s, finite30-step episode with terminal value0: exact undiscounted600s team-delay return, no bootstrap beyond900s.
14. Reuse existing wms.control.mappo.buffer compute_gae/ReturnScaler and ppo PPOConfig/clipped_surrogate/clipped_value_loss/sample_normalized; fresh categorical actor and observable-only critic.
15. Actor/critic 2x128 tanh; Adam actor3e-4/critic1e-3, clip.2/valueclip.2, epochs4, minibatch256frames, entropy.01,value.5,gradnorm.5,targetKL.03; running return scaling; fixed optimization RNG0.
16. Frozen speed-MAPPO implements a library action only; no selector warm start; single training replicate, no sweeps/checkpoint cherry-picking.
17. Fresh TRAIN18000000..18007999: first7774 unique production seeds18000000..18007773; smoke18007998/18007999; remaining224 unconsumed/unallocated.
18. Audit every registered interval incl umbrellas, all task-specific guards, and pruned safe filenames on cpuQ/gpuR before allocation; reject collision, never silently replace seeds.
19. Never964000..964199/962xxx/977xxx; disjoint from ALL registration blocks incl old C1SWITCH969100..969135 and the new C1SWITCH2 evaluation panel.
20. Production169updates x46episodes x900s =6,996,600sim-s (warmup2,332,200 +controlled4,664,400); smoke2x900=1,800sim-s, mechanics only/no gradients.
21. Evaluation16x900=14,400sim-s on the exact NEW S-C1SWITCH2 panel, loaded only from its registered hash-bound seed/cell manifest; do not reuse the old60s panel by assumption.
22. Exact fleet cap7,012,800sim-s =6,996,600production +1,800smoke +14,400evaluation; GPU0; no per-CAV budget is registered by this task.
23. Every warmup, partial/failure and smoke charged; precharge full episode, retain actual and reserved; no retries/replacement worlds/top-up/refunds; no additional comparator simulations funded here.
24. Training cells cycle all12 penetration(.05,.1,.2,.4) outer x demand(1900,2400,2700) inner with seeded per-update permutation; cell metadata never actor/critic input.
25. CPU hosts cpuQ/gpuR spare CPUs only, nice19/single-thread/CUDAhidden; one mechanics worker first, then throughput/admission-gated <=12workers, PPO<=4threads; never signal peers.
26. Throughput plan: time both smoke episodes and first46 production episodes retained in training; separately record warmup/control, controller/observation/PPO and total wall, sim-s/hour and completion projection.
27. cpuQ plain-episode reference ~1Msim-s/hour => ~7h planning baseline; publish measured six-controller performance, do not treat reference as a guarantee.
28. Stop at169updates, cap,24h total training wall, cooperative stop, nonfinite/clock/ownership/hash/ledger fault, collision or teleport; preserve completed work; no outcome-based early stopping.
29. Evaluate deterministic argmax (lowest library index for ties) ONLY final169 checkpoint, actor/scaler frozen before evaluation; no gradients or normalization updates during panel access.
30. Primary paired J600: WM-picker minus fleetPPO and fleetPPO minus bestFixed; same cadence/library/hand-over/observables/objective/new16worlds as S-C1SWITCH2.
31. Compare all6 fixed arms, WM and BB switching using complete hash-bound peer receipts; await missing peers rather than run unbudgeted comparators; per-CAV MAPPO is secondary from S-SELMAPPO2.
32. Equal-world paired means;2000 whole-world bootstrap draws,numpy seed0,percentile95%CIs; reselect best fixed by mean over6 inside each draw; preserve all-arm pairing.
33. Report road/source decomposition, switches, WALL exposure, collision/teleport,TTC<3vehicle-s and hard-braking; safety is not reward shaping; disclose16-world power/exploratory multiplicity.
34. Disclose one training replicate: evaluation world-bootstrap does not capture training-seed uncertainty; no claim that desired WM superiority is established in advance.
35. Fairness: fleet selector training6,996,600 +smoke1,800; eval14,400. Itemize WM/BB pretraining and inference and inherited speed-MAPPO training (~6,996,600successful +fail/smoke); no equal-budget claim.
36. Account common library pretraining once in aggregate; retain per-method simulation ledgers, source/checkpoint hashes, CPU-hours and GPU-seconds; disclose any WM/BB prediction-vs-full-team objective limitation.
37. Launch requires registered design/budget, zero-conflict exposure audit, CPU checks, frozen six-controller/observable integration receipt and resource admission; owner budget approval does not bypass missing WALL integration.
38. Never prohibited/truth/assignment reads, peer PID signals, new venvs, deletions, reset/clean/stash-drop or push; commit each completed step; preserve prior code/receipts.
