# S-SELMAPPO — prospective controller-selection baselines (2026-10-31)
1. Authority: OWNER ~23:30 / MAIN23:28; design/build/CPU only until MAIN appends `S-SELMAPPO BUDGET APPROVED` to the shared registration.
2. Compare fleet PPO (one RSU action for every current/entering CAV) and parameter-shared per-CAV MAPPO (one categorical action per CAV).
3. Library/order: native; CTRL0 zero.py SHA24095a1092eafb4c894804aa8142478a802eaea046142ec777e5f24f49f73596 / policy64f70af4; speed-MAPPO iter169 d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca; PI external-U; CIRCLES damping.
4. Use C1SWITCH frozen E3 b3c031ad runtime/wrappers, textbook population, native lanes, existing command bounds/safety modes; no new behavioural thresholds.
5. Every episode is fresh: native warmup300s + control600s; ten categorical decisions300+60j,j=0..9; no saved-state continuation.
6. Fleet hand-over is C1SWITCH Handover: release_all(live_ids), clear controller-specific state on change, preserve incumbent, no extra physics tick.
7. Per-CAV uses the same release/reset/preserve rule restricted to the changed CAV; independent command-owner/controller contexts prevent one CAV resetting another.
8. Entrants in fleet mode inherit the fleet action immediately; per-CAV entrants stay native until the next grid decision; departures retire memory, re-entries start native.
9. Inputs ONLY CAV self reports, detected neighbours, anonymous RSU merge/ramp receipts, coverage/age and observed entrance counters; no environment state reference in the selector.
10. Actor features are permutation-invariant sensor summaries plus normalized clock, previous controller one-hot and, per-CAV, self telemetry; IDs route actions but are never numeric features.
11. Central critic receives only the shared observable summary, training only; no hidden types/parameters, true penetration/demand, unseen vehicles, request schedules or global truth.
12. Missing entrance sensing has an explicit mask; never substitute RequestLedger/backlog truth. Team reward/metrics are a separate training/evaluation channel, never actor input.
13. Window reward = -(speed_deficit_vehicle_seconds + source_wait_vehicle_seconds), accumulated over all60 one-second ticks; no throughput/fairness/safety shaping.
14. Gamma=1.0 per60s and GAE lambda=1.0: finite10-step Monte-Carlo return covers the entire600s objective exactly; terminal value0, no post900s bootstrap.
15. All acting CAVs receive the shared team advantage, with one central value loss per decision; empty-CAV decisions still train the critic.
16. Reuse wms.control.mappo.buffer compute_gae/ReturnScaler and ppo clipped_surrogate/clipped_value_loss/sample_normalized/PPOConfig; new categorical head and observable critic.
17. Fresh selector actor/critic/Adam; no acceleration-policy warm start. The frozen speed-MAPPO is an action implementation, not selector initialization.
18. Fixed optimizer: actor3e-4,critic1e-3,clip.2,value clip.2,epochs4,minibatch256frames,entropy.01,value.5,grad norm.5,target KL.03; return normalization from existing stack.
19. Fresh TRAIN blocks: fleet18000000..18007999; per-CAV18100000..18107999; audit all registration intervals and pruned safe metadata; task-scoped guards require approval, exact seed/cell/role.
20. Disjoint from ALL registered blocks including umbrella blocks and C1SWITCH969100..969135; never964000..964199/962xxx/977xxx. Reject conflicts, never silently replace seeds.
21. Per variant: smoke seeds block_start+7998/+7999; production first7774 unique seeds, no reuse; remaining224 seeds unconsumed/unallocated.
22. Production169 iterations x46 episodes x900s =6,996,600sim-s EACH variant (4,664,400controlled +2,332,200warmup).
23. Mechanics smoke2 x900s=1,800sim-s EACH; deliberately exercises switches/entries, training seeds only, never contributes gradients or checkpoint selection.
24. Frozen final-checkpoint evaluation16 x900s=14,400sim-s EACH on C1SWITCH panel; no new fixed/WM/BB simulations funded here.
25. Exact requested per-variant cap7,012,800sim-s; combined14,025,600sim-s; all warmup/smoke/failures charged, zero retry/top-up reserve, GPU0.
26. Throughput measurement after approval: smoke then first46 production episodes retained in training; separately time startup, simulation/controller inference, observations, PPO and total wall.
27. cpuQ plain-episode reference ~1Msim-s/hour is only a planning baseline (~7h/variant); mixed per-CAV controller inference may be slower; publish measured sim-s/hour and projected completion.
28. CPU hosts cpuQ / gpuR spare CPUs only, nice19/single-thread workers/CUDA hidden; initially1 canary, then <=12 workers, yield admission to existing work; update<=4 threads.
29. Stop at169updates, exact sim cap,24h elapsed per variant, nonfinite/clock/slot/ownership/hash/ledger fault or collision/teleport; retain partial checkpoint, do not report partial efficacy.
30. No reward-based early stopping, sweeps or evaluation-based checkpoint choice; evaluate only frozen final169 checkpoint. Any failed simulation stops dispatch; retain actual and reserved charge.
31. Training cells: cycle all12 combinations penetration(.05,.1,.2,.4) outer x demand(1900,2400,2700) inner, seeded permutation per iteration; metadata never policy input.
32. Optimization RNG fleet0/per-CAV1 fixed; one training replicate each, disclose training-seed uncertainty is not captured by world bootstrap.
33. Evaluation seeds969112..969127: all12 cells once then(.2,2400),(.2,2700),(.4,2400),(.4,2700); identical ordering, seeds, warmup, executor and full-team objective as C1SWITCH.
34. Deterministic categorical argmax, lowest library index breaks ties; freeze actor/normalization before panel access; no gradients/normalization updates on evaluation.
35. Pair J600(road+source) against all5 fixed controllers, frozen WM switching and BB switching; use their complete hash-bound episode receipts only, await missing comparators rather than rerun or fabricate.
36. Primary fleetPPO-WM and fleetPPO-bestFixed; per-CAV contrasts secondary (richer action space). Best fixed minimizes equal-world mean over5 arms; reselect identity in each bootstrap draw.
37. Equal-world mean paired differences;2000 whole-world bootstrap draws,numpy seed0,percentile95%CIs; preserve pairing across all arms, disclose16-world power and exploratory multiplicity.
38. Report all5 fixed contrasts, per-world hindsight oracle descriptive only, per-cell/counts, road/source decomposition, switches, entrant-native exposure, collision/teleport,TTC<3 vehicle-s, hard-braking.
39. Fleet controller-selection matches WM action/cadence; per-CAV has more expressivity and compute. Selector return uses full team delay whereas frozen WM/BB predict on-road60s: disclose this objective advantage.
40. Fairness: selector TRAIN6,996,600 +smoke1,800 EACH; eval14,400 EACH. Separately itemize WM/BB pretraining and inference, and frozen library controller's inherited speed-MAPPO training (~6,996,600successful +fail/smoke); no equal-budget claim.
41. Reuse common library pretraining only once in aggregate accounting; retain method-specific simulation ledgers, actual/reserved failures, CPU-hours, GPU-seconds and all checkpoint hashes.
42. Before any SUMO: register this file SHA256 +exact budgets, exposure audit, CPU tests and frozen source binding; then poll shared registration every600s for MAIN approval.
43. No prohibited path/truth/assignment reads, peer PID signals, new venvs, deletions, reset/clean/stash-drop or push; commits after design, build/tests, registration and results.
