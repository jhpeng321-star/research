# S-COOPMWM — MAPPO-origin cooperative PPO (speed amendment, 2026-11-01)
1. Authority: MAIN16:22/16:25; COOPNATIVE from-native superseded before physics.
2. Both arms fork update0 from frozen speed-only MAPPO169 d9042cc5; native lanes.
3. Frozen MAPPO actor/memory executor plus trainable zero-mean residual adapter.
4. A: PPO shared team reward=-J, road speed deficit plus source waiting.
5. B: identical PPO plus WM replay credit; no behavioural shaping or exact reward labels.
6. All live CAVs choose speed/log-timegap residuals on MAPPO command anchors, W10s.
7. Zero residual bypasses tracking for the whole window: bitwise MAPPO command parity.
8. Reuse COOPFREE pre-safety injection, unchanged safety and CommandOwnership.
9. Actor uses whitelisted local telemetry/neighbours; zero-init4D RSU broadcast.
10. Central critic pools CAV features plus12RSU zone observations; no hidden channels.
11. Shared exploration rho.5, sigma4.5m/s/.5log-s; exact correlated joint density.
12. Reuse PPO4epochs/Adam3e-4/clip.2/KL.045; gamma exp(-10/600),lambda exp(-10/1200).
13. Critic GAE e-folding400s; complete900s task after300s warmup, terminal0.
14. WM fixed FULL-v4 source22cc32f6; later qualified v5 requires a new registered arm.
15. Reuse WMTEACH scale.futures K8 batching and option_controller executor/safety.
16. Adapt candidate roster to explicit chosen setpoints; no option-library action restriction.
17. Observable replay roster only; fitted joint FIT driver population and uniform32FIT source prior.
18. Cold motion/controller memory and unobserved source prior disclosed; no realised future inputs.
19. RSU group=first two CAVs ordered by distance to merge within inherited[-100,75.44].
20. Fewer than2 => no group credit; no new behavioural thresholds or positive-only selection.
21. At each W10 decision with a group: K8 CRN joint, each member-revert, all-revert.
22. Keep non-group chosen targets identical in all futures; after W10 release to MAPPO.
23. Primary H90 only. D_i=J(member-revert)-J(joint); D_G=J(all-revert)-J(joint).
24. C_G=D_G-sum(D_i); complete paired futures required, including negative values.
25. Reward=-J + ramp(u)*scale*(sum(D_i)+.5*C_G); w_coop<=w_ind.
26. Ramp fixed min(1,(u-1)/8): zero at update1, full at update9; never retuned.
27. Scale fitted nonnegative through-origin pooled exact-vs-WM credits on calibration windows.
28. Exact calibration labels never enter policy/critic features or optimizer batches.
29. Initial random subset: twelve TRAIN windows, RNG20261101, one per12cells.
30. Select uniformly from complete group windows in first TRAIN batch, before credit outcomes.
31. Exact COOPFREE full-prefix twins, joint/member-revert/all-revert, H90.
32. Publish sign agreement for C_G and D_i, Wilson95CI, world-bootstrap scale95CI.
33. B stops if either sign agreement Wilson lower<=.5, or positive scale cannot be fitted.
34. Zero/tie credits reported separately; no claim from ties. A may continue.
35. Training reuses UNUSED registered986000..986071 from superseded design, six worlds/cell.
36. Check986072..986083, one/cell, independent TRAIN; never optimizer/selection input.
37. Every4updates checks A+B vs cached once MAPPO controls, J600/J900/safety.
38. Dev984000..984023 at update4 and final; update0 uses existing24bitwise MAPPO bytes.
39. Paired world bootstrap10000/RNG20261101 for A/MAPPO,B/MAPPO,B/A; no dev tuning.
40. Final detector-first discovery uses unchanged COOPBEHAV detectors and complete exact twins.
41. Learning checks report team return/value/KL/gradient, residual execution and credit distribution.
42. Fixed80updates/arm,24episodes/update; no outcome-based checkpoint/hyperparameter choice.
43. Exact budget6,000,000 TOTAL: train4,608,000; checks590,400; dev115,200.
44. Mechanics2,400(two episodes); calibration240,000; discovery240,000; unallocated204,000.
45. Calibration/discovery reserve complete groups/prefixes before SUMO; insufficient phase=>STOP.
46. All failures/warmup/twins charged in full before attempts; no retries/refunds/duplicate dispatch.
47. Initial cpuQ two nice19 singlethread CPU workers; admission load<=20/RAM8GiB/disk2GiB.
48. WM GPUA or GPUR strict8GiB; nvidia-smi and WMPLAN priority release before allocation.
49. GPU CPU helpers disabled until separate frozen parity qualification; own PIDs only.
50. Both seed guards, fresh-name exposure, immutable sources/cache/approval bindings before physics.
51. No prohibited paths/truth/assignments, no newenv/deletion/push; commit every completed step.
