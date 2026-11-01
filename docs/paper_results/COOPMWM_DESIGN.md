# S-COOPMWM — WM-guided exploration, SUMO-graded (2026-11-01 17:55)
1. Authority: MAIN16:22/25/38/46/17:07/12/27/42/55; native-start and WM-reward arms superseded.
2. A/B/R fork update0 from frozen speed-only MAPPO169 d9042cc5; native lanes.
3. Frozen MAPPO actor/memory executor plus trainable zero-mean residual adapter.
4. A: PPO shared team reward=-J, road speed deficit plus source waiting.
5. B: same SUMO-only reward/advantage, WM-guided exploration; no WM or behavioural bonus.
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
16. R=B except uniform AVAILABLE O1-O4 proposer, no WM/saving filter; same executors/W10 setpoints.
17. Observable replay roster only; fitted joint FIT driver population and uniform32FIT source prior.
18. Cold motion/controller memory and unobserved source prior disclosed; no realised future inputs.
19. Inherited option group=two upstream mainline CAVs within100m; ordinary windows also sampled.
20. u1 historical p=1/900 dense,1/3600 ordinary; u2+ fixed throughput-fit p,p/4 shared by B/R/seeds.
21. FULL-v4 K8 CRN H300 scores policy action/MAPPO/O1-O4; propose iff saving vs BOTH baselines>0.
22. Keep non-group chosen targets identical in all futures; after W10 release to MAPPO.
23. H60/120/300/600. D_i=J(member-revert)-J(joint); D_G=J(all-revert)-J(joint).
24. C_G=D_G-sum(D_i); complete paired futures required, including negative values.
25. beta starts1 per training seed; immutable update receipts supply the next update's beta.
26. Log query route mass p, within-query execute beta(type,regime); own nonquery mass1-p.
27. Target own-nonquery mass1/query0: ratio=pi_new/[(1-p)*pi_old] or0, including declined/rejected queries.
28. Clipped PPO respects zero WM-route mass; executed proposals get beta-weighted joint-Gaussian NLL.
29. Initial random subset: twelve TRAIN windows, RNG20261101, one per12cells.
30. Select uniformly from complete group windows in first TRAIN batch, before credit outcomes.
31. Select t<=600 so all H600 twins finish by1200; full-prefix COOPFREE quartet once/window.
32. Publish C_G and D_i sign agreement, whole-world10000bootstrap95CI plus Wilson, scale95CI.
33. E2 runs in a2worker diagnostic thread beside PPO; serialized accounting; never reward/training gate.
34. Zero/tie credits reported separately; scale fitted from exact data, diagnostic only.
35. TRAIN986000..986071/987000..987071/988000..988071, six worlds/cell per seed.
36. Check986072..986083, one/cell, independent TRAIN; never optimizer/selection input.
37. Every4updates checks A/B/R vs once-only MAPPO controls, J600/J900/safety; forced counts/type and SNR.
38. Dev984000..984023 at update4 and final; update0 uses existing24bitwise MAPPO bytes.
39. Dev u4 primary A/B/R plus u24 all five arm-seeds; paired world bootstrap10000/RNG20261101; no tuning.
40. Final detector-first discovery uses unchanged COOPBEHAV detectors and complete exact twins.
41. Learning checks report team return/value/KL/gradient, beta, proposal routes/acceptance/O1-O4 counts.
42. A/B seeds20261101/2, R primary20261101 only; seed3 DROPPED; fixed24updates x24episodes/arm-seed.
43. Cap6,000,000 TOTAL: train3,456,000; checks446,400; dev230,400; primaryfinal172,800; Rtest43,200.
44. Mechanics3,600; calibration240,000; discovery240,000; seed2test86,400; unallocated1,081,200.
45. Calibration/discovery reserve complete groups/prefixes before SUMO; insufficient phase=>STOP.
46. All failures/warmup/twins charged in full before attempts; no retries/refunds/duplicate dispatch.
47. Primary A/B parallel first; cpuQ12nice19singlethread workers+2E2; load<=20/RAM8GiB/disk2GiB.
48. Both GPUs one8GiB worker each, owner17:55 release; max10000proposals/host+12E2+parity/timing; nvidia-smi.
49. GPU CPU helpers disabled until separate frozen parity qualification; own PIDs only.
50. Both seed guards, fresh-name exposure, immutable sources/cache/approval bindings before physics.
51. No prohibited paths/truth/assignments, no newenv/deletion/push; commit every completed step.
52. E2 diagnostic multi-H weights normalize measured agreement on horizons with sign95CI lower>.5.
53. Fit p=min(.5,dualqueries/s*initial48episodewall*4/12/(dense+ordinary/4)); complete-group counts only.
54. Per-window/H report K8 joint-return advantage variance and paired-credit variance in veh-s².
55. Frozen critic baseline is constant within window; variance is conditional WM-estimated noise.
56. Report actual PPO advantages across TRAIN windows separately; no empirical SUMO-noise claim.
57. Same-group CRN removes shared hidden-driver/arrival variation only to extent model supports it.
58. No credit outcomes inspected before random subset manifest; failed/incomplete quartet=>STOP.
59. FINAL_TEST989600..989635:36fresh worlds,3/cell; metadata manifest hash registered before SUMO.
60. Precharge4x36x1200=172800; MAPPO169/native/Aseed0u24/Bseed0u24, no model selection.
61. Primary four-arm test ONCE after BOTH primaryu24; publish, then R train/test, then seed2A/B train/test.
62. Final paired world-bootstrap95CI B-MAPPO/B-A/B-R/A-MAPPO J600/J900; percell/safety; controls reused.
63. Final B detector-first exact C_G twins share240000 discovery reserve; complete chronological groups.
64. Trust: H300 SUMO return+critic tail-baseline; Hajek executed(type,regime) minus own(regime) residuals.
65. 10000 world bootstrap95CI[L,U]; e=max(|mean-L|,|U-mean|); beta=min(old,max(0,U)/(|mean|+e)).
66. Eight type x regime normal-means EB buckets: tau²=max(0,var(bucketmeans)-mean(bootstrapSE²)); weight=tau²/(tau²+SE²).
67. Test liabilities protected; cpuP unreachable/excluded; registry/protocol+cpuQ/gpuA/gpuR audits qualify.
68. Regime=LEVERPROBE ramp+crosslane pair+union of four recorded upper-quartile CAV/density bands; observable only.
69. World-bootstrap EB residual contrasts are selection-conditional, not causal twins; missing bucket=>pooled; no pooled evidence=>keep beta.
