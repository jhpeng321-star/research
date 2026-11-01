# S-COOPMWM — acceleration plans, SUMO-graded (OWNER19:44 quality-first)
1. Authority through MAIN19:44; native-start/WM-reward/all prior set-point runs superseded and charged.
2. A/B/R fork update0 from frozen speed-only MAPPO169 d9042cc5; native lanes.
3. Frozen MAPPO actor/memory executor plus trainable zero-mean residual adapter.
4. A: PPO shared team reward=-J, road speed deficit plus source waiting.
5. B: same SUMO-only reward/advantage, WM-guided exploration; no WM or behavioural bonus.
6. All live CAVs choose1s acceleration residuals added to MAPPO raw commands; no gap/speed action coordinates.
7. Exact zero abstains before unchanged safety: actual frozen MAPPO executor bitwise parity CPU-tested.
8. Reuse COOPFREE pre-safety injection, unchanged safety and CommandOwnership.
9. Actor uses whitelisted local telemetry/neighbours; zero-init4D RSU broadcast.
10. Central critic pools CAV features plus12RSU zone observations; no hidden channels.
11. AR1 phi=exp(-1/5), spatialshared rho.5, sigma.5m/s²; logged conditional joint density/previous noise.
12. Reuse PPO4epochs/Adam3e-4/clip.2/KL.045; gamma exp(-1/600),lambda exp(-1/1200).
13. Critic GAE e-folding400s; complete900s task after300s warmup, terminal0.
14. WM fixed FULL-v4 source22cc32f6; later qualified v5 requires a new registered arm.
15. Reuse WMTEACH futures/executors; fixed-width batches, full1600-slot forward/source/controller; compact path rejected.
16. R=B except random acceleration profiles from identical bounds/plan space, matched count/regime/duration; noWM.
17. Roster only; joint FIT driver prior/uniform32FIT source: integer round(rate*t) pseudo-counts/exposure t.
18. Cold memory/unobserved source pseudo-exposure disclosed; not causal request conditioning; no future inputs.
19. Inherited option group=two upstream mainline CAVs within100m; ordinary windows also sampled.
20. All A/B/R restart MAPPOu0 in isolated acceleration namespaces;811640superseded charges retained.
21. MediumCEM32population x3iterations/elite8; K4H120screen,top8K8H300rescore; seeds O1-O4 xallsettings+random.
22. Plans per-member bounded acceleration knots10s/duration10..60s; committed; sharedK8bootstrap Thompson beats BOTH baselines.
23. H60/120/300/600. D_i=J(member-revert)-J(joint); D_G=J(all-revert)-J(joint).
24. C_G=D_G-sum(D_i); complete paired futures required, including negative values.
25. beta starts1 per training seed; immutable update receipts supply the next update's beta.
26. Log query route mass p, within-query execute beta(type,regime); own nonquery mass1-p.
27. Target own-nonquery mass1/query0: ratio=pi_new/[(1-p)*pi_old] or0, including declined/rejected queries.
28. Per-step clipped PPO respects singular guided-route mass; executed member residual imitation weightbeta.
29. Original initial12cell E2 retained; acceleration E2 uses registered first12random executed TRAIN plans, RNG20261101.
30. Exact/WM E2 keeps all background CAVs on MAPPO after decision, group commitment then MAPPO; no native reversion.
31. Select t<=600 so all H600 twins finish by1200; full-prefix COOPFREE quartet once/window.
32. Publish C_G and D_i sign agreement, whole-world10000bootstrap95CI plus Wilson, scale95CI.
33. E2 runs in a2worker diagnostic thread beside PPO; serialized accounting; never reward/training gate.
34. Zero/tie credits reported separately; scale fitted from exact data, diagnostic only.
35. TRAIN986000..986071 primary seed20261101 only, six worlds/cell; original fresh guards/registrations retained.
36. Check986072..986083, one/cell, independent TRAIN; never optimizer/selection input.
37. Every4updates checks A/B/R vs once-only MAPPO controls, J600/J900/safety; forced counts/type and SNR.
38. Dev984000..984023 every8updates A/B/R; existing24bitwise MAPPO bytes reused; no MAPPO dev rerun.
39. TRAIN checks every4updates and dev every8 are reporting only; no checkpoint selection/tuning.
40. Final detector-first discovery uses unchanged COOPBEHAV detectors and complete exact twins.
41. Learning checks report team return/value/KL/gradient, beta, proposal routes/acceptance/O1-O4 counts.
42. Fixed24updates x24episodes x1200 per A/B/R; seed2 dropped under19:38 first-drop rule (would cost6239240).
43. Cap6000000: superseded811640; newtrain2073600; checks259200; dev259200; primaryfinal460800; Rtest115200.
44. Newmechanics3600; calibration57600; discovery240000; allocated4280840/free1719160 not automatically spent.
45. Calibration/discovery reserve complete groups/prefixes before SUMO; insufficient phase=>STOP.
46. All failures/warmup/twins charged in full before attempts; no retries/refunds/duplicate dispatch.
47. Primary A/B parallel first; cpuQ12nice19singlethread workers+2E2; load<=20/RAM8GiB/disk2GiB.
48. Both GPUs batched services <=65%memory each; measured capacity; once/worker single parity; original24h deadline.
49. GPU CPU helpers disabled until separate frozen parity qualification; own PIDs only.
50. Both seed guards, fresh-name exposure, immutable sources/cache/approval bindings before physics.
51. No prohibited paths/truth/assignments, no newenv/deletion/push; commit every completed step.
52. E2 diagnostic multi-H weights normalize measured agreement on horizons with sign95CI lower>.5.
53. Eligible windows pause for GPU queue slots, p=.95 when admitted/dense priority; numericalNexec/throughput before restart.
54. Per-window/H report K8 joint-return advantage variance and paired-credit variance in veh-s².
55. Frozen critic baseline is constant within window; variance is conditional WM-estimated noise.
56. Report actual PPO advantages across TRAIN windows separately; no empirical SUMO-noise claim.
57. Same-group CRN removes shared hidden-driver/arrival variation only to extent model supports it.
58. Gate counts closed executed PLAN STARTS; R matches B count/regime/duration; unmet quota=>STOP/no extraepisodes.
59. FINAL96=989600..989635 plus fresh990600..990659,8/cell; SHAd74f685ef75a819248bbe7bb0bef89cf0295037ff3b03b334b13ccec75c40bcc.
60. Precharge4x96x1200=460800; MAPPO169/native/Au24/Bu24; laterR96x1200=115200; no selection/reruns.
61. Primary four-arm sealed test ONCE after BOTH primaryu24; publish, then R train/test; controls reused.
62. Final paired world-bootstrap95CI B-MAPPO/B-A/B-R/A-MAPPO J600/J900; percell/safety; controls reused.
63. First12executedBstarts sampled RNG20261101 across completed updates + detector-first final twins share240000reserve.
64. Trust: H300 SUMO return+critic tail-baseline; Hajek executed(type,regime) minus own(regime) residuals.
65. 10000 world bootstrap95CI[L,U]; e=max(|mean-L|,|U-mean|); beta=min(old,max(0,U)/(|mean|+e)).
66. Eight type x regime normal-means EB buckets: tau²=max(0,var(bucketmeans)-mean(bootstrapSE²)); weight=tau²/(tau²+SE²).
67. Test liabilities protected; cpuP unreachable/excluded; registry/protocol+cpuQ/gpuA/gpuR audits qualify.
68. Regime=LEVERPROBE ramp+crosslane pair+union of four recorded upper-quartile CAV/density bands; observable only.
69. World-bootstrap EB residual contrasts are selection-conditional, not causal twins; missing bucket=>pooled; no pooled evidence=>keep beta.
