# S-RANKCHECK — deadline measurement (incomplete; no launch authority)
As of 2026-10-30 21:12 +08:00 (live shards; interim); final verified code+evidence SHA `82dbd7bc7e32026998bf3ec5b912f9b23c5d67e6`.
Report-delivery SHA: `git log -1 --format=%H -- docs/paper_results/RANKCHECK.md`.
Measured source SHAs: `2dcb6af5` (PI/CIRCLES), `3cd6481e` (MAPPO/zero), `f396955a` (disjoint MAPPO tails), `05f7bfba` (retained K8).
## Contract and completeness
WM2 `b8797242` + learned mix ON; candidate-B resolved config, CTRL0 law/reference, all vehicles + entrants + road deficit + source waiting; no training/SUMO.
MAIN19:10 fallback: **K4 = first four original K8 CRN plans**, same dense-suffix/joint-q10 and gain-gated300s algebra; this is a reduced-K estimator, not K8 qualification.
Config SHA256 `78a3caec6813c9b8b87996162908a09d7bb3d747f8932d2484d2cd239bbcedd3`; WM inference SHA256 `f95f7c4cbb501ccb85bb3968e29f3d6ca53db4ecef5f586225f96df1f48193e4`.
Sixteen actual TRAIN-sampler starts exported; plan SHA256 `b522e6cb27788d6d4095f4a40f7a9b401f4b57855e9beb7e7b71d3add3a10382`. Completed counts appear below; no missing arm/future imputed.
Raw JSON/argv/ownership: `tools/candB/evidence/rankcheck_parallel/{circles,pi,mappo,mappo_proxy_tail,mappo_speed_tail,zero}/`; retained K8: `../rankcheck_saved_k8/CANDB_RANKCHECK.json`.
cpuQ read-only first-inspection export: `~/mappospeed/production2/policy_00025.pt`, iteration25/169, **PARTIALLY TRAINED**, frozen before outcomes; SHA256 `69e414e53e7f520bbd112ff292aa207d3da6388d73c3ac91e894079dfe40a543`.
Native lanes throughout; aecd7d70 proxy ignores its learned lane head. PI/CIRCLES use the published TimelineControllers; CPU/CUDA sensor-copy parity4/4 on BOTH hosts.
CIs: paired equal-WORLD percentile bootstrap,5000 draws/seed20261030; average futures/starts inside each world first. NE=not estimable from one world.
## Physical J (veh-s; lower is better)
Each row uses ALL its completed starts; dJ is paired to that row's actual CTRL0 futures. Different completion counts mean absolute row means are not a balanced ranking.
| Controller | starts/worlds/K | J90 [95% CI] | dJ90 vs CTRL0 | J300 [95% CI] | dJ300 vs CTRL0 |
|---|---|---:|---:|---:|---:|
| native | 11/11/K4 | 10869.3 [7674.8,14242.0] | +129.0 [+70.5,+195.4] | 44580.2 [32736.0,57002.4] | +1060.9 [+602.9,+1532.8] |
| CTRL0 | 11/11/K4 | 10740.3 [7537.5,14114.5] | +0.0 [+0.0,+0.0] | 43519.2 [31579.7,56098.2] | +0.0 [+0.0,+0.0] |
| MAPPO proxy aecd7d70 | 5/5/K4 | 10442.0 [6288.4,14595.6] | +12.9 [+2.1,+21.3] | 41639.3 [26590.1,56459.9] | -163.5 [-410.5,+83.5] |
| speed-only MAPPO partial i25 | 4/4/K4 | 11651.0 [6673.1,15377.3] | +18.4 [+7.8,+33.1] | 46218.6 [26759.8,59774.9] | -254.6 [-723.6,+214.5] |
| CIRCLES damping | 7/7/K4 | 12040.7 [8014.1,16326.8] | +323.7 [+201.2,+477.5] | 49085.0 [33710.2,65002.9] | +2450.4 [+1582.6,+3399.6] |
| PI external-U | 11/11/K4 | 10918.3 [7730.5,14262.2] | +178.1 [+115.5,+243.6] | 44481.0 [32604.0,56968.9] | +961.8 [+374.5,+1618.0] |
| fresh-zero B | identity: one 300s/K4 start | =CTRL0 | 0 | =CTRL0 | 0 |
Balanced common-start ranking (4 starts),90s: CTRL0 < MAPPO proxy aecd7d70 < speed-only MAPPO partial i25 < native < PI external-U < CIRCLES damping.
Balanced common-start ranking,300s: speed-only MAPPO partial i25 < MAPPO proxy aecd7d70 < CTRL0 < PI external-U < native < CIRCLES damping.
Cross-host CTRL0 reference max absolute future-J300 difference on common starts: 0.293 veh-s; every dJ remains paired within its own process.
## Actor objective (loss at scale1; lower is better)
L90 is TRAIN-resolved dense-prefix suffix/joint-q10 loss (observed caps [10.0, 60.0, 90.0]s); L300 adds registered gain-gated accrued300s credit. These are not physical J.
| Controller | starts/worlds/K | L90 [95% CI] | dL90 vs CTRL0 | L300 [95% CI] | dL300 vs CTRL0 |
|---|---|---:|---:|---:|---:|
| native | 1/1/K8 | -73.9 [NE] | +132.6 [NE] | -73.9 [NE] | +132.6 [NE] |
| CTRL0 | 11/11/K4 | -716.1 [-1618.7,40.2] | +0.0 [+0.0,+0.0] | -716.1 [-1618.7,40.2] | +0.0 [+0.0,+0.0] |
| MAPPO proxy aecd7d70 | 5/5/K4 | -1237.6 [-2757.8,141.9] | +30.7 [-9.4,+92.0] | -1237.6 [-2757.8,141.9] | +30.7 [-9.4,+92.0] |
| speed-only MAPPO partial i25 | 4/4/K4 | -1118.5 [-3200.1,541.9] | +10.7 [+4.4,+17.9] | -1202.0 [-3200.1,458.3] | -72.8 [-239.8,+16.9] |
| CIRCLES damping | 7/7/K4 | -404.6 [-1702.6,664.8] | +357.9 [+100.9,+637.6] | -1472.4 [-3754.3,454.9] | -710.0 [-2833.8,+518.2] |
| PI external-U | 11/11/K4 | -542.8 [-1413.9,193.2] | +173.3 [+51.8,+306.9] | -542.8 [-1413.9,193.2] | +173.3 [+51.8,+306.9] |
| fresh-zero B | identity: one 300s/K4 start | =CTRL0 | 0 | =CTRL0 | 0 |
Balanced common-start actor ranking,dense: CTRL0 < MAPPO proxy aecd7d70 < speed-only MAPPO partial i25 < PI external-U < CIRCLES damping.
Balanced common-start actor ranking,dense+300s: CIRCLES damping < speed-only MAPPO partial i25 < CTRL0 < MAPPO proxy aecd7d70 < PI external-U.
Native actor loss is the retained single-start K8 intervention; its larger-panel actor CI is missing. Native/CTRL0 physical rows are reused from actual reference rows, not extra rollouts.
Retained serial K8 first start: proxy dJ90+4.43,dJ300+249.59; native dJ90+91.91,dJ300+559.59 veh-s (one world,CI NE). All original saved arms remain on disk.
Fresh-zero B: one full300s scientific-state array equality PASS vs CTRL0, plus exact physical/actor equality. Current u191 head is nonzero; no construction-based current-A identity is claimed.
CIRCLES point estimates worsen all-CAV physical J300 while improving the group-weighted actor loss: these objectives rank it differently. R2 is evaluated on physical J, not actor-loss headroom.
## Existing SUMO evidence and R2
CTRLARM600s: CTRL0-native **-1956.53 [-4086.66,+144.50]**; lane-on MAPPO-CTRL0 **-623.95 [-2044.25,+664.51]** veh-s; both include0. Exact source: `E:/research-dense/docs/paper_results/CTRLARM.md:9-17`.
WM CTRL0-native300s: -1060.9 [-1532.8,-602.9] veh-s; point direction agrees with CTRLARM. Horizon/uncertainty differences remain.
These lane-on/600s reused-world results are contextual, not matched native-lane/300s speed-only efficacy; the partial speed-only export has no matched real result here.
No numeric PI/CIRCLES SUMO contrast in either named paper_results directory. `E:/research-dense/runs/active/C4-V5-20261014/REGISTRATION.md:96-99` registers damping/PI arms only; active C4-EVAL registrations are absent.
SETUPAUDIT2 physical all-vehicle/source/mix scoring agrees, but its TRAIN export calls train.sample directly (tools/setupaudit/inspect_inputs.py:46-58), bypassing B headroom/tilt/exclusions; no actor loss is saved. Fixedupdate190/horizon600 slots/future-specific lane uniforms also differ. It cannot substitute for this exact B panel.
**R2 proxy clause: NOT DEMONSTRATED in available reduced-K data. Full registered R2: NOT DEMONSTRATED (coverage incomplete; authorized K4 fallback disclosed); no matched real PI/CIRCLES/native-lane alternative establishes it.**
Available proxy dJ300: -163.5 [-410.5,+83.5] veh-s; point direction agrees with historical lane-on MAPPO, not matched speed-only proof. R2 needs upper95 below0.
Binding rule: `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md:4951-4956`. Point-estimate order/overlapping intervals alone do not establish the alternative.
## Execution and verification
Fresh immutable staging via tools/candB/stage.sh. MAIN21:08: no new gpuA scorers; existing shards finish. Retired only exact owned obsolete21:20 cutoff supervisors. New scoring is gpuR-only,total<=22GB; peers untouched.
Recovered tail guard had already failed before signalling; historical45000MiB guard was ineffective (gpuA45713MiB at21:08). Repeated prefix/tail starts are exact-array verified and counted once; all attempts retained.
16 CPU tests pass (2 local CUDA skips);4 CPU/CUDA sensor tests pass on each GPU host. Actual WM native/CTRL0/proxy J90/J300 K4 arrays bitwise match saved K8 first4; Ruff/diff PASS. All paired CIs independently recomputed.
Every receipt remains measure_only=true,complete=false,passed=false,launch_authorized=false; historical mismatched CCT-2 remains unqualified. No push/deletions.
