# S-RANKCHECK — deadline measurement (incomplete; no launch authority)
As of 2026-10-30 20:15 +08:00; final verified code+evidence SHA `c8bfebb54e6b94303f356b4e7cd1931af9ec469b`.
Report-delivery SHA: `git log -1 --format=%H -- docs/paper_results/RANKCHECK.md`.
Measured source SHAs: `2dcb6af5` (PI/CIRCLES), `3cd6481e` (MAPPO/zero), `05f7bfba` (retained serial K8).
## Contract and completeness
WM2 `b8797242` + learned mix ON; candidate-B resolved config, CTRL0 law/reference, all vehicles + entrants + road deficit + source waiting; no training/SUMO.
MAIN19:10 fallback: **K4 = first four original K8 CRN plans**, same dense-suffix/joint-q10 and gain-gated300s algebra; this is a reduced-K estimator, not K8 qualification.
Config SHA256 `78a3caec6813c9b8b87996162908a09d7bb3d747f8932d2484d2cd239bbcedd3`; WM inference SHA256 `f95f7c4cbb501ccb85bb3968e29f3d6ca53db4ecef5f586225f96df1f48193e4`.
Sixteen actual TRAIN-sampler starts exported; plan SHA256 `b522e6cb27788d6d4095f4a40f7a9b401f4b57855e9beb7e7b71d3add3a10382`. Completed counts appear below; no missing arm/future imputed.
Raw JSON/argv/ownership: `tools/candB/evidence/rankcheck_parallel/{circles,pi,mappo,zero}/`; retained K8: `../rankcheck_saved_k8/CANDB_RANKCHECK.json`.
cpuQ read-only first-inspection export: `~/mappospeed/production2/policy_00025.pt`, iteration25/169, **PARTIALLY TRAINED**, frozen before outcomes; SHA256 `69e414e53e7f520bbd112ff292aa207d3da6388d73c3ac91e894079dfe40a543`.
Native lanes throughout; aecd7d70 proxy ignores its learned lane head. PI/CIRCLES use the published TimelineControllers; CPU/CUDA sensor-copy parity4/4 on BOTH hosts.
CIs: paired equal-WORLD percentile bootstrap,5000 draws/seed20261030; average futures/starts inside each world first. NE=not estimable from one world.
## Physical J (veh-s; lower is better)
Each row uses ALL its completed starts; dJ is paired to that row's actual CTRL0 futures. Different completion counts mean absolute row means are not a balanced ranking.
| Controller | starts/worlds/K | J90 [95% CI] | dJ90 vs CTRL0 | J300 [95% CI] | dJ300 vs CTRL0 |
|---|---|---:|---:|---:|---:|
| native | 3/3/K4 | 10941.6 [4284.7,16699.5] | +97.1 [+79.8,+128.2] | 44770.9 [19474.8,63592.1] | +835.5 [+596.1,+1009.5] |
| CTRL0 | 3/3/K4 | 10844.5 [4201.3,16571.3] | +0.0 [+0.0,+0.0] | 43935.4 [18573.8,62582.6] | +0.0 [+0.0,+0.0] |
| MAPPO proxy aecd7d70 | 1/1/K4 | 16591.0 [NE] | +19.7 [NE] | 62454.2 [NE] | -128.3 [NE] |
| speed-only MAPPO partial i25 | 1/1/K4 | 16580.6 [NE] | +9.3 [NE] | 62719.1 [NE] | +136.6 [NE] |
| CIRCLES damping | 2/2/K4 | 14361.8 [11923.2,16800.5] | +195.8 [+162.3,+229.2] | 58746.8 [51969.2,65524.4] | +2130.6 [+1319.4,+2941.9] |
| PI external-U | 3/3/K4 | 10994.5 [4442.0,16683.8] | +150.0 [+96.8,+240.7] | 44418.7 [19680.8,62670.9] | +483.3 [+88.3,+1107.0] |
| fresh-zero B | identity: one 300s/K4 start | =CTRL0 | 0 | =CTRL0 | 0 |
Balanced common-start ranking (1 starts),90s: CTRL0 < speed-only MAPPO partial i25 < MAPPO proxy aecd7d70 < PI external-U < native < CIRCLES damping.
Balanced common-start ranking,300s: MAPPO proxy aecd7d70 < CTRL0 < PI external-U < speed-only MAPPO partial i25 < native < CIRCLES damping.
Cross-host CTRL0 reference max absolute future-J300 difference on common starts: 0.293 veh-s; every dJ remains paired within its own process.
## Actor objective (loss at scale1; lower is better)
L90 is TRAIN-resolved dense-prefix suffix/joint-q10 loss (observed caps [60.0, 90.0]s); L300 adds registered gain-gated accrued300s credit. These are not physical J.
| Controller | starts/worlds/K | L90 [95% CI] | dL90 vs CTRL0 | L300 [95% CI] | dL300 vs CTRL0 |
|---|---|---:|---:|---:|---:|
| native | 1/1/K8 | -73.9 [NE] | +132.6 [NE] | -73.9 [NE] | +132.6 [NE] |
| CTRL0 | 3/3/K4 | -125.2 [-964.6,1033.6] | +0.0 [+0.0,+0.0] | -125.2 [-964.6,1033.6] | +0.0 [+0.0,+0.0] |
| MAPPO proxy aecd7d70 | 1/1/K4 | -438.0 [NE] | +6.6 [NE] | -438.0 [NE] | +6.6 [NE] |
| speed-only MAPPO partial i25 | 1/1/K4 | -442.1 [NE] | +2.5 [NE] | -442.1 [NE] | +2.5 [NE] |
| CIRCLES damping | 2/2/K4 | 561.7 [184.3,939.0] | +267.2 [-94.6,+628.9] | -3175.9 [-7290.8,939.0] | -3470.4 [-6846.2,-94.6] |
| PI external-U | 3/3/K4 | 35.2 [-827.2,1050.1] | +160.4 [+16.5,+327.3] | 35.2 [-827.2,1050.1] | +160.4 [+16.5,+327.3] |
| fresh-zero B | identity: one 300s/K4 start | =CTRL0 | 0 | =CTRL0 | 0 |
Balanced common-start actor ranking,dense: CTRL0 < speed-only MAPPO partial i25 < MAPPO proxy aecd7d70 < PI external-U < CIRCLES damping.
Balanced common-start actor ranking,dense+300s: CIRCLES damping < CTRL0 < speed-only MAPPO partial i25 < MAPPO proxy aecd7d70 < PI external-U.
Native actor loss is the retained single-start K8 intervention; its larger-panel actor CI is missing. Native/CTRL0 physical rows are reused from actual reference rows, not extra rollouts.
Retained serial K8 first start: proxy dJ90+4.43,dJ300+249.59; native dJ90+91.91,dJ300+559.59 veh-s (one world,CI NE). All original saved arms remain on disk.
Fresh-zero B: one full300s scientific-state array equality PASS vs CTRL0, plus exact physical/actor equality. Current u191 head is nonzero; no construction-based current-A identity is claimed.
## Existing SUMO evidence and R2
CTRLARM600s: CTRL0-native **-1956.53 [-4086.66,+144.50]**; lane-on MAPPO-CTRL0 **-623.95 [-2044.25,+664.51]** veh-s; both include0. Exact source: `E:/research-dense/docs/paper_results/CTRLARM.md:9-17`.
These lane-on/600s reused-world results are contextual, not matched native-lane/300s speed-only efficacy; the partial speed-only export has no matched real result here.
No numeric PI/CIRCLES SUMO contrast in either named paper_results directory. `E:/research-dense/runs/active/C4-V5-20261014/REGISTRATION.md:96-99` registers damping/PI arms only; active C4-EVAL registrations are absent.
SETUPAUDIT2 physical all-vehicle/source/mix scoring agrees, but its TRAIN export calls train.sample directly (tools/setupaudit/inspect_inputs.py:46-58), bypassing B headroom/tilt/exclusions; no actor loss is saved. Fixedupdate190/horizon600 slots/future-specific lane uniforms also differ. It cannot substitute for this exact B panel.
Available proxy dJ300: -128.3 [NE] veh-s; speed-only R2 needs upper95 below0.
**R2 proxy clause: NOT DEMONSTRATED in available reduced-K data. Full registered R2: NOT DEMONSTRATED (coverage incomplete; authorized K4 fallback disclosed); no matched real PI/CIRCLES/native-lane alternative establishes it.**
Binding rule: `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md:4951-4956`. Point-estimate order/overlapping intervals alone do not establish the alternative.
## Execution and verification
Fresh source staging via tools/candB/stage.sh; gpuA <=2x.19, gpuR .40; own-PID/incarnation controls only. Peer WMREQUAL/SETUPAUDIT jobs untouched; all failed/timing attempts retained.
16 CPU tests pass (2 local CUDA skips);4 CPU/CUDA equivalence tests pass independently on each GPU host; Ruff/diff PASS. Paired CIs independently recomputed from every retained completed start.
Every receipt remains measure_only=true,complete=false,passed=false,launch_authorized=false; historical mismatched CCT-2 remains unqualified. No push/deletions.
