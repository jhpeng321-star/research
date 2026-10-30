# S-RANKCHECK — deadline measurement (incomplete; no launch authority)
As of 2026-10-30 20:34 +08:00; final verified code+evidence SHA `fe5e28f23986cc1bec703322f9fd89174bb7888f`.
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
| native | 6/6/K4 | 10277.4 [6728.5,13768.9] | +147.2 [+69.8,+237.9] | 41771.8 [28291.0,54356.8] | +1167.1 [+736.4,+1633.6] |
| CTRL0 | 6/6/K4 | 10130.2 [6572.9,13588.7] | +0.0 [+0.0,+0.0] | 40604.7 [27351.1,53100.4] | +0.0 [+0.0,+0.0] |
| MAPPO proxy aecd7d70 | 2/2/K4 | 14173.3 [11755.6,16591.0] | +7.2 [-5.3,+19.7] | 56305.9 [50157.5,62454.2] | -310.3 [-492.3,-128.3] |
| speed-only MAPPO partial i25 | 2/2/K4 | 14173.9 [11767.2,16580.6] | +7.8 [+6.3,+9.3] | 56830.7 [50942.2,62719.1] | +214.5 [+136.6,+292.4] |
| CIRCLES damping | 4/4/K4 | 9880.4 [5398.9,14361.8] | +343.1 [+194.0,+576.2] | 41278.0 [23809.2,58746.8] | +2546.2 [+1679.3,+3423.1] |
| PI external-U | 6/6/K4 | 10335.1 [6849.7,13761.1] | +204.9 [+127.0,+291.2] | 41578.1 [28357.1,54064.4] | +973.4 [+285.2,+1784.9] |
| fresh-zero B | identity: one 300s/K4 start | =CTRL0 | 0 | =CTRL0 | 0 |
Balanced common-start ranking (2 starts),90s: CTRL0 < MAPPO proxy aecd7d70 < speed-only MAPPO partial i25 < native < PI external-U < CIRCLES damping.
Balanced common-start ranking,300s: MAPPO proxy aecd7d70 < CTRL0 < PI external-U < speed-only MAPPO partial i25 < native < CIRCLES damping.
Cross-host CTRL0 reference max absolute future-J300 difference on common starts: 0.293 veh-s; every dJ remains paired within its own process.
## Actor objective (loss at scale1; lower is better)
L90 is TRAIN-resolved dense-prefix suffix/joint-q10 loss (observed caps [60.0, 90.0]s); L300 adds registered gain-gated accrued300s credit. These are not physical J.
| Controller | starts/worlds/K | L90 [95% CI] | dL90 vs CTRL0 | L300 [95% CI] | dL300 vs CTRL0 |
|---|---|---:|---:|---:|---:|
| native | 1/1/K8 | -73.9 [NE] | +132.6 [NE] | -73.9 [NE] | +132.6 [NE] |
| CTRL0 | 6/6/K4 | -912.2 [-2363.2,371.3] | +0.0 [+0.0,+0.0] | -912.2 [-2363.2,371.3] | +0.0 [+0.0,+0.0] |
| MAPPO proxy aecd7d70 | 2/2/K4 | 286.9 [-438.0,1011.8] | -7.6 [-21.8,+6.6] | 286.9 [-438.0,1011.8] | -7.6 [-21.8,+6.6] |
| speed-only MAPPO partial i25 | 2/2/K4 | 298.9 [-442.1,1039.9] | +4.4 [+2.5,+6.3] | 298.9 [-442.1,1039.9] | +4.4 [+2.5,+6.3] |
| CIRCLES damping | 4/4/K4 | -126.0 [-813.6,561.7] | +424.1 [+30.7,+817.4] | -1994.8 [-5670.2,499.6] | -1444.7 [-5095.6,+730.8] |
| PI external-U | 6/6/K4 | -625.1 [-2099.4,631.9] | +287.2 [+136.6,+450.3] | -625.1 [-2099.4,631.9] | +287.2 [+136.6,+450.3] |
| fresh-zero B | identity: one 300s/K4 start | =CTRL0 | 0 | =CTRL0 | 0 |
Balanced common-start actor ranking,dense: MAPPO proxy aecd7d70 < CTRL0 < speed-only MAPPO partial i25 < PI external-U < CIRCLES damping.
Balanced common-start actor ranking,dense+300s: CIRCLES damping < MAPPO proxy aecd7d70 < CTRL0 < speed-only MAPPO partial i25 < PI external-U.
Native actor loss is the retained single-start K8 intervention; its larger-panel actor CI is missing. Native/CTRL0 physical rows are reused from actual reference rows, not extra rollouts.
Retained serial K8 first start: proxy dJ90+4.43,dJ300+249.59; native dJ90+91.91,dJ300+559.59 veh-s (one world,CI NE). All original saved arms remain on disk.
Fresh-zero B: one full300s scientific-state array equality PASS vs CTRL0, plus exact physical/actor equality. Current u191 head is nonzero; no construction-based current-A identity is claimed.
CIRCLES point estimates worsen all-CAV physical J300 while improving the group-weighted actor loss: these objectives rank it differently. R2 is evaluated on physical J, not actor-loss headroom.
## Existing SUMO evidence and R2
CTRLARM600s: CTRL0-native **-1956.53 [-4086.66,+144.50]**; lane-on MAPPO-CTRL0 **-623.95 [-2044.25,+664.51]** veh-s; both include0. Exact source: `E:/research-dense/docs/paper_results/CTRLARM.md:9-17`.
WM CTRL0-native300s: -1167.1 [-1633.6,-736.4] veh-s; point direction agrees with CTRLARM. Horizon/uncertainty differences remain.
These lane-on/600s reused-world results are contextual, not matched native-lane/300s speed-only efficacy; the partial speed-only export has no matched real result here.
No numeric PI/CIRCLES SUMO contrast in either named paper_results directory. `E:/research-dense/runs/active/C4-V5-20261014/REGISTRATION.md:96-99` registers damping/PI arms only; active C4-EVAL registrations are absent.
SETUPAUDIT2 physical all-vehicle/source/mix scoring agrees, but its TRAIN export calls train.sample directly (tools/setupaudit/inspect_inputs.py:46-58), bypassing B headroom/tilt/exclusions; no actor loss is saved. Fixedupdate190/horizon600 slots/future-specific lane uniforms also differ. It cannot substitute for this exact B panel.
Available proxy dJ300: -310.3 [-492.3,-128.3] veh-s; point direction agrees with historical lane-on MAPPO, not matched speed-only proof. R2 needs upper95 below0.
**R2 proxy clause: HOLDS in the reduced-K diagnostic only. Full registered R2: NOT DEMONSTRATED (coverage incomplete; authorized K4 fallback disclosed); no matched real PI/CIRCLES/native-lane alternative establishes it.**
Binding rule: `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md:4951-4956`. Point-estimate order/overlapping intervals alone do not establish the alternative.
## Execution and verification
Fresh source staging via tools/candB/stage.sh; gpuA <=2x.19, gpuR .40; own-PID/incarnation controls only. Peer WMREQUAL/SETUPAUDIT jobs untouched; all failed/timing attempts retained.
16 CPU tests pass (2 local CUDA skips);4 CPU/CUDA sensor tests pass on each GPU host. Actual WM native/CTRL0/proxy J90/J300 K4 arrays bitwise match saved K8 first4; Ruff/diff PASS. All paired CIs independently recomputed.
Every receipt remains measure_only=true,complete=false,passed=false,launch_authorized=false; historical mismatched CCT-2 remains unqualified. No push/deletions.
