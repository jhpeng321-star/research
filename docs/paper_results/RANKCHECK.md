# S-RANKCHECK — deadline measurement (incomplete; no launch authority)
As of 2026-10-30 21:40 +08:00 (live shards; interim); final verified code+evidence SHA `fabe3e7d0d070dcd095e8c4e90061fd0d64edb1f`.
Report-delivery SHA: `git log -1 --format=%H -- docs/paper_results/RANKCHECK.md`.
Measured source SHAs: `2dcb6af5` (PI/CIRCLES), `3cd6481e` (MAPPO/zero), `f396955a` (MAPPO tails), `82dbd7bc` (PI/native resume), `05f7bfba` (retained K8).
## Contract and completeness
WM2 `b8797242` + learned mix ON; candidate-B resolved config, CTRL0 law/reference, all vehicles + entrants + road deficit + source waiting; no training/SUMO.
MAIN19:10 fallback: **K4 = first four original K8 CRN plans**, same dense-suffix/joint-q10 and gain-gated300s algebra; this is a reduced-K estimator, not K8 qualification.
Config SHA256 `78a3caec6813c9b8b87996162908a09d7bb3d747f8932d2484d2cd239bbcedd3`; WM inference SHA256 `f95f7c4cbb501ccb85bb3968e29f3d6ca53db4ecef5f586225f96df1f48193e4`.
Sixteen actual TRAIN-sampler starts exported; plan SHA256 `b522e6cb27788d6d4095f4a40f7a9b401f4b57855e9beb7e7b71d3add3a10382`. Completed counts appear below; no missing arm/future imputed.
Raw JSON/argv/ownership: `tools/candB/evidence/rankcheck_parallel/` (named arm and resume subdirectories); retained K8: `../rankcheck_saved_k8/CANDB_RANKCHECK.json`.
cpuQ read-only first-inspection export: `~/mappospeed/production2/policy_00025.pt`, iteration25/169, **PARTIALLY TRAINED**, frozen before outcomes; SHA256 `69e414e53e7f520bbd112ff292aa207d3da6388d73c3ac91e894079dfe40a543`.
Native lanes throughout; aecd7d70 proxy ignores its learned lane head. PI/CIRCLES use the published TimelineControllers; CPU/CUDA sensor-copy parity4/4 on BOTH hosts.
CIs: paired equal-WORLD percentile bootstrap,5000 draws/seed20261030; average futures/starts inside each world first. NE=not estimable from one world.
## Physical J (veh-s; lower is better)
Each row uses ALL its completed starts; dJ is paired to that row's actual CTRL0 futures. Different completion counts mean absolute row means are not a balanced ranking.
| Controller | starts/worlds/K | J90 [95% CI] | dJ90 vs CTRL0 | J300 [95% CI] | dJ300 vs CTRL0 |
|---|---|---:|---:|---:|---:|
| native | 12/12/K4 | 10805.1 [7902.8,13975.7] | +146.6 [+84.6,+211.7] | 44534.9 [33627.6,56281.0] | +1262.4 [+714.8,+1843.4] |
| CTRL0 | 12/12/K4 | 10658.5 [7768.8,13833.1] | +0.0 [+0.0,+0.0] | 43272.5 [32351.2,55107.7] | +0.0 [+0.0,+0.0] |
| MAPPO proxy aecd7d70 | 5/5/K4 | 10442.0 [6288.4,14595.6] | +12.9 [+2.1,+21.3] | 41639.3 [26590.1,56459.9] | -163.5 [-410.5,+83.5] |
| speed-only MAPPO partial i25 | 5/5/K4 | 10446.4 [6292.0,14600.8] | +17.2 [+8.8,+29.7] | 41747.3 [26180.3,56596.1] | -55.4 [-551.6,+440.8] |
| CIRCLES damping | 8/8/K4 | 11097.6 [7283.1,15326.9] | +301.3 [+193.3,+437.6] | 45648.9 [31382.0,61299.9] | +2179.9 [+1310.8,+3149.1] |
| PI external-U | 12/12/K4 | 10834.7 [7943.4,14004.8] | +176.3 [+117.8,+237.2] | 44223.4 [33403.9,56017.9] | +951.0 [+406.6,+1548.3] |
| fresh-zero B | identity: one 300s/K4 start | =CTRL0 | 0 | =CTRL0 | 0 |
Balanced common-start ranking (5 starts),90s: CTRL0 < MAPPO proxy aecd7d70 < speed-only MAPPO partial i25 < native < PI external-U < CIRCLES damping.
Balanced common-start ranking,300s: MAPPO proxy aecd7d70 < speed-only MAPPO partial i25 < CTRL0 < PI external-U < native < CIRCLES damping.
Cross-host CTRL0 reference max absolute future-J300 difference on common starts: 0.293 veh-s; every dJ remains paired within its own process.
## Actor objective (loss at scale1; lower is better)
L90 is TRAIN-resolved dense-prefix suffix/joint-q10 loss (observed caps [10.0, 60.0, 90.0]s); L300 adds registered gain-gated accrued300s credit. These are not physical J.
| Controller | starts/worlds/K | L90 [95% CI] | dL90 vs CTRL0 | L300 [95% CI] | dL300 vs CTRL0 |
|---|---|---:|---:|---:|---:|
| native | 1/1/K4 | -2246.6 [NE] | +118.2 [NE] | -2246.6 [NE] | +118.2 [NE] |
| CTRL0 | 12/12/K4 | -853.5 [-1686.5,-108.5] | +0.0 [+0.0,+0.0] | -853.5 [-1686.5,-108.5] | +0.0 [+0.0,+0.0] |
| MAPPO proxy aecd7d70 | 5/5/K4 | -1237.6 [-2757.8,141.9] | +30.7 [-9.4,+92.0] | -1237.6 [-2757.8,141.9] | +30.7 [-9.4,+92.0] |
| speed-only MAPPO partial i25 | 5/5/K4 | -1251.4 [-2750.6,150.7] | +16.9 [+6.4,+29.7] | -1318.2 [-2817.4,109.4] | -50.0 [-188.2,+27.4] |
| CIRCLES damping | 8/8/K4 | -348.8 [-1515.5,533.9] | +320.4 [+95.2,+576.9] | -1283.2 [-3302.9,302.7] | -614.0 [-2472.3,+462.3] |
| PI external-U | 12/12/K4 | -683.5 [-1500.5,60.2] | +169.9 [+58.3,+288.8] | -683.5 [-1500.5,60.2] | +169.9 [+58.3,+288.8] |
| fresh-zero B | identity: one 300s/K4 start | =CTRL0 | 0 | =CTRL0 | 0 |
Balanced common-start actor ranking,dense: CTRL0 < speed-only MAPPO partial i25 < MAPPO proxy aecd7d70 < PI external-U < CIRCLES damping.
Balanced common-start actor ranking,dense+300s: CIRCLES damping < speed-only MAPPO partial i25 < CTRL0 < MAPPO proxy aecd7d70 < PI external-U.
Native actor is a separate K4 intervention on its reported completed starts (not in the common-panel actor ranking); native/CTRL0 physical rows reuse actual reference rows.
Retained serial K8 first start: proxy dJ90+4.43,dJ300+249.59; native dJ90+91.91,dJ300+559.59 veh-s (one world,CI NE). All original saved arms remain on disk.
Fresh-zero B: one full300s scientific-state array equality PASS vs CTRL0, plus exact physical/actor equality. Current u191 head is nonzero; no construction-based current-A identity is claimed.
CIRCLES point estimates worsen all-CAV physical J300 while improving the group-weighted actor loss: these objectives rank it differently. R2 is evaluated on physical J, not actor-loss headroom.
## Existing SUMO evidence and R2
CTRLARM600s: CTRL0-native **-1956.53 [-4086.66,+144.50]**; lane-on MAPPO-CTRL0 **-623.95 [-2044.25,+664.51]** veh-s; both include0. Exact source: `E:/research-dense/docs/paper_results/CTRLARM.md:9-17`.
WM CTRL0-native300s: -1262.4 [-1843.4,-714.8] veh-s; point direction agrees with CTRLARM. Horizon/uncertainty differences remain.
These lane-on/600s reused-world results are contextual, not matched native-lane/300s speed-only efficacy; the partial speed-only export has no matched real result here.
No numeric PI/CIRCLES SUMO contrast in either named paper_results directory. `E:/research-dense/runs/active/C4-V5-20261014/REGISTRATION.md:96-99` registers damping/PI arms only; active C4-EVAL registrations are absent.
SETUPAUDIT2 physical all-vehicle/source/mix scoring agrees, but its TRAIN export calls train.sample directly (tools/setupaudit/inspect_inputs.py:46-58), bypassing B headroom/tilt/exclusions; no actor loss is saved. Fixedupdate190/horizon600 slots/future-specific lane uniforms also differ. It cannot substitute for this exact B panel.
**R2 proxy clause: NOT DEMONSTRATED in available reduced-K data. Full registered R2: NOT DEMONSTRATED (coverage incomplete; authorized K4 fallback disclosed); no matched real PI/CIRCLES/native-lane alternative establishes it.**
Available proxy dJ300: -163.5 [-410.5,+83.5] veh-s; point direction agrees with historical lane-on MAPPO, not matched speed-only proof. R2 needs upper95 below0.
Binding rule: `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md:4951-4956`. Point-estimate order/overlapping intervals alone do not establish the alternative.
## Execution and verification
Fresh immutable staging via tools/candB/stage.sh. MAIN21:08: no new gpuA scorers; existing shards finish. Retired only exact owned obsolete21:20 cutoff supervisors. New scoring is gpuR-only,total<=22GB; peers untouched.
Recovered tail guard had already failed before signalling; historical45000MiB guard was ineffective (gpuA45713MiB at21:08). Repeated prefix/tail starts are exact-array verified and counted once; all attempts retained.
gpuR PI exhausted its.40 allocator at plan index11 after11 saved starts. Fresh source82dbd7bc resumes11 on CPU and12-15 on GPU with native actor interventions; both use the same bound starts and first4 CRN plans.
gpuA.16 MAPPO tails also exited on allocator OOM after one saved start each; no retry there. Existing prefix/CIRCLES continue. CPU/CUDA resume differences are not bitwise parity claims; contrasts are always process-paired.
17 CPU tests pass (2 local CUDA skips);4 CPU/CUDA sensor tests pass on each GPU host. Actual WM native/CTRL0/proxy J90/J300 K4 arrays bitwise match saved K8 first4; Ruff/diff PASS. All paired CIs independently recomputed.
Every receipt remains measure_only=true,complete=false,passed=false,launch_authorized=false; historical mismatched CCT-2 remains unqualified. No push/deletions.
