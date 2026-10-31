# SPEEDHEAD — EXPLORATORY TRAIN/development study

## Frozen design and identity
Owner:2026-10-30 19:45; copy20:10; zero-step reuse20:30; PI/CIRCLES23:00; GPU WM02:30. Eight C4_V5_TRAIN963880–963887; textbook; native lanes.
Each arm-world:300s native+600s control; two worlds per p=.05/.1/.2/.4, demands1900/2400/2700. Both seed/exposure guards PASS; no freshness claim.
Design:`37f42e09c43c27addb29ea3c3b570937d4d8833777109fd67c5e47570652af21`; E3 manifest:`b3c031adb6bc2b5c2bca160bea308f83ba5567b97c1cd0e853eddcd8367a8193` (543/543).
CTRL0=exact E3 C1 forced-command zero residual/speedMode27; wrapper:`24095a1092eafb4c894804aa8142478a802eaea046142ec777e5f24f49f73596`; C1:`64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c`.
Observer:`b22af4623b5fe7fbb9a8920b8e8417c3591efae66a52bae7b72f4d899b4a7874`; four-arm worker:`e164f80f504a8889b58100b8c0fd38a59856ef542ea619ba51b601dea9215c2f`; baseline adapter:`c63b14bfaa49b993cade4608b0e69f714de6a286ea9cb1049816118541f4f3eb`.
MAPPO FINAL169:`d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca`; lane head absent. PI external-U31.29m/s; CIRCLES=frozen damping.
Real:SUMO1.27.1 CPU,nice19,one worker; gpuR CTRL0/PI/CIRCLES,gpuA MAPPO authorized fallback; original runtime unchanged.
Training J=transition-cohort road deficit at31.29m/s+post-insertion source backlog, integrated0.5s; team=E3 BranchTracker road+RequestLedger due-time source waiting.
Negative delta is better; equal-world means; K8 futures reduced within world; shared2,000 WORLD percentile draws, NumPy seed0,95%CIs;600s primary,300s descriptive.

## Per-world paired arm−CTRL0 delay (veh-s; J / team)
All entries are signed contrasts; CTRL0 is the shared zero reference. Exact absolute arm costs and120s components remain in per-arm receipts.
| Seed (p,demand) | Arm | Real300 | WM300 | Real600 | WM600 |
|---|---|---:|---:|---:|---:|
| 963880 (0.05,1900) | MAPPO | +2,830.1 / +2,830.1 | +393.2 / +393.1 | +7,770.3 / +7,769.9 | +1,211.8 / +1,211.8 |
| 963880 (0.05,1900) | PI | +578.8 / +578.8 | -100.8 / -101.0 | +946.8 / +946.8 | -2,460.0 / -2,460.3 |
| 963880 (0.05,1900) | CIRCLES | +981.5 / +981.0 | +434.2 / +433.6 | +2,030.6 / +2,030.2 | +1,820.5 / +1,819.8 |
| 963881 (0.05,2700) | MAPPO | -1,711.0 / -1,711.0 | -480.4 / -480.7 | -114.5 / -114.4 | -3,056.5 / -3,056.7 |
| 963881 (0.05,2700) | PI | +1,401.2 / +1,401.2 | +52.6 / +52.6 | +2,153.8 / +2,153.8 | -2,224.7 / -2,224.6 |
| 963881 (0.05,2700) | CIRCLES | -153.2 / -153.2 | +150.0 / +150.0 | -796.7 / -796.7 | +5.2 / +5.2 |
| 963882 (0.1,1900) | MAPPO | +1,150.6 / +1,150.5 | +97.0 / +96.3 | +1,341.1 / +1,341.0 | -1,982.1 / -1,986.0 |
| 963882 (0.1,1900) | PI | +96.3 / +96.3 | -117.8 / -117.9 | +1,125.6 / +1,125.6 | +652.1 / +648.1 |
| 963882 (0.1,1900) | CIRCLES | +1,644.1 / +1,643.7 | +333.3 / +332.5 | +4,311.9 / +4,311.5 | -79.5 / -84.0 |
| 963883 (0.1,2400) | MAPPO | -2,066.6 / -2,066.8 | -377.4 / -377.5 | -5,169.6 / -5,169.9 | -548.0 / -548.0 |
| 963883 (0.1,2400) | PI | +1,991.0 / +1,991.0 | -387.4 / -387.5 | +122.7 / +122.7 | +2,096.8 / +2,096.6 |
| 963883 (0.1,2400) | CIRCLES | +63.9 / +63.9 | +1,139.8 / +1,139.7 | +699.2 / +699.2 | +4,652.9 / +4,652.8 |
| 963884 (0.2,2400) | MAPPO | +1,525.6 / +1,525.5 | +445.8 / +445.6 | +8,023.3 / +8,023.3 | -108.0 / -108.3 |
| 963884 (0.2,2400) | PI | +1,255.3 / +1,255.3 | +670.5 / +670.3 | +1,127.2 / +1,127.2 | +3,262.0 / +3,261.7 |
| 963884 (0.2,2400) | CIRCLES | +3,307.7 / +3,307.7 | +2,123.5 / +2,123.3 | +11,325.0 / +11,324.6 | +12,599.4 / +12,598.9 |
| 963885 (0.2,2700) | MAPPO | -2,916.6 / -2,916.6 | -813.0 / -813.3 | -3,040.1 / -3,040.2 | +1,403.2 / +1,402.6 |
| 963885 (0.2,2700) | PI | -2,587.6 / -2,587.6 | -530.8 / -530.9 | -3,575.1 / -3,575.1 | +162.4 / +162.1 |
| 963885 (0.2,2700) | CIRCLES | -1,504.0 / -1,504.0 | +861.0 / +860.9 | -2,016.9 / -2,016.9 | +3,730.5 / +3,730.2 |
| 963886 (0.4,1900) | MAPPO | +59.9 / +59.8 | -987.2 / -987.8 | +1,317.4 / +1,317.1 | -7,000.7 / -7,001.0 |
| 963886 (0.4,1900) | PI | +468.6 / +469.0 | -37.2 / -42.5 | +3,161.1 / +3,161.9 | -3,444.5 / -3,451.6 |
| 963886 (0.4,1900) | CIRCLES | +5,824.3 / +5,805.6 | +3,218.9 / +3,210.8 | +20,147.2 / +20,112.1 | +8,262.1 / +8,250.6 |
| 963887 (0.4,2700) | MAPPO | -3,215.7 / -3,215.7 | +83.1 / +82.9 | -9,865.5 / -9,865.6 | +1,487.0 / +1,486.8 |
| 963887 (0.4,2700) | PI | -892.9 / -892.9 | +640.4 / +640.2 | -5,173.5 / -5,173.5 | +1,950.8 / +1,950.6 |
| 963887 (0.4,2700) | CIRCLES | +798.8 / +798.8 | +1,736.1 / +1,735.9 | +3,858.1 / +3,858.1 | +8,898.7 / +8,898.4 |

## Equal-world paired means and95%CIs (J / team)
- MAPPO300s real: -543.0 [-2,032.6, +886.7] / -543.0 [-2,032.7, +886.6]; WM: -204.9 [-559.5, +140.3] / -205.2 [-559.9, +140.1].
- MAPPO600s real: +32.8 [-3,920.4, +3,868.6] / +32.6 [-3,920.5, +3,868.4]; WM: -1,074.2 [-3,096.2, +575.0] / -1,074.9 [-3,096.9, +574.8].
- PI300s real: +288.9 [-689.4, +1,143.2] / +288.9 [-689.3, +1,143.3]; WM: +23.7 [-234.6, +308.3] / +22.9 [-236.0, +307.5].
- PI600s real: -13.9 [-2,065.7, +1,641.6] / -13.8 [-2,065.7, +1,641.9]; WM: -0.6 [-1,632.1, +1,560.1] / -2.2 [-1,634.5, +1,559.3].
- CIRCLES300s real: +1,370.4 [+48.9, +2,914.6] / +1,367.9 [+48.8, +2,907.5]; WM: +1,249.6 [+602.1, +1,923.6] / +1,248.4 [+601.7, +1,920.4].
- CIRCLES600s real: +4,944.8 [+750.3, +9,937.1] / +4,940.3 [+750.1, +9,923.8]; WM: +4,986.2 [+2,039.7, +7,921.5] / +4,984.0 [+2,036.5, +7,919.8].
- MAPPO300s WM−real bias J/team: +338.1 [-890.6, +1,655.7]; sign6/8 / +337.9 [-890.8, +1,655.5]; sign6/8.
- MAPPO600s WM−real bias J/team: -1,107.0 [-5,382.2, +3,824.8]; sign3/8 / -1,107.5 [-5,383.2, +3,824.8]; sign3/8.
- PI300s WM−real bias J/team: -265.2 [-1,176.7, +662.0]; sign3/8 / -266.0 [-1,176.8, +661.9]; sign3/8.
- PI600s WM−real bias J/team: +13.3 [-2,996.2, +2,944.9]; sign3/8 / +11.7 [-2,999.7, +2,944.2]; sign3/8.
- CIRCLES300s WM−real bias J/team: -120.8 [-1,122.8, +943.0]; sign6/8 / -119.6 [-1,118.9, +942.9]; sign6/8.
- CIRCLES600s WM−real bias J/team: +41.4 [-3,934.1, +3,458.4]; sign5/8 / +43.7 [-3,925.8, +3,458.3]; sign5/8.
WM2:`b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436`+learned mix; exact observable t300 states/histories/queues; aligned setupaudit cap/Euler/native-background comparator.
WM GPU/host adapters:`c36d9ed6fa5e29aaa5b9f91b52e68ad3a4cfaeeffcb5c3a7eaf7cb6183719444`/`952246e508d75bcda2fe550b7b0a6d234d54dc1832b6b71f5860c5a5b01dc784`; device-matched four-arm K8,policy_000190/update190/seed965140; <=8GiB allocator; CPU partials retained separately.

## Safety, accounting and provenance
CTRL08/8:collisions0,teleports0,hard-braking110.5veh-s,TTC<3s3125.5veh-s,dropped0.
MAPPO8/8:collisions0,teleports0,hard-braking120.5veh-s,TTC<3s2834.5veh-s,dropped0.
PI8/8:collisions0,teleports0,hard-braking120.0veh-s,TTC<3s3173.5veh-s,dropped0.
CIRCLES8/8:collisions0,teleports0,hard-braking229.5veh-s,TTC<3s4065.5veh-s,dropped0.
Allocations33; charged29,700/actual28,800sim-s; caps29,700/28,800; no additional SUMO. All arms lane requests/changeLane/setLaneChangeMode=0; matched initial-state hashes.
Original zero-step failure charged900/actual0 retained; one authorized reuse. Precharges fsynced before authorization; no further charged retries. Evidence:`tools/speedhead/evidence/` (runtime/export/start/task hashes, raw costs, CIs).
Final execution/evidence SHA:`1f58ab94686e946821deb877d1037344294e34a5` (report-only commit follows).
**Real speed-only headroom: inconclusive.**
