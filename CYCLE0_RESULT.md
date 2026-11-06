# Cycle 0 real-SUMO result — round-16 coop-ON policy

## Headline

**Policy − native total delay: +618,976.835 veh-s pooled across 26 × 300 s paired continuations / 13 worlds; 95% world-bootstrap CI [+503,848.282, +739,343.579].**
Gain (native − policy): **-618,976.835 veh-s**, CI [-739,343.579, -503,848.282]. Policy improves **0/13 worlds; 0/26 starts**.
Per requested-vehicle exposure: cost difference **+41.082 s/request**, CI [+33.413, +49.127]; gain **-41.082 s/request**, CI [-49.127, -33.413].

## Cohort, signs and estimand

- Existing cycle-0 collection only: cpuP seven worlds, cpuQ six; seeds968129–968141; CAV shares5%,10%,20%,40%; demands1900,2100,2400,2700 veh/h. Two saved native-state decision times360s/660s per world; each branch lasts300s. These are training-world results, not final evaluation.
- The two policy branches each restart from their saved native state. Pooling them does not establish performance of one continuously controlled600s episode. Prehistory/native base costs are excluded from branch metrics.
- Cost difference = treatment − reference, here POLICY − native B: negative improves delay, braking, TTC and incidents. Gain = native − policy: positive improves those costs. For throughput, more arrivals are better: a positive POLICY−B difference is an improvement; native−policy is only the signed complement, not a throughput benefit.
- Total delay J = authoritative on-road speed-deficit veh-s + observed entrance-waiting veh-s, including post-start entrants and queued requests. This is finite-horizon delay, not completed-trip travel time. On-road uses the collector’s fixed speed limit; no behavioral threshold is fitted.
- Normalized delay divides pooled J by 15,067 requested-vehicle exposures. `requested_population` counts requests due by each branch endpoint, including requests before branch start, and each world occurs at two endpoints; it is not a count of unique newly requested vehicles. Raw veh-s is the primary estimand.
- Hard braking uses the existing collector metric a<−3m/s² at0.5s, reconstructed from observed acceleration/presence arrays after each branch start through its endpoint. It measures vehicle-seconds of exposure, not unique braking episodes. Road-delay reconstruction matches every authoritative summary at rtol1e−6/atol1e−3, verifying population/time coverage; no drops in52/52 branches.
- TTC<3s vehicle-seconds, arrivals, collision involvements and starting teleports come directly from each branch’s authoritative ledger. No TTC reconstruction from lane geometry or driver/type files.

## Pooled metrics and paired world-cluster uncertainty

5,000 percentile bootstrap draws, NumPy default_rng seed20261026: sample13 whole worlds with replacement; retain both starts, both arms, numerators and denominators together. The pooled-total CI is for a13-world/26-start cohort total. Normalized ratios are recomputed in every draw. No vehicle/time-step pseudo-replication.

| Metric | Native B | POLICY | Cost difference POLICY−B [95% CI] | Gain B−POLICY [95% CI] |
|---|---:|---:|---|---|
| Total delay incl. entrance waiting (veh-s) | 578,391.062 | 1,197,367.898 | +618,976.835 [+503,848.282, +739,343.579] | -618,976.835 [-739,343.579, -503,848.282] |
| On-road delay (veh-s) | 257,401.360 | 460,993.696 | +203,592.335 [+128,253.037, +272,841.968] | -203,592.335 [-272,841.968, -128,253.037] |
| Entrance waiting (veh-s) | 320,989.702 | 736,374.202 | +415,384.500 [+316,265.475, +504,587.500] | -415,384.500 [-504,587.500, -316,265.475] |
| Throughput (arrivals) | 4,912.000 | 2,062.000 | -2,850.000 [-3,397.050, -2,316.950] | +2,850.000 [+2,316.950, +3,397.050] |
| Hard-brake exposure (veh-s) | 182.500 | 319.500 | +137.000 [+9.000, +291.512] | -137.000 [-291.512, -9.000] |
| TTC<3s exposure (veh-s) | 5,390.000 | 10,287.000 | +4,897.000 [+2,667.000, +7,529.050] | -4,897.000 [-7,529.050, -2,667.000] |
| Collision involvements | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | -0.000 [-0.000, -0.000] |
| Starting teleports | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | -0.000 [-0.000, -0.000] |

Mean throughput over pooled branch time: native 2,267.08, policy 951.69 arrivals/hour; POLICY−B -1,315.385 [-1,567.869, -1,069.362]; B−POLICY +1,315.385 [+1,069.362, +1,567.869].
Zero collision/teleport bootstrap bounds reflect zero observed incidents in this cohort; they do not bound incident risk in a different cohort.
Equal-world sensitivity: mean total difference +47,613.603 veh-s/world (two starts); mean of within-world normalized differences +41.268 s/request. Each world contributes two starts, so equal-world and equal-start raw-cost weighting agree.

## Per world

All entries below are POLICY−B; gain B−POLICY is the exact negative. Each row pools both300s starts. Uncertainty is estimated across worlds above; a single world cannot support a world-cluster CI.

| Seed / box | CAV share | Demand (veh/h) | Δ total delay (veh-s) | Gain total (veh-s) | Δ on-road | Δ entrance | Δ arrivals | Δ hard-brake veh-s | Δ TTC veh-s | Δ collision / teleport |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 968129 / cpuP | 5% | 1900 | +28,239.513 | -28,239.513 | -1,818.487 | +30,058.000 | -107 | -9.500 | +45.500 | +0 / +0 |
| 968130 / cpuQ | 10% | 1900 | +31,114.297 | -31,114.297 | +31,097.297 | +17.000 | -205 | +38.500 | +1,203.500 | +0 / +0 |
| 968131 / cpuP | 20% | 1900 | +55,130.783 | -55,130.783 | +20,376.283 | +34,754.500 | -254 | +13.500 | +290.500 | +0 / +0 |
| 968132 / cpuQ | 40% | 1900 | +66,888.379 | -66,888.379 | +20,836.879 | +46,051.500 | -314 | +4.500 | +288.500 | +0 / +0 |
| 968133 / cpuP | 5% | 2100 | +27,909.156 | -27,909.156 | +16,331.156 | +11,578.000 | -131 | +61.000 | +965.000 | +0 / +0 |
| 968134 / cpuQ | 10% | 2100 | +34,008.514 | -34,008.514 | +14,279.514 | +19,729.000 | -165 | +27.500 | +666.500 | +0 / +0 |
| 968135 / cpuP | 20% | 2100 | +57,449.114 | -57,449.114 | +15,523.614 | +41,925.500 | -257 | -10.000 | +183.500 | +0 / +0 |
| 968136 / cpuQ | 40% | 2100 | +71,760.779 | -71,760.779 | +25,488.779 | +46,272.000 | -322 | -3.000 | +131.000 | +0 / +0 |
| 968137 / cpuP | 5% | 2400 | +39,042.270 | -39,042.270 | +7,164.770 | +31,877.500 | -158 | +19.000 | +191.500 | +0 / +0 |
| 968138 / cpuQ | 10% | 2400 | +38,053.765 | -38,053.765 | +6,417.265 | +31,636.500 | -172 | +1.500 | +256.500 | +0 / +0 |
| 968139 / cpuP | 20% | 2400 | +63,446.856 | -63,446.856 | +23,273.856 | +40,173.000 | -312 | +0.500 | +466.500 | +0 / +0 |
| 968140 / cpuQ | 40% | 2400 | +73,690.729 | -73,690.729 | +26,677.729 | +47,013.000 | -320 | -5.500 | +65.000 | +0 / +0 |
| 968141 / cpuP | 5% | 2700 | +32,242.681 | -32,242.681 | -2,056.319 | +34,299.000 | -133 | -1.000 | +143.500 | +0 / +0 |

## CAV-share and demand strata

Strata are marginal observed subsets, not a balanced factorial grid or causal share/demand effects. The2700veh/h stratum has one world; its world-cluster CI is not estimable. Other strata have only3–4 worlds; bootstrap bounds are exploratory.

| Stratum | Worlds / starts | Δ total delay (veh-s) [95% CI] | Gain total [95% CI] | Δ delay (s/request) [95% CI] | Δ arrivals | Δ hard-brake veh-s | Δ TTC veh-s |
|---|---:|---|---|---|---:|---:|---:|
| CAV 5% | 4 / 8 | +127,433.620 [+112,297.338, +145,366.322] | -127,433.620 [-145,366.322, -112,297.338] | +26.615 [+23.490, +30.228] | -529 | +69.500 | +1,345.500 |
| CAV 10% | 3 / 6 | +103,176.575 [+93,342.890, +114,161.296] | -103,176.575 [-114,161.296, -93,342.890] | +30.781 [+29.660, +31.652] | -542 | +67.500 | +2,126.500 |
| CAV 20% | 3 / 6 | +176,026.753 [+165,392.349, +190,340.568] | -176,026.753 [-190,340.568, -165,392.349] | +52.064 [+51.126, +53.839] | -823 | +4.000 | +940.500 |
| CAV 40% | 3 / 6 | +212,339.886 [+200,665.136, +221,072.186] | -212,339.886 [-221,072.186, -200,665.136] | +59.882 [+55.869, +62.396] | -956 | -4.000 | +484.500 |
| Demand 1900veh/h | 4 / 8 | +181,372.971 [+118,707.619, +244,038.323] | -181,372.971 [-244,038.323, -118,707.619] | +43.980 [+29.267, +58.215] | -880 | +47.000 | +1,828.000 |
| Demand 2100veh/h | 4 / 8 | +191,127.563 [+123,835.340, +258,419.787] | -191,127.563 [-258,419.787, -123,835.340] | +43.095 [+28.613, +56.896] | -875 | +75.500 | +1,946.000 |
| Demand 2400veh/h | 4 / 8 | +214,233.620 [+154,192.070, +274,275.170] | -214,233.620 [-274,275.170, -154,192.070] | +42.147 [+30.557, +53.569] | -962 | +15.500 | +979.500 |
| Demand 2700veh/h | 1 / 2 | +32,242.681 not estimable (one world) | -32,242.681 not estimable (one world) | +22.626 not estimable (one world) | -133 | -1.000 | +143.500 |

## Real-SUMO cooperation credit: ONLY / GROUP / FLIP

Recorded credit points: **26**; points with actual real-SUMO credit arms: **21**; unavailable: **5**. These have their own30s horizon, not the300s policy/native comparison. FLIP count is0 by registered configuration.
Definitions in gain units: D_i=J(B)−J(ONLY:i), D_G=J(B)−J(GROUP), C_G=D_G−Σ_iD_i; positive C_G means extra group benefit beyond summed solo gains. Cost differences are −D_i/−D_G, and the cost-interaction counterpart is −C_G. Do not substitute WM-imagined credit or ALL−B for uncollected ONLY/GROUP arms.

Conditional sample: 21 points from 11 worlds; 5,000 whole-world bootstrap draws, seed20261026. Ineligible points are not imputed as zero. Missing points: 968129@360, 968129@660, 968138@360, 968141@360, 968141@660.
- D_i_gain: equal-world mean -12.102 veh-s, 95% CI [-15.591, -8.519]; cost-sign counterpart +12.102 veh-s, 95% CI [+8.519, +15.591]. Average members within point for D_i, points within world, then worlds.
- D_G_gain: equal-world mean -29.363 veh-s, 95% CI [-40.513, -18.708]; cost-sign counterpart +29.363 veh-s, 95% CI [+18.708, +40.513]. Average members within point for D_i, points within world, then worlds.
- C_G_gain: equal-world mean +4.519 veh-s, 95% CI [+2.033, +7.519]; cost-sign counterpart -4.519 veh-s, 95% CI [-7.519, -2.033]. Average members within point for D_i, points within world, then worlds.
Positive C_G indicates less harm together than the sum of solo harms here; mean D_i and D_G remain negative gains versus native. It does not establish a beneficial group intervention.

| Seed @ time | Group size | D_i gain (member:veh-s) | D_G gain | C_G gain | D_G cost difference | C_G cost interaction |
|---|---:|---|---:|---:|---:|---:|
| 968130@360 | 2 | mainline.185:-9.682; mainline.183:-7.934 | -14.485 | +3.131 | +14.485 | -3.131 |
| 968130@660 | 2 | mainline.293:-24.001; ramp.64:-0.009 | -24.010 | +0.000 | +24.010 | -0.000 |
| 968131@360 | 3 | mainline.112:-6.848; ramp.38:-3.309; mainline.109:-3.434 | -9.418 | +4.173 | +9.418 | -4.173 |
| 968131@660 | 3 | mainline.269:-2.772; mainline.253:-30.903; ramp.58:+0.000 | -33.702 | -0.028 | +33.702 | +0.028 |
| 968132@360 | 3 | ramp.33:-26.646; mainline.140:-8.922; mainline.138:-3.798 | -40.172 | -0.806 | +40.172 | +0.806 |
| 968132@660 | 3 | ramp.58:-12.371; mainline.291:-4.209; ramp.57:-64.080 | -79.575 | +1.086 | +79.575 | -1.086 |
| 968133@360 | 2 | mainline.134:-3.847; mainline.127:+1.579 | +4.177 | +6.446 | -4.177 | -6.446 |
| 968133@660 | 3 | mainline.250:-2.429; mainline.234:-5.645; mainline.225:-7.210 | -8.330 | +6.954 | +8.330 | -6.954 |
| 968134@360 | 2 | mainline.175:-5.672; mainline.171:-6.887 | -11.087 | +1.472 | +11.087 | -1.472 |
| 968134@660 | 3 | mainline.287:-6.034; mainline.257:-9.869; ramp.85:-62.521 | -70.490 | +7.934 | +70.490 | -7.934 |
| 968135@360 | 3 | mainline.127:-2.115; ramp.37:-33.333; mainline.135:-5.615 | -38.004 | +3.059 | +38.004 | -3.059 |
| 968135@660 | 3 | mainline.286:-10.346; mainline.284:-11.976; mainline.281:-15.352 | -17.624 | +20.049 | +17.624 | -20.049 |
| 968136@360 | 3 | mainline.131:-7.818; ramp.34:-2.099; mainline.128:-26.687 | -36.109 | +0.495 | +36.109 | -0.495 |
| 968136@660 | 3 | ramp.66:-15.889; ramp.65:-37.361; ramp.63:-23.903 | -48.764 | +28.388 | +48.764 | -28.388 |
| 968137@360 | 2 | mainline.165:-20.412; mainline.162:-13.458 | -22.352 | +11.518 | +22.352 | -11.518 |
| 968137@660 | 2 | mainline.343:-10.145; mainline.340:-6.190 | -14.746 | +1.589 | +14.746 | -1.589 |
| 968138@660 | 3 | ramp.64:-0.004; ramp.62:-0.422; mainline.305:-7.583 | -7.459 | +0.550 | +7.459 | -0.550 |
| 968139@360 | 3 | mainline.132:-7.520; ramp.37:-5.582; mainline.129:-13.987 | -37.015 | -9.925 | +37.015 | +9.925 |
| 968139@660 | 3 | mainline.311:-2.924; mainline.310:-6.480; mainline.309:-8.866 | -6.650 | +11.619 | +6.650 | -11.619 |
| 968140@360 | 3 | mainline.145:-8.069; ramp.36:+1.916; mainline.126:-13.862 | -23.259 | -3.245 | +23.259 | +3.245 |
| 968140@660 | 3 | mainline.287:-4.658; mainline.297:-88.610; mainline.284:-10.588 | -99.442 | +4.413 | +99.442 | -4.413 |

## Provenance and validation

- Accepted/training WM: round16 FINAL SHA256 `5b2569418bf651b8ef2c3a2abdc110ac6860c32dd92dca8cdf3b2f48b1c8cbb2`.
- Policy: cycle0 `policy/c00.a2/policy_latest.pt`, SHA256 `fa2427772a49e9462c0be3172e402c515616a15e2930a263e1eafeacb23a0d5d`;150 updates, frozen update10 gate PASS. These training gates do not imply SUMO improvement.
- Current-policy returns `/root/main14_loop/returns/returns_c00.a1.npz`, SHA256 `6756b3cb8a5295e8385cec3fbca9f4b5c2e64924c8074673e2b54f7e86f82cfa`;26/26 returns exactly equal negative authoritative POLICY J.52 summaries/26 complete B–POLICY pair groups,13 unique worlds, matched branch-start backlog and requested populations.
- Pairs `/root/main14_loop/data/pairs_c00.a1`; collector artifacts under `/root/main14_loop/worlds/c00/w<seed>.a1/artifact`; ledger `/root/main14_loop/cycle_ledger.jsonl`. Hash-checked78 summary/credit/calibration files against per-world `HASHES.json`; all208 named numeric array files hashed. Source JSON preserves per-summary paths/hashes, all real-credit records, collection receipts and ledger snapshot.
- Pairs manifest SHA256 `2f1fc055e5fd3c0e3e5fa8d2bfc88072441a3239b0742548c840ccddc8fc2d45`. Local input SHA256 `dabb3de4334b80e57fad13d09cb0bf5d10ecc8c667091bedbff667d194f49e89`; safety/array-validation SHA256 `ed3b135bc54b6fc7b11c567eea00d6312685144f4f84e32cae2e94c41c6fa446`.
- Reproduce: `python tools/mainstart/cycle0_result.py`; machine-readable values/stratum metrics/point-level arm costs are in `tools/mainstart/mainfix/CYCLE0_RESULT.json`. No new SUMO, evaluation seeds, restricted paths, driver truth or assignment files were used.
- Existing accounting:87,399 charged /28,353 actual sim-s; prior58,266 charged startup reservations retained; total cap210,000. Completed world `pair_sim_seconds` already includes base and extra branches; do not add base `simulated_seconds` again.

## S-MAINFIX operational status

**WAIT_OWNER.** The driver recorded its5400s WM timeout at2026-10-26 18:59:03UTC+8; driver93301/trainer94892 exited under driver handling. Owner decisions were appended verbatim to the shared registration before changes. The local staging invocation then hit Windows WinError206 before SSH creation. Counting the timeout plus this invocation failure, the second non-scientific failure stop rule was applied; no source/config/state changes, retry or overlap launch occurred. New receipt: `restart/MAINFIX_WAIT_OWNER.json`; original `restart/WAIT_OWNER.json` retained.
The live timeout remains1.5h, WM speed flags remain disabled, overlap remainsOFF. Source integration exists for SPEED2 and the earlier FASTLOOP schedule, but that schedule joins GPU policy before WM-FT; concurrent WM/policy scheduling needs an owner-authorized, tested extension and exact owned-PID guard before activation. Explicit owner direction is required to repair staging and continue.
