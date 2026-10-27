# E5 — matched cycle-1 cooperation-loss ablation

**Complete: 228/228 paired TRAIN/dev worlds;** the two policies have **exactly equal recorded outcomes in every world**. ON − OFF total delay at 120/300/600 s: **0 [0, 0] veh-s/world at each clock**.

## Panel, provenance and accounting

- Exact worlds 973000–973227; 19 in each of 12 CAV-share×demand cells: shares 5/10/20/40%; demands 1900/2400/2700 veh/h. All four arms share each world’s observable native initial-state SHA256. Native warmup 300 s; intervention clocks 120/300/600 s.
- coopON cycle-1 checkpoint `64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c`; coopOFF cycle-1 `e1fa4881bbc5c89d05298e979d98b23284308331ad1bbb12f9608fa67e105675`; MAPPO `aecd7d70`; both original native-relative gates pass. Frozen scientific source manifest `b3c031ad` (543 files) verified on both hosts.
- MAIN completion amendment was already registered; exactly six worlds 973217/219/221/223/225/227 completed once on cpuQ. All receipt-bound simulation predecessors exited. This resumed analysis dispatched zero simulations.
- E5 coopOFF: **210,600 charged / 205,200 actual sim-s** (cpuP 102,600/102,600; cpuQ 108,000/102,600). Six historical pre-SUMO failures and their 5,400 charges remain; each has only its previously authorized repair. Completion consumed exactly +5,400 charged/actual. Policy phase 415,800 charged / 410,400 actual of 615,600; 199,800 unused. Baseline+policy production 826,200 charged / 820,800 actual of 1,026,000.
- Complete result files, unique attempt charges, immutable input/source hashes, all standalone cumulative clocks, full-population/source/group conservation verified; maximum conservation residual 0.000132113688 veh-s. No complete-case selection.

## Estimands and uncertainty

**Cost convention:** treatment − reference, negative = less delay. Total delay = on-road speed-deficit vehicle-seconds + entrance/source waiting vehicle-seconds. **Saved-delay convention:** reference − treatment, positive = better for treatment. These are exact sign reversals, with CI endpoints reversed. Means weight the registered worlds equally. All intervals use the same fixed 5,000 whole-world bootstrap draws, resampling 19 worlds within each original cell, seed 20261027; arms, clocks and metrics remain paired. Cell/secondary intervals are pointwise, without multiplicity adjustment. The matched world panel is not replicated policy training: it compares these two cycle-1 checkpoints, rather than estimating a general training-seed ablation effect.

## Exact-tie provenance check

All **228 ON/OFF observable trajectory NPZ files are byte-identical**, and their complete measurement dictionaries are exactly equal. This is an observed deterministic tie, not merely a nonsignificant mean or an equivalence test. A zero-SUMO audit checked every deployment summary: the correct, distinct checkpoint paths were used, each for 600 policy ticks. Both stored checkpoints are update180 native-residual policies; their residual-head weights/bias differ (maximum tensor difference 0.0017030201), while all 26 group-encoder tensors match. Independent CPU loading reproduced each checkpoint’s stored policy tensors exactly, with deterministic execution, no hold and no added exploration noise. Observable requested and issued acceleration ledgers differ in **all 228 worlds**, with maximum ON/OFF difference **0.004012585 m/s²**; lane-command ledgers match. Thus the stored commands differ but recorded realised trajectories do not. The precise executor/plant mechanism producing the tie has not been isolated by this analysis. Do not infer that the cooperation objective is generally ineffective, or attribute either policy’s native-relative gain to cooperation. Per-world evidence: `IDENTITY_AUDIT.json` and `TELEMETRY_AUDIT.json` on both hosts.

## Total delay, all worlds

- **120 s:** ON − OFF +0.0 [+0.0, +0.0] veh-s; OFF − ON saved -0.0 [-0.0, +0.0] veh-s. Road contribution +0.0 [+0.0, +0.0]; entrance contribution +0.0 [+0.0, +0.0] veh-s.
- **300 s:** ON − OFF +0.0 [+0.0, +0.0] veh-s; OFF − ON saved -0.0 [-0.0, +0.0] veh-s. Road contribution +0.0 [+0.0, +0.0]; entrance contribution +0.0 [+0.0, +0.0] veh-s.
- **600 s:** ON − OFF +0.0 [+0.0, +0.0] veh-s; OFF − ON saved -0.0 [-0.0, +0.0] veh-s. Road contribution +0.0 [+0.0, +0.0]; entrance contribution +0.0 [+0.0, +0.0] veh-s.

Comparisons against native and MAPPO (cost convention; veh-s/world):

- **120 s:** ON − native -67.2 [-100.5, -34.6]; OFF − native -67.2 [-100.5, -34.6]; ON − MAPPO -0.7 [-23.0, +22.4]; OFF − MAPPO -0.7 [-23.0, +22.4].
- **300 s:** ON − native -323.2 [-505.5, -135.1]; OFF − native -323.2 [-505.5, -135.1]; ON − MAPPO +162.6 [-8.0, +333.6]; OFF − MAPPO +162.6 [-8.0, +333.6].
- **600 s:** ON − native -970.0 [-1,491.7, -446.3]; OFF − native -970.0 [-1,491.7, -446.3]; ON − MAPPO +938.7 [+454.9, +1,430.4]; OFF − MAPPO +938.7 [+454.9, +1,430.4].

## CAV share × demand at 600 s

ON − OFF total-delay cost, veh-s/world (n=19 each); all clocks, all comparator contrasts, marginal strata and reverse saved-delay signs are in the CSV.

- **5% CAV:** 1900 veh/h: +0.0 [+0.0, +0.0]; 2400 veh/h: +0.0 [+0.0, +0.0]; 2700 veh/h: +0.0 [+0.0, +0.0].
- **10% CAV:** 1900 veh/h: +0.0 [+0.0, +0.0]; 2400 veh/h: +0.0 [+0.0, +0.0]; 2700 veh/h: +0.0 [+0.0, +0.0].
- **20% CAV:** 1900 veh/h: +0.0 [+0.0, +0.0]; 2400 veh/h: +0.0 [+0.0, +0.0]; 2700 veh/h: +0.0 [+0.0, +0.0].
- **40% CAV:** 1900 veh/h: +0.0 [+0.0, +0.0]; 2400 veh/h: +0.0 [+0.0, +0.0]; 2700 veh/h: +0.0 [+0.0, +0.0].

## Throughput and safety

Absolute all-world means [95% CI] at 600 s; vph is arrivals × 3600 / intervention duration. Hard braking and TTC<3 use vehicle-seconds, not event counts.

- **coopON:** throughput +2,263.7 [+2,253.9, +2,273.2] veh/h; hard braking +15.0 [+13.9, +16.1] veh-s; TTC<3 +395.2 [+378.9, +411.8] veh-s.
- **coopOFF:** throughput +2,263.7 [+2,253.9, +2,273.2] veh/h; hard braking +15.0 [+13.9, +16.1] veh-s; TTC<3 +395.2 [+378.9, +411.8] veh-s.
- **native:** throughput +2,247.5 [+2,236.6, +2,258.7] veh/h; hard braking +15.8 [+14.7, +17.0] veh-s; TTC<3 +396.7 [+379.9, +413.8] veh-s.
- **mappo:** throughput +2,283.7 [+2,274.4, +2,292.8] veh/h; hard braking +12.7 [+11.8, +13.6] veh-s; TTC<3 +350.9 [+333.8, +368.3] veh-s.
- **ON − OFF, 120 s:** throughput +0.0 [+0.0, +0.0] veh/h (positive = more discharge); hard braking +0.0 [+0.0, +0.0] veh-s; TTC<3 +0.0 [+0.0, +0.0] veh-s (negative = less exposure).
- **ON − OFF, 300 s:** throughput +0.0 [+0.0, +0.0] veh/h (positive = more discharge); hard braking +0.0 [+0.0, +0.0] veh-s; TTC<3 +0.0 [+0.0, +0.0] veh-s (negative = less exposure).
- **ON − OFF, 600 s:** throughput +0.0 [+0.0, +0.0] veh/h (positive = more discharge); hard braking +0.0 [+0.0, +0.0] veh-s; TTC<3 +0.0 [+0.0, +0.0] veh-s (negative = less exposure).
- Across 912 arm-worlds at the full clock: collisions {'coopON': 0, 'coopOFF': 0, 'native': 0, 'mappo': 0}; teleports {'coopON': 0, 'coopOFF': 0, 'native': 0, 'mappo': 0}; dropped vehicle steps {'coopON': 0, 'coopOFF': 0, 'native': 0, 'mappo': 0}.

## Frozen COOPMEAS observable behaviours

Detector SHA256 `452d44a78a1c1280aa0fd6f3d9f58a7498d35432ac4158706ba40a922a04c9f4`; thresholds `9119cd77d27889c9546cfaacd91af5a3f7f89beda3a5b6616e988aa67860701f`, unchanged. Rates pool event counts / observed modelled-lane CAV-hours, rather than averaging per-world rates. Paired bootstrap resamples each world’s counts and arm-specific exposure together. Side-by-side is a subset of multi-CAV slowing; lane-vacate counts all registered 1→0 moves at −400..0 m, with the ramp-present courtesy subset additionally reported. Event starts and exposure are restricted to [300,300+h); each detector receives only that prefix. Saved tapes begin at 300.5 s (first 1 Hz frame 301 s), so detector rolling features warm up inside the intervention: pre-intervention behaviour history and events already underway at the boundary are unobserved. This common left-censoring is retained, not filled in.

Full 600 s pooled count / CAV-hour and rate [95% CI]:

- **Multi-CAV slowing:** coopON 72/592.111 h = +0.12 [+0.09, +0.15]; coopOFF 72/592.111 h = +0.12 [+0.09, +0.15]; native 278/610.046 h = +0.46 [+0.40, +0.51]; mappo 384/591.864 h = +0.65 [+0.58, +0.72] events/CAV-hour. ON − OFF +0.00 [+0.00, +0.00] events/CAV-hour.
- **Side-by-side:** coopON 13/592.111 h = +0.02 [+0.01, +0.03]; coopOFF 13/592.111 h = +0.02 [+0.01, +0.03]; native 53/610.046 h = +0.09 [+0.06, +0.11]; mappo 114/591.864 h = +0.19 [+0.16, +0.23] events/CAV-hour. ON − OFF +0.00 [+0.00, +0.00] events/CAV-hour.
- **Upstream metering:** coopON 319/592.111 h = +0.54 [+0.48, +0.60]; coopOFF 319/592.111 h = +0.54 [+0.48, +0.60]; native 910/610.046 h = +1.49 [+1.40, +1.58]; mappo 2303/591.864 h = +3.89 [+3.72, +4.07] events/CAV-hour. ON − OFF +0.00 [+0.00, +0.00] events/CAV-hour.
- **Lane vacate (all):** coopON 1180/592.111 h = +1.99 [+1.86, +2.12]; coopOFF 1180/592.111 h = +1.99 [+1.86, +2.12]; native 1218/610.046 h = +2.00 [+1.89, +2.10]; mappo 396/591.864 h = +0.67 [+0.61, +0.73] events/CAV-hour. ON − OFF +0.00 [+0.00, +0.00] events/CAV-hour.
- **Lane vacate (ramp-present courtesy subset):** coopON 866/592.111 h = +1.46 [+1.35, +1.58]; coopOFF 866/592.111 h = +1.46 [+1.35, +1.58]; native 891/610.046 h = +1.46 [+1.36, +1.56]; mappo 262/591.864 h = +0.44 [+0.39, +0.49] events/CAV-hour. ON − OFF +0.00 [+0.00, +0.00] events/CAV-hour.

These are whole-policy trajectory associations. They do not isolate causal cooperation, nor establish that any detected behaviour caused the delay contrast.

## Reproduction and evidence

- `tools/autonomy_e5/analyze_shard.py`: receipt-bound zero-SUMO validation and frozen detector, one nice19/single-thread detached process per CPU host; `analysis_ops.py` stages/polls/fetches without dispatching simulations.
- `python -B -m tools.autonomy_e5.summarize` regenerates estimates from `tools/autonomy_e5/receipts_done/{cpuP,cpuQ}/ANALYSIS_DATA.json`. Source trajectories remain at their recorded CPU-host paths; every accessed input is hashed in these receipts.
- `E5_COOPOFF_C1.csv`: full absolute physical/rate estimates, paired contrasts and saved-delay signs at all clocks and strata. `E5_COOPOFF_C1_worlds.csv`: all 2,736 arm×world×clock rows, physical metrics, observable detector counts/exposures and pairing hashes. `RESULTS.json` and `MANIFEST.json` preserve exact estimates and artifact hashes. Report/CSV copies are committed under `tools/autonomy_e5/receipts_done/`.
