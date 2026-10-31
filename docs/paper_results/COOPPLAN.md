# S-COOPPLAN — exact-model planner built; robust cooperation not established
Completed 2026-11-01 on c4/coopplan; frozen design SHA256 `53b0edf4ea8e6fc72165691727e923505ed6a805af9c0294efd63f2ab3389a5f`.
## Main finding
The pluggable cooperative planner runs end-to-end, executes useful group actions, and produces release-supported gap/yield behaviour.
It improves paired J600 on this fixed development panel, but J900 remains uncertain and mean sampled C_G is NEGATIVE.
Therefore this is a working perfect-model planning benchmark, **not a demonstrated robust superadditive-cooperation success**.
## Frozen experiment
All12 fresh development worlds983100–983111 completed, one per .05/.10/.20/.40 ×1900/2400/2700 cell; never used for fitting.
SUMO1.27.1/full-rerun oracle K1 vs pure frozen speed-only MAPPO169; identical world/arrivals/noise settings.
H120, cadence60, decisions t300..960, then MAPPO-only after1020 through1200; warmup300 excluded from scores.
O0..O4 inherited options/geometry/safety; COOPFREE had no learned checkpoint at freeze, so its reserved slot remained disabled.
Selection: minimum expected delay among strict paired P(beats MAPPO)>=.8 AND lower-mean candidates; exact K1 means strict improvement.
## Paired full-team delay
Saving = MAPPO minus planner (veh-s/world); full-team J includes road deficit plus source waiting.
J600: **+396.280**, 95% whole-world bootstrap CI **[+0.894,+1128.773]**.
J900: **+1195.148**, CI **[-568.178,+3724.114]**.
Registered10,000 percentile resamples, RNG20261101, world is the unit; 12 heterogeneous worlds, no episode/decision pseudoreplication.
The plotted world contrasts show concentration of benefit; J900 includes harmful worlds despite a positive pooled point estimate.
## Cooperation and executed decisions
144 decisions: MAPPO125; coordinated slow O1 five; yield gap O3 four; longer gap O4 ten; cross-lane wall O2 zero.
Nineteen executed group actions occur in six worlds; first executed group in each of those six received two exact member-alone reruns.
All sampled group savings D_G>0 by the oracle gate, but this does not imply superadditivity.
C_G=(J0-JG)-sum_i(J0-J_i), evaluated at H120 with fixed joint targets in the solos.
Mean sampled C_G **-38.379 veh-s**, world-bootstrap CI **[-94.759,-1.022]**, conditional on the six sampled worlds.
Positive C_G in **2/6** samples:983108/t600/O4 **+0.198734** (D_G+0.659999),983111/t300/O4 **+0.014360** (D_G+141.582352).
These are small positive examples; the other samples are -6.743784,-172.412662,-0.000012642,-51.333025. None excluded or replaced.
Unchanged COOPBEHAV on same-prefix sampled joint-vs-MAPPO traces: **1 paired-policy-mean-supported yield**, **18 supported gap events**,30 supported wave-damping events.
Whole planner tapes also contain25 observational yield,1404 gap,130 cross-lane-hold and2862 wave-damping candidates.
Those observational counts are not causal cooperation counts; no wall option was executed, and observed walls are not a wall-action success.
## Safety and mechanics
Across the12 scored900s episodes per arm: collisions0/0 and teleports0/0 (MAPPO/planner).
TTC<3 exposure7004.5/6623.0 veh-s; hard braking220.0/222.0 veh-s; these are descriptive totals, not a universal safety-improvement claim.
Planner intervention safety clamped587/1278 requested member-command records; original safety layer/native lane logic unchanged.
All247 full-prefix checks PASS (245 scientific oracle branches plus2 mechanics branches); first production baseline also matched the full1200s mechanics repeat bitwise.
Frozen runtime CPU tests16/16 on EACH host, real S-RB-DECIDE parity; current local CPU/analysis tests20/20; extra FREE adapter CPU ABI PASS with synthetic weights/SUMO blocked.
## Accounting, interface and artifacts
Actual=charged **220,980 sim-s**: cpuQ51,540 +gpuA94,200 +gpuR75,240; no failed/incomplete attempts or automatic physics retries.
Registered upper704,760<=800,000; actual483,780 below it through candidate pruning and fewer/earlier credit samples. Approved cap unspent579,020; none reallocated.
Same `Planner` accepts `SumoOraclePredictor` or `WMPhysPredictor`; WM stub requires RB-INT rollout plus paired hidden-driver/arrival future builder.
WM callback absence is explicit. CompleteDraw supports identity-bound tensor payloads; WM calibration/live efficacy remains untested here.
Code: `tools/coopplan/`; integration/execution handoff: `HANDOFF_COOPPLAN.md`; design: `COOPPLAN_DESIGN.md` (63lines).
Receipts/identity hashes/decisions/member effects: `coopplan/final/RECEIPTS.json`; paired CSV, SUMMARY/CREDITS/BEHAVIOURS JSON in the same directory.
Figures: [paired delay](coopplan/final/paired_delay.png), [sampled cooperation credit](coopplan/final/cooperation_credit.png).
No training, prohibited truth/assignment/path reads, new venv, deletion, push or peer PID actions; nice19 CPU-only single-thread execution on shared hosts.
