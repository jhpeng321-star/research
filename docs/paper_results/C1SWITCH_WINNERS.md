# C1SWITCH scenario winners — recorded panel complete; branching stopped
**Status (2026-10-31):** Step0a COMPLETE; Step0b MECHANICALLY STOPPED, 0/120 complete decision states. Main switching NOT STARTED. This is not a scientific no-headroom verdict.

## Recorded fixed-arm evidence (zero new SUMO)
- All 40 WMDATA worlds969400–969439 and 50 WMDATA2 worlds969550–969599, common five controllers only; generic-random excluded from this library comparison.
- 1,350 cumulative source receipts = 90 paired worlds x5controllers x3clocks; five-way initial-state SHA equality checked for every world/clock. Source byte hashes preserved in `tools/c1switch/evidence/RECORDED.json`.
- Outcome: minimum cumulative fleet+source team delay (veh-s), after300s native warmup, measured at120/300/600s of control; ties <=1e-6 split fractional shares in figures and remain explicitly tied in CSV.
- **At600s unique world wins: native22, CTRL0 17, MAPPO-speed19, PI29, CIRCLES3; no ties.** Every library controller wins somewhere; no recorded controller wins nearly everywhere.
- At120s: native14, CTRL0 37, MAPPO17, PI14, CIRCLES6, plus2four-way ties. At300s: native22, CTRL0 24, MAPPO20, PI21, CIRCLES3; no ties.
- Pooled best fixed by mean: CTRL0 at120s (8,168.60veh-s); PI at300s (24,809.42) and600s (62,422.09). Median best-to-second margin56.84/323.43/808.32veh-s at120/300/600s.
- WMDATA600s wins native7/CTRL0 7/MAPPO11/PI14/CIRCLES1; WMDATA2 native15/CTRL0 10/MAPPO8/PI15/CIRCLES2. Different demand-cell weights make pooling descriptive rather than a balanced population estimate.
- Cell-level changes are visible: in WMDATA600s, PI wins3/4 low-penetration(.05)/2700 worlds; MAPPO2/4 at(.4,1900); CTRL0 2/4 at(.4,2700); native2/3 at(.05,2400); CIRCLES1/3 at(.2,1900).
- These are fixed-episode outcomes, not same-state interventions. This panel establishes scenario diversity in recorded worlds; it cannot establish switching benefit, state-dependent predictability or a noise-resolved60s opportunity.

## Paper figure and machine-readable outputs
- [Winner by traffic condition, PNG](c1switch/recorded_winner_by_condition.png) and [vector PDF](c1switch/recorded_winner_by_condition.pdf): two datasets x3clocks; all12cells, cell sample sizes, exact controller legend, tie handling and units.
- [Per-world winners/costs/margins](c1switch/recorded_world_winners.csv):270rows, including all five costs. [Per-cell/controller win shares](c1switch/recorded_cell_winshares.csv):360rows. [Summary](c1switch/recorded_summary.json).
- Figure visually reviewed: labels, caption, legend and all bars visible. TRAIN/descriptive; small-n cells identified; no inferential CI or favourable-controller filtering.
- Reproduce: `python -m tools.c1switch.winners`; inputs exported once by `python -m tools.c1switch.ops recorded` from permitted cumulative receipts only. No source arrays/model training/GPU inference.

## Fresh-state mechanics and stop
- Proposed block969100–969135 passed both frozen seed guards36/36, exact registry-range audit, C1GAP corpus-ID audit and safe-name audits on gpuR/gpuA. Step0 first12, main last24; no seed replacement.
- Native live90/restored90 on969100 passed complete tape+metric bitwise equality, J=2,941.159261686626veh-s;300warmup+180continuation=480charged=actual. Engine includes the existing precision17/RNG/loaderTime fixes.
- Step0 then started only969100 (native trunk). Native90 and CTRL0 first60 ran after300warmup; at CTRL0's60s snapshot, `copy.deepcopy(ZeroResidual)` recursively delegated `__getattr__` before its executor attribute existed. No complete five-way decision, regret, rank, win share, observable-condition association or noise comparison was emitted.
- Durable STOP retained. Own dispatcher524236 and canary523094 are gone. No second world, retry, GPU process, main arm or model choice. Source/error/partial saved states retained on gpuR `/root/c1switch20261031`.
- Reconciliation: canary480 + failed-world450 = **930charged =930actual sim-s**,7precharged segments; GPU0hours. No refund. [Exact receipt](../../tools/c1switch/evidence/RECONCILIATION.json) is also at `tools/c1switch/evidence/RECONCILIATION.json`.
- Repair retains the raw frozen C1 executor/memory in snapshots and constructs the stateless ZeroResidual typed-output view only at command time. The frozen zero.py and executor weights are unchanged.
- CPU repair:26local tests PASS,9task tests on gpuR PASS, plus real native/CTRL0/MAPPO/PI/CIRCLES controller deepcopy checks PASS with SUMO start/step intercepted. Separate versioned repair source; old deployed source/STOP untouched. Receipt `tools/c1switch/evidence/CPU_REPAIR.json`.
- Full12-world state-dependent headroom/noise gate is **UNRESOLVED**. Do not interpret zero completed decisions as zero controller diversity or absent benefit. Never treat deterministic native repeats as an independent RNG divergence floor.

## Required continuation gates
- MAIN must acknowledge the mechanical stop and prospectively authorise a new attempt/version before dispatch. A complete12-world rerun would cost58,680; including retained930, Step0 total59,610<=60,000 and full planned total498,810<=500,000 (reserve1,190), but that amended allocation is NOT dispatched/claimed approved here.
- Existing native restore PASS need not be repeated. Preserve failed attempt, original ledger and source; bind repair hash/attempt IDs and both-world prefix/pairing checks before resumption. Recheck resource priority; one nice19CPU, no peer PID operations.
- Before main switching: complete Step0/noise/headroom gate; publish C1PIDIAG diagnosis and select eligible WM by frozen rule; verify observable-source/entries adapters for WM and frozen BB-traj and GPU-memory admission. RPC infrastructure is built; real-model adapter is currently unbound and unvalidated.
- Original designSHA `c4154bbe1a9281d61eb465de0e64123f232151b60dcf5a10a2fbdde30377967a` and498,360sim-s plan remain the registered baseline; owner approval already exists, so waiting for a second copy of the budget-approval phrase is unnecessary.
