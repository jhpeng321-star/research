# S-C1FLOOR — tiny-perturbation chaos reference (2026-10-31)
**Approved fallback (b) completed; an RNG-only divergence floor remains unavailable.**
Authority: owner S-C1FLOOR approval and MAIN17:25 fallback addendum; branch `c4/c1floor`.
## Fair comparison and intervention
COOPMEASURE's verified branching path was tried within the30-minute allowance: its original-seed restore also fails on969400 (900sim-s retained).
Every accepted reference reproduces all six recorded t/ID/position/speed/acceleration/lane arrays bitwise through t600.
Panel: first SIX registered C1CLOSED TRAIN/dev worlds969400–969405, native arm, t0=300s; two perturbation targets/world, same original SUMO seed.
Each draw reruns its complete300s trunk and matches its unperturbed t300 history/request/CAV signature before intervention.
Choose the two lowest-s HDVs with s<=-300m, stable-ID ties; each draw pulses one, without target replacement.
`TraCI.setPreviousSpeed(id,v+0.01,a)` changes instantaneous speed only, preserving acceleration and position; next .5s tick is native.
All12 API readbacks verify +0.01m/s to1e-10; other present vehicle speeds are unchanged at intervention; no persistent speed override.
Both target IDs are excluded from endpoint scoring in BOTH draws; aggregate fields/delay include all traffic and the direct pulse.
An earlier `setSpeed(t0_speed+0.01)` implementation had incorrect realized amplitude; ALL its draws are quarantined, not pooled.
## Horizon results and interpretation
Horizons10/30/60/90/120/180/300s; original-HDVs/common survivors; median within draw, over two draws, then across worlds.
Across6worlds, world-median position/speed errors are exactly0 through120s: sparse local perturbations need not move the median HDV.
At180s position3.73e-9m [0,0.01596], speed0m/s [0,0.000596]; only4worlds retain matched HDVs.
At300s NO original HDVs survive jointly: endpoint estimates/CIs are BLANK, n_worlds0, never imputed0.
Team-delay absolute-error medians at60/120/180/300s:0.06396/0.5193/2.706/55.61veh-s (all6worlds).
At300s team-delay95%CI[0.0000382,421.18]veh-s; density MAE1.1075veh/km road; speed-field MAE0.24425m/s; merge-queue AE4.25veh.
These are amplitude/target/arm-dependent chaos sensitivities, NOT an irreducible RNG lower bound or matched PI/MAPPO reference.
Six reused development worlds and late survivor loss limit inference; no fitting or tuning used these outcomes.
## Shared scorer, accounting and delivery
Import pinned C1CLOSED forecast/endpoint/errors/CSV and byte-identical label/field scorers; bootstrap changes ONLY panel dimension40→6.
Whole-world95% percentile bootstrap:2000resamples, seed0, both draws clustered; signed biases retain upstream mean convention.
`C1FLOOR_error_horizon.csv` has the exact14 C1CLOSED columns, all7horizons; per-world/per-draw CSVs accompany it.
Prior blocked publication preserved in `tools/c1floor/evidence/blocked_publication/`; `c1floor_pulse/` retains generated exports.
Cumulative ACTUAL21900sim-s = prior2400 + fallback(a)900 + quarantined11400 + accepted-pulse7200, including EVERY prefix.
Conservative precharges29700/30000, no refunds;300unallocated. All old failures, STOPs, quarantined data and ledgers retained.
Frozen E3 sources543/543 verified; both seed guards reused/rechecked; original exposure audit retained, no new world/RNG ranges.
One nice19 single-thread gpuR CPU worker, CUDA hidden, quota/load admission yielding to SUMOCF/COOPMEASURE; all owned launch PIDs exited.
CPU checks:10inherited mechanics+2request+2exact-pulse+4scientific contracts PASS; no peer signals, deletions or push.
Evidence: `tools/c1floor/evidence/fallback_1793439838485955000/pulse/{COMPLETE,EFFECT_AUDIT}.json`; accepted raw tapes gpuR `/root/c1floor20261031/jobs/pulse_s*`.
Canary commit`b02d364f`; run/scorer/receipts commit`1e1e4e96`; full prospective correction and retained costs in `FLOOR_DESIGN.md`.
