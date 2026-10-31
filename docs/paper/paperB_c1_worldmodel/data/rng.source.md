# S-ENSPROBE — conditional SUMO RNG ensemble

## What was run
- Five fresh TRAIN native-warmup states at t=300 s; paired native/speed-only MAPPO169, K8 seeds each, 600 s: 80 branches / 10 state-arm cells.
- Worlds 969040–969044; cells (.05,2400), (.10,2700), (.20,2400), (.40,2700), (.20,2700); continuation RNG 969086–969093. Two ordinary-share and three dense/high-share states; same physical starts across arms.
- Original plan: 49,500 sim-s; cache-gate recovery adds 300 s native prefix; owned scheduling stop retains 69 partial sim-s. **49,869 charged = 49,869 actual sim-s**; unused 131 never allocated.
- Frozen cpuQ e3final manifest b3c031ad, SUMO 1.27.1, textbook population; both guards and registry/filename exposure audit passed. Requests/routes/driver population stay fixed; only SUMO RNG changes.

## Physical restart validity
- 80/80 starts pass: max lane-position error 0 m, max speed error 0 m/s; xy/lanes/routes/route indices/due waiting objects/source queues/lane queues verified.
- Residual: getPendingVehicles() insertion-attempt cache differs at 80/80 loads. First attempt stopped at 0 ticks; zero-step diagnosis found 36/36 due vehicles still loaded/off-road/not departed with correct routes. Amended direct-object queue gate is pre-outcome and published.
- Repeated first warmup reproduces original physical trunk exactly. save-state.rng=false and each requested new --seed verified. CF/LC internal memory need not reproduce the un-restored trunk continuation: this is a physical-state-conditional restored ensemble.
- A second 0-tick stop exposed deepcopy memo reuse. Fresh memos fix it; first-state bookkeeping was reconstructed at 0 ticks from the same public t0 state/due waiters, with exact retained prehistory and native 600 s result. MAPPO uses the frozen current-frame state API and fresh controller memory; four later worlds retain full native Python histories. No completed physics rerun or result-based selection.
- Spare-CPU admission was corrected after an owned cooperative scheduling stop; only its incomplete 69 s prefix reran and all ticks remain charged. One nice-19 thread, load>20/own-YIELD pause; priority peers untouched.

## Distribution results
- Maximum within-state/arm 600 s team-delay SD = **0 veh-s**; maximum min–max range = **0 veh-s**. All eight half-second physical-tape hashes are identical within each of the ten state/arm cells.
- Same-state sometimes-breaks/sometimes-not: **0/10 state-arm cells** under the unchanged LRDIAG BIC detector (600 s upstream); onset and preexisting flags per seed in CSV. No speed cutoff or behavioral threshold fitted.
- Native has post-t0 BIC events in all five states (onsets 80/280/190/510/90 s in seed order), identical in all K8 futures. MAPPO has post-t0 events in 969041/969044 (40/520 s); 969043 receives a pre-t0 fit, and 969040/969042 no event. These are empirical detector selections, not validated physical onset labels.
- At 600 s, pooled within-seed variance native/MAPPO = 0/0 (veh-s)^2; between-state SD = 12,555.4/13,730.5 veh-s (five states only, descriptive; full variance estimates in CSV).
- Paired MAPPO-minus-native 600 s state-mean effects span -7,442.102 to +8,775.168 veh-s; mean across five states +76.490 veh-s. Largest within-state SD of that paired effect = 0 veh-s. Positive means MAPPO worse.
- All 120/300/600 s fleet/source/team means, sample SDs, minima/maxima, paired spreads and variance components are in the CSVs. Event timing uses full on-road mapped lanes, 10 s half-open vehicle-time bins and 60 s prehistory; missing/preexisting distinct.

## Plain-language answer and scope
- **No same-state mixed breakdown outcome was observed:** every tested RNG seed reproduced the same trajectory for its state/controller. Controller effects vary across states and are much larger than the observed zero RNG variation.
- This ensemble changes simulator RNG, not the future demand/driver realization. It cannot by itself measure the real-world metastable distribution conditional on an observable traffic state.
- The code audit identifies randomness in Python-generated departures, CAV identities and heterogeneous IDM/LC parameters, fixed before the continuation. IDM has no dawdling noise; sigma=0, speedDev=0, lcSigma defaults0; insertion best/max and fixed routes add no identified process noise.
- A degenerate seed ensemble therefore indicates insufficient configured continuation stochasticity, not proof that real traffic is deterministic or that a WM should have no uncertainty. Observable-state uncertainty and fresh future arrivals/drivers require a separately registered ensemble design.
- Complete audit and version-pinned documentation: [ENSPROBE_RANDOMNESS.md](ENSPROBE_RANDOMNESS.md). No assignment/truth/forbidden paths inspected; realized type-specific quantities remain undetermined.

## Artifacts
- [Figure: K8 delay distributions and controller effects](ensprobe/ENSPROBE.png).
- CSVs: ensprobe/{per_seed_delay,state_spread,breakdown_per_seed,breakdown_fraction,paired_arm_spread,variance_components,physical_restore}.csv.
- Provenance: tools/ensprobe/evidence/{collection,v4_stage,v4_cpu,queue_diagnosis,waiting_diagnosis}.json; ensprobe/summary.json includes exact detector source/hash. Old STOPs and all original attempts retained on cpuQ.
