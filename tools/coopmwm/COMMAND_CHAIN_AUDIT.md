# MAIN07:18 command-chain audit — activation is not qualified

## Existing completed evidence to adopt

- `COMMAND_CHAIN_NULL_FIXTURES.json`: original TRAIN requests, proposal receipts,
  and original/without-plan observable rows for worlds986051/986070 at t300.
- Both original K8/H420 forecasts report saving_vs_policy **exactly0** and
  centre_open_loop_gap **exactly0**. Their +164.137/+26.225 figures are versus
  whole-road MAPPO, a different contrast. Both top-k pools contain only
  zero-own-policy-gain finalists. The exact marginal zeros are consistent.
- `COMMAND_CHAIN_SOURCE_{cpuQ,gpuA}.json` and `COMMAND_CHAIN_LAWS_gpuA.json`
  pin public runtime source only. The WM already has the SUMO IDM/allowed-speed
  `alwayscmd.inferred_cap` ceiling. Do not add a second independent IDM ceiling.
- `CPU_coopmwm20261102_commandcpu3.json`:48PASS/zero skips/5.67s/zero SUMO.
  CPU1/2 failures are preserved: harness environment and absent newer GPU
  runtime in an unrelated cross-runtime test, not failed plant episodes.

## Prototype

`command_chain.ownership_target` mirrors actual `damping.direct_target` float64
arithmetic, independently tested through the real `CommandOwnership` writer.
`SlowDown` retains the one-second target timeline, including the odd half-step
and odd-tick new-CAV native fallback. The existing speedMode27 ceiling still
follows the timeline on every half-step. Bounds are frozen plant interface
constants, not behavioural selection cutoffs.

`plan_wm.evaluate(..., final_chain=False)` is deliberately OFF by default.
Neither the running GPU services nor any training binding enables it.

## Probe interpretation

`COMMAND_CHAIN_NULL_CALLBACK_PROBE.json` executes the actual frozen WM callback
on CPU on the GPU-A host. It does not bootstrap a WM or use GPU memory. For both
old and prototype chains, plan-versus-own-policy motion is identical on every
active callback. Absolute SUMO transition residuals remain up to0.208864 and
0.055090m/s. The one-second tape requires interpolated half-step neighbours;
the group resets to the observed state at each one-second decision. This is a
teacher-forced transition replay, NOT continuous full-road exact parity.

Do not relabel this as GPU parity or activate based on CPU writer tests alone.
Further numerical release needs genuine half-step sensed records, matching
motion/error accounting, source-bound GPU full/fast and batched/single gates,
and a registered boundary handover that adopts all assigned searches.
Do not rerun existing episodes/searches or spend unallocated budget implicitly.

## Live descriptive reporting

`noop_ops status` observes reporter3746050/start59055744 (`nullreport1`).
It checks every qualified same-B/seed TRAIN exact-zero original/without-plan
tape, keeps zeros/losses, writes immutable `physical_nulls/` classifications,
and publishes counts by update in `B_EXACT_VIABILITY_*.json`. These are reports
only; no optimizer, trust, grading, library, quota or acceptance changes.

Changing the positive-vs-MAPPO admission rule to require positive incremental
own-policy gain is a scientific-scope question recorded for MAIN/owner, not an
unregistered engineering fix. The later MAIN22:39 rule remains in force.
