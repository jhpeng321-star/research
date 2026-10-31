# S-SUMOCF — registered SUMO-oracle counterfactual learner

Authority: REGISTRATION.md 2026-10-31 16:00; branch c4/sumocf from 826da8ed.
Scientific goal: improve speed-only MAPPO and measure cooperation in real SUMO.

## Policy and intervention
- Load the exact MAPPO iter169 export d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca.
  Frozen E3 runtime b3c031ad, original observations/RSU reports, native lanes, safety
  layer, speedMode, and [-3,2.6] acceleration limits. Optimize only the existing
  final acceleration-mean linear row (65 parameters); all other parameters frozen.
  Zero updates use the original executor without numerical changes.
- At t=300 select the CAV nearest the merge centre and its nearest CAV neighbour
  (max two), using reported positions only. No hidden types or behaviour thresholds.
  Draw independent N(0,0.3^2) acceleration-latent exploration per present CAV.
  Intervention: the selected CAVs hold their sampled bounded raw acceleration for
  10 s, reapplying the original safety filter every control tick. Nonmembers follow
  policy mean. After 10 s everybody follows policy mean. B follows mean throughout.
  Thus the intervention is sampled-window action versus mean policy, not a second controller.
- Branching = full deterministic same-seed rerun, fresh policy/controller state and
  fresh simulator per branch. No save/load state or inferred controller snapshot.
  Every prefix is charged; initial observed state and captured branch inputs must match.

## Credit and learner
- H=600 s after t300, J=original cumulative_600 team_delay_vs (fleet + source),
  D(S)=J(B)-J(S), positive improvement. Measure ONLY:i, G, G\\i (for |G|=2 these
  are the other ONLY branch), ALL; C_G=D_G-sum(D_i); M_i=D_G-D_(G\\i).
  Singletons/empty groups are logged and receive no fabricated interaction credit.
- Local score-function policy gradient on the sampled latent window actions;
  continuation policy is frozen while estimating each update (a local surrogate,
  not an unbiased gradient of the entire closed-loop policy's J).
  Objective w_all D_all + w_group D_G + w_ind sum D_i + w_coop C_G + w_team sum M_i.
  u=1..32 curriculum q=(u-1)/31: weights (all,group,ind,coop,team)=
  (q/4,q/2,1,q/4,q/4). Hence w_coop<=w_ind and every expanded solo coefficient
  1-q/2 remains nonnegative. No outcome-dependent behavioural gates.
  One common RMS gain normalizer preserves these coefficients; Adam lr=1e-3,
  gradient norm clip1, one optimizer step/update, no advantage sign clipping.
- Choose score gradient over parameter ES: two solo branches yield two direct
  per-CAV action scores and the group/ALL samples reuse them. For eight samples,
  one world costs 1 B + 8*(2 ONLY+G+ALL)=33 episodes. ES with eight antithetic
  directions and the same four credit subsets costs >=64 episodes plus B and
  perturbs 65 shared parameters instead of directly crediting sampled CAV actions.
  600 s is affordable: 5,702,400 planned training sim-s leaves >2.29M reserve;
  300 s would save only one third of full-rerun cost, losing half the credit horizon.

## Worlds, exposure and measured cost
- Fresh TRAIN: 969200-969391 (192), six worlds/update, used once across updates;
  cell j at zero-based u is (2*j+u%2)%12 in p-major .05/.1/.2/.4 x d1900/2400/2700.
  Every two updates cover all 12 E3 cells. Eight independent sampled windows/world.
  Dev: fixed 969600-969623, two full 12-cell repeats. Canary: 969624 (.4,2700).
- Both frozen-runtime seed guards PASS:217 seeds x3 demands=651 cases. Local
  metadata/text audit excludes forbidden paths; 969200-391/600-624 identity hits
  are only hash/timestamp/decimal coincidences, no use. Reject old draft969000-199.
  Remote allowed filename scans: cpuQ209580, gpuA432109, gpuR233720 files, zero hits.
  Superseded drafts are exposure, not proof of execution; all audits retained.
- Reused *measured* cost: SUMOES_RUN baseline 16-35 wall-s per 900 sim-s episode;
  WMDATA 20 workers, ~6 min/195 post-canary episodes. Current cpuQ load30/32;
  gpuA cpu.max confirms20-core quota and load25+. Use cpuQ max2 nice19 workers,
  one numerical thread, CUDA hidden, yield before dispatch if load>28 or Track3B
  has active SUMO workers. Priority processes are never signalled. Fresh probe
  refines wall estimates before launch; no GPU-host oversubscription planned.
- Per update:6*33=198 episodes=178200 actual sim-s;32 updates=5702400.
  At16-35s/episode and2 workers,26-58min/update before priority/load waits.
  Expected32 updates in14-31h compute; checkpoint and fsynced receipt every update.

## Exact caps, evaluation, stop rules
- TRAIN hard cap8,000,000 actual sim-s; conservative precharge cap is also8,000,000.
  Canary four full episodes=3600, planned total5706000; all failed reservations
  retained, no automatic retry. Every job precharged under exclusive durable lock.
- DEV hard cap300,000 sim-s:24*(MAPPO+CTRL0) baseline=43200;
  check learned policy at u3,6,9,12,15,18,21,24,27,30,32:11*24*900=237600;
  total280800. No dev optimizer data. Paired world-bootstrap2000 draws/seed0,
  learned-minus-comparator J600, negative=better,95%percentile CIs.
- Stop on any nonfinite, failed determinism/runtime/lane audit, infrastructure
  failure, any dev-world collision count increase versus paired MAPPO, or cap.
  After the first three dev checks, stop after three consecutive checks with no
  new best mean paired learned-minus-MAPPO J600. Teleports/TTC are recorded;
  teleports/dropped-vehicle anomalies stop dispatch. MAIN STOP file honoured.
- Report CG>0 rows, DG/Di/Mi, continuous gap changes and observed ramp merge/yield
  event counts, raw/safe executed actions versus paired MAPPO, safety counters,
  sim-s/update and wall-s/update. These are descriptive TRAIN/dev findings.
  Record MAPPO's original training cost from its receipts; no equal-budget claim.

## Reuse and launch gate
Reuse tools/sumoes accounting conventions, frozen tools/ctrlarm/zero.py24095a10,
tools/speedhead/observer.py b22af462 and WMDATA MAPPO path without duplicating them.
CPU action identity + branch-state determinism + toy credit algebra, real identical-
action branch J identity, zero-update MAPPO J identity and throughput receipt are
mandatory. LAUNCH_READY.md and registration SHA binding precede autonomous launch.
