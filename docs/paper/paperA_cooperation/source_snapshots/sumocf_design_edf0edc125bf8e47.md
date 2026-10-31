# S-SUMOCF — cooperation-first SUMO-oracle counterfactual learner

## Binding amendment (owner 15:45, 15:58, 16:05; supersedes original design below)

Cooperation is PRIMARY. The original design is retained below as history, not launch
authority. Recovery 2026-10-31 found no live SUMOCF process on cpuQ/gpuR, no remote
receipts/ledger, and only staged pre-amendment Python files on cpuQ. Draft preserved
in commit 794956ff. No training or probe SUMO had been dispatched.

### Policy, observability, density and horizon
- Exact speed-only MAPPO iter169 SHA d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca;
  frozen E3 source b3c031ad, native lanes and original safety/limits. Train the final
  hidden linear layer and acceleration-mean row. Add an RSU branch with a trainable
  <=4-dimensional broadcast latent and a ZERO-initialised output to acceleration
  latent. Merge/ramp reports only: positions, speeds, lanes and gaps, no driver truth.
  Skip the zero addition at zero initialisation to preserve MAPPO bitwise; test it.
- RSU zone is the observable approach/merge geometry s in [-150,75.44] metres,
  all lanes (ramp included). Groups span lanes: nearest 2, nearest 4, or all CAVs
  present in this zone, deterministic distance-to-merge ordering. This defines an
  observation footprint, not a behavioural trigger. Missing/undersized groups are
  logged, never represented as zero interaction.
- Per-window exploration is a Gaussian shared draw plus independent draws, sigma=.3
  in the original unsquashed action latent. Mixing rho in {0,.5,1}; covariance
  sigma^2*((1-rho)I+rho*11'). All subsets use the SAME sampled vector; their score
  uses the subset's marginal JOINT density. rho=1 uses the rank-one support density
  (pseudoinverse); no fictitious independent scores. At rho1 off-support individual
  mean shifts have no ordinary density: this is explicitly a common-mode projected
  score surrogate. rho0/.5 use full-rank exact joint scores. Shared noise applies to
  group members; the remainder of ALL has independent noise. Actions are bounded and filtered
  by the original MAPPO safety operator each tick. Hold sampled latent for window;
  nonmembers and all continuations use the checkpoint mean. No shaped rewards.
- Window start is a deterministic private uniform integer draw in [300,600]. Every
  branch runs FULL H=600 after its start, ending t_start+600 (up to1200), not900.
  Prefix from300 uses current policy. Snapshot controller memory/dwell together
  with the COOPMEAS2 RNG/environment snapshot. Try saved-state only if a live-trunk
  versus loaded-identical-action 600s bitwise J/trajectory canary passes; otherwise
  use complete same-seed reruns including their prefixes, fully charged.

### SNR probe gate (pre-registered before probe/training)
- Grid: group {2,4,all zone} x window {10,30,60}s x rho {0,.5,1} =27 designs.
  Six TRAIN samples/design: first six registered training seeds969200-969205,
  first four the four high-demand/high-penetration cells, two lower cells.
  Same sampled start within a world across designs; fixed draws, no replacement.
  Reuse B/solo/flip branches within a design whenever subsets coincide.
- For every design measure D_i,D_G,C_G,M_i from 600s fleet+source J. Also collect
  merge-zone local delay of observable vehicles crossing the zone during window+120s.
  Local J NEVER stands alone: candidate control-variate credit is team D minus
  beta*(local D - its probe mean), beta=cov(team,local)/var(local), zero if variance0.
  Report raw and leave-one-world-out CV SNR; raw team J remains the default learner
  unless the cross-fitted CV improves SNR. Freeze beta/centering before training.
  The CV is an explicitly biased local surrogate if its expectation changes with
  policy; raw 600s primary cooperation is always reported alongside it.
- Selection rule: largest abs(mean D_G)/sample_sd(D_G), ties -> measured smaller
  actual sim-s/sample, then lexicographic (size,window,rho); zero sd and mean ->0.
  Both raw and CV candidate use the 600s outcome. No outcome-based sample enlargement.
  Probe rows and machine-readable 27-row SNR table will be appended here after gate.
  No measured SNR exists at this amendment; training is blocked until it does.
  The winning grid freezes group size/window; its rho is the nominal probe winner.
  As required by F, training continues to draw rho uniformly per window in{0,.5,1}
  with identity-keyed private RNG; its actual rho is used in the joint score.

### Credit, learner and updates
- D_S=J_B-J_S; C_G=D_G-sum_i D_i; M_i=D_G-D_(G\\i), D_all separately.
  Score-gradient maximises w_all D_all+w_group D_G+w_ind sum D_i+w_coop C_G+
  w_team sum M_i. u1..32 q=(u-1)/31; weights(q/4,q/2,1,q/4,q/4).
  Thus w_coop<=w_ind; expanded coefficients: solo=1-q/4, group=q/2+q/4+n*q/4,
  flip=-q/4, all=q/4 (combine coincident subsets). One common RMS normaliser,
  Adam1e-3, norm clip1. No sign clipping or behaviour thresholds.
  Trainable RSU initialization uses torch seed20261031; actor/group trunk remains
  the hash-bound export. Every simulator half-second is independently counted
  against its worker's already-precharged prefix+branch allowance, including failures.
- Six fresh worlds/update, up to31 updates. Before launch choose samples/world
  S=min(8,max(1,floor((8M-probe_and_canary_charges-12000)/(31*6*selected_measured_sim_s*1.2))));
  this cost-only rule spends useful remaining budget without outcome-based tuning.
  Each world contributes S independent sampled start-times/noises within its one
  update; no world reused across updates. CG CIs resample WORLD clusters, retaining
  all within-world windows. The same checkpoint controls every sample in an update.
  Each sample costs B + n ONLY + n FLIP + G + ALL, reduced for duplicate subsets.
  Score-gradient credits the actual sampled actions through final hidden+RSU layers;
  parameter ES would require at least two such branch sets per direction in thousands
  of parameters. Direct action credit is cheaper per useful scalar signal. Probe
  measures actual sim-s/wall before the choice is frozen; no WM rollouts.
- Training cells use a16-entry cycle: all12 E3 cells once plus the four .2/.4 x
  2400/2700 cells a second time (2:1 per-cell oversampling), interleaved high/other
  so every six-world update includes three high cells. Dev unchanged24 worlds
 969600-969623 (two complete12-cell repeats); canary969624. Probe consumes first6
  TRAIN seeds; updates use969206 onward, at most31 six-world updates within969391.
  Registered guard/exposure receipts in tools/sumocf/evidence/PREFLIGHT.json retained;
  re-run both frozen guards on actual allocations before dispatch.

### Accounting, capacity and stops
- Training hard cap8,000,000 actual AND reserved sim-s; dev hard cap300,000.
  Every prefix and branch precharged under fsync+exclusive lock, no refunds/retries.
  Dynamic observed group size determines branch reservation BEFORE any branch runs;
  if remaining cap cannot fit the complete sample, stop without partial training.
  First3 updates guaranteed only if measured probe costs leave headroom; cap wins.
- Saved-state sample cost t_start+600*(number of distinct subsets), full-rerun cost
  (t_start+600)*(number of distinct subsets). Expected n4: up to11 subsets ->
  7050sim-s/sample,42300/update; n20:43 ->26250/sample,157500/update.
  Probe upper allocation2,000,000, canaries<=12,000 retained under8M; after probe
  compute exact maximum affordable updates from observed chosen-design branch cost.
  Prior SUMOES measured16-35wall-s/900sim-s; fresh throughput supersedes this.
- Dev controls start300; MAPPO,CTRL0 and each checkpoint scored at600 AND900 after
  control start (episodes1200).24*(2 baselines+6 checks)*1200=230,400sim-s.
  Checks u3,7,12,18,24,31 (last affordable update may replace a planned slot,
  never adding a7th check). Each check additionally measures raw600s cooperation
  on three preselected high-cell dev worlds, rotating the four high cells: fixed
  observable PAIR diagnostic (explicitly distinct from selected training group),
  selected window, private-random t300..600 and rho, all5 subsets includingALL.
  Worst-case6*3*(600+5*600)=64,800sim-s; total295,200<=300,000. This supplies
  C_G/DG/sumDi/Mi distributions at every dev check without sacrificing long horizons.
  Missing pairs are disclosed, not replaced/zero-filled. No dev credit gradients.
  Paired world bootstrap2000 draws seed0,
  learned-minus-comparator, negative better. Report 600/900 ranking agreement.
  Consume S-COOPEVAL harness/figures when delivered; do not duplicate its analysis.
- Hosts measured at recovery: gpuR quota25 CPUs, load9.55 and7 WM calibration
  processes; cpuQ load11.74/32, no live SUMOCF or SUMOES match. Start conservative,
  determine safe workers from measured active CPU usage+quota, nice19 one numerical
  thread, CUDA hidden. Recheck capacity before every dispatch; cpuQ Track3B first,
  yield on load/peer pressure. Never signal peer PIDs. CPU-only stages may run first
  on cpuQ's existing environment; gpuR workers need byte-verified frozen runtime.
- Stops: nonfinite, any bitwise/guard/lane failure, worker failure, teleports/dropped
  anomalies, any dev-world collisions above MAPPO, cap, MAIN STOP. After first3 dev
  checks stop after3 consecutive no-improvement checks. No automatic retries.

### Stable cooperation handoff
gpuR /root/sumocf20261031/traces/<job>/b<subset-index>.jsonl.gz contains per-second
observable vehicle reports (id,s,speed,lane,gap,is_cav), group membership, sampled
latent, policy/MAPPO/executed commands and window timing, from start-30 through
window_end+30. Schema/manifest and per-sample credits accompany immutable checkpoints
and receipts each update. S-COOPBEHAV consumes these for gap/yield/wall detection;
S-COOPEVAL owns dev harness/figures; S-COOPMEASURE owns controller audit. Trainer
reports raw CG distribution(mean/world-CI/share>0), DG,sumDi,Mi PRIMARY, dev J and
safety SECONDARY. MAPPO speed-only training6,996,600sim-s plus earlier lane-on init;
no equal-budget superiority claim.

Remote paired-index handoff (after first checkpoint):
gpuR:/root/sumocf20261031/COOPBEHAV_INDEX.json . Immutable per-update indices sit
beside it. Kind is **policy_mean_reference**, NOT native_member_release: the existing
S-COOPBEHAV native-release adapter must not relabel this intervention. Full observed
fleet, leaders/followers, observable lengths and issued commands are exported. V2
training traces separate issued_safe from realized physical acceleration frames;
probe-v1 field actions.executed is explicitly an issued-safe-command alias. Any
native warmup prefix reconstructed from packed history marks lengths unknown.

Capacity refinement before SUMO: cpuQ now has17+ S-WMFIX workers and an active
24-worker Track3B WM screen, so its SUMOCF admission yields completely. gpuR has
quota25/load10.39 with7 calibration workers; use up to12 one-thread nice19 workers
there, reducing admission if measured load leaves fewer than3 spare quota cores.
Reuse its byte-verified frozen runtime at /root/autodl-tmp/c4/speedhead20261030/e3final,
wrappers at the parent speedhead20261030, exact MAPPO export copied to the owned
/root/sumocf20261031/inputs/mappo.pt. Root disk6GB free; gzip observation traces,
stop new dispatch below1GB free. No cpuQ SUMO started before this capacity switch.

## Original 16:00 draft (superseded by the binding amendment above)

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
