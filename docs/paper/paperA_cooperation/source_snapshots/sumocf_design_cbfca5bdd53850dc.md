# S-SUMOCF — cooperation-first SUMO-oracle counterfactual learner

## Corrected probe PASS / prospective learner resumption (supersedes old launch)

ARTIFACT_CHECK.md and evidence/ARTIFACT_PASS.json document the mechanics PASS.
Saved-state engine rejected; every future branch is a full same-seed prefix rerun.
Corrected table is evidence/ARTIFACT_PROBE.json (all27 labels;3distinct estimable
pair/rho.5 designs, n2). Measured DG mean/SD by10/30/60s:0/0;
.000719547/.001017594;.001293659/.001829510veh-s. PooledDi mean/SD:0/0;
.000359774/.000719547;.000646830/.001293659. CG mean/SD all0/0. Raw SNR0,
.7071067811865475,.7071067811865476. Exact maximum selects **pair/60s** (grid7),
not a scientifically distinguishable improvement over30s. Near-zero effect in one
world and exactly0 in the other; no evidence of cooperation. No SNR cutoff was
registered. All-zone labels are exact identity aliases;4-member groups unavailable;
other rho explicitly not sampled after the prospective budget reduction.

Artifact72881sim-s charged=actual, inside100k nested in TRAIN8M. Training total
charged795791 includes all old/quarantined reservations (actual791777;4014never-
started prefix seconds retained). Original DEV48 baseline actual=charged57600,
reused bitwise, never rerun. Old u01 worlds969206–211 are exposed and SKIPPED.
Resume from original immutable zero policy_u00 (bitwise MAPPO), fresh Adam only
because no optimizer update ever completed. No prior invalid credits are read.
Fresh learner block969212–391 gives30updates x6worlds; cell cycle starts at offset6
and retains the same2:1 per-cell high-demand/high-penetration oversampling.

Cost-only formula (original31 denominator, conservative for30updates) selects
**6windows/world** from corrected measured5317.5sim-s/sample. Expected191430sim-s/
update,6,538,691TRAINtotal including all gates/sunk reservations; hard8M. Worst
case36samples x5branches x1200sim-s x30 +795791=7,275,791<8M. Expected wall from
measured157.3s/sample and20admitted workers:~283s/update plus stragglers/load.
No truncation: each window starts300..600 and every branch runs its full600s.
Sigma.3/noise_scale1 unchanged; shared/independent rho0/.5/1 sampled per window.
Score denominator uses the actual Gaussian variance, including explicit scale
in diagnostic jobs; rho1 remains the labelled support-projected surrogate.

Full prefixes increase DEV cost. Use **five** checks u3/7/12/20/30:24worlds x1200s
per checkpoint plus3fixed-pair long-horizon diagnostics, each at most5x1200s.
57600baseline+5*(28800+18000)=**291600<=300000DEVsim-s**. Preserve both J600/J900,
paired bootstrap and cooperation distributions; first3-check/no-improvement and
collision/nonfinite/teleport/cap stops unchanged. No DEV optimizer input.

gpuR measured quota25,peer~2cores,no throttling,3.94GB free. Resume max-worker
ceiling22, actual=min(22,floor(25-3-measured_peer_cores)) (currently20), nice19,
one numerical thread,CUDA hidden; remeasure every5s, reserve3cores and yield to
peers. cpuQ untouched. Baseline/checkpoint/proof/source hashes in new
ARTIFACT_LAUNCH_READY.json must validate before releasing durable optimizer HOLD.
CPU suite18PASS/2.52s includes fresh seed exclusion, full-rerun DEV cost, and
exact reconstruction of the actual immutable zero checkpoint including RSU RNG.

## MAIN17:17 artifact amendment (binding before further SUMO)

Optimizer HOLD enforced, inherited DEV baseline retained; see ARTIFACT_CHECK.md.
**Saved-state engine now REJECTED:** two later TRAIN states (t465,t596) with
zero offsets have identical issued actions yet diverge in their first physical
step after load; false DG+1233.34/-1555.78veh-s explains the original probe.
The old t300 canary was insufficient. Use complete same-seed prefix reruns for
EVERY branch (including all prefix costs) from here. No learned update completed.

**Prospective fallback cost correction, before any fallback SUMO:** failed saved
gate7061sim-s retained. Full-rerun gate6samples each<=5*(600+600)=6000sim-s,
<=36000. Reduce corrected probe to pair x{10,30,60}s x rho.5 on the same two
fixed states969201/969204:6samples<=36000. All-zone exact aliases reused; other
rho designs explicitly NOT SAMPLED (not zeros), size4 unavailable.27 labels,
3distinct estimable designs. Combined<=79061<100000 with prior failed gate;
nested100k hard cap unchanged. Largest raw-DG SNR among these estimable designs,
ties cheaper/grid; beta0. Training still samples rho0/.5/1 per owner F.
New immutable attempt prefix art2_; art1_ results preserved, never retried.
The saved-state allocation paragraph below is historical, superseded here.

The original intervention held an absolute branch-start mean plus noise. Correct
it to **current policy mean + persistent sampled latent offset**, injected into
the original Torch head/decoder/safety/memory path. A zero offset skips addition
and is exactly the original policy at every tick. No NumPy action re-decoding or
second command-memory write. Score credit remains the local window-start
Gaussian mean surrogate with the original joint density; it does not claim an
unbiased derivative of the full closed-loop continuation. Tail/RSU capacity and
all cooperation weights/anti-gaming/horizon contracts are unchanged.

Within the original TRAIN cap, an additional hard **100,000 sim-s** allocation
covers artifact canaries AND corrected probe, enforced in durable precharge:
- Gate on original TRAIN states969203 and969204 (original registered t0/cells),
  pair,60s,rho.5, scales0/.25/4 of original sigma. Six complete samples, each at
  most600+5*600=3600sim-s: <=21600. Zero-noise must match B in J600 AND trajectory
  hash on every subset, with every D/CG/M exactly0. Compare small/large executed
  deviation, raw deviation, clamping and D on these identical starting states.
- Corrected small probe: prospectively fix original states969201 and969204,
  both having exactly two observable-zone CAVs, to fit the owner100k constraint.
  Pair x{10,30,60}s x rho{0,.5,1}:18samples<=64800. All-zone groups are EXACT
  identity aliases of these pairs (verify geometry again, reuse identical
  receipts); four-member designs have no eligible state and report NA.
  Thus all27 design labels remain reported but only9 distinct designs estimable.
  n=2 is explicitly low-power; this is bug validation/reselection, not evidence
  of population cooperation or superiority. Select max abs(mean RAW DG)/sd,
  ties cheaper then original grid order. Freeze CV beta0 (no fitting a control
  variate on two worlds). Combined worst-case86400<=100000, no retries/refunds.
- Trace schema sumocf_observable/3 adds per-action current raw mean, actual raw
  intervention, persistent offset, and exact safety-clamped flag. Distinguish
  raw-clamped share, fraction of nonzero raw exploration erased, and unchanged
  issued-command share. Stable path remains /root/sumocf20261031/traces/.
- Old pre-HOLD u01 simulations/receipts are quarantined, never optimizer input.
  Their reservations and actual seconds remain charged. Training restart and
  any seed-schedule shift require a new passing artifact/launch registration.

Measured CPU gate after correction:14PASS/2.64s on existing gpuR environment,
including zero-offset changing-state bitwise action/memory identity, exact Torch
decode with nonzero offsets, and optimizer HOLD-before-dispatch regression.

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
  Report raw and leave-one-world-out CV diagnostics; selection uses RAW D_G as the
  owner registered. Freeze any accepted beta/centering before training.
  The CV is an explicitly biased local surrogate if its expectation changes with
  policy; raw 600s primary cooperation is always reported alongside it.
- Selection rule: largest abs(mean D_G)/sample_sd(D_G), ties -> measured smaller
  actual sim-s/sample, then lexicographic (size,window,rho); zero sd and mean ->0.
  The CV is diagnostic, not a substituted D_G mean for selection. No sample enlargement.
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
  selected window, private-random t300..600, all5 subsets includingALL. Dev credit
  compares DEPLOYED learned means against MAPPO means during the intervention
  window (same observed state and actual prior command history), with learned
  continuation thereafter; it is not credit for random exploration around itself.
  TRAIN/probe credit remains sampled-action versus same-checkpoint mean as registered.
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

Admission refinement while the probe was idle: global host load rose32..44 while
the25-core cgroup used only7.02 cores with ZERO throttled periods. Global load is
from other containers and is not quota occupancy. Subsequent admission samples
cgroup cpu.stat and exact OWN worker/child CPU ticks; target=min(15,floor(25-3-
measured_peer_cores)), recomputed5s. It yields to peer quota usage and leaves3cores.
The idle old supervisor was replaced only after PID/starttime and zero live-child
checks. Completed probe results are read unchanged; already reserved but never
started jobs keep their same reservation/spec. Any started incomplete job refuses
continuation. This is no simulation retry or reallocation. Probe numerical worker
stays unchanged throughout; richer trace/half-second accounting v3 is canaried
  separately before learner launch:2400sim-s for arm/branch identity plus3300sim-s
  for zero-update deployed-mean dev subsets (all must be bitwise MAPPO with CG=0).
  Combined canaries8100sim-s are inside the12000 allocation and8M TRAIN cap.

## Completed SNR probe — measured table and authoritative selection

162/162 fixed jobs complete. Probe plus original canaries: charged=actual606309
TRAIN sim-s, DEV0. No completed sample rerun. Full precision table:
`tools/sumocf/evidence/SNR_TABLE.csv`; original and corrected receipts are retained.
Units for means/SDs: veh-s. D_i pools eligible member rows; DG/CG use eligible
worlds. Each design attempted six worlds. Requested four-member groups existed in
only one world, so their between-world SD/SNR is not estimable (NA, not zero).

Measured table (CSV; `all` = all observed merge-zone CAVs):
```csv
grid,group,window_s,rho,n,mean_Di,sd_Di,mean_DG,sd_DG,mean_CG,sd_CG,raw_DG_SNR,sim_s_per_eligible_sample
0,2,10,0,4,-80.767,1057.485,-80.922,1141.976,80.611,1142.453,.070862,3424
1,2,10,.5,4,-80.767,1057.486,-80.922,1141.977,80.611,1142.453,.070861,3424
2,2,10,1,4,-80.767,1057.486,-80.922,1141.977,80.611,1142.453,.070861,3424
3,2,30,0,4,-80.598,1057.726,-80.588,1142.493,80.608,1142.457,.070537,3424
4,2,30,.5,4,-80.598,1057.726,-80.586,1142.495,80.611,1142.454,.070535,3424
5,2,30,1,4,-80.599,1057.725,-80.587,1142.494,80.611,1142.454,.070536,3424
6,2,60,0,4,-80.608,1057.725,-80.607,1142.491,80.609,1142.457,.070554,3424
7,2,60,.5,4,-80.601,1057.725,-80.589,1142.497,80.613,1142.450,.070537,3424
8,2,60,1,4,-81.697,1057.635,-82.783,1142.296,80.611,1142.454,.072470,3424
9,4,10,0,1,1233.029,NA,1232.096,NA,-3700.019,NA,NA,7065
10,4,10,.5,1,1233.029,NA,1232.098,NA,-3700.019,NA,NA,7065
11,4,10,1,1,1233.029,NA,1232.098,NA,-3700.019,NA,NA,7065
12,4,30,0,1,1233.370,NA,1233.451,NA,-3700.028,NA,NA,7065
13,4,30,.5,1,1233.370,NA,1233.456,NA,-3700.025,NA,NA,7065
14,4,30,1,1,1233.368,NA,1233.448,NA,-3700.023,NA,NA,7065
15,4,60,0,1,1233.369,NA,1233.444,NA,-3700.033,NA,NA,7065
16,4,60,.5,1,1233.369,NA,1233.441,NA,-3700.034,NA,NA,7065
17,4,60,1,1,1233.368,NA,1233.440,NA,-3700.032,NA,NA,7065
18,all,10,0,4,482.422,1028.391,-80.922,1141.976,-1769.398,4634.386,.070862,5524
19,all,10,.5,4,482.422,1028.391,-80.922,1141.977,-1769.398,4634.386,.070861,5524
20,all,10,1,4,482.422,1028.391,-80.922,1141.977,-1769.398,4634.386,.070861,5524
21,all,30,0,4,482.518,1028.468,-80.586,1142.495,-1769.401,4634.390,.070535,5524
22,all,30,.5,4,482.519,1028.468,-80.584,1142.497,-1769.400,4634.389,.070534,5524
23,all,30,1,4,482.518,1028.467,-80.587,1142.494,-1769.399,4634.388,.070536,5524
24,all,60,0,4,482.513,1028.471,-80.607,1142.491,-1769.402,4634.393,.070554,5524
25,all,60,.5,4,482.517,1028.468,-80.592,1142.491,-1769.402,4634.393,.070541,5524
26,all,60,1,4,481.890,1028.787,-82.785,1142.293,-1769.401,4634.392,.072472,5524
```

**Owner-rule winner: grid26, all-zone CAVs,60s, nominal rho1.** Raw SNR0.07247238
is only marginally above grid8's0.07247040; the exact registered maximum selects26,
not a claim that the difference is significant. Training rho still sampled per window.
Raw CG mean-1769.40,SD4634.39,share>0=.5; no established beneficial cooperation.
Measured cost5524sim-s and209.901wall-s per eligible sample on the probe host.

Selection defect disclosed: first receipt incorrectly ranked cross-fitted CV means
as DG and selected grid6. CV means drifted (e.g. grid6+7623.58 vs raw DG-80.61),
so this was not the owner's registered statistic. `SNR_PROBE_ORIGINAL.json` is
preserved unchanged; `SNR_PROBE_OWNER_RULE.json` deterministically re-ranks the same
162 samples by abs(mean RAW DG)/sd(RAW DG). No new samples or exclusions. Regression
test prevents CV means substituting for DG again. The selected design's CV SD2806.80
exceeds raw SD1142.29: **beta=0**, no local credit used in learning. Local delay and
all rejected CV diagnostics remain in receipts. These weak/noisy signals are
reported as such; the owner specified no SNR cutoff or sample enlargement.

D_all here means the policy-eligible CAV identities present at window start;
new arrivals follow the continuation mean. Dev fixed-pair diagnostics measure
learned-mean versus MAPPO-mean window effects, not random exploration credit.

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
