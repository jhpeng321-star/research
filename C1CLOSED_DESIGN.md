# S-C1CLOSED — prospective closed-loop C1 protocol (2026-10-31)

Authority: owner contribution-1 request, approximately 16:15; branch c4/c1closed,
base f07bd01e. This registration precedes scientific inference. No SUMO, fitting,
or black-box retraining. TRAIN/dev recorded-world reuse is explicit.

## Question and closest evidence

Does a frozen current WM predict the effects of controller *code* executed on
its own predicted observations, more accurately than the original frozen C1
black boxes? Historical unseen-wall 60 s results were teacher-forced; C1FAIR
used 13E, and WMREQUAL3 replayed original longitudinal commands. None answers
this current-model, full-feedback question. Read sources: the main worktree's
docs/paper/CLAIM_EVIDENCE_CHECKLIST.md C1 entries; c4/evalprep E1/E2 exports;
c4/c1fair C1FAIR.md; c4/speedhead wm_worker.py; c4/wmcalib worker;
c4/wmrequal3 report/audit; c4/wmdata and c4/wmdata2 completed-data reports.

## Frozen models and exposure eligibility

- Primary WM2 SHA256 b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436
  plus learned entrant mix ON, noise scale 1; current SPEEDHEAD execution-parity
  runtime, physical exits, native lane ownership and source queues. Hash the
  config, source closure, entrant sampler and MAPPO export before inference.
- Frozen BB-traj afaaa32cb15079014d1a9e08f78b26e74fd341298d1aa0292a24009d9f783742;
  BB-id e12c81c008e3bb29e366db5597061754e0fa56092dbe5a995a9baa1fbfcce61d.
  Their legacy no-PI corpus is a domain-shift limitation, not matched training.
- Historical C1 structured 87ed02e0707e5e4eefdd2dbb0fd282f834c1288776f7f395b41ab4d13826aaef
  is secondary, only if checkpoint and genuine observed history survive.
- S-C1GAP's hash-linked exposure audit is an eligibility dependency. Separate
  unseen controller algorithm, unseen policy weights, and unseen world. A
  controller exposed anywhere in the WM lineage cannot enter the zero-shot
  algorithm headline; retain any measured exposed arm as diagnostic. Unknown
  provenance is pending, not clean exposure. Never tune from outcomes.

## Outcome-independent panels and controllers

- Primary recorded panel: all 40 S-WMDATA seeds 969400–969439, t0=300 s,
  genuine observed 60 s history, native/PI external-U/CIRCLES/MAPPO-speed iter169.
  Raw jobs cpuQ /home/ubuntu/wmdata20261031/jobs; pre-t0 starts already mirrored
  on gpuA/gpuR in /root/wmcalib/data. MAPPO SHA256
  d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca.
  These worlds were collected for later WM-fix fitting, not unseen-world claims.
- WMDATA2 969550–969599 is a reserve input-recovery panel only, never selected
  by error. Its recorded six arms also contain no wall. Do not pool panels.
- Wall-v2.1 requires separate recorded wall worlds and proposal membership,
  request time, set speed/hold and pre-t0 executor state. Recover through the
  existing C1 safe manifest and S-C1GAP audit. No future-derived group proposals,
  snapshot-history substitution, new SUMO, or non-wall surrogate. Register exact
  eligible wall IDs before its inference; absent inputs mean unavailable wall.
- PI/CIRCLES reuse real published controller classes at 1 s control / .5 s plant;
  MAPPO uses its deterministic speed-only feedback bridge. All initial/entrant
  CAVs use the same arm. No original command replay, recorded future lanes,
  presence/exits, HDV motion or entrant schedule is supplied to predictions.
- BB-traj receives the same external CAV controller/plant where its API permits;
  any incompatible autonomous-entry API is a reported blocker, not silently
  changed semantics. BB-id's original ID-conditioned learned-CAV baseline is
  separate: PI/wall/damping IDs only where defined, never claim it executes
  arbitrary MAPPO feedback or map unseen MAPPO to a fabricated controller ID.

## Computation and metrics

- Tiny CPU actual-world test precedes the main panel; verify finite states,
  feedback dependence, actual duration, native ramp delegation, identity mapping,
  component additivity, and no future-input use. Then K8 paired WM draws at
  10/30/60/90/120/180/300 s (600 half-second ticks); common streams across arms.
  BB point-parameter forecasts are a single frozen-model draw, explicitly labelled.
- Endpoint HDV position/speed absolute errors: stable t0 IDs, same surviving
  real+predicted IDs for paired model comparisons, median within world/arm then
  median across worlds. Report each survival count and unavailable long horizons;
  no zero imputation or matching synthetic entrants to real identities.
- 2,000 whole-world bootstrap resamples, seed 0, percentile 95% intervals;
  paired WM-S minus BB-traj is computed within world on common IDs before
  aggregation, not by subtracting marginal intervals. Preserve all horizons.
- Team J = integrated all-present speed-deficit delay + source waiting, veh-s,
  left-boundary .5 s rectangles; region partition from frozen network/scoring
  constants: source + upstream + merge + downstream/discharge. Verify exact
  additivity; use common total-J median draw weights for component contrasts.
  Describe errors, not causal mediation. Retain discharged-count error separately.
- E2 sign convention: treatment minus native cost, negative is improvement.
  Direction excludes exactly-zero true effects only from its denominator;
  within-world ranking/decisive-pair accuracy and oracle regret minimise J,
  deterministic native-first ties. No borrowed qualification threshold or sigma.

## Resources, preservation, output

Use existing Python environments only. gpuA-host CPUs at nice19/one thread per
worker where cores are free; GPU only after nvidia-smi admission, <=12 GB TOTAL
owned memory, leave >=1 GB physical reserve, no peer signals. gpuR preferred
after WMCALIB exits. Record own PID/start ticks and per-task runtime/failures;
write-once task outputs/resume only matching identities. No denied-path access,
truth/assignment reads, prohibited world seeds, data deletion, push, or resets.

Deliver docs/paper_results/C1CLOSED.md (<=60 lines), raw/per-world and bootstrap
CSV, error-vs-horizon PNG (unsupported BB-id cells explicitly unavailable),
and completion registration. Commit each step. If feasible, separately registered
same-panel physics / +residual / +settings ablation; never replace the primary.

Initial prerequisites: S-C1GAP report absent; local branch lacks later tools/data
that remain accessible as immutable git objects. At initial probes gpuA had
~12 GB free GPU memory and load ~22/208 CPUs; gpuR ~4.8 GB free GPU memory.
No scientific inference launched at this design registration.

## Prospective compatibility amendment (after tiny CPU checks, before panel)

The actual frozen BB-traj rejects `native_lane_change` because it has no
structured first-passage head (2 s CPU failure retained). Therefore the paired
WM2/BB-traj/historical panel uses the existing C1FAIR **common external CAV
plant**: no ordinary CAV lane changes, explicit gap-based ramp merge, shared
execution-parity longitudinal dynamics. This is a scoped approximation to the
recorded native-lane plant, not full production parity. No BB module is added,
retrained or replaced. Separately retain WM2 production-native K1 diagnostic
on all 40 same worlds; never pool the two plants or compare unlike plants as a
paired model win. BB-id remains its explicit learned-ID contract (no feedback
MAPPO/CTRL0). Checkpoint TRAIN_RECEIPTs accompany each legacy checkpoint to
resolve its original physics; the failed no-sidecar attempt is not evidence.
These changes are fixed before any main-panel forecasts or recorded outcome
extraction. K8 common-plant WM2 + K1 BB-traj/BB-id/historical, plus K1 WM2
production diagnostic = 480 world/future tasks, five arms except BB-id's three.

## Owner secondary analysis addendum (fresh-session recovery, before panel)

The primary checkpoint, K8, panel, endpoints and horizons above are unchanged.
At 60/120/180/300 s also measure aggregate fidelity against the same recorded
SUMO: merge-zone queue (present vehicles at -200 <= s <= 75.44 m with v < 5 m/s),
100 m by 10 s speed/density fields on fixed s edges -1000:100:500 m (all lanes
pooled, density veh/km of road, speed occupancy-weighted; empty cell speed=0,
with empty/occupied mismatch exported), cumulative merge crossings and the four
additive delay components. Score cumulative space-time field MAE through each
horizon; queue at the endpoint and time-mean queue are both reported. Bootstrap
paired model error contrasts using the same 2,000 whole-world draws, seed 0.
All fields include autonomous entrants; no matching of their synthetic IDs.
Use medians over WM draws for aggregate fields/queues/crossings, and shared
total-J median-order weights for additive delay components as specified above.
Long-horizon chaotic divergence is a motivation, not an established floor.

If S-WMFIXMERGE v2-DATA or S-WMFIX-ES passes its OWN registered verdict, bind
its checkpoint, verdict and exposure receipt before evaluating the exact same
panel/metrics as a secondary comparison. Adapted controllers stay labelled
adapted; this never selects/replaces WM2 for the C1 headline. A failed/pending
verdict leaves the slot unavailable. Reserve an explicitly unavailable SUMO
seed-to-seed divergence-floor series in the figure/data schema. MAIN must
supply approved recorded same-start/different-RNG results and provenance;
this task launches no SUMO and does not fabricate an irreducible floor.

Recovery: four 2 s CPU model outputs exist; no panel outputs, DISPATCH or
COMPLETE receipt exists. The uncommitted dispatcher is incomplete and has not
been launched. S-C1GAP C1_GAPS.md now supports PI/wall/MAPPO algorithm absence
in audited WM2 corpora, with pre-R16 initialization exclusion still incomplete;
eligibility is provisional, not an unbroken-lineage zero-shot certificate.

## Recovery receipts after launch (primary unchanged)

One gpuA dispatcher PID 211623 / start ticks 881579797 owns the fixed 480 tasks.
Observed full-300 s task runtime is approximately 348 s, not the tiny-case
throughput extrapolation. Two GPU lanes remain bounded; CPU lanes yield until
load <18. No shortened K or selected-world substitute is authorized.
Independent recorded-label extraction verifies all 200 world/arm starts. Tapes
start at t300.5, so the observed t300 history/state supplies the first left
integration boundary. This indexing correction changes neither estimands nor
model inputs. See tools/c1closed/labels.py and its exported SHA256.
Checkpoint-linked S-C1GAP metadata explicitly gives 940 audited training worlds
and zero seed overlap with 969400–969439, while retaining the pre-R16 lineage
gap. This is not a sealed/unseen-world certification.

Wall recovery located 120 genuine 60 s-history starts, not zero; all have empty
controller maps and no persistent executor snapshot, and cached futures only
through 120 s. Membership/request summaries also contain post-t0 release/abort
outcomes, which are forbidden as forecast inputs. Causal executor restoration
has not been established. docs/paper_results/C1CLOSED_WALL_INPUTS.md records this
specific remaining blocker. No wall forecasts or wall panel selection occurred.

## Owner 19:12 throughput expansion (primary and frozen inference unchanged)

Recover and retain the live gpuA dispatcher and sole collector. Add four gpuR
GPU lanes only after exact full-300 s parity of completed world0/future0/common
against gpuA (exclude only PID and runtime from equality). Keep gpuR's total
owned allocation <=8,000,000,000 bytes, physical reserve >=1 GiB; gpuA retains
two GPU lanes and two spare-core CPU lanes under its original <=12 GB budget.
Per-process allocator remains the frozen 3 GiB. Shared peer sources are never
overwritten. Missing inherited config files may be copied byte-identically;
hash the full config chain and concrete checkpoint/policy inputs.

The original dispatcher has no cross-host claims. To make allocation exclusive,
pause ONLY its receipt-verified PID/start incarnation, let all its own children
finish, then freeze deterministic disjoint remaining-task lists for gpuA/gpuR.
Keep every completed output. Expansion runners execute identical worker.py,
aggregate code, inputs, K, controller order and seeds. A publication bridge moves
only checksum-verified, successfully completed gpuR outputs atomically into the
existing gpuA panel. The existing collector/finalizer source hashes and process
remain unchanged. The bridge publishes successful COMPLETE only at480/480,
then retires the paused original owned scheduler. All signals require the exact
registered owned PID/start identity; never signal a peer or by process name.
The separate parity run is never counted as a new panel task. No scientific
outcome inspection, sample reduction, post-hoc selection or retraining is added.

### Numerical-parity gate amendment, before gpuR panel allocation

The stricter-than-owner bitwise gate failed on the different 4090/4090D GPUs:
same-host full300s repeats are bitwise identical, but cross-host roundoff grows
after60s. Preserve that FAIL and every diagnostic; do not relabel it exact PASS.
The model/source/input/package hashes match; only the resolved config symlink
path differs in non-executable metadata. No recorded-SUMO comparison was used.

Before computing the following check, register this separate operational gate:
all non-timing task metadata/identity exact except that verified config alias;
through60s require identical discrete states/counts/queues, endpoint |ds|<=.001m,
|dv|<=.0001m/s, fields speed<=.0001m/s, component/total delay<=.001veh-s.
At90/120/180/300s require each arm's scalar aggregate discrepancy (all four
components, total delay, crossings, queue and mean queue) no larger than the
max-minus-min spread of its already-completed frozen world0 K8 draws, plus
.001 rounding tolerance for continuous integrals (.0001 for mean queue, zero
for integer counts). For speed/density fields, compare mean absolute host
difference per arm over all space-time cells to mean K8 cellwise range plus
.0001 units. For jointly present original IDs, require endpoint differences
within the same-ID K8 range plus the early numerical tolerances. Check all
arms/horizons; no selected metric waiver or SUMO-error-based tolerance.

Only this prospective numerical/ensemble check passing permits disjoint gpuR
allocation. Export both its verdict and the original exact failure; describe
the mixed-hardware numerical sensitivity as a limitation, not a SUMO chaos
floor. If it fails, retain the original gpuA panel/collector without expansion.

Allocation detail before assignment: preserve every drained CPU baseline task;
remaining frozen legacy forecasts may use gpuR CUDA after all three original
10s CPU paths match CUDA values under the same early numerical tolerances.
This changes device only, never weights/controller/plant/K. Existing spare-core
CPU tasks are allowed to finish. For new tasks, use six GPU lanes: gpuA2/gpuR4.
Because measured parity runtime is ~57s on gpuR versus minutes on shared gpuA,
assign every ninth remaining WM task to gpuA and the rest to gpuR; gpuR also
receives remaining legacy tasks, oldest roster first. This is deterministic
runtime-based allocation, with no recorded-error or controller-arm selection.
Slow in-flight CPU tasks may drain concurrently with the new, disjoint GPU
assignments: reserve their exact task identities outside both assignment lists.
All prior CUDA children must finish before replacing GPU lanes. The complete
gate still requires every reserved CPU output; no in-flight task is duplicated.
