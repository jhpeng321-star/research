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
