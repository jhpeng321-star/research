# S-WMVALUE: frozen offline design (2026-10-31)

## Question and estimand
Owner direction: REGISTRATION.md S-WMVALUE/OWNER 17:37, world model central.
No simulator execution. Control starts at absolute t=300; the endpoint is t=900
(600 s after control start), NOT absolute t=600. At t0=300,360,420,480,540,
predict remaining fleet speed-deficit integral plus source waiting, in veh-s.
Compare all ten unordered pairs of native/CTRL0/MAPPO/PI/CIRCLES, fixed order
native,ctrl0,mappo,pi,circles; difference = second minus first (lower is better).
Use each arm's recorded state at t0; shared warmup only guarantees equality at 300.
This is policy-continuation prediction, not a randomized new intervention at t0.

## Data and exclusion
FIT: WMDATA seeds 969400..969431; WMDATA2 seeds 969550..969599, all recorded
arms including generic_random. Test once: WMDATA 969432..969439 (8 registered
reserved FIT worlds); never fitting, preprocessing fit, model selection or tuning.
Optional S-SUMOCF traces are excluded from this first frozen experiment.
Only explicit allowed episode/view/start/result/objective paths may be read.
Never open paths containing sealed, holdout, heldout, cal_split, chunk5, j4b,
inc1, hetdata_truth, vehicle_params, driver_params, or vType assignment files.
Never evaluation seeds 964000..964199, 962xxx, 977xxx. No scenario files.
Full raw observable tails must be used; cropped views alone are insufficient.

## Observation and label gate (before fitting)
Whitelist kinematics, presence, self CAV identity, causal detected neighbours,
merge-zone RSU sensing, entrance insertion/request observations and queue counts.
Do not use global vehicle coverage as if it were an RSU observation. Audit the
recording's sensor coverage, per-time source queues, and fleet/source cost clock.
Require recorded sensor masks or a documented reproducible observation adapter;
require actual causal queue observations, not seed/demand-generated future demand.
Metadata demand, penetration, arm names, seed/world IDs, hidden driver types and
future realized actions/states are NOT features. Seeds/arms only join and split.
Global recorded kinematics may supervise fleet delay labels, never bypass sensing.
Labels may use authoritative real cost clocks; reconstruct only independently
auditable real integrals, and check against objective_600.json (not a WM label).
If missing telemetry makes this contract impossible, record INCOMPLETE, show NA
for unavailable methods, and deliver fail-closed code/receipts. Do not fabricate
queues, silently substitute on-road delay, or fit a global-truth value model.

## Frozen models and predictions
State features: lane x spatial segments [-500,-250,0,100,300,600] m, visible
count/density, mean/std speed and mean acceleration; visible CAV count; observed
mainline/ramp entrance queues; causal insertion/request counts; time remaining.
These are geometric bins, not behavioral speed/gap thresholds. Missing coverage
must be explicitly masked and identical on real and imagined states.
V: sklearn HistGradientBoostingRegressor(loss=squared_error, max_iter=150,
max_leaf_nodes=15, learning_rate=.05, l2_regularization=10, random_state=20261031),
no hyperparameter search. Fit targets remaining J through 900 at 30 s grid
300..870 plus terminal 900=0. Weight worlds equally, then arms/states equally.
Terminal prediction at 900 is exactly zero. CPU single-thread fitting.
A: frozen WM2 b8797242 + learned mix, rollout from t0 through 900.
B/C: same WM and CRNs, integral through t0+60/90 plus V of sensed terminal state.
D: separate same fixed regressor, state plus first 1 s command summary (CAV
issued acceleration/target-speed count,mean,std), evaluated without WM.
Only present decision commands, no future realized motion/actions or arm ID.
E: FIT-only arm x t0 mean remaining real J, with no reserved-world labels.
WM K=8 identity-keyed paired futures; average costs before differencing; same
entry/exit, fleet+source rectangle convention and controller bindings for A/B/C.
Bind WM/config/mix/policy/data/code hashes in receipts; never retrain WM here.

## Metrics and decisive gate
400 rows = 8 worlds x 5 times x 10 arm pairs; all methods use the exact same rows.
Sign = exact np.sign agreement; real zero included, predicted zero correct only
for real zero. Report zero counts, Spearman (average ranks; constant => 0), MAE.
World-cluster bootstrap 10,000 draws, rng seed 20261031, percentile 95% CIs,
resampling eight whole worlds with replacement, keeping all times/arms together.
For B and C separately versus EACH of A,D,E: bootstrap sign(B)-sign(baseline)
and MAE(baseline)-MAE(B); all SIX lower bounds must be strictly >0 for PASS.
At least one of B/C must satisfy all six; report both without selecting/tuning.
No claim of final-evaluation generalization; these are reserved TRAIN worlds.
No policy training before PASS and a separate owner-visible registration.

## Delivery and execution
Commit design + shared registration before fitting; commit code/CPU tests; commit
results (WMVALUE.md <=60 lines), hashes, recommendation and shared verdict line.
Use only existing Python environments; nice19 single-thread remote CPU work;
GPU inference only if nvidia-smi shows free compute, otherwise CPU. Own PIDs only.
If PASS, map V bootstrap and WM counterfactual group credit to policy14_cli and
provide a measured GPU-hour estimate before registering any training run.

## Registered MAIN amendment (18:12), execution freeze on resume
Supersedes the source-feature requirement in lines 25-45: no source queue,
backlog, request count, inferred demand, or seed/demand feature is permitted.
Rebuild on-road observations using the existing belief.py projection: own CAV
reports, nearest same-lane front/rear within 100 m bumper gap (nominal 5 m
length, never length_true_m), and RSU zones mainline [-800,225.44), ramp
[-378.66,75.44). Outside RSU, neighbour acceleration is unobserved (zero).
Coverage masks identify geometric bins intersecting RSU/detection coverage;
they do not imply full global sensing. No behavioural threshold is added.
Boundary FLOW = cumulative and trailing-60s first-entry detector counts per
entrance from the prefix only. Registry IDs are used only for deduplication,
never as features. Require first appearances at the physical network boundary;
later first appearances inside the road fail the reconstruction audit.
Known CAVs with recorded internal-junction lane=-1 remain in the self count;
they contribute no lane/segment aggregate (no future lane backfill or truth lane).
V state replaces queue/request channels with these four flow counts. D retains
only commands issued at t0 (the first 1s decision), not realized accelerations;
read explicit nullable commands from the recorded action log to avoid zero
sentinel ambiguity. Fit at t900 uses no command and exact zero terminal value.
Global speed histories and independently reproduced request schedules may be
used ONLY to construct LABELS. Full objective uses the recorded observer's
transition-union/end-speed fleet cost (births included, exits carry last speed)
and end-step source rectangles. Reconcile each objective_120/300/600 clock;
never interpolate missing source clocks or replace full J with on-road J.
Freeze all feature definitions and tests before real fitting. Reserved labels
remain unopened until the complete A-E adapter/coverage gate is ready.
A may reuse only matched seed/start/arm/future/end-clock predictions. Existing
t300-only cumulative tails cannot stand in for later-state conditional starts.
The MAIN amendment authorizes new WM inference only at 60/90s. If reused A is
incomplete, record the exact missing cells and request the smallest explicit
amendment; do not run unregistered new long inference or report a false FAIL.

## Registered MAIN amendment (18:52), adapter-first execution
The shared registration's MAIN18:52 supersedes the long-inference restriction
above: missing A cells may be computed with frozen WM2+mix, K8, through absolute
t900, including native and all five decision times on seeds969432..969439.
B/C60/90 inference is authorized under the same frozen test. V/D/E and all
features, labels, hyperparameters, comparisons and bootstrap rules remain frozen.
Before reserved outcomes or inference, CPU-audit real/imagined sensing parity,
prefix-only 60s restart history, entry/exit tracking, objective clocks and CRN.
Reuse A only when its full restart/controller/entry/cost binding matches; a
different adapter cannot silently mix cached and newly computed predictions.

### Adapter execution contract (frozen before reserved outcomes)
Late restarts align identities across arms and preserve all historical neighbours
while encoding the native121-frame60s history; only current-union columns enter
the rollout. PI/CIRCLES memory is replayed on real prefix observations from300
through t0-1. MAPPO's previous issued acceleration is seeded from t0-1; lane
intent is zero for its frozen native-lane binding. Entrants receive the same
closed-loop controllers. All arms share identity columns, plan/draw seeds and
entry candidate keys; full-long stochastic plans are consumed by prefix for B/C.
V terminal X uses exactly the real projection and real-prefix + imagined-birth
boundary events, never an actual future entry or state. Model training unchanged.
For the WM's existing internal source model only, retain recorded warmup queue
q300 and rate=(inserted300+q300)/300; late initial queue is round(max(0,
q300+rate*(t0-300)-(inserted_t0-inserted300))). This fixed mean prior uses no
seeded request schedule or late real source clock. No queue/rate enters V/D.
Real source request reconstruction remains LABEL ONLY. Cost is transition-union,
end-speed with exits carrying their last speed, plus post-step source rectangles.
Cached WMCALIB J uses predicted exit-step speed instead; its aggregates omit the
per-step exit frames needed to repair that difference. Its present-only t300
slot layout also differs from the late-arm identity-union CRN contract. Prior
coverage was explicitly keys-only, not parity certification. Preserve all cached
files, but classify uncertified A keys as missing under18:52 and compute the
matched A/B/C trajectories together. This is an adapter/clock gate, not selection
by prediction accuracy; no reserved label has been opened at this freeze.

## Registered MAIN amendment (21:12): short-prefix-first stopping rule
Compute all200states' K8 B/C60/90 checkpoints and frozen D/E first. Preserve
full-long stochastic plans, padding, controller and cost clocks; truncate only
execution after90s. Certified completed long trajectories supply their existing
checkpoints; never duplicate an active original task. Frozen V/D/E are not refit.
Use the same400pairs and10,000world-cluster bootstrap draws/seed/95% bounds.
If neither B nor C has strictly positive sign-gain AND MAE-reduction lower bounds
against EACH D and E, the full registered conjunction necessarily FAILS; publish
FAIL with missing A explicitly NA. Only otherwise complete A and apply the full
A/D/E gate. This order change does not relax the PASS criterion. One retry for
an infrastructure exit without Python exception is authorized only after proving
the original process is gone and its immutable output is absent.
