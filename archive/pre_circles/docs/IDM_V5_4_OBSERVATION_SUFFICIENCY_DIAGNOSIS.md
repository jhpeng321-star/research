# IDM v5.4 observation-sufficiency diagnosis

**Status:** completed development-only observation discovery  
**Date:** 2026-08-31  
**Data:** seeds 4401--4520 only  
**Treatment:** frozen IDM damping plus the frozen v5.4 P30 moving wall  
**Not a validation result:** this analysis neither reads nor reinterprets the
frozen 4601--4660 prospective cohort. It does not open 5001--5060, train a new
bundle, change either threshold, or generate a SUMO branch.

## Decision

The current scalar state is **not sufficient for a paper-ready safety-and-benefit
gate**. Dose delivery is partly observable, benefit has real but unstable
information, and safety remains the bottleneck. More independent seeds would
tighten uncertainty but are unlikely to repair the missing safety information
if they repeat the same observation contract.

The next justified experiment is therefore a decision-time, deterministic
10-second shadow rollout of frozen damping versus identical damping plus the
proposed wall. It must project member/follower trajectories, treatment dose,
command authority and inflow response before the real action. The existing saved
rows cannot reconstruct that rollout because they do not contain the required
per-vehicle state or serialized damping-controller state.

This conclusion does not change the frozen v1 prospective question. It only
defines the exact observation repair to consider if that prospective test fails.

## 1. Frozen population integrity

All requested counts reproduce.

| Population | Cases | Independent seeds |
|---|---:|---:|
| Declared decision windows | 480 | 120 |
| Gate-F admissions | 199 | 106 |
| At least 80% requested duration | 192 | 105 |
| Full compact-wall dose qualified | 179 | 101 |
| Full dose-contract failures | 20 | 17 |
| Safe, dose-qualified | 106 | 70 |
| Unsafe, dose-qualified | 73 | 53 |
| Safe and beneficial | 53 | 45 |
| Safe but nonbeneficial | 53 | 38 |
| Joint nonbeneficial, dose-qualified | 126 | 80 |

The duration target and the fitting cohort are intentionally distinct. Seven
admissions fail the 80%-duration target, while 20 fail at least one part of the
full dose contract (duration, coverage, assembly/hold, or compact geometry).
The current Gate-B comparison remains the frozen 179 full-dose cases.

Delay is used only as the frozen noninferiority guardrail. Queue area is not a
selection criterion.

## 2. Feature provenance and leakage audit

The source stores 279 raw feature names. With decision step and penetration
context, 281 fields were audited. Of these, 279 are finite scalars, but only 242
are nonconstant finite physical candidates in the A--G union.

- 37 fields are constant in this cohort.
- 16 additional fields have one value in at least 95% of rows.
- Nine exact-duplicate groups and 60 pairs with absolute Spearman correlation
  at least 0.98 were found.
- `health_path_chronological` is a list in every row and is rejected as a model
  scalar.
- `placement_lane_2_nearest_cav_to_merge_m` is absent/nonfinite in every row and
  is rejected.
- Penetration is fixed at 0.30, required/member lane geometry is fixed at two
  lanes, and lane-2 fields are zero or absent. This cohort cannot identify
  penetration or three-lane effects.

Important duplicate/redundant examples include active-CAV count versus preflight
active-CAV count, decision premerge CAV count versus placement count, local lane
CAV counts versus placement lane counts, several identical time coordinates,
and postmerge count/density/occupancy transforms. The flow-speed product is also
redundant here because proposed wall speed is fixed at 10 m/s.

Every admitted model feature is generated from the common damping prefix through
the declared decision or from deterministic Gate-F preflight at that decision.
`preflight_*` and `obs_v2_dose_*` are treatment-aware proposals/projections, not
realized outcomes. The analyzer never puts these postdecision objects into a
model vector:

- `treatment_dose`, actual completion and release reason;
- treatment/control arm outcomes or realized control deltas;
- `safety_service_oracle`, oracle labels or counterfactual effects;
- seed identity or absolute vehicle identity.

The A--F families are physically interpretable comparison blocks but overlap by
design: rear-safety summaries also occur in temporal, lane, geometry and
preflight blocks. The leave-family-out analysis therefore measures removal of a
physical information block, not a disjoint causal decomposition.

## 3. Current-model errors at the frozen thresholds

The current selected development model is the primary shallow booster. At the
unchanged safety threshold 0.70 and conditional-benefit threshold 0.75:

| Decision | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| Safety head on 179 full-dose cases | 29 | 15 | 77 | 58 |
| Benefit head on the 106 observed-safe cases | 14 | 8 | 39 | 45 |
| Final joint command | 5 | 5 | 48 | 121 |

The joint rule commands ten cases across nine seeds. Eight are safe and five are
jointly beneficial. Thus its development precision is 0.80 for safety and 0.50
for joint benefit, with only 0.094 benefit recall. These are sparse exploratory
development results, not calibrated or prospective claims.

Opposite outcomes occur in physically close cross-seed cases. Examples include
seed 4450/step 700 versus 4464/step 680 for safety and seed 4445/step 620 versus
4492/step 620 for benefit. In the latter pair, the wall changes discharge in
opposite directions despite similar observed state. The remaining differences
are mostly CAV placement, lane inflow, compression persistence, preflight span,
and local acceleration. Those are exactly the quantities a member/follower
trajectory projection should resolve more directly.

## 4. Univariate information

Univariate AUCs are direction-invariant and accompanied in the machine report
by signed rank correlation, mutual information, and 200 seed-clustered bootstrap
resamples. They are discovery statistics, not selected-model performance.

### Safety

The best single safety signals are weak:

- lag-30 soft-band lane coverage: AUC 0.630, clustered 95% interval 0.562--0.688;
- lag-30 soft-band CAV count: 0.624, 0.553--0.694;
- 60-second premerge gap autocorrelation: 0.619, 0.536--0.702;
- lag-5 density: 0.612, 0.538--0.704;
- postmerge speed slope: 0.612, 0.532--0.704.

These describe CAV availability and traffic persistence, but none directly
projects what the selected wall will do to its followers.

### Conditional benefit

Benefit contains stronger state information:

- 60-second change in premerge speed standard deviation: AUC 0.725,
  clustered 95% interval 0.641--0.813;
- lane-1 speed standard deviation: 0.682, 0.571--0.771;
- lane-1 mean speed: 0.673, 0.561--0.770;
- 60-second premerge speed change: 0.664, 0.565--0.768;
- postmerge speed variance: 0.654, 0.551--0.785.

These support the physical idea that a wall's value depends on wave growth,
lane-specific state and remaining downstream service, not a single health value.
They do not yet yield a stable deployable policy.

### Duration dose

The strongest duration-dose signals are preflight quantities:

- minimum front TTC: AUC 0.923, clustered 95% interval 0.793--0.995;
- minimum projected acceleration: 0.899, 0.806--0.969;
- damping-command count: 0.818, 0.715--0.891;
- minimum rear TTC: 0.802, 0.613--0.934;
- release-boundary margin: 0.783, 0.651--0.938.

Only seven admissions fail duration, so these estimates are promising but
data-sparse. Full dose-contract failure has 20 cases across 17 seeds.

## 5. Seed-grouped family/model screen

Every family uses the same five outer seed-grouped folds. Model choice between a
regularized logistic model and the existing shallow booster occurs only inside
each outer training fold.

| Family | Features | Dose AUC | Safety AUC | Benefit AUC | Commands/seeds | Safety precision | Benefit precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| A static traffic | 25 | 0.525 | **0.618** | 0.546 | 6/6 | 0.833 | 0.167 |
| B temporal summaries | 117 | 0.406 | 0.518 | 0.575 | 12/11 | 0.417 | 0.333 |
| C lane-resolved traffic | 53 | 0.455 | 0.505 | 0.527 | 12/12 | 0.417 | 0.000 |
| D wall geometry/authority | 29 | 0.523 | 0.484 | 0.494 | 7/6 | 0.714 | 0.429 |
| E preflight dose/assembly | 14 | 0.789 | 0.450 | 0.527 | 2/2 | 0.500 | 0.000 |
| F follower/rear safety | 27 | 0.730 | 0.443 | 0.621 | 3/3 | 0.333 | 0.000 |
| G combined physical | 242 | **0.800** | 0.539 | **0.628** | 16/16 | 0.688 | 0.500 |

No row passes both frozen policy thresholds as a useful policy family. Family A
attains safety precision by commanding only six cases and has almost no benefit
recall. The combined family commands 16 independent seeds, but remains below the
0.70 safety and 0.75 benefit precision requirements.

The stricter fully nested selection, in which feature family and model are both
selected inside each outer fold, gives:

- duration dose AUC 0.712;
- safety AUC **0.485**;
- conditional-benefit AUC **0.569**;
- nine commands on nine seeds, 0.556 safety precision, 0.222 joint-benefit
  precision, and 0.038 benefit recall.

This collapse is the decisive warning against reading the best individual
family after screening as a verified improvement. The current selected model's
development AUCs are 0.562 for safety and 0.642 for conditional benefit. The new
combined benefit result (0.628) is not a 0.03 improvement, and the apparent
static-safety result does not survive fully nested family selection.

Leave-family-out results agree with the physics:

- removing temporal summaries reduces combined benefit AUC from 0.628 to 0.537;
- removing preflight dose projections reduces dose AUC from 0.800 to 0.435;
- removing follower/rear summaries reduces dose AUC to 0.509;
- no safety ablation establishes a robust family; all safety results remain
  close to chance.

Shuffled-label grouped controls center near 0.5. Their wide intervals, especially
for duration dose, reflect small negative support and reinforce the need for
independent-seed sensitivity rather than a training correlation claim.

## 6. Observation ceiling

Under cross-seed nearest-neighbor evaluation of the 242-feature combined state:

| Target | 1-NN accuracy | 5-NN AUC | Opposite label among closest quartile |
|---|---:|---:|---:|
| Duration dose | 0.925 | 0.508 | 0.060 |
| Safety | 0.559 | 0.561 | 0.400 |
| Conditional benefit | 0.491 | 0.602 | 0.519 |

The current primary and expanded contracts show the same pattern: safety is
approximately chance and close observations often have opposite safety or
benefit labels. This is evidence of an observability ceiling for the saved
aggregates, not proof that the underlying event is fundamentally unpredictable.
Metastability creates real uncertainty, but the present observations also omit
the variables most likely to resolve local divergence: selected-member and
follower trajectories, command saturation, assembly convergence, replacement
availability and the wall's predicted inflow effect.

## 7. What can and cannot be reconstructed now

The saved rows can recover current member/lane geometry, t0 min gap/TTC,
projected static span, lane/bin traffic aggregates, causal flow/discharge
history, storage proxies, speed gradients and compression persistence. They can
also derive a 60-second excess-demand-rate proxy as upstream arrival flow minus
downstream discharge.

They cannot reconstruct an exact dynamic shadow rollout. Missing information
includes:

- per-vehicle position, speed, acceleration, IDM parameters and route state;
- leader/follower identities and member-specific gaps/relative speeds;
- serialized damping references, filtered actions, cooldown/release state and
  projected commands;
- same-lane reserve candidates and their headroom;
- projected TTC/gaps at 1, 2, 5 and 10 seconds;
- required follower deceleration/compression, lane TTC quantiles;
- assembly/convergence time, authority/saturation, replacement stability;
- projected wall-induced inflow reduction and true short-horizon shock motion.

`obs_v2_dose_flow_speed_product` is not an inflow-reduction estimate. Actual
postdecision control deltas could reveal some missing quantities but are
forbidden leakage and must not be used.

## 8. Evidence-ranked recommendation

### Retain as compact current context

The following ten scalars are the smallest defensible current discovery block,
after avoiding clocks, constants and obvious duplicate transforms:

1. `pattern60_pre_merge_speed_std_mps_change`
2. `pattern60_pre_merge_speed_mps_change`
3. `local_lane_1_speed_std_mps`
4. `obs_v2_premerge_gap_lag1_autocorr_60s`
5. `traffic_lag_30_soft_band_lane_coverage_fraction`
6. `traffic_lag_5_density_scaled`
7. `preflight_min_front_ttc_s_capped`
8. `preflight_min_rear_ttc_s_capped`
9. `preflight_min_acceleration_mps2`
10. `obs_v2_dose_release_boundary_margin_m`

This list is a compact starting context, not a frozen v2 model and not evidence
that ten current scalars alone are sufficient.

### Add as the highest-priority new information

Add a deterministic, side-effect-free 10-second shadow with two twins from one
decision capture: frozen damping and identical damping plus the frozen v5.4
wall. At 1, 2, 5 and 10 seconds emit projected dose/assembly, member span and
coverage, front/rear TTC and gaps, follower compression/required deceleration,
affected-vehicle count, command saturation/authority, reserve availability,
crossing/inflow reduction, excess-demand/storage delta and shock persistence.

### Reject or collapse

Exclude raw clock duplicates, constant/near-constant health fields, dormant
lane-2 fields, duplicate count/scaled variants and the fixed-speed flow product.
Never admit realized dose, completion/release reason, treatment outcomes, oracle
labels, counterfactual effects, seed or absolute vehicle identity.

## 9. Exact next experiment if prospective v1 fails

1. Preserve the failed v1 result without reinterpretation or threshold changes.
2. Freeze the review-only v2 extractor and shadow equations in
   `docs/IDM_V5_4_OBSERVATION_V2_REVIEW_PROTOCOL.md`.
3. Recapture all 480 decision states for the already-opened development seeds
   4401--4520 from their common damping prefixes. Compute shadows only for the
   199 Gate-F admissions and join them to the already-frozen labels. Do not add
   new outcomes in this repair screen.
4. Before fitting, pass determinism, prefix identity, order invariance,
   outcome-mutation leakage, toy safety monotonicity and extractor-hash tests.
5. Repeat the exact nested seed-grouped screen with logistic and shallow
   boosting models. Keep 0.70/0.75 thresholds. Require either safety or benefit
   AUC to improve by at least about 0.03 **and** require meaningful improvement
   in precision/coverage without collapsing command count or independent seeds.
6. Only after that gate passes, freeze one v2 contract/bundle and evaluate on
   60--120 newly allocated independent seeds. Exclude used v1 seeds 4601--4660
   and reserved paper seeds 5001--5060.

Estimated four-worker time is 1.5--2.5 hours to recapture 120 development
prefixes, under about 15 minutes for pure shadows on 199 admissions, and about
3--6 hours for an untouched 60-seed paired validation after a v2 freeze. Benchmark
the first four seeds before scaling.

## 10. Artifacts

- Machine report:
  `runs/eval/idm_wall_v5_4_observation_sufficiency_diagnosis/report.json`
- Cross-fitted predictions:
  `runs/eval/idm_wall_v5_4_observation_sufficiency_diagnosis/model_predictions.json`
- Review-only machine contract:
  `runs/eval/idm_wall_v5_4_observation_sufficiency_diagnosis/proposed_v2_observation_contract.json`
- Reusable analysis:
  `counterfactual/idm_observation_sufficiency.py`
- Runner:
  `tools/analyze_idm_v5_4_observation_sufficiency.py`
- Focused tests:
  `tests/test_idm_observation_sufficiency.py`

The machine report records `validation_or_final_outcomes_read: false` and
`new_sumo_branches_generated: false`.
