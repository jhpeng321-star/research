# IDM v5.4 shadow-observation v2 development results

## Decision

The frozen development-only observation repair is a **confirmatory no-go**.
No v2 observation block passed the complete leakage-safe observation and
threshold-policy contract. The exact 10-second wall shadow improved
conditional-benefit ranking, and the combined trajectory-plus-shadow block
slightly improved safety ranking, but neither produced a sufficiently safe and
precise deployable escalation policy.

This is an observation-model result, not evidence that the verified wall
mechanics or deterministic feasibility supervisor failed. No validation or
final partition was opened, no threshold or label was changed after seeing the
result, and no additional SUMO data were generated for the analysis.

## Frozen population and integrity

- Development seeds: 4401--4520 only; 120/120 seed captures valid.
- Decision records: 480 total; Gate-F-admitted wall shadows: 199 across 106
  independent seeds.
- Dose: 179 successes and 20 failures.
- Safety, among dose-qualified cases: 106 safe and 73 unsafe.
- Conditional joint benefit, among safe dose-qualified cases: 53 beneficial
  and 53 nonbeneficial.
- All observations end at the declared decision except the separately declared
  10-second treatment shadow; the shadow uses exact damping and wall twins from
  the same decision state.
- All outer and inner folds were grouped by complete seed. Model-family and
  threshold selection occurred inside grouped training folds.
- Validation seeds 4601--4660 and final seeds 5001--5060 were not read.

The completed capture contract SHA-256 is
`720117ab1b01dd471828dd87684aea217b341c233a6c626a434a2d46420d7b4c` and
the immutable manifest SHA-256 is
`2eed9f0c574b9b869a471ec3c7642e45563ed9127ff1e54a051133fa4bda99ce`.

## Nested grouped prediction results

The exact historical v1 block was recomputed on the same cases and the same
five outer/four inner complete-seed folds. AUC gains below are matched against
that exact v1 result, not against a value from a different cohort or split.

| Observation block | Dose AUC | Safety AUC | Conditional-benefit AUC | Safety gain | Benefit gain |
|---|---:|---:|---:|---:|---:|
| Exact v1 baseline | 0.730 | 0.580 | 0.625 | -- | -- |
| Deployable v1 without health | 0.730 | 0.579 | 0.641 | -0.001 | +0.015 |
| Exact 10-second wall shadow | **0.978** | 0.578 | **0.677** | -0.002 | **+0.051** |
| Physical trajectory, 20 seconds | 0.641 | 0.563 | 0.607 | -0.017 | -0.019 |
| Physical trajectory plus shadow | 0.962 | **0.601** | 0.636 | **+0.021** | +0.010 |

The shadow-only conditional-benefit gain remained positive under every
single-seed removal (minimum gain +0.033), but its seed-clustered 95% bootstrap
interval was wide and crossed zero (-0.094 to +0.180), and held-out log loss
deteriorated. It therefore did not qualify as a calibrated confirmatory
improvement. The combined block's safety gain also remained positive under
single-seed removal (minimum +0.005), but was smaller than the frozen +0.03
materiality threshold and its bootstrap interval crossed zero (-0.082 to
+0.120).

The main interpretable distinction is:

- **Treatment delivery is observable.** The exact shadow identifies dose
  success very well (AUC 0.978 versus 0.730).
- **Treatment safety is still weakly observable.** The best shallow safety AUC
  is only 0.601.
- **Some conditional-benefit information exists in the proposed-action
  shadow**, but it is not sufficient to support the required joint
  safety-and-benefit policy.

## Cross-fitted threshold-policy results

Thresholds were chosen independently inside each grouped training fold and
then applied to its held-out seeds. These are pooled held-out decisions.

| Observation block | Commands | Command seeds | Dose success | Safety | Joint-benefit precision | Unsafe | Nonbeneficial | Excess collision/teleport |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact v1 baseline | 11 | 9 | 100.0% | 54.5% | 45.5% | 5 | 6 | 0 |
| Deployable v1 without health | 8 | 7 | 87.5% | 50.0% | 50.0% | 4 | 4 | 0 |
| Exact 10-second wall shadow | 21 | 18 | 100.0% | 61.9% | 47.6% | 8 | 11 | 0 |
| Physical trajectory, 20 seconds | 11 | 11 | 90.9% | **72.7%** | 36.4% | 3 | 7 | 0 |
| Physical trajectory plus shadow | 19 | 18 | 100.0% | 68.4% | 42.1% | 6 | 11 | 0 |

Every v2 policy met the minimum command/seed count and avoided excess
collision or teleport. None met both frozen deployment rates: at least 90%
selected safety and at least 50% selected joint-benefit precision. None even
met the explicitly nonconfirmatory 80% safety / 45% benefit enrichment rule.
Selected-case inventories, including every unsafe, nonbeneficial,
physically-harmful and failed-dose command, are preserved with the machine
report.

## Raw causal sequence screen

A bounded temporal convolutional model was nested by complete seed and tested
on the anonymous masked histories. It overfit badly and did not recover safety
or benefit:

| History window | Safety AUC | Conditional-benefit AUC | Mean training safety AUC | Mean training benefit AUC |
|---|---:|---:|---:|---:|
| 20 seconds | 0.518 | 0.520 | 0.983 | 0.999 |
| 60 seconds | 0.532 | 0.540 | 0.963 | 0.983 |

The 20-second sequence model improved dose AUC over the physical trajectory
block (0.689 versus 0.641), but both sequence windows were inferior to the
shallow physical representation for safety and benefit. Shuffled-label AUC
distributions remained near chance. Per the protocol, the smaller physical
representation is retained and the raw sequence encoder is rejected.

## Diagnosis and scientific consequence

The v2 observations separate dose feasibility much better than treatment
outcome. In the combined feature space, 48.9% of close safety neighbors and
44.4% of close conditional-benefit neighbors from different seeds have the
opposite label; five-neighbor AUC is 0.594 for safety and 0.512 for benefit,
versus 0.775 for dose. This is consistent with unresolved metastable outcome
variation rather than a simple missing-history problem.

Accordingly:

1. observation repair is **not promoted** to validation or final evaluation;
2. no paper-ready learned escalation predictor is claimed;
3. deterministic Gate F, the damping-preserving wall mechanics, treatment-dose
   boundary and negative learned-gate evidence remain valid contributions;
4. further feature searching on this frozen treatment/population should stop
   unless the scientific target, intervention, or measurement design is
   separately justified and frozen in advance.

## Post-screen safety-mechanism audit

All 73 unsafe dose-qualified cases were audited from the paired development
records and the captured event samples.  The classification below uses the
first wall-arm hard-brake onset or low-TTC exposure that is absent from its
matched damping arm.  It is an event-alignment diagnosis, not proof that the
named phase caused the later event.

| Earliest evidenced divergence | Cases | Seeds | Time after activation (min/median/max) | Main interpretation |
|---|---:|---:|---:|---|
| Assembly-induced braking | 0 | 0 | -- | Not supported as the first stored divergence. |
| Follower compression during HOLD | 0 | 0 | -- | Not supported as the first stored divergence. |
| Merge/downstream disturbance | 40 | 35 | 25/48.5/165 s | Delayed remote divergence: 39 hard-brake events and one low-TTC event; 25 affected HDVs and 14 affected CAV hard-brakers. |
| Wall-member handoff | 0 | 0 | -- | Every unsafe case records zero handoffs. |
| Taper/release-localized | 16 | 16 | 16/16/26 s | Twelve begin during taper and four within five seconds after the last override; eight affect wall members and one an immediate follower. |
| Already unrecoverable at decision | 0 | 0 | -- | Every case records `sustained_congestion=false` at the decision. |
| Other delayed premerge divergence | 17 | 16 | 28/41/93 s | All first events are on premerge lane 0, are not wall-adjacent, and occur 7--72 seconds after release. |

The label composition is 65 hard-brake-only failures, five hard-brake plus
low-TTC failures, and three low-TTC-only failures.  No unsafe case adds a
collision or teleport.  The taper/release-localized cases are the only group
with a direct, plausibly repairable controller association: ten record an
explicit safety abort, and their median onset acceleration is -4.5 m/s2.
The merge/downstream cases begin a median 27.5 seconds after the actual
release, while the exact ten-second shadow predicts no median queue or
discharge response.  The present observation horizon therefore cannot assign
those delayed events to release timing, a particular merge, or a follower
chain.

The two largest descriptive groups contain 56/73 cases (76.7%), but they are
not two established repairable mechanisms.  Only 16/73 cases (21.9%) are
localized to taper/release.  The audit consequently does **not** establish
that at least 70% of unsafe outcomes share one or two controller repairs.
Complete postdecision trajectories immediately before each adverse event were
not retained, so that stronger causal claim is intentionally withheld.

## Density and collapse-headroom interpretation

The current 2700-veh/h dose-qualified development population contains 175
cases where neither arm sustains congestion, two wall preventions, two cases
where both arms sustain congestion, and no wall-induced collapse.  Four
control collapses and two preventions do not provide the three-seed positive
and three-seed negative diversity required for a primary collapse-prevention
claim.

Independent fresh IDM sensitivity evidence gives a reason to test density and
duration rather than add more identical 2700-veh/h cases:

| Mainline demand | No-op collapse | Damping collapse | Wall collapse | Interpretation |
|---:|---:|---:|---:|---|
| 2200 veh/h | 8/12 | 9/12 | 2/12 | Transition headroom exists in the inspected cohort. |
| 2400 veh/h | 12/12 | 12/12 | 4/12 | Wall response remains diverse. |
| 2700 veh/h | 12/12 | 12/12 | 12/12 | Structural-overload/service stress unless new evidence shows recoverable headroom. |

These are empirical regimes, not a declaration that one nominal demand is a
universal critical density.  The next experiment must persist nominal demand
and measured predecision flow, density, occupancy, queue, speed variance,
discharge, ramp-arrival phase and available wall runway.

## Consolidated decision

The present no-go is a combination with a clear order:

1. **Treatment safety and density-specific treatment headroom are primary.**
   Raw completed-treatment safety is far below the required 90%, and most
   unsafe divergence occurs after wall authority in remote traffic rather
   than in an observable formation defect.
2. **Observation insufficiency is secondary.**  The shadow contains useful
   benefit and dose information, but no learned gate should be asked to rescue
   an actuator with this raw safety rate.
3. **Insufficient duration is plausible but unverified.**  The 20-second
   request includes assembly and the five-second taper, leaving a short HOLD
   in difficult formations.  A matched 20/30/40-second screen is warranted,
   but it must include an early/late timing contrast and a common terminal
   horizon.
4. **The first conditional revision is closed-loop adaptive duration.**  It is
   authorized for design only if a longer fixed arm helps safely.  Until that
   discriminator passes, no adaptive release controller is implemented.

The approval-bound experiment is frozen in
`docs/IDM_WALL_V5_5_DENSITY_DURATION_HEADROOM_PROTOCOL.md`.  Gate F remains the
deterministic safety/feasibility/dose supervisor.  Gate B remains a later
conditional-benefit learner among Gate-F-admissible treatments; the learned
model is not responsible for compensating for unsafe treatment mechanics.

## Reproducible artifacts

- Frozen protocol: `docs/IDM_V5_4_SHADOW_OBSERVATION_V2_PROTOCOL.md`
- Capture summary:
  `runs/eval/idm_wall_v5_4_shadow_observation_v2_development/capture/summary.json`
- Machine-readable analysis:
  `runs/eval/idm_wall_v5_4_shadow_observation_v2_development/analysis/report.json`
- Out-of-fold predictions:
  `runs/eval/idm_wall_v5_4_shadow_observation_v2_development/analysis/oof_predictions.json`
- Selected-case inventories:
  `runs/eval/idm_wall_v5_4_shadow_observation_v2_development/analysis/selected_case_inventories.json`

SHA-256 checksums verified during the post-screen audit are:

- `report.json`: `834dd6867ee2b56fb5a0731ab509132fcd69f24d9c5921c73eb5ce7760a0ef89`;
- `oof_predictions.json`: `5452ffb4c65ae7e392c5f32dc7600b479cc119d4feeef66360871af0f944d3b8`;
- `selected_case_inventories.json`: `6cfdc9b98a0fb8618e6fddef5dd1ecdeff90079a0dc0288cdee8a814419224e0`.
