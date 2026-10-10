# Counterfactual wall-win sensitivity (local-v2)

## Bottom line

The frozen label calls soft a win only when it is Pareto-nonworse on pre-merge sustained congestion, pre-merge queue area, aggregate incremental delay, and merge discharge, with at least one strict improvement. Safety is then imposed as a separate hard gate.

That definition is too restrictive for an inflow-control wall: the wall deliberately creates upstream holding, and both queue area and SUMO timeLoss can therefore rise even when bottleneck service is preserved. More importantly, the frozen congestion label is measured on the same pre-merge edge where the wall operates, so it can mix an intentional control queue with an uncontrolled bottleneck collapse.

The recommended candidate label is **merge service, exact throughput**: the soft arm must be safety-admissible, must not lower paired merge discharge or mean merge speed, and must strictly improve at least one of those two quantities. Queue and delay remain mandatory reported secondary costs, but they do not veto the label. The one-vehicle throughput-tolerance version is a sensitivity check, not the primary claim, until a longer recovery horizon demonstrates catch-up.

## What each rule counts

| Rule | Definition |
|---|---|
| Frozen Pareto rule | No worse pre-merge congestion, queue area, delay, or merge discharge; strictly better on at least one. |
| 2% delay non-inferiority | Frozen rule except aggregate delay may increase by at most 2% when another outcome strictly improves. |
| 5% delay non-inferiority | Frozen rule except aggregate delay may increase by at most 5% when another outcome strictly improves. |
| Delay excluded | No worse pre-merge congestion, queue area, or merge discharge; delay is reported but does not determine the label. |
| Pre-merge stability + throughput | No induced pre-merge sustained congestion and no merge-discharge loss; strictly improve congestion state or discharge. Queue and delay are reported costs. |
| Merge service, exact throughput | No loss of mean merge speed or merge discharge; strictly improve at least one. Pre-merge queue, pre-merge congestion, and delay are reported costs. |
| Merge service, one-vehicle tolerance | Mean merge speed must not fall and merge discharge may fall by at most one vehicle over the outcome horizon; strictly improve speed or discharge. |
| Collapse rescue, one-vehicle tolerance | The soft wall must turn an off-arm pre-merge congestion event into a non-event, with at most one fewer merge discharge. |

## Development paired cases

Eligible paired cases: 90.

| Rule | Traffic wins | Safe wins | Rescues | Induced pre-merge congestion | Policy discharge gain (veh/case) | Policy merge-speed gain (m/s) | Policy delay change (s/case) | Policy queue change (veh-s/case) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen Pareto rule | 36 | 32 | 9 | 0 | 2.93 | 0.169 | -424.31 | -457.97 |
| 2% delay non-inferiority | 37 | 33 | 9 | 0 | 2.93 | 0.169 | -423.95 | -460.02 |
| 5% delay non-inferiority | 38 | 34 | 9 | 0 | 2.96 | 0.173 | -418.21 | -461.62 |
| Delay excluded | 38 | 34 | 9 | 0 | 2.96 | 0.173 | -418.21 | -461.62 |
| Pre-merge stability + throughput | 44 | 40 | 10 | 0 | 3.21 | 0.184 | -361.57 | -390.28 |
| Merge service, exact throughput | 43 | 41 | 9 | 1 | 3.14 | 0.201 | -341.00 | -383.51 |
| Merge service, one-vehicle tolerance | 48 | 45 | 9 | 1 | 3.10 | 0.219 | -343.52 | -399.04 |
| Collapse rescue, one-vehicle tolerance | 10 | 10 | 10 | 0 | 0.87 | 0.047 | -117.93 | -127.93 |

For the recommended candidate:

- 41 safe soft cases are selected; 2 additional traffic-effective cases fail the frozen safety budget.
- The oracle-labelled policy changes mean merge discharge by 3.14 vehicles per case and mean merge speed by 0.201 m/s versus always off.
- Delay rises in 9 selected cases (mean positive cost 732.75 s; maximum 1493.17 s).
- Pre-merge queue area rises in 10 selected cases (mean positive cost 579.40 veh-s; maximum 1325.00 veh-s).

## Locked paired cases

Eligible paired cases: 30.

| Rule | Traffic wins | Safe wins | Rescues | Induced pre-merge congestion | Policy discharge gain (veh/case) | Policy merge-speed gain (m/s) | Policy delay change (s/case) | Policy queue change (veh-s/case) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen Pareto rule | 8 | 8 | 1 | 0 | 1.67 | 0.062 | -343.85 | -441.87 |
| 2% delay non-inferiority | 9 | 8 | 1 | 0 | 1.67 | 0.062 | -343.85 | -441.87 |
| 5% delay non-inferiority | 9 | 8 | 1 | 0 | 1.67 | 0.062 | -343.85 | -441.87 |
| Delay excluded | 9 | 8 | 1 | 0 | 1.67 | 0.062 | -343.85 | -441.87 |
| Pre-merge stability + throughput | 11 | 10 | 1 | 0 | 2.00 | 0.103 | -270.95 | -330.87 |
| Merge service, exact throughput | 8 | 8 | 0 | 0 | 1.60 | 0.115 | -215.09 | -225.37 |
| Merge service, one-vehicle tolerance | 8 | 8 | 0 | 0 | 1.60 | 0.115 | -215.09 | -225.37 |
| Collapse rescue, one-vehicle tolerance | 1 | 1 | 1 | 0 | 0.23 | -0.007 | -38.63 | -42.53 |

For the recommended candidate:

- 8 safe soft cases are selected; 0 additional traffic-effective cases fail the frozen safety budget.
- The oracle-labelled policy changes mean merge discharge by 1.60 vehicles per case and mean merge speed by 0.115 m/s versus always off.
- Delay rises in 3 selected cases (mean positive cost 552.15 s; maximum 943.88 s).
- Pre-merge queue area rises in 3 selected cases (mean positive cost 381.33 veh-s; maximum 754.00 veh-s).

## Can the current features predict the revised win?

No. The new label is better aligned with the wall mechanism, but relabeling does not make the current pre-action features predictive.

- Nested grouped development: merge-speed gain R2=-0.085, discharge gain R2=-0.079; both are worse than predicting the development mean.
- The cross-fitted gate activates 62 of 90 cases, with precision 41.9% and recall 63.4%. It also makes 10 unsafe activations under the observed frozen safety budget.
- Its development policy averages 0.93 additional merge discharges and 0.051 m/s merge-speed gain per case versus always off, but this does not transfer.
- On the already-consumed retrospective split, it activates 24 of 30 cases with precision 29.2%, including 1 unsafe activation.
- That retrospective predicted policy loses 0.97 merge discharges per case and 0.044 m/s mean merge speed, while adding 225.49 s delay and 94.47 veh-s queue area per case versus always off.

Therefore the revised rule is a label/design improvement, not evidence of a working deployable gate. The next model must use early post-pulse response features or a randomized probe-and-commit design; more seeds alone will not fix a feature set whose grouped signal is currently below baseline.

## Interpretation for the paper

Use Krauss to test whether the wall can preserve bottleneck service under stochastic car-following heterogeneity. Do not claim that its timeLoss response reproduces the full economic cost of a real traffic breakdown. Present the available IDM collapse result as a model-form stress test showing that, under a car-following model with stronger collapse-delay amplification, the temporary wall cost can be offset by avoided breakdown loss. Insert the measured IDM effect size and uncertainty before making that numerical claim.

This analysis is post-hoc and oracle-labelled: it changes what an observed paired case counts as, but it does not repair the failed pre-action predictor. If this label is adopted, retrain a two-head gate for merge-speed gain and discharge gain and evaluate it once on new untouched seeds. Also add a merge-edge sustained-collapse trace metric in the next data generation so the primary stability endpoint is not measured inside the wall's holding zone.

## Sources

- Development data: `D:\research-dense\runs\probe\counterfactual_local_v2_dev_combined_s101_130_steps120_180_240.jsonl`
- Development safety audit: `D:\research-dense\runs\probe\counterfactual_local_v2_dev_s101_130_steps120_180_240.readiness.json`
- Locked data: `D:\research-dense\runs\probe\counterfactual_local_v2_locked_s201_210_steps120_180_240.jsonl`
- Locked safety audit: `D:\research-dense\runs\probe\counterfactual_local_v2_locked_s201_210_steps120_180_240.readiness.json`
- Revised gate diagnostic: `D:\research-dense\runs\probe\counterfactual_merge_service_gate_local_v2.report.json`
