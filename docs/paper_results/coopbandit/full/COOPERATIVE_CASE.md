# A measured beneficial interaction (selected TRAIN example)

State `s981057_t420`, O3 receiving-lane yield gap, setting3 = target headway3s
for60s, then MAPPO. Full configured-SUMO reruns match the baseline prefix bitwise.
The fixed nearest-two CAVs are mainline.154 and mainline.156 in lane1,
positions -3.53/-11.12m and speeds0.68/0.00m/s. Their observed front gaps are
2.75/2.59m. Lane1 has16 observed vehicles,14 slow-queue detections, mean1.55m/s;
four ramp-lane vehicles are observed. The adjacent mainline lane averages28.32m/s.
These are descriptive observations, not a learned execution threshold.

At600s after decision, fleet+source delay saving relative to MAPPO:
- BOTH members together: D_G = +755.3875507190533 veh-s.
- Member0 alone, same joint context: D_0 = -5203.751958260327 veh-s.
- Member1 alone, same joint context: D_1 = -1481.942826651306 veh-s.
- Sum individual effects = -6685.694784911633 veh-s.
- C_G = D_G - sum(D_i) = +7441.082335630686 veh-s.

This is the ONE available O2/O3 measured solo-credit case in the complete panel
with both D_G>0 and C_G>0. Twelve available cases are measured; the other68
registered option/state credit cases are unavailable exact no-ops. This selected
example demonstrates useful non-additivity in TRAIN twins; it is not evidence
of a successfully learned live selection policy or a representative average.

Exact rows: `exam/group_credit.csv` and
`registered-summary/beneficial_interaction_cases.csv`. Observable input is in
`exam/states.jsonl`; all fields/settings are documented in `../EXAM_SCHEMA.md`.

Full1Hz observable/action tapes for S-COOPPLAN / S-COOPBEHAV:
- Reference cpuQ `/home/ubuntu/coopbandit20261031_v3/jobs/s981057_screen/observable.jsonl.gz`.
- Joint gpuA `/root/autodl-tmp/coopbandit20261031_labels_v4/jobs/s981057_t420_O3/observable.jsonl.gz`.
- Solo0 same gpuA root `/jobs/s981057_t420_O3_solo0/observable.jsonl.gz`.
- Solo1 same gpuA root `/jobs/s981057_t420_O3_solo1/observable.jsonl.gz`.
Each originating job RESULT.json binds its tape SHA; CLOSE.json records actual
and precharged duration. Geometry is only the full RSU/CAV-observable roster;
do not infer unobserved neighbours/driver truth or native-release attribution.
