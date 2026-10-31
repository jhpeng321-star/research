# Exact counterfactual EXAM v1 — consumer contract

The fixed complete panel is 60 fresh TRAIN worlds x five decision clocks
300/360/420/480/540. Partial exports contain only completed all-option states.
`MANIFEST.json` hashes original file BYTES (CSV CRLF and JSON LF/CRLF preserved).
Do not normalize newlines before checking hashes. The Git attributes preserve bytes.

## Inputs and metadata
`states.jsonl` has one record/state: `state_id`, `world_id`, `t0`, `features`,
`observed_vehicles`, `prefix_sha256`. The roster includes sensed vehicle ID,
longitudinal position `s_m`, speed `v_mps`, lane0/1/mainline or2/ramp, known CAV
telemetry flag. It is RSU/CAV-observable, not a complete hidden simulator state.
Source IDs/world IDs are joins, never predictors. `cell`, demand, true share and
world seeds must never enter a learned predictor. Missing group-member features
use -1; no observed leader/follower uses the recorded1000m gap sentinel.
Lane density is sensed count per0.17544km; queue is sensed speed<5m/s count,
not a hidden source-queue total. Gap features use nominal5m geometry.

The action setting is part of the context: 0=low/30s,1=low/60s,
2=high/30s,3=high/60s. O1/O2 speed decrement low3/high6m/s;
O3/O4 target headway low2/high3s. Group is the nearest TWO observable upstream
mainline CAVs in[-100,0]m; keep fixed identities during that option window.
O1=L2 metering, O2=L3 cross-lane wall (members span mainline lanes),
O3=L4 receiving-lane yield gap (both lane1 and observed ramp in merge zone),
O4=L1 longer gap. O0 is the unchanged MAPPO169 speed-only controller.
O3 turns off when its observed ramp/lane condition no longer holds.
Exact target construction/availability is `tools/coopbandit/protocol.py` plus
the frozen leverprobe intervention contract; native lanes/safety are retained.

## Labels and credit
`labels.csv` is long state x optionO0..O4 x horizon120/300/600 seconds AFTER t0.
Primary label `D=J(O0)-J(option)` (veh-s saved; positive helpful), where J is
fleet speed-deficit + source wait, with components `D_road` and `D_source`.
`available=False` means measured exact MAPPO/no-op, not an imputed missing effect.
Safety columns are outcome-horizon counters; `override_share` is the whole
option-branch intervention audit share, repeated across its horizon rows.
Unavailable/no-request options have override_share0.
`group_credit.csv` covers only the registered solo subset: O2/O3 at40states.
Solos preserve the joint membership/targets and alter only the named member.
`C_G=D_G-(D_0+D_1)`; positive C_G alone is not beneficial cooperation.
Require BOTH positive D_G and positive C_G when claiming beneficial interaction.

## Splits and inference
Split by world_id across all clocks/options. Registered OOF world fold is in
`learned/oof_predictions.csv`; all rows of a world share its fold. Live dev
981080–981091 is absent from fitting/calibration and must remain so.
`worlds.json` retains the full prospective selection even in partial snapshots;
actual partial membership comes from `states.jsonl`, not that selection list.
Primary summaries weight each world equally. Complete worlds have five clocks;
partial equal-world and equal-state helpful rates can therefore differ.
Use availability-aware sign/rank scoring, report unavailable no-ops separately,
and distinguish training/model development from any independent WM evaluation.
Full-state reruns give exact configured-SUMO effects, not an assertion that those
effects are identifiable from the observable state. Preserve provenance hashes.
