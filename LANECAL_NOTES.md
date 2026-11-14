# LANECAL_NOTES: imagination lane-feasibility rule calibrated to SUMO acceptance (C4)

Problem: `wms.control.lanes.LaneRules` (imagination mask) passed requests that SUMO
(`CavSumoEnv`, laneChangeMode 513, `changeLane(..., 1.0)`) then refused. Goal: make the mask predict SUMO acceptance.

## Data (tools/lanecal_collect.py, local under runs/active/C4-LANECAL/, not committed)
- 36 SUMO episodes: seeds 4107-4112 x demand {2400, 2700} x penetration {0.1, 0.2, 0.4}, warmup 300 s +
  120 s window, env `direct`. Scripted policy: AR(1) random accelerations (clipped [-1, 0.8]) for all CAVs and, per
  CAV and tick, a left/right request with probability 0.25 if the 3 s dwell and the mask geometry allow (the gap
  rule is NOT applied, so infeasible requests are sampled too). 21,673 requests issued by the env (edge-end
  deferrals excluded; the edge guard is unchanged), 0 collisions, 0 teleports.
- Per request, WM-frame features at request time (nominal 5 m length as in imagination): ego v, target-lane front/rear
  bumper gaps and speeds, lane, s, distance to the next edge boundary; label = lane after the 1 s step == target
  (the env's own "completed"). True vType tau/decel of the follower are logged for diagnosis only.
- Split by seed: train 4107-4110 (14,489 mainline requests), held-out 4111-4112 (6,957). Ramp (lane 2) requests: 227, only
  8 accepted, reported separately (the bridge's forced-merge rule handles the merge).
- Overall SUMO acceptance of these deliberately unfiltered requests: 13.5 %. With both target gaps > 60 m: 92 %.
  The acceptance rises with the rear gap only at ~30-70 m (3 % below 0 m, 6 % at 2-8 m, 45 % at 30-45 m, 69 % at 45-70 m,
  96 % above 70 m), i.e. SUMO's follower-safety check (secure gap incl. closing speed) is far stricter than 6 m + 1 s.

## Method
Parameters of the existing linear form, with the rear check made independent (backward-compatible extension):
front `>= gap_base_m + gap_time_s*v_ego + closing_time_s*max(0, v_ego - v_front)`,
rear `>= rear_gap_base_m + rear_gap_time_s*v_rear + rear_closing_time_s*max(0, v_rear - v_ego)`
(`rear_*` default to the front values when `None`; a mapping without `rear_*` keys, i.e. every old checkpoint payload,
is read as the legacy symmetric rule). Fit (tools/lanecal_fit.py): stochastic hill-climb, 6 parameters, maximise recall
subject to TRAIN precision >= 0.92, 4 restarts (all agree: front ~1 m / 1.2 s / 4.5 s, rear ~3 m / 1.25 s / 6 s), rounded.
A quadratic closing term (v^2 differences) fit equally well and was not adopted; the shared (symmetric) form
loses ~3 recall points at equal precision (e.g. 3 m / 1.25 s / 5 s: held-out 0.955 / 0.637).
`edge_guard_m` and `dwell_s` unchanged (edge deferrals were pre-filtered; all issued requests respect them).

## Result (precision = P(SUMO executes | mask says feasible); recall = P(mask feasible | SUMO executes))
Old = 6 m / 1.0 s / 1.0 s symmetric. New = front 1 / 1.2 / 4.5, rear 3 / 1.25 / 6 (chosen).

| split | subset | n req | SUMO accept | old prec | old rec | new prec | new rec | new n pred |
|---|---|---|---|---|---|---|---|---|
| train | all mainline | 14489 | 0.128 | 0.659 | 0.620 | 0.911 | 0.610 | 1239 |
| train | left (1->0) | 9638 | 0.090 | 0.517 | 0.549 | 0.866 | 0.539 | 543 |
| train | right (0->1) | 4851 | 0.202 | 0.819 | 0.683 | 0.947 | 0.673 | 696 |
| held-out | all mainline | 6957 | 0.153 | 0.722 | 0.674 | **0.952** | 0.657 | 733 |
| held-out | left (1->0) | 4675 | 0.109 | 0.619 | 0.625 | **0.945** | 0.603 | 325 |
| held-out | right (0->1) | 2282 | 0.243 | 0.833 | 0.718 | **0.958** | 0.706 | 408 |
| held-out | ramp (2->1) | 74 | 0.054 | (1 pred) | | (1 pred, refused) | | |

Closed-loop check (new seeds 4113-4116, demand/penetration 2400/0.4 and 2700/0.2, p_request 0.5, only mask-feasible
requests issued, mask with the real `lane_feasibility`): NEW rule 813 issued, 746 executed = 91.8 % (left 93.1 %,
right 90.6 %, ramp 11 requests 36 %); OLD rule on seeds 4113-4114: 740 issued, 474 executed = 64.1 % (left 55.6 %,
right 76.0 %). 0 collisions, 0 teleports in all 12 episodes.

Cost of the stricter rule: the mask passes ~40 % fewer requests than the old one (733 vs 992 held-out) at about the same
recall of accepted requests (0.66 vs 0.67): the removed requests were almost all refused ones.

## Chosen values (configs/c4_policy.toml [lanes], and LaneRules dataclass defaults)
gap_base_m 1.0, gap_time_s 1.2, closing_time_s 4.5, rear_gap_base_m 3.0, rear_gap_time_s 1.25,
rear_closing_time_s 6.0, dwell_s 3.0, edge_guard_m 10.0. Old checkpoints keep their saved symmetric rule via
`LaneRules.from_mapping` (no `rear_*` keys).

## Remaining refusals (precision 0.91-0.95, not 1.0) and why
- The mask uses only the nearest front/rear vehicle of the target lane at the decision instant. SUMO also lets HDVs change
  lanes (their own LC model) and other CAVs act in the same second, so the same gap can be taken by someone else
  (not modelled in imagination; apply_lane_choices only resolves CAV-CAV conflicts).
- Refusal is higher when the ego is tailgating its own leader (current front gap < 10 m: 21 % false positives vs 6 % when > 50 m);
  the left (1->0) direction has about twice the false-positive rate of right (10 % vs 5 % on all data).
- SUMO uses per-vehicle tau/decel/length (rear tau 1.0-1.5 s here) while imagination uses nominal 5 m length and no driver
  parameters; the mask is a population-level boundary tuned for ~95 % precision, not exact.
- Ramp-lane requests are essentially always refused by SUMO under mode 513 (8/227 executed): the forced-merge
  mechanism, not the lane head, should do ramp merges; the mask still allows "merge now" inside the zone (unchanged).
- Misses (SUMO accepts, mask refuses; recall ~0.66): mostly stop-and-go states (v ~ 0-12 m/s, gaps -1 to 25 m), where
  SUMO accepts small gaps; deliberately not admitted to keep precision.

## Budget note
Collection: 36 episodes, ~1045 s summed episode wall (4 parallel; 4 min 51 s elapsed). Closed-loop validation: 12
episodes, 4 parallel, 9 min elapsed (summed episode wall 2099 s: the per-request K=192 mask evaluation in my harness and
CPU contention made it slow). Together this EXCEEDS the 25 min SUMO allowance if summed episode wall is counted
(~52 min; ~14 min elapsed). No further SUMO was run. Tools: tools/lanecal_collect.py, lanecal_analyze.py, lanecal_fit.py,
lanecal_report.py.
Tests: tests/unit/test_c4_policy2.py updated (rear parameters, legacy mapping); tests/unit/test_c4_*.py: 160 passed, 1
failed (`test_c4_loop.py::test_refresh_config_and_knob_validation`, asserts WM config `lane_rule == forced_merge_only`; it
concerns WM train configs, unrelated to this change).
