# A75 hold-forms plan schema (executor side) - SPEC v1, 2026-11-13

Executor: NEW module `tools/coopmwm/a75_holdforms.py` (repo `E:/agent-work/a54-mappo60b`, branch `c4/a66-mappo60c`),
runner arms `A75_PLAN` (one family) / `A75_BANK` (every row of the cell). It is a subclass of the A44 executor
(`a44_lcvariant`, itself the A37 executor + lc roles). A37 / A44 / oracle_plan code is NOT edited; their digests
are unchanged. A plan with none of the new fields goes through the inherited A44 code object untouched
(byte-identical commands, same strategy_key) - unit-tested on synthetic + recorded rows, SUMO identity fixture in
MECH_FIXTURE.md.

Bank format is unchanged (rows + `clean5c.templates[*].offer_condition`); new fields are OPTIONAL keys in
`offer_condition` (plan level) and in `offer_condition.roles[*]` (role level). Same names in the search-side
strategy spec (`strategy['roles'][*]`, internal role keys lane/rank/v_w/dmax/hold_s/onset_s/lc_target as today).

## Geometry / frames (as today, for reference)
- SUMO lanes: 0 = closing lane, 1 = middle, 2 = leftmost. SUMO x = WM s_m + S_OFF; lane-drop end X_B = 2696.0 m
  (WM s_m = 75.44). Every member is released when it reaches X_B ('at_drop').
- Decisions every 1 s, warmup 300 s, law `a = clip(target - v, -dmax, +1.0)` (m/s^2) sent as the absolute
  pre-safety request through the existing single writer (`desired[vid] = requested_raw`, CommandOwnership / MAPPO
  safety downstream, speed mode unchanged). Members not in an active hold step get NO override (base U0 controller);
  their lane request is their own current lane (G5 pin = no-op), exactly as today pre-onset. HDVs never commanded.

## Canonical form and strategy_key (IMPORTANT for the search side)
- Write a new field ONLY when its value differs from the default. The executor rebuilds the spec with the old keys
  plus every new key present in the offer_condition, and `strategy_key` = sha256 over
  `{trigger, zone, spread_m, cooldown_s, slots, roles}` (+ gate/start_gate/phases when set, as today)
  + `zone_d` when set; roles carry their new keys when present. Plans without new keys -> today's key.
- Use `a75_holdforms.canonical(spec)` (drops default-valued new fields) then `a75_holdforms.strategy_key(spec)`,
  and `a75_holdforms.bank_doc(plans, cell)` to write banks. `a75_holdforms.executor_refusal(spec)` returns None if
  executable, else the refusal class.

## 1. Short holds (IMPLEMENTED; no code change was needed)
- `hold_s`: int seconds, executor accepts 1..900 (refusal `bad_holdform` outside). Nothing in the executor
  assumes >= 90 (hold ends at `onset_s + hold_s` elapsed, releases at drop). `cooldown_s` any int >= 0.
- Suggested grid: hold_s in {5, 10, 20, 30, 45, 60, 90, 120, 180}; cooldown_s in {hold_s, hold_s+30, 60, 120}.

## 2. Vehicle-conditioned ("gap") start (IMPLEMENTED) - role level
| field | type / allowed | default | meaning |
|---|---|---|---|
| `start` | `"onset"` \| `"gap"` | `"onset"` | onset = today (hold runs `[onset_s, onset_s+hold_s)` after selection). gap = role is ARMED (no command) from `onset_s` for `gap_arm_s` seconds; a hold event fires at the first armed second the gap condition is true. |
| `gap_dx_lo`, `gap_dx_hi` | float m, -100 <= lo < hi <= 100 | required if gap | window on dx = x_veh - x_cav (positive = vehicle ahead of the CAV). |
| `gap_drop_m` | float m > 0 or null | null | additionally require the closing-lane vehicle within this distance upstream of X_B (X_B - x_veh <= gap_drop_m). null = no constraint. |
| `gap_arm_s` | int s >= 1 | required if gap | armed period length (from selection + onset_s). Session releases 'gap_arm_end' when it ends and no hold event is running. |
| `gap_release` | `"timer"` \| `"clear"` | `"timer"` | timer: event lasts hold_s. clear: event ends at the first second (>= 1 s after firing) the condition is false, at most hold_s. |
| `gap_events` | int >= 1 | 1 | max hold events per session; after an event ends the role re-arms while still inside the armed period (an event that starts inside the period always runs to its end). |

Gap condition at second t (role must be on SUMO lane 1 - the only lane that senses the closing lane; else refusal
`gap_role_lane`): some vehicle (HDV or CAV, any type - type is NOT used) with WM lane 2 (= SUMO lane 0, the closing
lane), upstream of the drop (s_m < 75.44), that THIS member CAV detects with its own local sensing (|ds| <= 100 m,
|d lane| <= 1: the project 100 m sensing; the same rule as `observed_rows`/N1 cav_reports, restricted to the member
itself), with gap_dx_lo <= x_veh - x_cav <= gap_dx_hi and (gap_drop_m null or X_B - x_veh <= gap_drop_m).
Inputs: member self row (x, v, lane) + its detected neighbours (x, lane). No type, no route, no intention, no
global state. The executor already has every plant row; it filters to the member's own sensing (stricter than the
RSU union `observed_rows`, which the selection gates use today).
- Suggested grid (search must not fix tuned values): gap_dx_lo in {-20, -10, 0}; gap_dx_hi in {20, 40, 60};
  gap_drop_m in {null, 150, 300, 600}; gap_arm_s in {30, 60, 120}; gap_release in {timer, clear};
  gap_events in {1, 3}; hold_s in {5, 10, 15, 20}; usually with `dv_rel` (below) and zone near the merge.

## 3. Zones (IMPLEMENTED) - plan level
- `zone` names (selection window on SUMO x of members, `[lo, hi]`):
  `far` [X_B-1800, X_B-1000] (today), `full` [X_B-1800, X_B-400] (today), `near` [X_B-1200, X_B-400] (already in
  the A37 executor, never used by the search), NEW `mid` [X_B-1400, X_B-600], NEW `merge` [X_B-600, X_B-100].
- Parametric: `zone_d`: `[d_up, d_down]` metres upstream of X_B, d_up > d_down >= 0, d_up <= 2400; requires
  `zone: "custom"`; window [X_B-d_up, X_B-d_down]. Grid: d_up in {1800, 1400, 1000, 700, 400}, d_down in
  {1000, 600, 400, 200, 100}, d_up - d_down >= 200.
- Default: no `zone_d`, zone as today. The new zones apply to both triggers ('lanes', 'wall').

## 4. Other cheap forms - role level
| field | allowed | default | status | meaning |
|---|---|---|---|---|
| `dv_rel` | float m/s > 0 or null | null | IMPLEMENTED | relative target: target = max(v_w, v0 - dv_rel), v0 = member speed at the first hold second (v_w becomes a floor; set v_w 0.0 for a pure relative hold). Grid {2, 4, 6, 8}. |
| `taper_in_s` | int >= 0 | 0 | IMPLEMENTED | target ramps linearly from v0 to the hold target: at hold second h (0-based) target = v0 + (T - v0) * min(1, (h+1)/taper_in_s). Grid {0, 5, 10, 20}. |
| `taper_out_s` | int >= 0 | 0 | IMPLEMENTED | in the last taper_out_s seconds of a timed hold, target ramps back toward v0: factor min(1, (hold_s-1-h)/taper_out_s) (last second target = v0). taper_in_s + taper_out_s <= hold_s. Not applied to an early 'clear'/'at_drop' end. Grid {0, 5, 10, 20}. |
| `pulse_on_s`, `pulse_off_s` | ints >= 1 (both or neither) | absent | IMPLEMENTED | pulsed hold: inside the hold, the law is issued only when h mod (on+off) < on; off seconds = no override (base controller), member stays in the session. Grid on {5, 10, 20}, off {5, 10, 20}; with hold_s >= on+off. |
| wider `v_w` | float 0..33 | - | IMPLEMENTED (no change needed) | executor has no v_w limit; suggested grid {8, 10, ..., 24}. |
| rolling / sequential holds | `onset_s` any int >= 0 | - | IMPLEMENTED (no change needed) | per-role onsets already independent; suggested onset grid {0, 10, 20, 40, 60, 90}. |
| lc_target + any new field | - | - | PROPOSED only | refused (`lc_with_holdform`); lc roles keep the A44 semantics. |
| gap trigger on RSU-union sensing / closing-lane CAV as gap requester / speed-matching to the merging vehicle / plan-level gap selection | - | - | PROPOSED only | not implemented. |

Allowed combinations: `start=gap` combines with dv_rel, taper_*, pulse_* (the event is the hold timeline, h = t -
t_fire). Every role field is independent per role (mixed onset/gap roles in one plan are allowed).

## Refusals added (executor_refusal / bank admission)
`bad_holdform` (out-of-range / inconsistent value or unknown key), `gap_role_lane` (gap role not on SUMO lane 1),
`lc_with_holdform`, `bad_zone` (unknown zone name / zone_d without zone custom). Old refusals unchanged.

## Receipts (A75)
Inherited A37/A44 receipts + `holdforms` = list of hold events per member: {family, vid, start_kind, armed_t,
fire_t, end_t, end_why, trigger_vid, dx_m, v0, issued_s} and `holdform_summary` (sessions, events, fired, armed
without event, early clear ends). Release reasons: hold_end, at_drop, left, gap_arm_end.

## Example offer_conditions (one per form)
Short hold: `{"trigger":"lanes","zone":"far","spread_m":300.0,"cooldown_s":30,"slots":1,"gate":null,"start_gate":null,"phases":null,"situation":{"unconditional":true},"roles":[{"sumo_lane":2,"rank":0,"v_w":18.0,"dmax":1.5,"hold_s":10,"onset_s":0,"lc_target_sumo":null}]}`
Gap: roles `[{"sumo_lane":1,"rank":0,"v_w":0.0,"dmax":1.5,"hold_s":10,"onset_s":0,"lc_target_sumo":null,"start":"gap","gap_dx_lo":-10.0,"gap_dx_hi":40.0,"gap_drop_m":300.0,"gap_arm_s":60,"gap_release":"clear","gap_events":3,"dv_rel":4.0}]`, zone "merge", cooldown_s 60.
Zone: `"zone":"custom","zone_d":[1000.0,400.0]` (any roles).
Pulsed: role `{..., "v_w":16.0, "hold_s":60, "pulse_on_s":10, "pulse_off_s":10}`.
Tapered: role `{..., "v_w":16.0, "hold_s":60, "taper_in_s":10, "taper_out_s":10}`.
Rolling: roles on lane 2 rank 0/1 and lane 1 with onset_s 0 / 30 / 60.
(Exact JSON of the unit-tested examples: `E:/research-dense/experiments/a75_holdforms/EXAMPLE_PLANS.json`.)

## Status (2026-11-13)
Implemented in commit 921a850e (branch c4/a66-mappo60c), a75_digest d1af7e0ff795f8cd1eb8345c097cc6c113791acd85754c3f664b4ad7dde7dfae. Unit tests only; NOTHING run in SUMO. Plain example plans (ex_short, ex_rolling) execute through the inherited A44 path; ex_gap/ex_zone/ex_pulse/ex_taper through HoldFormCore.
