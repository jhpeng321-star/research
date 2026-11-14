# A75 hold forms: what the frozen WM-S can and cannot score (search side, 2026-11-13)

This note is about SPEC_HOLDFORMS.md v1. Field names are the spec's own; no second format was made.
- Code: `search/ship/a75ss.py` (generator, key, refusal mirror, selection) and `search/ship/a75_wmforms.py` (WM
  translation wrapper).
- The WM checkpoint `wm_s.pt` 6da9d333 is frozen and the WM tree (code_v12r7b) is not edited.
- The wrapper is bound through the WMA backend `api` override and is layered on E1-FIX `wm_fix_crn.evaluate`.

## How the WM scores a plan
The WM search scores ONE activation of the plan at t0 of each start state, with H = 300 s against the base. This is
the same as every previous search.

During that rollout, the controller callback sees the full predicted state every 0.5 s step: `s_m`, `v`, `lane` and
`present` for all slots, entrants included. So a role's hold start CAN depend on a neighbouring vehicle inside the
rollout.

## Representability per form
| form (spec field) | WM status | reason / caveat |
|---|---|---|
| short holds (`hold_s` 1..900) | as is | The original HoldSpeedController takes any `hold_s`. |
| wider `v_w` | as is | Law imposed on the member. Below 15 and above 22 m/s is outside the WM data's hold moves (see flags). |
| rolling / sequential onsets (`onset_s`) | as is | Per-role `start_s`. |
| zones `mid` / `merge` / `custom` + `zone_d` | as is (selection) | Members are picked from the t0 CAV rows with the executor's window (`a75ss.select_members` = executor `select_members`). |
| `start='gap'` + `gap_*` | wrapper (HoldFormController) | Fires on predicted closing-lane vehicles inside the rollout. Same rule as `gap_candidate`: WM lane 2, upstream of the drop, present, member's own 100 m / ±1 lane sensing, dx window, `gap_drop_m`; type never read. Checked at whole seconds; timer / clear / `gap_events` / `gap_arm_s` follow `HoldFormCore._member`. |
| `dv_rel`, `taper_in_s`, `taper_out_s` | wrapper | Uses the executor's `hold_target` formula with v0 = the member's speed in the firing second. |
| `pulse_on_s` / `pulse_off_s` | wrapper | Executor `pulse_on`. Off-seconds use WM native motion; see limitation 2. |
| `lc_target` + any new field | not needed | The executor refuses it (`lc_with_holdform`), and so does the search mirror. |
| plan-level gap selection, closing-lane CAV as gap requester, speed matching (spec: PROPOSED only) | not implemented | Not executable either. |

Unit tests: `search/tests/test_a75_wmforms.py`.
- They compare the controller's per-second commands with a reference timeline built from the executor's own
  `gap_candidate` / `hold_target` / `pulse_on` / `law`, on scripted scenes. These cover gap timer, gap clear with
  3 events, drop window, taper, pulse, dv_rel and release at the drop.
- Plain members are bit-identical to the original controller.
- CPU smoke on the real `wm_s.pt` (one D2800_p60 state, H 20-60 s): gap events fire in plan rows and never in base
  rows. A plain plan gives the same J through both APIs, and results do not depend on action order.

## Does the WM predict human merges into an opened gap?
**Yes, mechanistically.**
- Closing-lane HDVs merge through the learned merge_pivot first-passage law. Its boundary is the clearance ratio
  of the target-lane leader and follower gaps (`merge_pivot.FirstPassageNet.boundary`), plus the learned follower
  courtesy and yield modules. A member easing off on SUMO lane 1 enlarges the follower gap and so raises the
  predicted merge hazard.
- **Not verified:** how accurate this response is for deliberate gap-opening. The WM data moves were 15-22 m/s,
  90-180 s holds in zone [X_B-1800, X_B-400]. Short gap events near the merge ([X_B-600, X_B-100]) are outside that.

## Limitations (main session: test these directly in SUMO)
1. **Repeated activations are not represented.** The SUMO executor re-fires a plan after `cooldown_s` / the busy
   rule; the WM scores a single activation.
   - `cooldown_s` therefore has NO effect on the WM score. Cooldown variants share the activation spec, cost no
     extra rollouts and get identical scores under CRN.
   - Short holds with short cooldowns (many firings per episode) are scored per firing only. Rank them by SUMO
     episode effect, not by WM per-activation gain. Choosing cooldown is a SUMO question.
2. **Off-pulse, armed (pre-fire) and pre-onset seconds:**
   - SUMO uses the base controller (MAPPO u0).
   - The WM uses its native HDV law, with the member's lane held for the whole rollout.
   - This is the existing pre-onset approximation, but pulsed and gap forms spend more time in it.
3. **Decision cadence:** the law is recomputed every 0.5 s WM step (as all existing WM holds), while SUMO commands
   every 1 s.
4. **E1-FIX CRN:** entrant CAV share is fixed at 0 (`ENTRANT_CAV_SHARE`), so WM entrants are never CAVs. This only
   matters for the episode, not for selection at t0.
5. A gap plan whose events never fire in the WM rollout still counts as "activated" (members were selected). Its
   gain is near 0, not exactly 0, because members keep their lanes.
