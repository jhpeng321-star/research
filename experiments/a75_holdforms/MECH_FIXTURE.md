# A75 hold-forms - proposed SUMO mechanics fixture (NOT run; for the main session / CPU box)

Purpose: mechanics only (does the executor do what SPEC_HOLDFORMS.md says inside SUMO). Not efficacy evidence.
Executor: `tools/coopmwm/a75_holdforms.py` (a75_digest d1af7e0f..., recompute on the box), runner arms
`A75_PLAN` / `A75_BANK` (runner edited by `ops/a75/host/patch_runner_a75.py arms <runner>`; spec helper
`ops/a75/a75_specs.py`). Plans: `EXAMPLE_PLANS.json` (`bank` = ready bank doc, cell key D2800_p60; for the
20 % cell rebuild with `a75_holdforms.bank_doc(plans, 'D2800_p20')`).

## Seeds / budget (main to register before running)
- World seeds: a FRESH audited mechanics block assigned by main (A44/A73 used 992400-992499; avoid every FORBIDDEN /
  sealed / acceptance block). Not the verify/confirm blocks.
- Horizon 1,200 s per episode (as A73). Proposed 20 episodes = 24,000 sim-s (= the A73 MECH_CAP):

| stage | arm(s) | plan | cell | worlds | episodes |
|---|---|---|---|---|---|
| I identity | A37_PLAN vs A75_PLAN | a plain hold (e.g. a44_fix_hold, or ex_short) | D2800_p60 (U0 = 60c binding) | 2 | 4 |
| F1 short | A75_PLAN | ex_short | D2800_p60 | 2 | 2 |
| F2 gap | A75_PLAN | ex_gap | D2800_p60 and D2800_p20 | 2 + 2 | 4 |
| F3 zone | A75_PLAN | ex_zone | D2800_p60 | 2 | 2 |
| F4 pulse | A75_PLAN | ex_pulse | D2800_p60 | 2 | 2 |
| F5 taper | A75_PLAN | ex_taper | D2800_p60 | 2 | 2 |
| F6 rolling | A75_PLAN | ex_rolling | D2800_p60 | 2 | 2 |
| F7 bank | A75_BANK | all six | D2800_p60 | 2 | 2 |

## Checks (from the record: vehicle log + `a75` receipts + safety)
All stages: `a75_sha256_loaded == registered a75_sha256`; collisions = 0; teleports = 0 (also emergency braking
count reported vs U0 on the same world); no command / lane target on any HDV (hook raises otherwise); refused = {}.
- I: decompressed vehicle logs, J (team delay) and executor receipts (minus schema/executor/arm keys and the A75
  extras which must be empty: extended = [], holdforms = []) EXACTLY equal between A37_PLAN and A75_PLAN, per world.
  Any difference = stop (identity broken).
- F1: every member session issues exactly hold_s (10) commanded seconds unless released 'at_drop'/'left'; release
  reason 'hold_end'; >= 1 activation per world.
- F2 (the key one): `holdform_summary.gap_fired >= 1` per world (trigger fired); for each fired event: trigger_vid
  was in SUMO lane 0 at fire_t within [x_cav-10, x_cav+40] and <= 300 m from the drop (re-derive from the vehicle
  log); member speed drops after fire_t (v at fire_t+5 < v0); GAP OPENED: the CAV's leader gap in lane 1 (or
  headway to the trigger vehicle's projected position) at end_t > at fire_t, and the trigger vehicle changes into
  lane 1 ahead of the CAV within 20 s of fire_t (count merged / not merged); RELEASED: every event has end_t and
  end_why in {clear, timer, at_drop}; every session released (hold_end / gap_arm_end / at_drop / left) before
  episode end; issued_s <= hold_s per event; gap_events <= 3 per session.
- F3: activation x of every member within [X_B-1000, X_B-400] (receipts act x).
- F4: commanded seconds per session follow the 10 on / 10 off pattern (from issued list / vehicle log).
- F5: commanded acceleration in the first 10 s is smaller in magnitude than an un-tapered step to v_w would be
  (target ramp), and the last 10 s target returns toward v0.
- F6: per-act member onsets 0 / 30 / 60 s honoured (first command second per member).
- F7: no member in two family sessions; all six families can activate.

Stop rule: any collision/teleport attributable to a hold member, identity mismatch, or a command on an HDV -> stop,
report, no efficacy runs. Unverified until this fixture runs: the gap trigger frequency in real traffic (synthetic
traffic fired 19 gap events in 11 sessions), whether eased CAVs actually open gaps that merging HDVs use, and
behaviour of very short holds under the MAPPO safety filter.
