# IDM v5.6.2 phase-split-authority source audit

Status: **complete; jerk-transition contract frozen and four-case outcome gate executed**

## Action-path findings

| Concern | Authoritative path | v5.6.2 result |
| --- | --- | --- |
| Assembly projection authority | `CounterfactualEscalationPolicy._project_compact_formation()` | First projected adaptive assembly command uses `compact_initial_assembly_servo_a_max_dec_mps2` (1.0 m/s²); later projection uses the low cap. |
| One-step feasibility | `_one_step_assembly_reaches_compactness()` / `_compact_selection()` | A candidate is vetoed before activation if the one allowed high-authority command cannot achieve the frozen speed-spread compactness condition. |
| Assembly runtime authority | `_apply_wall_actuator()` | `ASSEMBLING` uses the explicit initial cap and increments the assembly-command count. |
| Formation failure | `_advance_compact_assembly()` | At the next control boundary, an unready wall aborts with `assembly_abort:high_authority_step_not_compact`; the following step is damping, never a repeated 1.0 command. |
| HOLD runtime authority | `_apply_wall_actuator()` | Any state other than `ASSEMBLING` uses `compact_servo_a_max_dec_mps2` (0.3 m/s²). |
| Taper runtime authority | `_apply_taper_actuator()` | Uses `compact_servo_a_max_dec_mps2` (0.3 m/s²). |
| Target / headway changes | `_adaptive_member_targets()`, `_safety_project_target()`, `WallHoldPolicy.act_with_targets()`, tau blending | Unchanged from the frozen compact-wall treatment; only in-band latched wall members receive an override. |
| Release | `_complete_taper()` / `_stop_wall()` | Restores wall tau and returns to the frozen damping policy; it does not release CAVs to uncontrolled SUMO. |

## Why Candidate A rejected 4405

Candidate A set the generic adaptive servo cap to 0.3 m/s² in every phase.
The trajectory-coherent projection then reported a 59.50 m maximum longitudinal
span against the frozen 50 m cap, so the treatment was vetoed before its wall
could form.  The v5.4 1.0 m/s² formation profile had admitted the four
development cases.  v5.6.2 restores that authority only for its first assembly
command and adds an explicit no-repeat failure rule.

## Observed original assembly duration

The v5.4 controller audit shows one `ASSEMBLING` command for each of 4401,
4402, 4404 and 4405.  Their initial member speed spreads were respectively
0.96, 1.54, 0.01 and 0.90 m/s, all within the frozen 2.0 m/s compactness
tolerance once the required first assembly command had been issued.

## Frozen wall-transition rule

Neither `policies/counterfactual_escalation.py` nor
`policies/inflow_control.py` contains a jerk limit, previous-acceleration
state, or an assembly-to-HOLD settling rule.  Both paths only limit the amount
by which a **speed target** can fall in one control step through `a_max_dec`.
Thus there is no existing numerical jerk bound to preserve when changing from a
possible 1.0 m/s² assembly deceleration to the 0.3 m/s² HOLD envelope.

The owner chose the separately declared `SETTLING_TO_LOW_AUTHORITY` lifecycle:
0.5 m/s³, actual-acceleration-based, maximum two in-window commands, with tau
frozen and hard safety taking precedence. Its implementation and final
fingerprint are recorded in `IDM_V5_6_2_PHASE_SPLIT_AUTHORITY_PROTOCOL.md`.
