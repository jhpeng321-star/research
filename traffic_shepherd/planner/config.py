"""PlannerConfig: every tunable constant of the planner decision loop.

Defaults are the owner-declared values from output/planner_v1/PL01_PLAN.md
("Declared constants", owner decisions 2026-09-25; the PL-06 native-collision
default, the PL-07 window-only relative rule, and the v2 wall bar values
follow the owner decisions of 2026-09-27). The dataclass is frozen;
later PL experiments tune through ``dataclasses.replace``. Decision logic must
read every threshold from here -- no magic numbers in rules/loop/grouping.

Zero simulation: no SUMO/traci/libsumo import anywhere in this package.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from .interface import ACTIONS


#: Allowed ``PlannerConfig.absolute_ttc_rule`` values (rule C, then rule A).
RULE_TTC_RELATIVE_TO_NATIVE: str = "relative_to_native"
RULE_TTC_ABSOLUTE: str = "absolute"
ABSOLUTE_TTC_RULES: Tuple[str, ...] = (RULE_TTC_RELATIVE_TO_NATIVE, RULE_TTC_ABSOLUTE)


@dataclass(frozen=True)
class PlannerConfig:
    """All planner constants (PL01_PLAN "Declared constants" table)."""

    # Delay horizon [s] (plan: test 300/600/900).
    horizon_s: float = 600.0
    # Part A length [s]; None = let the world model use its own part_a_horizon_s.
    trajectory_s: Optional[float] = 60.0
    # Absolute safety floor on min TTC in every M*K cell.
    ttc_floor_s: float = 1.5
    # Fraction of M*K cells that must be collision-free with min TTC >= floor
    # (owner: 1.0 = ALL cells).
    abs_safety_quantile: float = 1.0
    # Allowed increase vs native in TTC<3 s exposure and hard braking [veh*s]
    # (owner: 0.0 now, tune in perfect-sim).
    rel_safety_tol_s: float = 0.0
    # Relative rule: mean increase <= tol AND increase <= tol in >= this
    # fraction of cells (owner: 0.90).
    rel_safety_quantile: float = 0.90
    # PL-06 (OWNER DECISION 2026-09-27, option A = the default): False -- a
    # NATIVE predicted collision in a cell does NOT veto the plan: the plan
    # must still pass its OWN absolute gate (no plan collision, min TTC >=
    # floor in ALL cells, the v2 future rules) and the relative rule vs
    # native, and the native collision is recorded as a logged
    # 'native_collision' diagnostic (SafetyVerdict.native_collision_cells +
    # a decision reason) instead. True (option B, the pre-2026-09-27
    # behaviour) restores the native-collision veto
    # (safety.fail_closed_mask). Every other native fail-closed term --
    # native_cell_valid / native_future_cell_valid False, NaN native
    # delays, native NaN TTC with a closing pair -- stays fail-closed
    # REGARDLESS of this flag.
    native_collision_vetoes_plan: bool = False
    # Abstain keeps the current plan UNLESS it is predicted unsafe (owner).
    abstain_safety_exception: bool = True
    # Step 5 drops group changes until the combined plan is safe (owner).
    combined_plan_fallback: bool = True
    # Cadence [s]: full re-decide at most this often; between cadence ticks the
    # loop keeps the current plan and only re-checks safety (owner: 1.0;
    # perfect-sim targeted experiments use 30-60 s).
    decide_every_s: float = 1.0
    # Switch rule: fraction of M*K cells with saving > 0 (owner: 0.80).
    agreement: float = 0.80
    # Switch rule margin: margin_floor_s + margin_k * std(paired saving).
    margin_floor_s: float = 60.0
    margin_k: float = 1.0
    # Higher bar for wall candidates (PL01-B); declared here per the plan.
    wall_agreement: float = 0.90
    wall_margin_k: float = 2.0
    # Max group size; larger components split at their weakest edge.
    group_cap: int = 5
    # Wall prediction cell aggregation ("any" or "majority" of M*K cells) for
    # the assembled / safety_abort events of a wall.
    wall_cell_rule: str = "any"
    # Wall member liveness/frame freshness (PL01-E4): a proposed member must
    # have an observed frame at most this old at decision time (0.0 = a frame
    # at this tick). Source: the state samples' optional ``frames`` mapping
    # (planner-side interface request to WD-01) or the hybrid tracked layer.
    wall_member_max_frame_age_s: float = 0.0
    # Interaction edge if the median-over-cells max-over-time deviation of the
    # other CAV exceeds the position ... or the speed threshold.
    interaction_pos_m: float = 2.0
    interaction_speed_mps: float = 0.5
    # Abstain if the fraction of state samples with merge_observed < this.
    abstain_min_observed: float = 0.8
    # Wall proposer grid (PL01-B); declared here per the plan. Every speed
    # must be strictly above the dispatcher floor 2.0 m/s (mixed_assignment/3
    # wall validation).
    wall_speeds_mps: Tuple[float, ...] = (12.0, 16.0, 20.0, 24.0)
    # Hold-duration grid [s] after assembly completes; the None option is the
    # OPEN HOLD (interpretation 4): hold until the planner releases the wall
    # (by wall omission, mixed_assignment/3) -- NO automatic release. It is
    # encoded as wall_open_hold_s so the dispatcher never auto-releases
    # within an episode.
    wall_release_options_s: Tuple[Optional[float], ...] = (60.0, 120.0, 240.0, None)
    wall_open_hold_s: float = 3600.0
    # Post-assembly hold-vs-release rule (PL01-B3; perfect-sim experiments
    # test both). "median_saving_gt0" releases when the median paired
    # release-vs-hold saving exceeds wall_release_margin_s (the default:
    # the historical behavior at margin 0.0); "switch_rule" releases only
    # when the standard switch rule passes (median saving > margin_floor_s
    # + margin_k * std AND agreement). An unsafe hold always force-releases.
    wall_release_rule: str = "median_saving_gt0"
    wall_release_margin_s: float = 0.0
    wall_max_member_gap_m: float = 100.0
    # None = unlimited walls (PL01-B); 0 = walls OFF (PL-04e2 --walls off:
    # no wall proposals at all -- the proposer skips, never proposes, so no
    # lookahead is spent on wall candidates).
    max_walls: Optional[int] = None
    # PL-04e2: cap on the individual stage's decision units -- only the N
    # roster CAVs nearest upstream of the merge that are present in the
    # current RSU frames are considered (the others keep their current
    # action). None = uncapped.
    max_decision_units: Optional[int] = None
    # PL-04j: the per-CAV action vocabulary the decision stages may PROPOSE
    # (a non-empty subset of interface.ACTIONS, same order = same tie-break).
    # Default: the full vocabulary. A runner whose installed dispatcher
    # cannot EXECUTE an action pins the executable subset instead of letting
    # the lookahead evaluate plans the dispatcher refuses -- the real
    # closed-loop wiring binds no PI session (mixed_assignment/3 raises
    # "plan assigns pi-external-u but no PI session is bound" on apply), so
    # it proposes native+damping only. This restricts candidates, never the
    # safety gates or the abstain rule.
    actions: Tuple[str, ...] = ACTIONS
    # Dispatcher commit horizons (dispatch.py): WM-40 mixed_assignment/3 fixes
    # commit_horizon_s at 0 (the planner re-decides every tick, dwell 0); the
    # mixed_assignment/2 fallback needs a minimal positive horizon.
    v3_commit_horizon_s: float = 0.0
    v2_commit_horizon_s: float = 10.0
    # ---- predict contract v2 (wd01_predict/2; PL-05; PL01_PLAN "Contract v2
    # safety beyond the commit window", owner decisions 2026-09-26) ----
    # Commit window length [s] (prediction semantics only: the planner still
    # re-decides every tick; NOT a dispatch dwell).
    commit_s: float = 30.0
    # Branch continuation after the commit window: "planner" (pi_k, identified
    # by meta.continuation_policy_id) or "native" (all CAVs native after the
    # commit window).
    continuation: str = "planner"
    # Operating mode (G-1 gating): "offline" and "controlled_experiment" may
    # consume v1/v1.1 predictions and unvalidated future heads; "live" and
    # "closed_loop" refuse them (named refusal before deciding).
    mode: str = "offline"
    # Controlled-experiment override: consume unvalidated future_min_ttc_q05
    # (and v1/v1.1 predictions) in "controlled_experiment" mode; every use is
    # logged in the decision record.
    allow_unvalidated_future_ttc: bool = False
    # Gating item G-2 (contract v2 section 8, 2026-09-27): controlled-
    # experiment override for UNVALIDATED relative-safety verdicts (the
    # paired TTC<3 exposure / hard-braking deltas, window and future) of a v2
    # prediction whose meta reports relative_safety_validated=False (or
    # omits the field). "live" and "closed_loop" refuse regardless of this
    # flag (safety.enforce_future_gate); "offline" never refuses and logs a
    # relative_safety_validated=False diagnostic instead. Every override use
    # is logged in the decision record.
    allow_unvalidated_relative_safety: bool = False
    # Future-collision gates (owner, 2026-09-26): veto when p_future_collision
    # exceeds the absolute max in ANY cell, or the native's in the paired cell
    # by more than the relative tolerance.
    p_future_collision_abs_max: float = 0.01
    p_future_collision_rel_tol: float = 0.0
    # v2 wall assembly bar (OWNER-APPROVED 2026-09-27 at 0.9 / 0.1): a
    # proposed wall additionally requires median p_assembly_completes >= this
    # and median p_assembly_abort <= the abort cap.
    wall_min_p_assembly: float = 0.9
    wall_max_p_abort: float = 0.1
    # PL-07 (OWNER DECISION 2026-09-27): True -- for contract v2
    # predictions the 30 s WINDOW difference alone (plan_m - native_m for m
    # in {ttc_lt3_exposure_s, hard_brake_s}) must ALSO pass the same
    # mean/quantile relative test with rel_safety_tol_s, IN ADDITION to the
    # summed window+future rule; a window-only failure vetoes and is
    # recorded in the safety verdict / why-vetoed reason as 'window-only
    # relative' (SafetyVerdict.window_relative_veto). False restores the
    # summed rule alone. v1/v1.1 predictions (no future terms) are
    # unaffected: their window rule is the only relative test either way.
    window_relative_nonincreasing: bool = True
    # T-A (owner, PL-EXP1b addendum 2): the TTC threshold of the relative
    # exposure rule. 1.5 s is the PRIMARY metric (the evaluation's
    # ttc_lt1p5_exposure_s; predictions carry the optional
    # ttc_lt1p5_exposure_s fields); 3.0 s selects the frozen
    # ttc_lt3_exposure_s fields (the pre-T-A behaviour, kept as the 3 s
    # SHADOW verdict when 1.5 is selected). Allowed values are exactly
    # {1.5, 3.0}; safety.py fails closed with a named error when the
    # selected field is absent on the prediction.
    relative_ttc_threshold_s: float = 1.5
    # Safety RULE C (owner, PL-EXP1c): how the ABSOLUTE min-TTC floor is
    # applied. "relative_to_native" (the default, rule C): a cell passes iff
    # collision-free AND plan min TTC >= min(ttc_floor_s, the NATIVE min TTC
    # of the same cell) - ``ttc_rule_tol_s``; a NaN native min TTC (no
    # closing pair) leaves the floor at ttc_floor_s. The future block uses
    # the same rule against its native twin. "absolute" is the old rule A
    # (plan min TTC >= ttc_floor_s everywhere), reproduced exactly.
    absolute_ttc_rule: str = "relative_to_native"
    # Numerical slack of rule C's plan-vs-native floor comparison (plan and
    # native TTC come from the same simulator, so identical physics compares
    # equal up to float noise). Not used by rule "absolute".
    ttc_rule_tol_s: float = 1e-6

    def __post_init__(self):
        if self.horizon_s <= 0:
            raise ValueError("horizon_s must be positive")
        if self.trajectory_s is not None and self.trajectory_s < 0:
            raise ValueError("trajectory_s must be >= 0 or None")
        if self.ttc_floor_s < 0:
            raise ValueError("ttc_floor_s must be >= 0")
        if not 0.0 <= self.abs_safety_quantile <= 1.0:
            raise ValueError("abs_safety_quantile must be in [0, 1]")
        if self.rel_safety_tol_s < 0:
            raise ValueError("rel_safety_tol_s must be >= 0")
        if not 0.0 <= self.rel_safety_quantile <= 1.0:
            raise ValueError("rel_safety_quantile must be in [0, 1]")
        if not isinstance(self.native_collision_vetoes_plan, bool):
            raise ValueError("native_collision_vetoes_plan must be a bool")
        if not isinstance(self.window_relative_nonincreasing, bool):
            raise ValueError("window_relative_nonincreasing must be a bool")
        if float(self.relative_ttc_threshold_s) not in (1.5, 3.0):
            raise ValueError(
                "relative_ttc_threshold_s must be exactly 1.5 (the primary "
                "ttc_lt1p5_exposure_s metric) or 3.0 (the frozen "
                "ttc_lt3_exposure_s metric)"
            )
        if self.absolute_ttc_rule not in ABSOLUTE_TTC_RULES:
            raise ValueError(
                f"absolute_ttc_rule must be one of {ABSOLUTE_TTC_RULES}, "
                f"got {self.absolute_ttc_rule!r}"
            )
        if not 0.0 <= float(self.ttc_rule_tol_s) < 1.0:
            raise ValueError("ttc_rule_tol_s must be in [0, 1)")
        if self.decide_every_s <= 0:
            raise ValueError("decide_every_s must be positive")
        if not 0.0 <= self.agreement <= 1.0:
            raise ValueError("agreement must be in [0, 1]")
        if not 0.0 <= self.wall_agreement <= 1.0:
            raise ValueError("wall_agreement must be in [0, 1]")
        if self.margin_floor_s < 0:
            raise ValueError("margin_floor_s must be >= 0")
        if self.margin_k < 0 or self.wall_margin_k < 0:
            raise ValueError("margin multipliers must be >= 0")
        if self.group_cap < 1:
            raise ValueError("group_cap must be >= 1")
        if self.interaction_pos_m <= 0 or self.interaction_speed_mps <= 0:
            raise ValueError("interaction thresholds must be positive")
        if not 0.0 <= self.abstain_min_observed <= 1.0:
            raise ValueError("abstain_min_observed must be in [0, 1]")
        if (
            not isinstance(self.actions, tuple)
            or not self.actions
            or any(action not in ACTIONS for action in self.actions)
            or len(set(self.actions)) != len(self.actions)
        ):
            raise ValueError(
                f"actions must be a non-empty duplicate-free tuple subset of {ACTIONS}"
            )
        if not all(
            isinstance(s, (int, float))
            and not isinstance(s, bool)
            and math.isfinite(float(s))
            and float(s) > 2.0
            for s in self.wall_speeds_mps
        ):
            raise ValueError(
                "every wall_speeds_mps entry must be a finite "
                "number > 2.0 (mixed_assignment/3 wall floor)"
            )
        for option in self.wall_release_options_s:
            if option is None:
                continue
            if isinstance(option, bool) or not math.isfinite(float(option)) or float(option) <= 0:
                raise ValueError(
                    "numeric wall_release_options_s entries must "
                    "be finite and positive (hold_s after "
                    "assembly); None is the open hold"
                )
        if self.wall_release_rule not in ("median_saving_gt0", "switch_rule"):
            raise ValueError("wall_release_rule must be 'median_saving_gt0' or 'switch_rule'")
        if (
            isinstance(self.wall_release_margin_s, bool)
            or not math.isfinite(float(self.wall_release_margin_s))
            or float(self.wall_release_margin_s) < 0
        ):
            raise ValueError("wall_release_margin_s must be a finite number >= 0")
        if (
            isinstance(self.wall_open_hold_s, bool)
            or not math.isfinite(float(self.wall_open_hold_s))
            or float(self.wall_open_hold_s) <= 0
        ):
            raise ValueError("wall_open_hold_s must be finite and positive")
        if self.wall_max_member_gap_m < 0:
            raise ValueError("wall_max_member_gap_m must be >= 0")
        if self.max_walls is not None and (
            isinstance(self.max_walls, bool) or int(self.max_walls) < 0
        ):
            raise ValueError("max_walls must be an int >= 0 (0 = walls off) or None")
        if self.max_decision_units is not None and (
            isinstance(self.max_decision_units, bool) or int(self.max_decision_units) < 1
        ):
            raise ValueError("max_decision_units must be an int >= 1 or None")
        if self.wall_cell_rule not in ("any", "majority"):
            raise ValueError("wall_cell_rule must be 'any' or 'majority'")
        if (
            isinstance(self.wall_member_max_frame_age_s, bool)
            or not math.isfinite(float(self.wall_member_max_frame_age_s))
            or float(self.wall_member_max_frame_age_s) < 0
        ):
            raise ValueError("wall_member_max_frame_age_s must be a finite number >= 0")
        if self.v3_commit_horizon_s < 0:
            raise ValueError("v3_commit_horizon_s must be >= 0")
        if self.v2_commit_horizon_s <= 0:
            raise ValueError("v2_commit_horizon_s must be positive")
        if (
            isinstance(self.commit_s, bool)
            or not math.isfinite(float(self.commit_s))
            or float(self.commit_s) <= 0
        ):
            raise ValueError("commit_s must be a finite number > 0")
        if self.continuation not in ("planner", "native"):
            raise ValueError("continuation must be 'planner' or 'native'")
        if self.mode not in ("offline", "controlled_experiment", "live", "closed_loop"):
            raise ValueError(
                "mode must be one of 'offline', 'controlled_experiment', 'live', 'closed_loop'"
            )
        if not 0.0 <= self.p_future_collision_abs_max <= 1.0:
            raise ValueError("p_future_collision_abs_max must be in [0, 1]")
        if not 0.0 <= self.p_future_collision_rel_tol <= 1.0:
            raise ValueError("p_future_collision_rel_tol must be in [0, 1]")
        if not 0.0 <= self.wall_min_p_assembly <= 1.0:
            raise ValueError("wall_min_p_assembly must be in [0, 1]")
        if not 0.0 <= self.wall_max_p_abort <= 1.0:
            raise ValueError("wall_max_p_abort must be in [0, 1]")
