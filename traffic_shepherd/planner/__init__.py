"""Traffic Shepherd planner: decision loop over a predicted world model.

PL01-A scope (output/planner_v1/BRIEF_PL01_A.md): the planner decision loop
WITHOUT walls, run against a mocked world model. ``WallSpec``/wall types
exist in interface.py per the frozen contract wd01_predict/1, but wall
proposal/lifecycle is PL01-B; the loop carries an existing current plan's
walls (locked members) through untouched.

Zero simulation: this package imports no SUMO/traci/libsumo machinery
anywhere, and nothing from archive/.
"""
from .config import PlannerConfig
from .dispatch import (
    DISPATCH_V2_SCHEMA,
    DISPATCH_V3_SCHEMA,
    AssignmentPlanV2,
    DispatcherPlanV3,
    DispatcherWallV3,
    WallAssignmentError,
    plan_id_for,
    to_dispatcher_v3,
    to_mixed_assignment_v2,
)
from .grouping import (
    InteractionEdge,
    build_edges,
    cap_split,
    components,
    influences,
    trajectory_deviation,
)
from .interface import (
    ABORT_REASONS,
    ACTIONS,
    ACTION_RANK,
    CONTRACT_VERSION,
    PART_A_SCOPES,
    WALL_PHASES,
    PlanCandidate,
    Prediction,
    StateSample,
    WallPrediction,
    WallSpec,
    WorldModel,
    locked_entries,
    part_a_scope,
    plan_key,
    with_actions,
    with_wall,
    with_wall_phase,
    without_wall,
)
from .loop import DecisionReason, Planner, PlannerDecision
from .mock import MockWorldModel
from .rules import (
    AbstainVerdict,
    NativeReturnVerdict,
    PartAScopeMismatchError,
    SwitchVerdict,
    abstain_check,
    candidate_sort_key,
    compare_part_a,
    native_return_check,
    paired_saving,
    part_a_scope_mismatch_reason,
    switch_check,
)
from .safety import (
    RelativeMetricVerdict,
    SafetyVerdict,
    absolute_safety_mask,
    safety_check,
)
from .walls import (
    WallCandidate,
    WallRecord,
    WallRegistry,
    cells_rule,
    fold_wall_group,
    propose_walls,
    t0_lane_position,
    wall_id_for,
)

__all__ = [
    "ABORT_REASONS",
    "ACTIONS",
    "ACTION_RANK",
    "CONTRACT_VERSION",
    "DISPATCH_V2_SCHEMA",
    "DISPATCH_V3_SCHEMA",
    "WALL_PHASES",
    "AbstainVerdict",
    "AssignmentPlanV2",
    "DecisionReason",
    "DispatcherPlanV3",
    "DispatcherWallV3",
    "InteractionEdge",
    "MockWorldModel",
    "NativeReturnVerdict",
    "PART_A_SCOPES",
    "PartAScopeMismatchError",
    "PlanCandidate",
    "Planner",
    "PlannerConfig",
    "PlannerDecision",
    "Prediction",
    "RelativeMetricVerdict",
    "SafetyVerdict",
    "StateSample",
    "SwitchVerdict",
    "WallAssignmentError",
    "WallCandidate",
    "WallPrediction",
    "WallRecord",
    "WallRegistry",
    "WallSpec",
    "WorldModel",
    "abstain_check",
    "absolute_safety_mask",
    "build_edges",
    "candidate_sort_key",
    "cap_split",
    "cells_rule",
    "compare_part_a",
    "components",
    "fold_wall_group",
    "influences",
    "locked_entries",
    "native_return_check",
    "paired_saving",
    "part_a_scope",
    "part_a_scope_mismatch_reason",
    "plan_id_for",
    "plan_key",
    "propose_walls",
    "safety_check",
    "switch_check",
    "t0_lane_position",
    "to_dispatcher_v3",
    "to_mixed_assignment_v2",
    "trajectory_deviation",
    "wall_id_for",
    "with_actions",
    "with_wall",
    "with_wall_phase",
    "without_wall",
]
