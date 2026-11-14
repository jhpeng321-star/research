"""P1 ``ctrl``: batched CAV controller plug-in for Idea A.

Public API
----------
``CtrlConfig`` : controller constants (config.py).
``CavMover`` / ``CavStepInput`` / ``CavStepOutput`` / ``CtrlState`` : the
batched per-step CAV mover (mover.py).
``execution_speed`` : the measured Part A execution rule (mover.py).
``WallLaw`` / ``register_wall_law`` / ``resolve_wall_law`` /
``wall_law_status`` : the wall-law hook (wall_hook.py).
``MergeReleaseWallLaw`` / ``s_road_m`` / ``MERGE_START_S_ROAD_M`` : the
built-in wall-v2.2 merge-release law (released at the merge start
``s_road = 2644.2 m`` in HOLD only; hold capped there).
Frozen wall-v2.2 law surface (wall_v22.py): ``MERGE_START_S`` /
``MERGE_END_S`` / ``NETWORK_END_S`` / ``MERGE_RELEASE_X_M`` /
``MERGE_RELEASE_POINT_S`` / ``MIN_MEANINGFUL_HOLD_S`` /
``ASSEMBLY_TIME_ASSUMED_S`` / ``FIXED_ARM_DRAWS_S`` /
``REQUEST_SPAN_CAP_M``, the proposer ``min_proposal_distance_m``
(D_min = 70 * v_wall) / ``dmin_refusal`` (``dmin_unreachable``) /
``check_pair`` (2 abreast, span cap, D_min), the hold cap ``hold_cap_s``
+ ``hold_cap_record`` log row, and the once-only
``MergeReleaseSession`` / ``merge_release_event``.
"""
from __future__ import annotations

from .config import (
    ACTION_CODES,
    CAV_ACTIONS,
    CIRCLES_MODEL_SHA256,
    DAMPING,
    NATIVE,
    NONE,
    PI,
    WALL,
    WALL_LAW_V21,
    WALL_LAW_V22,
    CtrlConfig,
)
from .mover import (
    WALL_PHASE_ASSEMBLY,
    WALL_PHASE_HOLD,
    WALL_PHASE_NONE,
    CavMover,
    CavStepInput,
    CavStepOutput,
    CtrlState,
    execution_speed,
)
from .wall_hook import (
    MERGE_START_S_ROAD_M,
    MergeReleaseWallLaw,
    ROAD_LENGTHS_M,
    ROAD_OFFSETS_M,
    WallLaw,
    WallLawUnavailable,
    clear_registry,
    ensure_builtin_laws,
    register_wall_law,
    resolve_wall_law,
    s_road_m,
    wall_law_status,
)
from .wall_v22 import (
    ASSEMBLY_TIME_ASSUMED_S,
    FIXED_ARM_DRAWS_S,
    MERGE_END_S,
    MERGE_RELEASE_POINT_S,
    MERGE_RELEASE_X_M,
    MERGE_START_S,
    MIN_MEANINGFUL_HOLD_S,
    NETWORK_END_S,
    REQUEST_SPAN_CAP_M,
    MergeReleaseSession,
    check_pair,
    dmin_refusal,
    hold_cap_record,
    hold_cap_s,
    merge_release_event,
    min_proposal_distance_m,
)

ensure_builtin_laws()

__all__ = [
    "ACTION_CODES", "CAV_ACTIONS", "CIRCLES_MODEL_SHA256", "DAMPING",
    "NATIVE", "NONE", "PI", "WALL", "WALL_LAW_V21", "WALL_LAW_V22",
    "CtrlConfig", "CavMover", "CavStepInput", "CavStepOutput", "CtrlState",
    "WALL_PHASE_NONE", "WALL_PHASE_ASSEMBLY", "WALL_PHASE_HOLD",
    "execution_speed", "MERGE_START_S_ROAD_M", "MergeReleaseWallLaw",
    "ROAD_LENGTHS_M", "ROAD_OFFSETS_M", "WallLaw", "WallLawUnavailable",
    "clear_registry", "ensure_builtin_laws", "register_wall_law",
    "resolve_wall_law", "s_road_m", "wall_law_status",
    "MERGE_START_S", "MERGE_END_S", "NETWORK_END_S", "MERGE_RELEASE_X_M",
    "MERGE_RELEASE_POINT_S", "MIN_MEANINGFUL_HOLD_S",
    "ASSEMBLY_TIME_ASSUMED_S", "FIXED_ARM_DRAWS_S", "REQUEST_SPAN_CAP_M",
    "min_proposal_distance_m", "dmin_refusal", "check_pair", "hold_cap_s",
    "hold_cap_record", "MergeReleaseSession", "merge_release_event",
]

