"""P3 ``rollout2``: controller-in-the-loop, strategy-agnostic rollout engine."""
from __future__ import annotations

from .baseline import Params, V2GCore, WorldRows, load_params
from .engine import (IdeaABundle, Rollout2Engine, rollout2,
                     world_from_batch)
from .smoke import j4_smoke_contract, j4_smoke_world
from .state import (Rollouts, Rollout2Config, Rollout2State,
                    StrategyActions, WorldState)
from .step import StepAux, step_once
from .validity import (ValidityCounts, integrate_ballistic, integrate_d1,
                       integrate_d1_vnew, project_rollout_step,
                       project_same_lane_gaps)

__all__ = [
    "IdeaABundle", "Rollout2Engine", "rollout2", "Rollouts",
    "Rollout2Config", "Rollout2State", "StrategyActions", "WorldState",
    "StepAux", "step_once",
    "ValidityCounts", "integrate_d1", "integrate_d1_vnew",
    "integrate_ballistic", "project_rollout_step", "project_same_lane_gaps",
    "Params", "V2GCore", "WorldRows", "load_params", "world_from_batch",
    "j4_smoke_contract", "j4_smoke_world",
]
