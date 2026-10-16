"""Load a MAPPO actor into the C4 deployment path.

The trainer writes ``policy_latest.pt`` / ``policy_XXXXX.pt`` in the ``c4_dreamer/3`` payload
schema (``trainer = "mappo"``), so ``wms.control.policy.load_policy_fn`` (and therefore
``python -m wms.control.deploy --controller policy --policy <file>`` and the evaluation
campaign) loads MAPPO exactly like the C4 policy: a ``PolicyExecutor`` with
``exe(state)`` -> accel and ``exe.act_with_lane(state)`` -> (accel, lane targets).
``load_policy_fn`` here additionally accepts a trainer checkpoint (``checkpoints/*.pt``).
"""

from __future__ import annotations

from pathlib import Path

import torch

from wms.control.lanes import LaneRules
from wms.control.policy import PolicyExecutor
from wms.control.policy import load_policy_fn as _load_c4

from .agent import build_networks


def load_policy_fn(path: Path | str, *, deterministic: bool = True, device: str = "cpu"
                   ) -> PolicyExecutor:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("format") != "c4_mappo_trainer/2":
        return _load_c4(path, deterministic=deterministic, device=device)
    policy, encoder, _ = build_networks()
    policy.load_state_dict(payload["policy"])
    encoder.load_state_dict(payload["group_encoder"])
    policy.eval()
    encoder.eval()
    rules = LaneRules.from_mapping(payload["config"]["lane_rules"])
    return PolicyExecutor(policy.to(device), encoder.to(device), lane_rules=rules,
                          deterministic=deterministic, device=device)
