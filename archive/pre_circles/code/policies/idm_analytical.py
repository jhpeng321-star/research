"""Policy adapter for frozen analytical IDM selected-absorber damping."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any, Dict, Optional

import numpy as np

from env.idm_residual import IDMResidualActionEngine, IDMResidualConfig
from policies import DirectControlPolicy


FROZEN_IDM_ANALYTICAL_CONTROL_ARM_SCHEMA = "idm_analytical_control_arm/1"
FROZEN_IDM_ANALYTICAL_CONTROL_ARM_ID = "idm_analytical_disturbance_p30_v1"
FROZEN_IDM_ANALYTICAL_REFERENCE_MODE = "disturbance_p30"
FROZEN_IDM_ANALYTICAL_CONTROL_ARM_FINGERPRINT_SHA256 = (
    "75649f14ea355869a40558f8dfb0afc6057385b55c182a8235b6fb2c9b8fe2c7"
)


def frozen_idm_analytical_config() -> IDMResidualConfig:
    """Return a fresh copy of the exact analytical wall-base configuration.

    The wall runners historically constructed this configuration inline as
    ``IDMResidualConfig(reference_mode="disturbance_p30")``.  Keeping that
    equation and every dataclass default unchanged, this factory gives the
    counterfactual control and treatment arms one versioned construction path.
    """

    return IDMResidualConfig(reference_mode=FROZEN_IDM_ANALYTICAL_REFERENCE_MODE)


def _control_arm_payload() -> Dict[str, Any]:
    return {
        "schema": FROZEN_IDM_ANALYTICAL_CONTROL_ARM_SCHEMA,
        "controller_id": FROZEN_IDM_ANALYTICAL_CONTROL_ARM_ID,
        "policy_class": "policies.idm_analytical.IDMAnalyticalDampingPolicy",
        "engine_class": "env.idm_residual.IDMResidualActionEngine",
        "configuration": asdict(frozen_idm_analytical_config()),
        "reference_mode": FROZEN_IDM_ANALYTICAL_REFERENCE_MODE,
        "residual_action_input": "exact_empty_mapping_zero_residual",
        "direct_control": True,
        "learned_checkpoint": None,
        "fallback_controller": None,
        "legacy_krauss_fallback": False,
    }


def frozen_idm_analytical_manifest() -> Dict[str, Any]:
    """Return the canonical control-arm manifest and deterministic fingerprint."""

    payload = _control_arm_payload()
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()
    if fingerprint != FROZEN_IDM_ANALYTICAL_CONTROL_ARM_FINGERPRINT_SHA256:
        raise RuntimeError(
            "frozen analytical damping control-arm contract changed; "
            "create a new version instead of silently changing v1"
        )
    return {
        **payload,
        "contract_fingerprint_sha256": fingerprint,
    }


class IDMAnalyticalDampingPolicy(DirectControlPolicy):
    """Apply analytical damping directly and return no environment actions.

    The direct TraCI command is intentional: a wall overlay may subsequently
    override only treatment members while every nonmember keeps the command
    already applied by this identical base controller.
    """

    def __init__(
        self,
        config: Optional[IDMResidualConfig] = None,
        *,
        reference_mode: str = "disturbance_p30",
    ) -> None:
        self.config = config or IDMResidualConfig(reference_mode=reference_mode)
        self.engine = IDMResidualActionEngine(self.config)
        self.env: Any = None
        self.last_diagnostics: Dict[str, Any] = {}

    def reset(self, env: Any, seed: int = None) -> None:
        self.env = env
        self.engine.reset()
        self.last_diagnostics = {}

    def act(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, object]:
        cav_ids = getattr(self.env, "_active_cavs", None) if self.env is not None else None
        self.last_diagnostics = self.engine.apply_all(
            {}, cav_ids=obs_dict.keys() if cav_ids is None else cav_ids
        )
        return {}

    def close(self) -> None:
        self.env = None


def build_frozen_idm_analytical_policy() -> IDMAnalyticalDampingPolicy:
    """Build the only analytical damping policy admitted by IDM wall runners."""

    policy = IDMAnalyticalDampingPolicy(config=frozen_idm_analytical_config())
    if policy.config.reference_mode != FROZEN_IDM_ANALYTICAL_REFERENCE_MODE:
        raise RuntimeError("frozen analytical damping reference mode changed")
    return policy
