"""RLlib policy wrapper for trained checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from policies import Policy

POLICY_ID_DEFAULT = "cav_policy"


def is_rllib_checkpoint(path: Union[str, Path]) -> bool:
    """Return True if path looks like an RLlib Algorithm/Policy checkpoint."""
    p = Path(path)
    if not p.exists():
        return False
    if p.is_file():
        # Custom torch .pt / .pth checkpoints are not RLlib.
        return False
    markers = (
        "rllib_checkpoint.json",
        "algorithm_state.pkl",
        "policies",
    )
    if any((p / m).exists() for m in markers):
        return True
    # algo.save() may nest under checkpoint_* /
    for child in p.rglob("rllib_checkpoint.json"):
        return True
    for child in p.rglob("algorithm_state.pkl"):
        return True
    return (p / "policies").is_dir()


def resolve_rllib_checkpoint_dir(path: Union[str, Path]) -> Path:
    """Resolve a user path to the directory Policy.from_checkpoint expects."""
    p = Path(path).resolve()
    if (p / "rllib_checkpoint.json").exists() or (p / "algorithm_state.pkl").exists():
        return p
    if (p / "policies").is_dir():
        return p
    # Walk one level for algo.save(".../best") -> .../best/checkpoint_000000
    candidates = sorted(p.glob("checkpoint_*"))
    if candidates:
        return candidates[-1]
    nested = list(p.rglob("rllib_checkpoint.json"))
    if nested:
        return nested[0].parent
    return p


def checkpoint_custom_model_config(
    checkpoint_path: Union[str, Path], policy_id: str = POLICY_ID_DEFAULT
) -> Optional[dict]:
    """Read the custom model config stored in an RLlib checkpoint.

    Returns e.g. ``{"max_neighbors": 20, "hidden": 128, "max_cavs_critic": 64,
    "local_obs_dim": 112}`` so eval harnesses can size the env's packed
    observation to match the checkpoint. None if unreadable.
    """
    import pickle

    ckpt = resolve_rllib_checkpoint_dir(checkpoint_path)
    pkl = ckpt / "policies" / policy_id / "policy_state.pkl"
    if not pkl.exists():
        return None
    try:
        with open(pkl, "rb") as f:
            state = pickle.load(f)
    except Exception:
        return None
    cfg = state.get("policy_spec", {}).get("config", {})
    custom = cfg.get("model", {}).get("custom_model_config", {}) or {}
    return dict(custom)


class RLlibPolicy(Policy):
    """Load a trained Ray RLlib checkpoint and use it for evaluation.

    Default policy id is ``cav_policy`` (shared CAV policy from train.py).
    """

    def __init__(
        self,
        checkpoint_path: str,
        policy_id: str = POLICY_ID_DEFAULT,
        explore: bool = False,
    ):
        self.checkpoint_path = checkpoint_path
        self.policy_id = policy_id
        self.explore = explore
        self._policy = None
        self._continuous_actions = False

    def reset(self, env: Any, seed: int = None) -> None:
        try:
            import gymnasium as gym

            self._continuous_actions = isinstance(env.action_space, gym.spaces.Box)
            if self._continuous_actions:
                self._act_low = float(env.action_space.low[0])
                self._act_high = float(env.action_space.high[0])
        except ImportError:
            self._continuous_actions = False
        if self._policy is None:
            try:
                from ray.rllib.policy.policy import Policy as RLlibPolicyBase
            except ImportError as exc:
                raise ImportError("ray[rllib] is required to use RLlibPolicy") from exc

            from models import register_mappo_model, register_selfish_ppo_model

            register_selfish_ppo_model()
            register_mappo_model()
            ckpt = str(resolve_rllib_checkpoint_dir(self.checkpoint_path))
            loaded = RLlibPolicyBase.from_checkpoint(ckpt, policy_ids=[self.policy_id])
            if isinstance(loaded, dict):
                if self.policy_id not in loaded:
                    raise KeyError(
                        f"Policy id {self.policy_id!r} not in checkpoint; "
                        f"found {list(loaded.keys())}"
                    )
                self._policy = loaded[self.policy_id]
            else:
                self._policy = loaded

    def act(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, object]:
        if self._policy is None:
            raise RuntimeError("Policy has not been reset. Call reset(env) first.")

        actions = {}
        for cav_id, obs in obs_dict.items():
            action = self._policy.compute_single_action(obs, explore=self.explore)[0]
            if isinstance(action, (tuple, list)):
                # Hybrid composite: (mode, dial) — pass through as
                # (int, clipped float) for the env parser.
                actions[cav_id] = (
                    int(np.asarray(action[0]).reshape(-1)[0]),
                    float(
                        np.clip(
                            float(np.asarray(action[1], dtype=np.float32).reshape(-1)[0]),
                            -1.0,
                            1.0,
                        )
                    ),
                )
            elif self._continuous_actions:
                action = np.clip(
                    np.asarray(action, dtype=np.float32),
                    self._act_low,
                    self._act_high,
                ).reshape(-1)
                # Box(1) -> scalar (legacy); multi-dim Box (hybrid) -> vector.
                actions[cav_id] = float(action[0]) if action.shape[0] == 1 else action
            elif isinstance(action, dict):
                actions[cav_id] = int(action.get("mode", 0))
            else:
                actions[cav_id] = int(np.atleast_1d(action)[0])
        return actions

    def close(self) -> None:
        self._policy = None
