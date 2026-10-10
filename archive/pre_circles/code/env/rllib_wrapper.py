"""RLlib MultiAgentEnv wrapper around TrafficShepherdMultiAgentEnv."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from env.core import TrafficShepherdMultiAgentEnv

POLICY_ID = "cav_policy"


class TrafficShepherdRLlibEnv(MultiAgentEnv):
    """Multi-agent wrapper with shared spaces and stable RLlib agent IDs.

    SUMO vehicle IDs can disappear for a step then reappear, or leave and
    (rarely) collide with bookkeeping. RLlib forbids the same agent_id from
    returning after terminated/truncated=True. This wrapper:

    1. Maps each live SUMO CAV to a unique RLlib agent_id.
    2. On true exit, emits one terminal transition then never reuses that id.
    3. If the same SUMO vehicle is seen again after done, assigns a fresh id.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        # Old-stack RLlib hands the env creator an EnvContext whose
        # worker_index / vector_index are *attributes*, not dict keys
        # (ray 2.55); read both channels so the inner env can diversify its
        # master RNG seed per env runner.
        ctx_worker_index = getattr(config, "worker_index", None)
        ctx_vector_index = getattr(config, "vector_index", None)
        config = dict(config or {})

        worker_index = config.pop("worker_index", ctx_worker_index)
        vector_index = config.pop("vector_index", ctx_vector_index)
        worker_index = int(worker_index) if worker_index is not None else 0
        vector_index = int(vector_index) if vector_index is not None else 0
        config.pop("num_workers", None)
        config.pop("remote", None)

        # Ports are allocated fresh on every reset via find_free_port().
        config.pop("sumo_port", None)
        config.pop("sumo_port_base", None)
        config.setdefault("worker_index", worker_index)
        config.setdefault("vector_index", vector_index)
        # Ensure RLlib never sees an empty-agent episode at reset.
        config.setdefault("cav_warmup_steps", 300)

        self._env = TrafficShepherdMultiAgentEnv(config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        self._sumo_to_agent: Dict[str, str] = {}
        self._agent_to_sumo: Dict[str, str] = {}
        self._done_agents: set = set()
        self._agent_seq = 0

        self._agent_ids: set = set()
        self.agents: list = []
        self.possible_agents: list = []

    def _fresh_agent_id(self, sumo_id: str) -> str:
        self._agent_seq += 1
        return f"{sumo_id}__{self._agent_seq}"

    def _bind_sumo(self, sumo_id: str) -> str:
        """Return an active RLlib agent id for this SUMO vehicle."""
        existing = self._sumo_to_agent.get(sumo_id)
        if existing is not None and existing not in self._done_agents:
            return existing
        aid = self._fresh_agent_id(sumo_id)
        self._sumo_to_agent[sumo_id] = aid
        self._agent_to_sumo[aid] = sumo_id
        return aid

    def _actions_to_sumo(self, action_dict: Dict[str, object]) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for aid, action in action_dict.items():
            if aid in self._done_agents:
                continue
            sumo_id = self._agent_to_sumo.get(aid)
            if sumo_id is None:
                # Defensive: allow raw SUMO ids if a caller bypasses mapping.
                sumo_id = aid
            out[sumo_id] = action
        return out

    def _map_dict(self, data: Dict[str, Any], sumo_ids) -> Dict[str, Any]:
        mapped: Dict[str, Any] = {}
        for sumo_id in sumo_ids:
            if sumo_id not in data:
                continue
            aid = self._bind_sumo(sumo_id)
            if aid in self._done_agents:
                # Should not happen for continuing agents; skip if stale.
                continue
            mapped[aid] = data[sumo_id]
        return mapped

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._sumo_to_agent.clear()
        self._agent_to_sumo.clear()
        self._done_agents.clear()
        self._agent_seq = 0
        self.possible_agents = []

        obs_sumo, info = self._env.reset(seed=seed, options=options)
        obs = {}
        for sumo_id, value in obs_sumo.items():
            aid = self._bind_sumo(sumo_id)
            obs[aid] = value

        self._agent_ids = set(obs.keys())
        self.agents = list(self._agent_ids)
        self.possible_agents = list(self.agents)
        infos = {aid: dict(info) for aid in self.agents} if self.agents else {}
        return obs, infos

    def step(self, action_dict: Dict[str, object]):
        sumo_actions = self._actions_to_sumo(action_dict)
        obs_s, rewards_s, term_s, trunc_s, info = self._env.step(sumo_actions)

        obs: Dict[str, np.ndarray] = {}
        rewards: Dict[str, float] = {}
        terminated: Dict[str, bool] = {}
        truncated: Dict[str, bool] = {}

        # Continuing / new agents first.
        for sumo_id, value in obs_s.items():
            is_term = bool(term_s.get(sumo_id, False))
            is_trunc = bool(trunc_s.get(sumo_id, False))

            existing = self._sumo_to_agent.get(sumo_id)
            if existing is not None and existing in self._done_agents:
                if is_term or is_trunc:
                    # Stale re-send of an already-finished agent: drop it.
                    continue
                # True revival of the SUMO vehicle after a prior done → new id.
                self._sumo_to_agent.pop(sumo_id, None)

            aid = self._bind_sumo(sumo_id)
            obs[aid] = value
            if sumo_id in rewards_s:
                rewards[aid] = rewards_s[sumo_id]
            terminated[aid] = is_term
            truncated[aid] = is_trunc

        # Terminal-only keys (left network): may already be in obs_s with dummy.
        for sumo_id, flag in term_s.items():
            if sumo_id == "__all__" or not flag:
                continue
            if sumo_id in obs_s:
                continue
            aid = self._bind_sumo(sumo_id)
            obs.setdefault(aid, np.zeros(self.observation_space.shape, dtype=np.float32))
            rewards.setdefault(aid, float(rewards_s.get(sumo_id, 0.0)))
            terminated[aid] = True
            truncated[aid] = bool(trunc_s.get(sumo_id, False))

        for sumo_id, flag in trunc_s.items():
            if sumo_id == "__all__" or not flag:
                continue
            aid = self._sumo_to_agent.get(sumo_id)
            if aid is None:
                continue
            truncated[aid] = True

        terminated["__all__"] = bool(term_s.get("__all__", False))
        truncated["__all__"] = bool(trunc_s.get("__all__", False))

        # Record newly finished agents; they must never appear again.
        newly_done = []
        for aid in list(obs.keys()):
            if terminated.get(aid, False) or truncated.get(aid, False):
                newly_done.append(aid)
                self._done_agents.add(aid)

        infos = {aid: dict(info) for aid in obs.keys()}

        # Active agent list for the *next* step excludes finished ones.
        self.agents = [
            aid
            for aid in obs.keys()
            if not (terminated.get(aid, False) or truncated.get(aid, False))
        ]
        self._agent_ids = set(self.agents)
        for aid in obs.keys():
            if aid not in self.possible_agents:
                self.possible_agents.append(aid)

        return obs, rewards, terminated, truncated, infos

    def close(self) -> None:
        self._env.close()

    def get_agent_ids(self):
        return set(self.agents)


def make_traffic_shepherd_env(env_config: Dict[str, Any]) -> TrafficShepherdRLlibEnv:
    """Factory for ray.tune.registry.register_env."""
    return TrafficShepherdRLlibEnv(env_config)
