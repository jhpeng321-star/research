"""Eval-time wrapper for the trained RSU wall supervisor (2026-08-21).

Loads the single-agent supervisor checkpoint and acts as a direct-control
policy in the probe harness: each step it assembles the corridor observation
(env.wall_supervisor.corridor_state — the same features the supervisor was
trained on), asks the supervisor for the wall state, and applies it via
WallHoldPolicy instances (probe-identical execution). Returns {} so the env
skips the per-CAV mode machinery.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from policies import DirectControlPolicy
from policies.inflow_control import WallHoldPolicy
from policies.traci_control import release_to_sumo

from env.wall_supervisor import RUNGS, corridor_state


class SupervisorPolicy(DirectControlPolicy):
    direct_control = True

    def __init__(self, checkpoint_path: str, policy_id: str = "default_policy", rungs: dict = None):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.policy_id = policy_id
        # v4.2: per-variant rung override (the eval must replay the same wall
        # config the run trained with).
        self._rungs = dict(RUNGS if rungs is None else rungs)
        self._policy = None
        self._obs_dim = None  # set on checkpoint load; None = full width
        self._walls: Dict[int, WallHoldPolicy] = {}
        self._env = None
        self._prev_h: Optional[float] = None
        self._rung = 0
        self._engaged = 0
        self._step_i = 0
        self._inflow_tracker: dict = {}

    def reset(self, env, seed: int = None) -> None:
        self._env = env
        self._prev_h = None
        self._h_hist: list = []
        self._rung = 0
        self._engaged = 0
        self._step_i = 0
        self._inflow_tracker = {}
        self._walls = {k: WallHoldPolicy(**v) for k, v in self._rungs.items()}
        for w in self._walls.values():
            w.reset(env, seed)
        if self._policy is None:
            if self.checkpoint_path.endswith(".pt"):
                # Raw torch BC net (diagnostic: does the warm-start CONTENT
                # alone reproduce the trigger, without PPO in between?)
                import torch

                sd = torch.load(self.checkpoint_path)
                saved_dim = int(sd["0.weight"].shape[1])
                net = torch.nn.Sequential(
                    torch.nn.Linear(saved_dim, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 2),
                )
                net.load_state_dict(sd)
                net.eval()
                self._bc_net = net
                # The saved net may predate the obs widening — slice the
                # corridor obs to ITS input width (the h lags are appended
                # last, so a tail slice restores the old layout exactly).
                self._obs_dim = saved_dim
                self._policy = "bc"  # marker: act() uses the BC net
            else:
                from ray.rllib.policy.policy import Policy as RLlibPolicyBase

                from models import register_mappo_model, register_selfish_ppo_model

                register_selfish_ppo_model()
                register_mappo_model()
                from policies.rllib_policy import resolve_rllib_checkpoint_dir

                ckpt = str(resolve_rllib_checkpoint_dir(self.checkpoint_path))
                loaded = RLlibPolicyBase.from_checkpoint(ckpt, policy_ids=[self.policy_id])
                self._policy = loaded[self.policy_id] if isinstance(loaded, dict) else loaded
                # v4 widened the corridor obs to 16 (h lags); older
                # checkpoints expect their original width — the lags are
                # appended last, so a tail slice restores them exactly.
                self._obs_dim = int(self._policy.observation_space.shape[0])

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, object]:
        sup_obs = corridor_state(
            self._env,
            self._prev_h,
            self._rung,
            self._engaged,
            self._step_i,
            self._inflow_tracker,
            self._h_hist,
        )
        if self._obs_dim is not None and sup_obs.shape[0] > self._obs_dim:
            sup_obs = sup_obs[: self._obs_dim]  # match the checkpoint's width
        self._prev_h = float(sup_obs[0])
        self._h_hist.append(float(sup_obs[0]))
        if self._policy == "bc":
            import torch

            with torch.no_grad():
                action = int(
                    self._bc_net(torch.tensor(sup_obs, dtype=torch.float32)).argmax().item()
                )
        else:
            action = int(self._policy.compute_single_action(sup_obs, explore=False)[0])
        if action in self._walls:
            self._walls[action].act(obs_dict)
            self._engaged += 1
        else:
            import traci

            live = traci.vehicle.getIDList()
            for cav_id in obs_dict.keys():
                if cav_id not in live:
                    continue
                try:
                    release_to_sumo(cav_id)
                except Exception:
                    pass
        self._rung = action
        self._step_i += 1
        return {}
