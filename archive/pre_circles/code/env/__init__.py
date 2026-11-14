"""SUMO ↔ RL environment package.

Main pieces:
  core.py / rllib_wrapper.py — Gym + Ray multi-agent wrappers
  rewards.py                 — Shepherd/Flat-MARL + Selfish reward functions
  actions.py                 — Shepherd discrete modes + continuous acceleration
  features.py                — observations (neighbors, trust, zones)
  vehicles.py                — CAV selection + HDV driver types
"""

import os
import sys

# Put SUMO's python tools (traci, sumolib, ...) on sys.path if SUMO_HOME is set.
_sumo_home = os.environ.get("SUMO_HOME")
if _sumo_home:
    _tools_path = os.path.join(_sumo_home, "tools")
    if _tools_path not in sys.path:
        sys.path.append(_tools_path)

from env.core import TrafficShepherdMultiAgentEnv  # noqa: E402

__all__ = ["TrafficShepherdMultiAgentEnv"]
