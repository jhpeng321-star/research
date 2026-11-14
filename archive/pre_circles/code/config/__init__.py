"""Configuration package: defaults + shared constants."""

from config.constants import (
    ACCELERATION_MAX,
    ACCELERATION_MIN,
    MODE_NAMES,
    NUM_MODES,
    RL_SPEED_MODE,
)
from config.defaults import DEFAULT_CONFIG, merge_config

__all__ = [
    "DEFAULT_CONFIG",
    "merge_config",
    "MODE_NAMES",
    "NUM_MODES",
    "RL_SPEED_MODE",
    "ACCELERATION_MIN",
    "ACCELERATION_MAX",
]
