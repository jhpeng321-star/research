"""WM-S step-aware, physics-informed next-state trainer."""

from .blackbox_traj import BlackBoxTrajModel
from .model import PhysicsConfig, WmSModel

__all__ = ["BlackBoxTrajModel", "PhysicsConfig", "WmSModel"]
