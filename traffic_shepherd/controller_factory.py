"""Explicit opt-in controller selection for the supported evaluation workflow."""

from .damping import CirclesController
from .pi_session import PISession


VERSION = "controller_factory/2"


def create_controller(controller_name, config, *, desired_speed_mps=None):
    if controller_name == "native":
        if desired_speed_mps is not None:
            raise ValueError("desired speed is only valid for pi-external-u")
        return CirclesController(config), {"factory_version": VERSION, "controller": "native"}
    if controller_name == "damping":
        if desired_speed_mps is not None:
            raise ValueError("desired speed is only valid for pi-external-u")
        return CirclesController(config), {"factory_version": VERSION, "controller": "damping"}
    raise ValueError("unsupported controller")


def create_pi_session(config, cav_ids, *, desired_speed_mps):
    return PISession(config=config, cav_ids=cav_ids, desired_speed_mps=desired_speed_mps)
