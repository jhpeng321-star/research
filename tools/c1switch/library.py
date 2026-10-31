"""Expanded library for new consumers; historical experiment design stays pinned."""
import importlib
from pathlib import Path

from .design import CONTROLLERS

SWITCHABLE_CONTROLLERS = (*CONTROLLERS, "wall")


def _new_wall():
    module = importlib.import_module('traffic_shepherd.c4.switchable_wall')
    return module.SwitchableWallController()


def install_wall_extension():
    """Expose additive staged modules beside a hash-pinned runtime, in process.

    No runtime files or manifests are rewritten. Stage the new modules at their
    repository-relative paths beside this tools package. Existing runtime
    modules retain import precedence. Bind these extension sources separately
    in the consuming experiment's source receipt.
    """
    root = Path(__file__).resolve().parents[2]
    for name, relative in [('traffic_shepherd.c4', 'traffic_shepherd/c4'),
                           ('wms.control', 'src/wms/control')]:
        package = importlib.import_module(name)
        path = str(root / relative)
        if path not in package.__path__:
            package.__path__.append(path)
    bridge = importlib.import_module('wms.sim_bridge')
    if not hasattr(bridge, 'make_switchable_wall'):
        bridge.make_switchable_wall = _new_wall


def make_handover(inputs):
    install_wall_extension()
    from .core import Handover
    from .sumo import Controller

    return Handover(lambda name: Controller(name, inputs), controllers=SWITCHABLE_CONTROLLERS)


def make_wall_rollout(model, *, fallback, execute=None, start_s=0.):
    install_wall_extension()
    from wms.control.wall import WallRolloutController
    return WallRolloutController(model, fallback=fallback, execute=execute, start_s=start_s)
