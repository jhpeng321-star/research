"""Expanded library for new consumers; historical experiment design stays pinned."""
from .design import CONTROLLERS

SWITCHABLE_CONTROLLERS = (*CONTROLLERS, "wall")


def make_handover(inputs):
    from .core import Handover
    from .sumo import Controller

    return Handover(lambda name: Controller(name, inputs), controllers=SWITCHABLE_CONTROLLERS)
