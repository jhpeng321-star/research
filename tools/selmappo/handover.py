"""C1SWITCH ownership/reset semantics with the S-WALLSW lifecycle contract."""
from .protocol import LIBRARY


class FleetHandover:
    def __init__(self, factory):
        self.factory, self.name, self.controller = factory, 'native', None
        self.switches = 0

    def select(self, name, env):
        if name not in LIBRARY:
            raise ValueError('unregistered controller')
        if name == self.name:
            return self.controller
        controller = self.factory(name)
        if self.controller is not None and hasattr(self.controller, 'stop'):
            self.controller.stop(now_s=getattr(env, 'now', 0.))
        env.ownership.release_all(live_ids=env.api.vehicle.getIDList())
        env.ownership.controller = env._circles
        env.pi_session, env._direct_pi = None, False
        if hasattr(controller, 'start'):
            controller.start(now_s=getattr(env, 'now', 0.))
        self.name, self.controller = name, controller
        self.switches += 1
        return controller
