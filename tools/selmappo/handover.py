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
        if self.name == 'wall':
            self.controller.stop(env)
        env.ownership.release_all(live_ids=env.api.vehicle.getIDList())
        env.ownership.controller = env._circles
        env.pi_session, env._direct_pi = None, False
        controller = self.factory(name)
        if name == 'wall':
            controller.start(env)
        self.name, self.controller = name, controller
        self.switches += 1
        return controller
