"""Real frozen-controller CPU integration; simulator start/step are intercepted."""
import argparse
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from .protocol import LIBRARY, safe_path
from .runtime import library


def check(root):
    lib = library(root)
    inputs = lib.setup()
    import traci
    import numpy as np
    from tools.c1switch.library import make_handover
    with patch.object(traci, 'start', side_effect=AssertionError('CPU gate cannot start SUMO')), \
            patch.object(traci, 'simulationStep', side_effect=AssertionError('CPU gate cannot step SUMO')):
        h = make_handover(inputs)
        if h.controllers != LIBRARY:
            raise ValueError('wrong published library order')
        owner = SimpleNamespace(release_all=Mock(), controller=None)
        frame = SimpleNamespace(s=np.array([-600., -605.]), v=np.array([15., 15.]),
                                a=np.zeros(2), present=np.ones(2, bool), is_cav=np.ones(2, bool),
                                junction=np.zeros(2, bool), lane=np.array([0, 1]),
                                lane_carry=np.array([0, 1]), vid_index=np.array([0, 1]))
        env = SimpleNamespace(now=300., api=SimpleNamespace(vehicle=SimpleNamespace(
            getIDList=lambda: ['a', 'b'])), ownership=owner, _circles=object(),
            pi_session=None, _direct_pi=False, frame=frame,
            builder=SimpleNamespace(vehicle_ids=['a', 'b']), cav_ids={'a', 'b'})
        # Real ctrl0/MAPPO constructors load their exact frozen checkpoints on CPU.
        for name in LIBRARY:
            h.select(name, env)
        wall = h.controller
        command, kwargs = wall.command(env)
        if command.shape != (1, 2) or not np.isfinite(command).all() or kwargs != {'fallback': 'damping'}:
            raise ValueError('real WALL request integration failed')
        if wall.exe.law.n_walls != 1:
            raise ValueError('synthetic observable roster failed to form WALL')
        h.select('wall', env)
        if h.controller is not wall:
            raise ValueError('incumbent WALL memory discarded')
        env.now = 320.
        h.select('pi', env)
        if wall.exe.running or wall.exe.law is not None:
            raise ValueError('outgoing WALL memory/requests retained')
        if owner.release_all.call_count != 6:
            raise ValueError('handover releases not exactly once per change')
    return dict(status='CPU_INTEGRATION_PASS', library=list(LIBRARY),
                real_checkpoint_constructors=True, real_wall_formed=True,
                wall_incumbent_preserved=True, wall_stop_reset=True, new_sumo_seconds=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    print(json.dumps(check(safe_path(parser.parse_args().root))))
