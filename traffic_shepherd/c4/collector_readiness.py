"""Check reset-time controller assets without a seed, scenario build, or SUMO."""
from __future__ import annotations

import hashlib
import importlib
import shutil
import subprocess


def check_controller_assets(config, controller_factory=None):
    path = config.model_path
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != config.model_sha256:
        raise ValueError("CIRCLES model checksum mismatch")
    if controller_factory is None:
        from traffic_shepherd.damping import CirclesController
        controller_factory = CirclesController
    controller_factory(config)  # also validates ONNX runtime/session creation, no inference tuning
    return {"model_path": str(path), "model_sha256": actual, "onnx_session": "ready"}


def check_runtime_dependencies():
    """Resolve the actual worker binary and eager reset imports; never allocate a world."""
    binary = shutil.which('sumo')
    if binary is None:
        raise RuntimeError('SUMO binary not found on PATH')
    version = subprocess.run([binary, '--version'], capture_output=True, text=True, timeout=30, check=True)
    for name in ('traffic_shepherd.controller_factory', 'traffic_shepherd.fleet_switch',
                 'traffic_shepherd.c4.policy_pairs', 'traffic_shepherd.c4.sumocredit'):
        importlib.import_module(name)
    from traffic_shepherd.c4.env import CavSumoEnv
    env = CavSumoEnv(controller='direct', population='textbook')
    env._setup_controllers('direct')  # eager reset imports, no scenario, simulator start or seed
    env.close()
    return {'sumo_binary': binary, 'sumo_version': version.stdout.splitlines()[0],
            'reset_imports': 'ready', 'actual_sumo_sim_s': 0}
