"""Bind the accepted G5 env change into the frozen SUMO runtime that episodes actually use.

Every native episode calls tools.leverprobe.backend.setup(job['runtime'], ...), which
hash-verifies the frozen runtime (/home/ubuntu/e3final) and PREPENDS it to sys.path,
so ``traffic_shepherd.c4.env`` resolves to the frozen, pre-G5 env (no ``lane_actors``).
Clean3 needs exactly the accepted G5 change (9fd32b4e: additive ``lane_actors``,
per-vehicle lane telemetry, byte-identical behaviour when the argument is absent)
on top of THAT frozen env, not the repository's later env version.

``g5_runtime_env.pysrc`` = frozen env.py + the accepted G5 hunks (identical +/- lines,
see the CPU test). ``install`` verifies the frozen base and the overlay by SHA256
(line-ending normalised) and registers the overlay as ``traffic_shepherd.c4.env``
inside the frozen ``traffic_shepherd.c4`` package, before any env import. All other
frozen modules are untouched. Opt-in per process via COOPMWM_G5_RUNTIME=1 (the clean3
chain sets it for G1, smoke, gate and training owners); absent -> no effect.
"""
import hashlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys

FLAG = 'COOPMWM_G5_RUNTIME'
MODULE = 'traffic_shepherd.c4.env'
BASE_REL = Path('runtime')/'traffic_shepherd'/'c4'/'env.py'
BASE_SHA256 = '59aecd0bff2c0218ea33450c0726b06bb3497e6fe029a3e3797ee5151b505040'
OVERLAY = Path(__file__).with_name('g5_runtime_env.pysrc')
OVERLAY_SHA256 = '5aac1769679964e6df6950cfaf115902c2787ac34d5a20622d8bf67bc7a8bcde'


def normalized_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def enabled():
    return os.environ.get(FLAG) == '1'


def install(runtime):
    """Register the bound G5 env for this process; idempotent; refuses any mismatch."""
    if not enabled():
        return None
    current = sys.modules.get(MODULE)
    if current is not None:
        if getattr(current, 'G5_RUNTIME_OVERLAY_SHA256', None) == OVERLAY_SHA256:
            return current
        raise ValueError('traffic_shepherd.c4.env was imported before the G5 runtime binding')
    base = Path(runtime)/BASE_REL
    if normalized_sha(base) != BASE_SHA256:
        raise ValueError('frozen runtime env differs from the G5 overlay base')
    if normalized_sha(OVERLAY) != OVERLAY_SHA256:
        raise ValueError('G5 runtime overlay changed')
    import traffic_shepherd.c4 as package
    if Path(list(package.__path__)[0]).resolve() != base.parent.resolve():
        raise ValueError('traffic_shepherd.c4 must resolve to the hash-verified frozen runtime')
    loader = importlib.machinery.SourceFileLoader(MODULE, str(OVERLAY))
    spec = importlib.util.spec_from_file_location(MODULE, str(OVERLAY), loader=loader)
    module = importlib.util.module_from_spec(spec)
    module.G5_RUNTIME_OVERLAY_SHA256 = OVERLAY_SHA256
    sys.modules[MODULE] = module
    try:
        loader.exec_module(module)
    except BaseException:
        del sys.modules[MODULE]
        raise
    package.env = module
    return module
