"""ZERO-SUMO setup-only check of a bound frozen MAPPO trunk through the TRAINING shim (frozen_trunk).

Usage (cpuV/cpuZ, inside a deployed source, env.sh sourced, nice 19, CUDA_VISIBLE_DEVICES=):
  python ops/clean5a_trunk_setup_check.py --runtime /home/ubuntu/e3final --checkpoint <file> --sha256 <sha>
Binds COOPMWM_FROZEN_TRUNK for this process only, runs frozen_trunk.setup (leverprobe's frozen checks with
only the trunk pin swapped), then loads the executor exactly as the plant does and reports native lanes.
"""
import argparse
import json
import os

p = argparse.ArgumentParser()
p.add_argument('--runtime', required=True); p.add_argument('--checkpoint', required=True); p.add_argument('--sha256', required=True)
a = p.parse_args()
os.environ['COOPMWM_FROZEN_TRUNK'] = f'{a.checkpoint}|{a.sha256}'
from tools.coopmwm import frozen_trunk as ft  # noqa: E402
from tools.leverprobe import backend as lp  # noqa: E402
before = lp.MAPPO_SHA
ft.setup(a.runtime, a.checkpoint)
from wms.control.policy import load_policy_fn  # noqa: E402  (frozen runtime path prepended by setup)
exe = load_policy_fn(a.checkpoint, device='cpu')
out = dict(status='PASS' if (getattr(exe, 'native_lanes', False) and not getattr(exe, 'residual_on_native', False)) else 'FAIL',
           shim='training (tools.coopmwm.frozen_trunk)', sha256=a.sha256, sha256_loaded=ft.file_sha256(a.checkpoint),
           frozen_default_pin=before, module_pin_after=lp.MAPPO_SHA, pin_swapped=a.sha256 != before,
           native_lanes=bool(getattr(exe, 'native_lanes', False)), residual_on_native=bool(getattr(exe, 'residual_on_native', False)),
           sumo_steps=0, sim_s=0)
assert out['module_pin_after'] == before, 'module pin must never be mutated'
print(json.dumps(out, sort_keys=True))
