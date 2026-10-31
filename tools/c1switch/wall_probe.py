"""Synthetic CPU callback integration probe against an existing runtime tree.

No checkpoint, recorded data, simulator or training is used. Extension modules
are overlaid in this process only; the target worktree is never modified.
"""
import argparse
from dataclasses import fields
import importlib
import json
from pathlib import Path
import re
import sys


def probe(runtime_root, *, wm_phys=False):
    root = Path(runtime_root).resolve()
    if re.search(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1', str(root), re.I):
        raise ValueError('restricted runtime path')
    extension = Path(__file__).resolve().parents[2]
    # Run as a fresh Python process, before importing model/runtime packages.
    if any(name in sys.modules for name in ('wms.model.wms', 'traffic_shepherd.c4.baselines')):
        raise ValueError('probe needs a fresh interpreter')
    sys.path[:0] = [str(root / 'src'), str(root)]
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(20261101)
    from .library import install_wall_extension
    install_wall_extension()
    wall_module = importlib.import_module('wms.control.wall')
    from wms.model.wms import WmSModel
    from wms.model.spec import ModelSpec
    from wms.physics.config import PhysicsConfig
    from wms.train.cav_drive import TimelineControllers
    cf_available = 'cf_model' in {field.name for field in fields(PhysicsConfig)}
    if wm_phys:
        physics = PhysicsConfig(merge_model='sumo', lane_learnable=False,
                                **({'cf_model': 'sumo_idm'} if cf_available else {}))
        spec = ModelSpec(rollout_mode='driver_params', lane_model='lc2013', lane_head='off',
                         residual_accel_off=True, temporal_accel_off=True)
        model = WmSModel(spec, physics=physics).eval()
    else:
        model = WmSModel().eval()
    state = dict(s_m=torch.tensor([[-1000., -1050., -500., -200.]]),
                 v_mps=torch.tensor([[20., 20., 18., 22.]]),
                 a_mps2=torch.zeros(1, 4), lane=torch.tensor([[0, 1, 2, 1]]),
                 present=torch.ones(1, 4, dtype=torch.bool),
                 is_cav=torch.tensor([[True, True, True, False]]),
                 length_m=torch.full((1, 4), 5.))
    timeline = dict(controller=torch.ones(1, 4, 4, dtype=torch.long),
                    acceleration_mps2=torch.zeros(1, 4, 4),
                    target_speed_mps=torch.zeros(1, 4, 4))
    damping = TimelineControllers(model, timeline)
    callback = wall_module.WallRolloutController(model, fallback=damping, start_s=301.)
    with torch.inference_mode():
        result = model.rollout(state, n_steps=4, cav_controller=callback)
    assert torch.isfinite(result['positions']).all()
    assert torch.isfinite(result['speeds']).all()
    assert callback.step_index == damping.step_index == 4
    assert callback.controllers[0].law.n_walls == 1
    return dict(passed=True, synthetic=True, checkpoint_loaded=False, new_sim_s=0,
                runtime_root=str(root), runtime_model=sys.modules['wms.model.wms'].__file__,
                runtime_bridge=sys.modules['wms.sim_bridge'].__file__,
                extension_root=str(extension), wm_phys=wm_phys, physics_ticks=4,
                cf_model_available=cf_available,
                wm_phys_complete=bool(wm_phys and cf_available),
                wall_count=callback.controllers[0].law.n_walls, device='cpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', required=True)
    parser.add_argument('--wm-phys', action='store_true')
    args = parser.parse_args()
    print(json.dumps(probe(args.runtime_root, wm_phys=args.wm_phys), indent=2))
