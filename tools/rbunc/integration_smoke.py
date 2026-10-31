"""Independent synthetic admission identity smoke against an integrated runtime.

PYTHONPATH must select the integrator's src; this task's sampler is loaded by
absolute path. No peer worktree is modified. No scientific data/SUMO/training.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import inspect
import json
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fitted-directory', type=Path)
    parser.add_argument('--lane-directory', type=Path)
    args = parser.parse_args()
    assert args.out.parent.is_dir()
    torch.set_num_threads(1)
    here = Path(__file__).resolve().parents[2]
    path = here/'src/wms/control/hidden_state.py'
    module_spec = importlib.util.spec_from_file_location('wms.control.hidden_state', path)
    hidden = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = hidden
    module_spec.loader.exec_module(hidden)
    fitted = None
    if args.fitted_directory is not None:
        path_fit = here/'src/wms/control/fitted_hidden.py'
        spec_fit = importlib.util.spec_from_file_location('wms.control.fitted_hidden', path_fit)
        mod_fit = importlib.util.module_from_spec(spec_fit)
        sys.modules[spec_fit.name] = mod_fit
        spec_fit.loader.exec_module(mod_fit)
        fitted = mod_fit.FittedHiddenState.from_directory(args.fitted_directory,
            lane_directory=args.lane_directory)
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel
    from wms.physics.config import PhysicsConfig
    from wms.model.entries import EntranceQueue, build_entrance_history
    from wms.model.driver_encoder import PARAMETER_NAMES
    from wms.model.sumo_insertion import SumoInsertion
    model = WmSModel(ModelSpec(rollout_mode='driver_params', lane_model='lc2013'),
        physics=PhysicsConfig(merge_model='sumo', **({} if fitted is None else {'cf_model':'sumo_idm'}))).eval()
    state = dict(s_m=torch.zeros(1, 16), v_mps=torch.zeros(1, 16),
        a_mps2=torch.zeros(1, 16), length_m=torch.full((1, 16), 5.),
        lane=torch.zeros(1, 16, dtype=torch.long), present=torch.zeros(1, 16, dtype=torch.bool),
        is_cav=torch.zeros(1, 16, dtype=torch.bool))
    state['s_m'][0, :3] = torch.tensor([-2500., -2400., -2200.])
    state['v_mps'][0, :3] = torch.tensor([10., 12., 14.])
    state['lane'][0, :3] = torch.tensor([0, 1, 1])
    state['present'][0, :3] = True
    history = torch.zeros(1, 16, model.history_len, 6)
    history[..., 0] = state['v_mps'][..., None] / 30
    history[..., 2] = 1
    history[..., 5] = 1
    mask = state['present'][..., None].expand(-1, -1, model.history_len).clone()
    histories = [build_entrance_history(counter=np.zeros((2, 30)))]
    queues = [EntranceQueue(np.ones(2), np.zeros(2), cav_share=np.zeros(2))]
    hashes, assignments = [], []
    before_rng = torch.get_rng_state().clone()
    step_parameters = []
    original_step = model.step
    def checked_step(state, **kwargs):
        if fitted is not None and args.lane_directory is not None:
            step_parameters.append({n:v.clone() for n,v in kwargs['driver_params'].items()})
        return original_step(state, **kwargs)
    model.step = checked_step
    for _ in range(2):
        extra = {} if fitted is None else dict(fitted_state=fitted,
            vehicle_ids=torch.arange(16)[None], t0_s=300, row_keys=[969400])
        f = hidden.plan_hidden_state_futures(model, state,
            dict(history=history, history_mask=mask), future_sampling='hidden_state',
            n_samples=1, seed=35, entrance_histories=histories, queues=queues, **extra)[0]
        mix = f.kwargs['entrant_driver_sampler']
        expected = [mix.candidate(0, e, 0) for e in (0, 1)]
        with torch.inference_mode():
            roll = model.rollout({k: v.clone() for k, v in state.items()}, n_steps=5,
                history=history, history_mask=mask, **f.kwargs)
        entries = f.kwargs['entries']
        assert isinstance(entries.cfg.sumo_insertion, SumoInsertion)
        assert entries.cfg.learned_admission is None and entries.cfg.learned_entry is None
        assert entries.stats[0].inserted == 2 and not entries.pending.any()
        for entrance, slot in enumerate((3, 4)):
            actual = {name: float(roll['driver_parameters'][0, slot, j])
                      for j, name in enumerate(PARAMETER_NAMES)}
            assert actual == {n:expected[entrance][n] for n in PARAMETER_NAMES}
            assert entries.cfg.sumo_insertion.candidate(0, entrance, 0) == expected[entrance]
            if step_parameters:
                for values in step_parameters[-5:]:
                    for n in mod_fit.LC_ORDER:
                        assert float(values[n][0, slot]) == float(torch.tensor(expected[entrance][n]))
        hashes.append(hashlib.sha256(b''.join(roll[k].numpy().tobytes()
            for k in ('positions', 'speeds', 'lanes', 'masks'))).hexdigest())
        assignments.append(expected)
    assert hashes[0] == hashes[1] and torch.equal(before_rng, torch.get_rng_state())
    result = dict(status='PASS', scientific_rollouts=0, synthetic_replays=2,
        ticks_per_replay=5, exact_admission_motion_identity=True, global_rng_unchanged=True,
        tape_sha256=hashes, assigned=assignments,
        sampler_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        runtime_sources={inspect.getfile(obj): hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest()
                         for obj in (WmSModel, SumoInsertion)},
         cf_model=getattr(model.physics, 'cf_model', 'legacy'))
    if fitted is not None:
        result.update(fitted_artifacts=fitted.provenance,
            fitted_lc_persistent=bool(step_parameters),
            fitted_reader_sha256=hashlib.sha256(path_fit.read_bytes()).hexdigest())
    with args.out.open('x') as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
