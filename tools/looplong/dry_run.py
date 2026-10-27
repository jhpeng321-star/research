"""Synthetic CPU engineering dry run only: tiny frozen-core WM, no traffic jobs."""
from __future__ import annotations

import cProfile
import copy
import io
import json
import os
from pathlib import Path
import pstats
import time

import numpy as np
import torch

from wms.model.wms import WmSModel
from wms.model.spec import ModelSpec
from wms.model.residual_net import HdvConfig
from wms.physics.config import PhysicsConfig
from wms.train.settings import TrainSettings
from wms.train.losses.compose import build_loss
from wms.train.unroll import UnrollRunner


def model(courtesy=False):
    torch.manual_seed(71003)
    m = WmSModel(ModelSpec(rollout_mode='residual', lane_head='off', follower_courtesy=courtesy,
                         follower_courtesy_intent='none'), physics=PhysicsConfig(max_gap_steps=8),
        hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16))
    for name, p in m.named_parameters():
        p.requires_grad_(name.startswith(('residual_net', 'follower_courtesy')))
    return m


def batch(steps=120):
    t = torch.arange(steps + 1).float()[None, :, None]
    states = torch.zeros(2, steps + 1, 3, 7)
    states[..., 0] = torch.tensor([-600., -650., -700.])[None, None] + 6. * t
    states[..., 1] = 1.
    states[..., 4] = 12.
    states[..., 6] = 5.
    states[1, 1:, 0, 4] -= 1.
    present = torch.ones(states.shape[:3], dtype=torch.bool)
    cav = torch.zeros_like(present)
    cav[..., 0] = True
    teacher = torch.stack((states[..., 0], states[..., 1:4].argmax(-1).float(),
                           states[..., 4], states[..., 5]), -1)
    return {'states': states, 'present': present, 'loss_valid': present,
            'slot_index': torch.arange(3).expand(2, steps + 1, -1), 'is_cav': cav,
            'cav_teacher': teacher, 'road_present': present,
            'road_continue': present[:, :-1], 'road_valid': torch.ones(2, steps + 1, dtype=torch.bool),
            'pair_ref': torch.tensor([0, 0]), 'pair_group': torch.tensor([0, 0]),
            'mix_source': ['loop14_pairs'] * 2}


def tiny_ft(longwm=True, updates=2):
    torch.set_num_threads(1)
    m = model()
    settings = TrainSettings(device='cpu', threads=1, longwm_enabled=longwm,
                             truncation_steps=60, state_noise_std_m=0., state_noise_std_mps=0.)
    runner = UnrollRunner(m, settings, build_loss(settings, physics=m.physics),
                          projection=None, device=torch.device('cpu'))
    optimizer = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=3e-5)
    frozen = {n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad}
    records = []
    for _ in range(updates):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        row = runner.run(batch(), unroll_steps=120, backward=True)
        optimizer.step()
        records.append({'seconds': time.perf_counter() - start, 'loss': row['loss'],
                        'longwm': row.get('longwm')})
    assert all(torch.equal(p, frozen[n]) for n, p in m.named_parameters() if n in frozen)
    return records


def speed_probe():
    from wms.data.window_cache import WindowCache
    timing = {}
    array = np.arange(601 * 32 * 7, dtype=np.float32).reshape(601, 32, 7)
    def decode():
        value = array.copy()
        for _ in range(30):
            value = np.maximum(value * .999, 0.)
        return {'states': value}
    for on in (False, True):
        cache = WindowCache()
        start = time.perf_counter()
        result = [cache.get(0, decode) if on else decode() for _ in range(16)]
        timing['decode_on' if on else 'decode_off'] = time.perf_counter()-start
        assert np.array_equal(result[0]['states'], result[-1]['states'])
    os.environ.pop('WMS_LOOPLONG_SPEED', None)
    return timing


def actual_ft_speed_probe(*, steps=120, truncation=120, courtesy=False, repeats=1):
    """Same full-path paired-return WM objective/optimizer/RNG, speed OFF/ON."""
    from unittest.mock import patch
    timings, states, logs, rngs = {}, {}, {}, {}
    torch.set_num_threads(1)
    initial = model(courtesy).state_dict()
    for on in (False, True):
        os.environ['WMS_LOOPLONG_SPEED'] = str(int(on))
        m = model(courtesy)
        m.load_state_dict(initial)
        settings = TrainSettings(device='cpu', threads=1, truncation_steps=truncation,
            state_noise_std_m=0., state_noise_std_mps=0.,
            paired_diff_weight=1., paired_diff_return_weight=1.,
            follower_courtesy_weight=9.067 if courtesy else 0.,
            follower_courtesy_contrast=courtesy, follower_courtesy_absolute_weight=4.525 if courtesy else 0.)
        runner = UnrollRunner(m, settings, build_loss(settings, physics=m.physics),
                              projection=None, device=torch.device('cpu'))
        optimizer = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=3e-5)
        calls = []
        original_rollout = m.rollout
        def replay(*args, **kwargs):
            calls.append(1)
            return original_rollout(*args, **kwargs)
        probe = batch(steps)
        # Include partnered ramp/mainline rows, so the courtesy objective and
        # its repeated recorded-state replay actually execute.
        if courtesy:
            probe['states'][..., 1] = 0.
            probe['states'][..., 2] = 1.
            probe['states'][:, :, 0, 2] = 0.
            probe['states'][:, :, 0, 3] = 1.
            probe['cav_teacher'][..., 1] = probe['states'][..., 1:4].argmax(-1).float()
        times = []
        for _ in range(repeats):
            optimizer.zero_grad(set_to_none=True)
            start = time.perf_counter()
            with patch.object(m, 'rollout', replay):
                log = runner.run(probe, unroll_steps=steps, backward=True)
            optimizer.step()
            times.append(time.perf_counter()-start)
        timings[str(on)] = {'seconds': float(np.median(times)), 'step_seconds': times,
                            'rollout_calls': len(calls)}
        states[on] = copy.deepcopy(m.state_dict())
        logs[on] = log['loss']
        rngs[on] = torch.get_rng_state()
    assert all(torch.equal(v, states[True][n]) for n, v in states[False].items())
    assert logs[False] == logs[True] and torch.equal(rngs[False], rngs[True])
    os.environ.pop('WMS_LOOPLONG_SPEED', None)
    return {**timings, 'objective_checkpoint_rng_byte_equal': True,
            'physics_steps': steps, 'truncation_steps': truncation,
            'batch_size': 2, 'courtesy': courtesy, 'optimizer_steps_per_arm': repeats}


def acceptance_probe():
    """Real acceptance rollout/safety kernels on two synthetic CPU windows."""
    import importlib.util
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('looplong_accept_probe', root / 'tools/loop14_checks/accept.py')
    acc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(acc)
    probe = batch()
    probe['states'][..., 1] = 0.
    probe['states'][..., 2] = 1.
    probe['cav_teacher'][..., 1] = 1.
    records = [{n: probe[n][i].numpy() for n in (
        'states', 'present', 'slot_index', 'is_cav', 'cav_teacher')} for i in range(2)]
    counts, timings, metrics = {}, {}, {}
    for on in (False, True):
        os.environ['WMS_LOOPLONG_SPEED'] = str(int(on))
        calls = []
        class Dataset:
            def __getitem__(self, i):
                calls.append(i)
                return copy.deepcopy(records[i])
        settings = TrainSettings(device='cpu', threads=1, state_noise_std_m=0., state_noise_std_mps=0.)
        arms = {name: (None, settings, model().eval()) for name in ('candidate', 'reference')}
        start = time.perf_counter()
        metrics[on] = acc.rollouts(arms, [Dataset()], [(0, 0), (0, 1)])
        timings[str(on)] = time.perf_counter() - start
        counts[str(on)] = len(calls)
    assert metrics[False] == metrics[True]
    os.environ.pop('WMS_LOOPLONG_SPEED', None)
    return {'seconds': timings, 'window_decodes': counts, 'metrics_identical': True,
            'scope': 'rollout/safety kernels only; not a production acceptance verdict'}


def memory_projection():
    # Rough total-memory scaling; static allocations and entrant growth are unknown.
    base = 36.512
    return {'reference_gib_60s_k2': base, 'gpu_marketed_gb': 48, 'allocator_cap_gib': 43.58,
        'linear_projection_gib_k8': {str(h): base * h / 60 * 8 / 2 for h in (120, 300)},
        'linear_projection_gib_k2': {str(h): base * h / 60 for h in (120, 300)},
        'assumption': 'same batch/rows/checkpointing; linear time and sample graphs; excludes entrant growth',
        'qualification': 'projection only; no GPU allocation or resource qualification'}


def policy_probe():
    """Three 90s toy-WM updates with eight distinct WM members, real trainer path."""
    import importlib.util
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('looplong_coop_fixture', root / 'tests/unit/test_coopw.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    from wms.control.coopw import ExplorationConfig
    trainer = helper.trainer_for(helper.hdv_speed_reward,
        models=[helper.CommandWM(gain=.01*(i+1)) for i in range(8)], horizon=90,
        cf={'n_group': 0, 'n_flip': 0}, low_penetration_horizon_cap=False)
    trainer.policy = helper.NoisePolicy()
    trainer.actor_parameters = list(trainer.policy.parameters()) + list(trainer.group_encoder.parameters())
    trainer.actor_optimizer = torch.optim.Adam(trainer.actor_parameters, lr=1e-3)
    trainer.exploration = ExplorationConfig(enabled=True, rho=[.5], long_persistence=True)
    trainer.looplong_logging = True
    trainer.long_credit_requested = True
    trainer.long_credit_trusted_s = 60
    trainer.long_credit = {'horizons': {'90': {'qualified': True, 'lower_scale': -.5,
                           'upper_scale': .5, 'correction_scale': 1.}}}
    output = []
    for _ in range(3):
        row = trainer.update()
        assert all(np.isfinite(row[k]) for k in ('actor_loss', 'critic_loss', 'actor_grad_norm'))
        output.append({k: row[k] for k in ('update', 'actor_loss', 'critic_loss',
            'imagination_horizon_s', 'exploration_persistence_s', 'exploration_correlated_active',
            'metering_like_holds_ge30s', 'metering_like_sampled_starts')})
    return output


def main():
    torch.set_num_threads(1)
    profile = cProfile.Profile()
    profile.enable()
    rows = tiny_ft()
    profile.disable()
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).sort_stats('cumtime').print_stats(15)
    result = {'kind': 'synthetic_cpu_engineering_only', 'wm_updates': rows,
              'policy_updates': policy_probe(),
              'acceptance_rollout_probe': acceptance_probe(),
              'speed_probe': speed_probe(), 'actual_ft_speed': actual_ft_speed_probe(),
              'tbptt_ft_speed': actual_ft_speed_probe(steps=240, truncation=120, courtesy=True, repeats=3),
              'memory': memory_projection(),
              'profile_top': stream.getvalue(), 'cuda_used': False}
    out = Path('tools/looplong/DRY_RUN.json')
    out.write_text(json.dumps(result, indent=1, allow_nan=False))
    print(json.dumps(result, indent=1, allow_nan=False))


if __name__ == '__main__':
    main()
