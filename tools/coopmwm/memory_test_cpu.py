"""Recovery regressions on real warm-runner control flow; physics is forbidden."""
import copy
import gc
import json
from pathlib import Path
import tempfile
import weakref
from unittest.mock import patch

import torch

from . import replay_start as rs, warm_runner as wr


def put(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v))


def test_distinct_resume_and_precharge_bindings():
    with tempfile.TemporaryDirectory() as d:
        r = Path(d)
        put(r / 'EXACT_OPERATIONAL_BINDING.json', {})
        put(r / 'WARM_B_RUNTIME_BINDING_V4.json', {'operational': {}})
        put(r / 'AWR_BINDING.json', {'config': {'coefficient': 1}})
        v = dict(budget={}, gpu_deadline=1, collection_root=d, demand_phase_end_s=1200,
                 resume_updates={'B': 18, 'R': 17},
                 precharged_unattempted={'B': ['B19'], 'R': ['R18']})
        with patch('tools.coopmwm.resource_config.clocks', side_effect=lambda r, b: b):
            b = rs.binding(r, v, 'B'); c = rs.binding(r, v, 'R')
        assert b['resume_update'] == 18 and c['resume_update'] == 17
        assert b['precharged_unattempted'] == ['B19']
        assert c['precharged_unattempted'] == ['R18']
        assert b['stop_after_update'] == c['stop_after_update'] == 24
        assert b['imitation_epochs'] == c['imitation_epochs'] == 5


def test_previous_cyclic_rollout_is_dead_before_next_wave():
    class Episode(dict):
        pass
    seen = []
    def wave(*args):
        if seen:
            assert seen[-1]() is None, 'previous rollout still live at admission boundary'
        ep = Episode(id='dummy', steps=[dict(time_s=700, sample={})])
        ep['cycle'] = ep
        seen.append(weakref.ref(ep))
        return [ep]
    with tempfile.TemporaryDirectory() as d:
        r = Path(d)
        policy = torch.nn.Linear(1, 1)
        opt = torch.optim.Adam(policy.parameters())
        b = dict(resume_update=17, stop_after_update=19, runtime_receipt_suffix='_MEMORY_TEST',
                 b_replay=True, accel_bounds={}, awr={'coefficient': 1})
        weights = {'counts': {'positive': 0}}
        with patch.object(wr, 'verify', return_value=b), \
             patch.object(wr, 'budget', return_value={'updates': {'B': 19}}), \
             patch.object(wr, 'load', return_value=(policy, opt, torch.Generator(), {'update': 17})), \
             patch.object(wr, 'assert_running'), patch.object(wr, 'charged', return_value=0), \
             patch.object(wr, 'wave', side_effect=wave), \
             patch.object(wr, 'routing', new=lambda *a: {'sampling': [], 'durations': [], 'executed': 0}), \
             patch.object(wr, 'evidence', new=lambda *a: []), \
             patch.object(wr, 'viability', return_value={}), \
             patch.object(wr, 'advantages', return_value=[(1, 2), (2, 3)]), \
             patch.object(wr, 'save'), \
             patch.object(wr.search_learning, 'empty', return_value={}), \
             patch.object(wr.search_learning, 'exact_update', return_value={}), \
             patch('tools.coopmwm.replay_bank.bank_pairs', side_effect=lambda *a, **k: ([], copy.deepcopy(weights))), \
             patch('tools.coopmwm.replay_bank.bank_quality', return_value={}), \
             patch('tools.coopmwm.replay_bank.record_lesson', return_value={}), \
             patch('tools.coopmwm.exact_grade.exact_update', new=lambda *a, **k: []):
            wr.run_arm(r, 'B')
        assert len(seen) == 2
        assert json.loads((r / 'B_UPDATE_019.json').read_text())['update'] == 19
        gc.collect()


def test_adopt_precharge_does_not_recharge_and_claim_is_terminal():
    from . import accel_runner as ar
    class AdmissionReached(Exception):
        pass
    with tempfile.TemporaryDirectory() as d:
        r = Path(d)
        key = 'replay0735_warm_u019_B_986000'
        job = r / 'jobs' / key
        job.mkdir(parents=True)
        b = dict(precharged_unattempted=[key], runtime={}, checkpoint='', accel_bounds={},
                 gpu_deadline=1, workers=1)
        policy = torch.nn.Linear(1, 1)
        with patch.object(ar, 'ledger', return_value=[{'id': key}]), \
             patch.object(ar, 'reserve') as reserve, \
             patch('tools.coopmwm.policy_wm.actor_state', return_value={}), \
             patch('tools.coopmwm.runner.admission', side_effect=AdmissionReached):
            try:
                ar.wave(r, b, {'B': policy}, [{'seed': 986000}], 'train',
                        'replay0735_warm_u019', 'train')
            except AdmissionReached:
                pass
            else:
                raise AssertionError('admission not reached')
            reserve.assert_not_called()
            put(job / 'CLAIM.json', {'pid': 1})
            try:
                ar.wave(r, b, {'B': policy}, [{'seed': 986000}], 'train',
                        'replay0735_warm_u019', 'train')
            except ValueError as e:
                assert 'attempted job cannot be retried' in str(e)
            else:
                raise AssertionError('attempted precharge was retried')
            reserve.assert_not_called()


def test_checkpoint_loader_restores_parameters_adam_and_generator_exactly():
    from .accel_learner import Policy, save, load
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'memory_checkpoint.pt'
        policy = Policy()
        opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
        for parameter in policy.parameters():
            parameter.grad = torch.ones_like(parameter)
        opt.step()
        rng = torch.Generator().manual_seed(20261101)
        torch.rand(37, generator=rng)
        save(p, policy, opt, rng, 18, 0)
        other, adam, generator, payload = load(p)
        assert payload['update'] == 18
        for k, v in policy.state_dict().items():
            assert torch.equal(v, other.state_dict()[k])
        for k, v in opt.state_dict()['state'].items():
            for field, value in v.items():
                assert torch.equal(value, adam.state_dict()['state'][k][field])
        assert opt.state_dict()['param_groups'] == adam.state_dict()['param_groups']
        assert torch.equal(rng.get_state(), generator.get_state())
