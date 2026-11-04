"""Cross-branch contracts exercised by the clean1 production launch."""
import pytest

from . import practice_weighting as pw, rerun_cpus as rc
from . import track2_reward as reward
from .test_track2_backend_cpu import cfg as backend_cfg, cpu_threads


def test_structured_practice_registration_is_explicit_track2_only():
    block = dict(rc.TRACK_SLOTS['practice_weighting'])
    cfg = dict(track2=True, practice_starts=True, practice_weighting=block)
    assert pw.effective_fraction(cfg) == .5
    assert pw.options(cfg) == dict(practice_weighting=True, practice_weighted_fraction=.5)
    with pytest.raises(ValueError, match='bool'):
        pw.options(dict(cfg, track2=False))
    with pytest.raises(ValueError, match='registration'):
        pw.options(dict(cfg, practice_weighting=dict(block, fraction=.75)))


def test_p1_p2_post_action_envelope_and_json_packet_roundtrip():
    import json
    from dataclasses import asdict
    from .merge_features import SensorReport, ObservedVehicle
    own = SensorReport(301., ObservedVehicle('c1', 0., 1, 10., 5.), (0, 1, 2),
        (ObservedVehicle('h1', 20., 1, 5., 5.),))
    cfg = dict(track2_reward=dict(weight=1., provenance='registered-test'))
    sample = dict(context=dict(members=['c1']))
    episode = dict(id='e1', arm='B', steps=[dict(time_s=300, reward=-999., sample=sample,
        reward_observations={'c1':dict(own=asdict(own), partner_reports={}, partner_ids=[])})])
    envelope = reward.reward_update(cfg, 'B', 1, json.loads(json.dumps([episode])))
    assert envelope['schema'] == 'coopmwm-track2-reward-v1'
    row, = envelope['rows']
    assert row['episode_id'] == 'e1' and row['time_s'] == 300
    assert row['source'] == 'sumo'
    assert row['own_reward'] == pytest.approx(-(1-10/31.29))
    assert row['neighbour_rewards'] == {'h1':pytest.approx(-(1-5/31.29))}
    assert row['reward'] == pytest.approx(row['own_reward']+row['neighbour_rewards']['h1'])
    assert envelope['receipt']['global_reward_used'] is False
    assert episode['steps'][0]['reward'] == -999.
    del episode['steps'][0]['reward_observations']
    with pytest.raises(ValueError, match='terminal post-action'):
        reward.reward_update(cfg, 'B', 1, [episode])


def test_production_config_keeps_canonical_registration_and_prior_budget():
    from copy import deepcopy
    from . import launch_production as launch, track2_backend as p1, track2_monitor as p3
    cfg = rc.runtime_config({}, root='/home/ubuntu/test_clean1',
        reference_root='/home/ubuntu/reference', source_root='/home/ubuntu/source',
        common_checkpoint_sha256='a'*64, track2=True)
    cfg.update(launch_optimization=deepcopy(launch.OPTIMIZATION),
               worlds_by_update={'1':[dict(seed=986000), dict(seed=986001)]})
    before = deepcopy(cfg)
    native = launch.backend_config(cfg)
    for arm in ('A', 'B', 'R'):
        p1._check(native, arm)
    assert native['dense_teacher'] is native['group_credit'] is False
    assert native['budget']['total'] == 5000000
    assert native['prior_charged_sim_s'] == 960
    assert sum(native['budget']['phases'].values())+960 <= 5000000
    assert cfg == before and rc.check_pins(cfg)
    assert p3._components(launch.monitor_config(cfg))['track2_reward'] is True
    assert launch.backend_config(cfg, smoke=True)['phase_map']['train'] == 'track2_smoke'
    cfg['launch_optimization']['trigger_lr'] *= 2
    with pytest.raises(ValueError, match='optimization'):
        launch.backend_config(cfg)


def test_isolated_loader_reconstructs_p1_layer_trigger_not_base_only(tmp_path):
    from copy import deepcopy
    import torch
    from . import launch_production as launch, track2_backend as p1
    from .test_track2_backend_cpu import IntentPolicy
    from .accel_learner import save
    cfg = dict(run_root=str(tmp_path), v2v=True, v2v_dim=48,
               training_seed=20261101, **deepcopy(launch.OPTIMIZATION))
    base = IntentPolicy()
    save(tmp_path/'B_checkpoint_001.pt', base, torch.optim.Adam(base.parameters(), lr=.0003),
         torch.Generator().manual_seed(11), 1, 960)
    policy = p1.EpisodePolicy(base, cfg, 'B')
    with torch.no_grad():
        policy.layer.trigger_gain.fill_(.123)
        policy.trigger.trigger_head[-1].bias.fill_(.456)
    torch.save(dict(arm='B', update=1, worker=policy.worker_state()), tmp_path/'B_track2_001.pt')
    packed = launch.checkpoint_weights(cfg, 'B', 1)
    restored = launch.load_policy(packed, cfg)
    assert isinstance(restored, p1.EpisodePolicy)
    assert restored.layer.trigger_gain.item() == pytest.approx(.123)
    assert restored.trigger.trigger_head[-1].bias.item() == pytest.approx(.456)
    assert all(torch.equal(v, restored.layer.base.state_dict()[k]) for k,v in base.state_dict().items())
    torch.save(dict(arm='R', update=1, worker=policy.worker_state()), tmp_path/'B_track2_001.pt')
    with pytest.raises(ValueError, match='arm/update'):
        launch.checkpoint_weights(cfg, 'B', 1)


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_verified_bank_move_ce_actually_updates_trigger_with_frozen_base(backend_cfg, monkeypatch, arm):
    from contextlib import contextmanager
    from copy import deepcopy
    from types import SimpleNamespace
    import torch
    from . import launch_production as launch, track2_backend as p1, bank_v2v
    from .test_track2_backend_cpu import sample, rows
    cfg = backend_cfg
    policy, recorded, controller = sample(cfg)
    plan = dict(members=['c1', 'c2'], duration_s=30,
        perturbation_profile=[[-.3]*30, [-.3]*30], profile=[[-.3]*30, [-.3]*30])
    before = deepcopy(policy.layer.base.state_dict())
    head_before = deepcopy(policy.trigger.trigger_head.state_dict())
    pair = (dict(recorded, intent_link=dict(t=300)), 1.)
    def pairs(*args, example_consumer, **kwargs):
        assert kwargs['imitation_target_space'] == 'effective'
        assert kwargs['imitation_bank_filter'] == 'cooperative_only'
        example_consumer(receipt={}, example=dict(admitted=True, cooperative=True),
            plan=plan, pairs=[pair], trace='fixture')
        return [pair], {}
    @contextmanager
    def bank_view(config):
        yield SimpleNamespace(bank_pairs=pairs)
    monkeypatch.setattr(p1, '_bank_view', bank_view)
    monkeypatch.setattr(bank_v2v, 'trace_frames', lambda path:
        {299:dict(vehicles=rows()), 300:dict(vehicles=rows())})
    runtime = SimpleNamespace(cfg=cfg, policy=policy, arm=arm)
    report = launch.bank_move_supervision(runtime)
    assert report['examples'] == 1 and report['final_ce'] < report['initial_ce']
    assert all(torch.equal(v, policy.layer.base.state_dict()[k]) for k,v in before.items())
    assert any(not torch.equal(v, policy.trigger.trigger_head.state_dict()[k]) for k,v in head_before.items())
    assert all(p.grad is None for p in policy.layer.base.parameters())
