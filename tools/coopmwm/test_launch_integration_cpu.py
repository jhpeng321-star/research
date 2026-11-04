"""Cross-branch contracts exercised by the clean1 production launch."""
import pytest

from . import practice_weighting as pw, rerun_cpus as rc
from . import track2_reward as reward
from .test_track2_backend_cpu import cfg as backend_cfg, cpu_threads
from .test_bank_perseed_cpu import ps as per_seed_bank


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


@pytest.mark.parametrize('update', [8, 24])
def test_native_shared_planner_eval_complete_grid_and_paired_bootstrap(update):
    from copy import deepcopy
    from . import planner_eval, protocol
    worlds = protocol.worlds('train', 0)+protocol.worlds('train', 1)[:24]
    registration = rc.final_world_registration(dict(final_evaluation=dict(role='train',
        worlds=[dict(world_id=str(w['seed']), world=w) for w in worlds])))
    contract = deepcopy(rc.FINAL_EVALUATION)
    if update == 8:
        contract.update(update=8, phase='v2v_interim')
    checkpoints = {a: dict(path=f'/home/ubuntu/policies/{a}_{update}.pt', file_sha256='a'*64,
        update=update, components=contract['checkpoint_components'][a]) for a in ('A', 'B', 'R')}
    pairs = planner_eval.prepare_track2(registration=registration, checkpoints=checkpoints, contract=contract)
    assert len(pairs) == 288 and sum(s['sim_s'] for s, _ in pairs) == 345600
    records = [dict(s, status='graded', J_veh_s={'A':100., 'B':90., 'R':95.}[s['arm']],
        safety=dict.fromkeys(contract['safety_counters'], 0)) for s, _ in pairs]
    report = planner_eval.analyse_track2(registration=registration, checkpoints=checkpoints,
        records=records, contract=contract)
    assert report['paired_ci']['B-A']['mean'] == -10.
    assert report['paired_ci']['B-A']['n'] == 96
    assert report['paired_ci']['B-A']['low'] == report['paired_ci']['B-A']['high'] == -10.
    with pytest.raises(ValueError, match='exactly once'):
        planner_eval.analyse_track2(registration=registration, checkpoints=checkpoints,
            records=records[:-1], contract=contract)
    records[0]['window'] = dict(start_s=0, end_s=60)
    with pytest.raises(ValueError, match='binding'):
        planner_eval.analyse_track2(registration=registration, checkpoints=checkpoints,
            records=records, contract=contract)


def test_training_release_refuses_mismatched_or_unsafe_final_gate(tmp_path):
    from . import launch_training as training
    from .exact_grade import _seal
    from .protocol import digest
    cfg = dict(run_root=str(tmp_path), source_manifest_sha256='a'*64)
    training.write(tmp_path/'SMOKE_PASS.json', _seal(dict(status='SMOKE_PASS',
        config_sha256=digest(cfg)), 'receipt_sha256'))
    gate = _seal(dict(status='PASS', config_sha256=digest(cfg),
        safety=dict(collisions=0, teleports=0)), 'receipt_sha256')
    training.write(tmp_path/'GATE_PASS.json', gate)
    training.release(cfg)
    with pytest.raises(ValueError, match='binding'):
        training.release(dict(cfg, source_manifest_sha256='b'*64))
    (tmp_path/'GATE_PASS.json').write_text(__import__('json').dumps(_seal(dict(
        gate, safety=dict(collisions=1, teleports=0)), 'receipt_sha256')))
    with pytest.raises(ValueError, match='zero collisions'):
        training.release(cfg)


def test_native_all8_seed_renumbering_requires_real_parent_snapshot_mapping(tmp_path, monkeypatch, per_seed_bank):
    from tools.coopmwm import practice_starts as ps, sumo_verify as sv, practice_weighting as actual_pw
    from .test_practice_weighting_cpu import augment
    from .test_practice_starts_cpu import bank_with_snapshots
    from .exact_grade import _seal
    import json
    bank_with_snapshots(tmp_path, vids=('v0',))
    row, manifest = augment(tmp_path, 'B')
    bank = tmp_path/'replay_bank/B'
    row['parent_example_id'] = 'v0#s2'
    source = __import__('torch').load(bank/'snapshots'/f'{row["state_id"]}.pt', weights_only=False)
    sha = sv.snapshot_sha(source)
    row['source_snapshot_sha256'] = sha
    link_path = bank/'decisions'/(row['state_id']+'.json')
    link = json.loads(link_path.read_text())
    link['augmentation'] = dict(parent_example_id='v0#s2', source_snapshot_sha256=sha)
    link_path.write_text(json.dumps(link))
    original = ps.load_snapshot
    def load(root, arm, item):
        return source if item['example_id'] == 'v0#s2' else original(root, arm, item)
    monkeypatch.setattr(ps, 'load_snapshot', load)
    manifest = _seal(dict(manifest, rows=[row]), 'practice_augmented_sha256')
    (bank/'PRACTICE_AUGMENTED.json').write_text(json.dumps(manifest))
    parents = ps.examples(tmp_path, 'B')
    rows, _ = actual_pw.augmented_examples(tmp_path, 'B', parents)
    assert len(rows) == 1
    link['augmentation']['source_snapshot_sha256'] = '0'*64
    link_path.write_text(json.dumps(link))
    with pytest.raises(ValueError, match='invalid own-bank'):
        actual_pw.augmented_examples(tmp_path, 'B', parents)


def test_real_post_sensing_departure_is_explicit_terminal_occupancy_not_missing_frame(backend_cfg, monkeypatch):
    from . import track2_backend as p1
    from .test_track2_backend_cpu import sample, rows, fake_load
    cfg = backend_cfg
    policy, recorded, controller = sample(cfg)
    event = controller.events[-1]
    post = [r for r in rows() if r['vid'] != 'c1']
    p1.record_outcomes(policy, event, post)
    assert event['reward_observations']['c1']['terminal_exit']['present_after'] is False
    episode = dict(id='exit', arm='B', actual_sim_s=301, steps=[dict(event, reward=-10.)])
    envelope = reward.reward_update(cfg, 'B', 1, [episode])
    assert envelope['receipt']['terminal_exits'] == 1
    assert envelope['receipt']['terminal_fallback_steps'] == 0
    assert next(r for r in envelope['rows'] if r['cav_id'] == 'c1')['reward'] == 0.
    fake_load(monkeypatch)
    runtime = p1.build_arm_runtime(cfg, 'B')['runtime']
    key = ('exit', 300)
    targets = runtime._reward_targets(envelope, {key:recorded}, {key})
    assert targets[key][recorded['context']['members'].index('c1')] == 0.
    event['reward_observations']['c1']['terminal_exit']['present_after'] = True
    with pytest.raises(ValueError, match='terminal SUMO'):
        reward.reward_update(cfg, 'B', 1, [episode])


def test_production_entrypoint_imports_do_not_construct_or_dispatch_workers(monkeypatch):
    import concurrent.futures
    def refuse(*args, **kwargs):
        pytest.fail('preparation/import must not start a worker')
    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', refuse)
    from . import launch_qualification, launch_preparation, launch_augmentation, launch_evaluation
    assert callable(launch_qualification.gate_pairs)
    assert callable(launch_preparation.prepare)
    assert callable(launch_augmentation.run)
    assert callable(launch_evaluation.scheduled)
