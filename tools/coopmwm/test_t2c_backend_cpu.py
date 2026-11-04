"""Production wiring tests: native V2V/controller/PPO, fake SUMO boundary only."""
from copy import deepcopy
import json
from pathlib import Path
import types

import pytest
import torch

from tools.coopmwm import rerun_cpus as rc, rerun_tracks_backend as prod
from tools.coopmwm.test_t2c_runtime_cpu import (Move, Trigger, Layer, Teacher, Credit, Augment,
                                  Weighting, install_moves, RealisedReward)
from tools.coopmwm.rerun_tracks_runtime import state_sha
from tools.coopmwm.v2v_intent import IntentPolicy
from tools.coopmwm.test_v2v_runtime_cpu import rows


class Features:
    def encode(self, obs):
        assert all(r['time_s'] < obs['time_s'] for r in obs['v2v_reports'])
        return dict(values=[2.], missing_share=float(not obs['v2v_reports']))


class ObservableCredit:
    def estimate(self, obs, move, base):
        return .2+move.depth*len(obs['cav_ids'])


@pytest.fixture
def production(tmp_path, monkeypatch):
    moves = install_moves(monkeypatch)
    moves.candidates = lambda obs, rng, n: [Move('none', tuple(obs['cav_ids'])),
                                          Move('gap_open', tuple(obs['cav_ids']), .2)]
    modules = {name: 'tools.coopmwm.'+name+':adapter' for name in prod.COMPONENTS}
    bindings = dict(component_factories=modules, move_to_plan='tools.coopmwm.moves:to_native',
        release=dict(stub=False, cpu_only=True, qualified=True, single_writer=True,
                     frozen_base=True, default_off_byte_identical=True))
    cfg = rc.runtime_config(dict(production=bindings, runtime='fake-runtime', checkpoint='fake-checkpoint',
        accel_bounds=[-3., 2.6], gpu_deadline=9999999999), root=str(tmp_path/'run'),
        reference_root=str(tmp_path/'refs'), source_root=str(tmp_path/'src'),
        common_checkpoint_sha256='a'*64, track2=True)
    rc.init_root(cfg['run_root'], track2=True)
    def factory(name):
        def build(*, base, components, **kw):
            return dict(merge_features=lambda: Features(), trigger_head=lambda: Trigger(),
                coop_layer=lambda: Layer(components['trigger_head']), group_credit=lambda: ObservableCredit(),
                dense_teacher=lambda: Teacher(), track2_reward=lambda: RealisedReward(), plan_augment=lambda: Augment(),
                practice_weighting=lambda: Weighting())[name]()
        return build
    def convert(move, *, observation, cfg):
        from tools.coopmwm.test_replay_bank_confirm_cpu import plan
        p = plan('native_'+move.family, move.depth)
        p['members'] = list(move.members)
        p.update(family=move.family, depth=move.depth)
        return p
    functions = {modules[name]: factory(name) for name in prod.COMPONENTS}
    functions[bindings['move_to_plan']] = convert
    monkeypatch.setattr(prod, 'entrypoint', functions.__getitem__)
    return cfg


def fake_episode(job, *, controller_hook=None, policy_loader=None):
    """Drive the real native controller at every native tick; replace SUMO only."""
    from tools.coopmwm.acceleration import Controller
    from tools.coopmwm.v2v_runtime import controller_class
    model = (policy_loader or prod.load_policy)(job['weights'])
    controller = controller_class(Controller)(model, torch.Generator().manual_seed(job['noise_seed']),
                                               job['deterministic'])
    prefixes, steps = {}, []
    Path(job['path']).mkdir(parents=True, exist_ok=True)
    if controller_hook:
        controller_hook(controller, job, prefixes)
    sensor = rows()
    commands = {r['vid']: .2 for r in sensor if r['is_cav']}
    for t in range(300, 1200):
        prefixes[str(t)] = dict(t_s=t)
        desired = controller.apply(sensor, commands, t)
        assert set(desired).issubset(commands)
        if controller_hook and job['arm'] != 'A':
            scoped = prod.observation(sensor, None, t, job['accel_bounds'])['cav_ids']
            assert all(desired[v]['requested_raw'] <= 1.2 for v in scoped if v in desired)
        steps.append(dict(controller.events[-1], reward=1.))
        if controller_hook and hasattr(controller_hook, 'after_step'):
            controller_hook.after_step(t, sensor)
        controller.observe_issued(dict(time_s=t, vehicles=[dict(r, issued_acceleration_mps2=.2) for r in sensor]))
    return dict(id=job['id'], world=job['world'], arm=job['arm'], steps=steps, prefixes=prefixes,
        totals=dict(team_delay_vs=900., collisions=0, teleports=0, hard_brake=0))


def test_native_worker_runs_full_window_components_and_A_native_ppo(production, tmp_path, monkeypatch):
    from tools.coopmwm import accel_plant
    from tools.coopmwm import exploration
    monkeypatch.setattr(exploration, 'regime', lambda rows: 'ordinary')  # external plant detector
    monkeypatch.setattr(accel_plant, 'episode', fake_episode)
    monkeypatch.setattr(prod, 'safety', lambda *a, **kw: dict(collisions=0, teleports=0, hard_brakes=0, ttc_events=0))
    cfg = production
    base = IntentPolicy().requires_grad_(False)
    before = state_sha(base)
    comp = rc.track2_components(cfg, 'B', adapters=prod.factories(cfg, base))
    comp['coop_layer'].correction.data.fill_(.05)
    weights = prod.component_weights(comp)
    job = dict(id='native_fake_B', path=str(tmp_path/'episode_B'), world=dict(seed=986000), role='train', arm='B',
        weights=base.state_dict(), runtime='fake', checkpoint='fake', noise_seed=17, update=1, episode_index=0,
        deterministic=False, accel_bounds=[-3., 2.6])
    out = prod.episode_worker(job, cfg, weights)
    assert len(out['training']['steps']) == 900  # native 300s warmup + 900 controlled seconds
    assert len(out['rows']) == 15*sum(r['is_cav'] for r in rows())
    assert all(r['reward_steps'] > 0 and r['realised_team_reward'] > 0 for r in out['rows'])
    assert all('snapshot_path' not in r for r in out['rows'])
    assert state_sha(base) == before
    assert any(r['family'] != 'none' for r in out['rows'])
    for row in out['rows']:
        assert set(row['move'].members) <= set(row['observation']['cav_ids'])
        assert all(r['time_s'] < row['observation']['time_s'] for r in row['observation']['v2v_reports'])
    # The production A path uses the native own-action PPO/intent objective, not
    # the numerical test backend's scalar synthetic optimizer.
    model = IntentPolicy()
    job.update(arm='A', id='native_fake_A', path=str(tmp_path/'episode_A'), weights=model.state_dict())
    a = prod.episode_worker(job, cfg, {})
    assert not a['rows'] and len(a['training']['steps']) == 900
    backend = prod.Backend(cfg, prep_dir=tmp_path, stage_dir=tmp_path)
    backend.models['A'] = model
    backend.optimizers['A'] = torch.optim.Adam(model.parameters(), lr=3e-4)
    backend.generators['A'] = torch.Generator().manual_seed(17)
    original = state_sha(model)
    metrics = backend.ppo_update(model, [a['training']])
    assert metrics and state_sha(model) != original
    backend.save_checkpoint(tmp_path/'A.pt', model, {})
    assert backend.checkpoint_components(tmp_path/'A.pt') == ['base', 'optimizer']
    backend.models['B'] = base
    backend.generators['B'] = torch.Generator().manual_seed(17)
    backend.save_checkpoint(tmp_path/'B.pt', base, comp)
    assert backend.checkpoint_components(tmp_path/'B.pt') == ['base', 'layer', 'trigger', 'optimizer']
    restored = prod.restore_components(cfg, 'B', base, backend.diagnostic_weights(base, comp))
    assert torch.equal(restored['coop_layer'].correction, comp['coop_layer'].correction)


def test_local_observation_excludes_remote_truth_and_current_partner_reports():
    sensor = rows()
    sensor[0]['private_driver_kind'] = 'never forward'
    far = dict(next(r for r in sensor if r['is_cav']), vid='remote', s_m=1000000.)
    sensor.append(far)
    previous = dict(time_s=399, vehicles=[dict(r, issued_acceleration_mps2=.4) for r in sensor])
    obs = prod.observation(sensor, previous, 400, [-3., 2.6])
    assert 'remote' not in obs['cav_ids']
    assert 'private_driver_kind' not in json.dumps(obs)
    assert all(r['time_s'] == 399 for r in obs['v2v_reports'])
    assert not prod.observation(sensor, dict(previous, time_s=400), 400, [-3., 2.6])['v2v_reports']


def test_native_job_construction_exact_reserved_id_and_launch3_before_dispatch(production, tmp_path, monkeypatch):
    from tools.coopmwm import v2v_launch3
    cfg = production
    cfg['production'].update(launch_binding_path='fake-binding', launch_binding_sha256='b'*64,
                             control_root='fake-control', mover_root='fake-mover')
    backend = prod.Backend(cfg, prep_dir=tmp_path, stage_dir=tmp_path)
    rc.charge(Path(cfg['run_root']), [dict(id='t2c_u001_A_episode_000', sim_s=1200)], 'v2v_train',
              budget=cfg['budget'], track2=True)
    calls = []
    monkeypatch.setattr(v2v_launch3, 'before_launch', lambda *a, **kw: calls.append(('launch3', kw)))
    def dispatch(fn, job, got_cfg, weights):
        assert calls[-1][0] == 'launch3'
        assert fn is prod.episode_worker and job['id'] == 't2c_u001_A_episode_000'
        assert job['v2v'] and job['sim_s'] == 1200 and job['runtime'] == 'fake-runtime'
        calls.append(('dispatch', job))
        return dict(native=True)
    monkeypatch.setattr(backend, '_dispatch', dispatch)
    assert backend.run_episode(arm='A', update=1, index=0, start=None, model=IntentPolicy(),
                               components={}, cfg=cfg) == dict(native=True)
    assert calls[0][1]['phase_costs'] == {'v2v_train': 0}
    with pytest.raises(ValueError, match='reservation missing'):
        backend._gate_reserved('unfunded', 1200, 'v2v_train')
    assert len(calls) == 2


def test_track2_launch3_uses_new_own_budget_without_legacy_superseded_charge(production):
    from tools.coopmwm import v2v_launch3
    cfg = production
    root = Path(cfg['run_root'])
    assert v2v_launch3.bind_runtime(dict(config=cfg), root=root, source_root=cfg['source_root']) == cfg
    out = v2v_launch3.requote(root, cfg)
    assert out['charged_to_date_sim_s'] == 0 and out['quote_total_sim_s'] == rc.quote(track2=True)['total']
    assert out['ceiling_sim_s'] <= 5000000
    rc.charge(root, [dict(id='closed', sim_s=1200)], 'v2v_train', budget=cfg['budget'], track2=True)
    assert v2v_launch3.requote(root, cfg, phase_costs={'v2v_train': 0})['charged_to_date_sim_s'] == 1200


def test_verification_native_specs_preserve_full_window_and_grade_targets(production, tmp_path):
    from tools.coopmwm.test_replay_bank_confirm_cpu import snapshot, plan
    cfg = production
    backend = prod.Backend(cfg, prep_dir=tmp_path, stage_dir=tmp_path)
    snap = snapshot(600)
    path = tmp_path/'state.pt'
    torch.save(snap, path)
    proposals = [Move('none'), Move('gap_open', depth=.1)]
    payload = dict(id='fixed_sample', arm='B', snapshot_path=str(path), snapshot_file_sha256=rc.sha(path),
        window=dict(start_s=600, end_s=1200), native_plans=[plan('p', .1)], target=proposals[1], proposals=proposals)
    pairs = backend.expand_verification('teacher', payload, cfg=cfg)
    assert all(s['sim_s'] == 1200 and s['window'] == payload['window'] for s, _ in pairs)
    assert sum(s['sim_s'] for s, _ in pairs) <= rc.GATE_SIM_S
    result = backend.grade_verification('teacher', payload,
        [dict(status='graded', J_veh_s=10.), dict(status='graded', J_veh_s=5.)])
    assert result['label_correct'] and result['advantage'] == 5.
    payload['snapshot_file_sha256'] = '0'*64
    with pytest.raises(ValueError, match='snapshot bytes changed'):
        backend.expand_verification('teacher', payload, cfg=cfg)


def test_real_backend_loads_native_checkpoint_and_complete_eval_through_plant(production, tmp_path, monkeypatch):
    from tools.coopmwm import accel_plant
    from tools.coopmwm.accel_learner import save
    from tools.coopmwm.v2v_runtime import load_checkpoint
    from tools.coopmwm.acceleration import Policy
    cfg = production
    legacy = Policy()
    path = tmp_path/'common.pt'
    save(path, legacy, torch.optim.Adam(legacy.parameters(), lr=3e-4),
         torch.Generator().manual_seed(7), 0, 0)
    backend = prod.Backend(cfg, prep_dir=tmp_path, stage_dir=tmp_path)
    backend.hooks = dict(load_checkpoint=load_checkpoint)
    loaded = backend.load_base(path, arm='A', cfg=cfg)
    assert isinstance(loaded, IntentPolicy) and next(loaded.parameters()).device.type == 'cpu'
    comp = rc.track2_components(cfg, 'B', adapters=backend.adapters(loaded))
    comp['coop_layer'].correction.data.fill_(.05)
    backend.models['B'] = loaded
    backend.generators['B'] = torch.Generator().manual_seed(7)
    checkpoint = tmp_path/'complete.pt'
    backend.save_checkpoint(checkpoint, loaded, comp)
    seen = []
    def plant(job, *, controller_hook=None):
        assert isinstance(controller_hook, prod.EpisodeHook)
        assert isinstance(controller_hook.selected, prod.AllMembers)
        assert controller_hook.components['coop_layer'].correction.item() == pytest.approx(.05)
        seen.append(job)
        return dict(totals=dict(team_delay_vs=42.))
    monkeypatch.setattr(accel_plant, 'episode', plant)
    monkeypatch.setattr(prod, 'safety', lambda *a, **kw: dict(collisions=0, teleports=0, hard_brakes=0, ttc_events=0))
    spec = dict(branch='shared_final', arm='B', world_id='w0', world=dict(seed=986000),
                checkpoint=dict(path=str(checkpoint), file_sha256=rc.sha(checkpoint)),
                sim_s=1200, window=dict(start_s=0, end_s=1200), final_worlds_sha256='f'*64)
    result = prod.branch_worker(spec, {}, cfg, str(tmp_path/'evaluation'))
    assert result['J_veh_s'] == 42. and result['checkpoint'] == spec['checkpoint']
    assert result['world'] == spec['world'] and result['final_worlds_sha256'] == 'f'*64
    assert seen[0]['v2v'] and seen[0]['sim_s'] == 1200 and seen[0]['deterministic']


def test_bank_native_callback_preserves_source_seed_weight_and_effective_targets(production, tmp_path, monkeypatch):
    from tools.coopmwm import replay_bank as rb, bank_v2v, practice_starts, v2v_intent
    from tools.coopmwm.test_replay_bank_confirm_cpu import snapshot, plan
    cfg = production
    backend = prod.Backend(cfg, prep_dir=tmp_path, stage_dir=tmp_path)
    model = IntentPolicy()
    backend.models['B'] = model
    backend.components['B'] = dict(merge_features=Features())
    sensor = rows()
    members = prod.observation(sensor, None, 600, [-3., 2.6])['cav_ids']
    p = plan('own_plan', .1)
    p.update(family='gap_open', depth=.1)
    p['members'] = members
    context = dict(members=members, x=[[0.]*29 for _ in members], rsu=[0.]*12,
                   v2v={v: [0.]*48 for v in members})
    sample = dict(context=context, action=[[.8] for _ in members],
        binding_limits={v: .3 for v in members}, behaviour=dict(executed=True, proposal_members=members),
        intent_link=dict(t=600))
    receipt = dict(verification_id='vid', window=dict(start_s=600, end_s=900), receipt_sha256='a'*64)
    example = dict(example_id='vid#s1', seed_index=1)
    snap = snapshot(600)
    monkeypatch.setattr(practice_starts, '_links', lambda *a: dict(vid=dict(state_id='own_state')))
    monkeypatch.setattr(rb, '_decision_snapshot', lambda *a: deepcopy(snap))
    monkeypatch.setattr(bank_v2v, 'trace_frames', lambda *a: {
        t: dict(time_s=t, vehicles=sensor) for t in (599, 600)})
    monkeypatch.setattr(v2v_intent, 'refresh_bank_intents', lambda *a: None)
    def native_pairs(root, arm, seed, *, v2v, example_consumer):
        assert v2v and arm == 'B'
        example_consumer(receipt=receipt, example=example, plan=p, pairs=[(sample, 3.25)], trace='fake')
    backend.hooks = dict(bank_pairs=native_pairs)
    examples = backend.bank_examples('B', 0, cfg=cfg)
    assert len(examples) == 1 and examples[0]['weight'] == 3.25
    assert examples[0]['seed'] == snap['episode']['world']['seed']
    assert examples[0]['effective_sample']['native_sample'] == sample
    assert examples[0]['effective_sample']['binding_limits'] == sample['binding_limits']
    from tools.coopmwm.rerun_tracks_bank import training_batch
    batch = training_batch(examples, [], arm='B', encoder=Features())
    assert batch[0]['weight'] == 3.25 and batch[0]['effective_sample']['action'] == [.8]*len(members)
    assert examples[0]['practice_row']['state_id'] == 'own_state'


def test_native_replay_branch_uses_reviewed_runner_and_mixed_policy(production, tmp_path, monkeypatch):
    from tools.coopmwm import sumo_verify as sv, replay_bank as rb
    from tools.coopmwm.policy_synergy import MixedPolicy
    from tools.coopmwm.test_replay_bank_confirm_cpu import snapshot, plan
    cfg = production
    model = IntentPolicy()
    components = rc.track2_components(cfg, 'B', adapters=prod.factories(cfg, model))
    weights = dict(model.state_dict(), **prod.component_weights(components))
    snap = snapshot(600)
    snap.update(weights=weights, u3_weights=model.state_dict(),
                policy_synergy=dict(arm='B', group=['c1', 'c2'], new_members=['c1'], background_old=True))
    native = sv.LocalBranchRunner
    calls = []
    class Runner(native):
        def run(self, spec, snapshot):
            controller = types.SimpleNamespace(policy=model, base_policy=model, apply=lambda *a: {})
            self.controller_hook(controller, {}, {})
            assert isinstance(controller.base_policy, MixedPolicy)
            assert controller.base_policy.new_members_only == {'c1'}
            assert callable(controller.apply)
    monkeypatch.setattr(sv, 'LocalBranchRunner', Runner)
    # Exercise the actual hook without running its per-tick plant.
    monkeypatch.setattr(prod.EpisodeHook, '__call__',
        lambda self, c, j, p: setattr(c, 'apply', lambda *a: {}))
    def run_branch(spec, snapshot, runner):
        assert isinstance(runner, native)
        runner.run(spec, snapshot)
        calls.append(spec)
        return dict(status='graded', J_veh_s=1., window=dict(collisions=0, teleports=0, hard_brake=0))
    monkeypatch.setattr(sv, 'run_branch', run_branch)
    monkeypatch.setattr(rb, 'ttc_events', lambda *a: 0)
    spec = dict(branch='solo0', verification_id='native_replay', window=dict(start_s=600, end_s=1200))
    result = prod.branch_worker(spec, snap, cfg, str(tmp_path/'branch'))
    assert calls == [spec] and result['safety']['ttc_events'] == 0


def test_native_all8_move_is_relative_to_each_branch_base_not_static_profile(production, monkeypatch):
    from tools.coopmwm import exploration
    from tools.coopmwm.acceleration import Controller
    from tools.coopmwm.v2v_runtime import controller_class
    monkeypatch.setattr(exploration, 'regime', lambda rows: 'ordinary')
    sensor = rows()
    members = prod.observation(sensor, None, 300, [-3., 2.6])['cav_ids']
    move = Move('gap_open', tuple(members), .2)
    plan = prod.native_move_plan(move, {}, production)
    model = IntentPolicy()
    c = controller_class(Controller)(model, torch.Generator().manual_seed(3), True)
    c.active_plans = [dict(plan=plan, start_s=300)]
    prod.relative_plan_hook(c, dict(accel_bounds=[-3., 2.6]), {})
    commands = {r['vid']: .2 for r in sensor if r['is_cav']}
    first = c.apply(sensor, commands, 300)
    c.observe_issued(dict(time_s=300, vehicles=sensor))
    commands = {v: .7 for v in commands}
    second = c.apply(sensor, commands, 301)
    for vid in members:
        assert first[vid]['requested_raw'] == pytest.approx(.4)
        assert second[vid]['requested_raw'] == pytest.approx(.9)
