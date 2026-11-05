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


def test_raw_mappo_u0_diagnostic_matches_actual_intent_episode_loader(tmp_path):
    from copy import deepcopy
    import torch
    from . import launch_production as launch, track2_backend as p1, v2v_intent
    from .acceleration import Policy
    from .accel_learner import save
    cfg=dict(run_root=str(tmp_path),training_seed=20261101,
        **v2v_intent.config_keys(),**deepcopy(launch.OPTIMIZATION))
    base=Policy()
    save(tmp_path/'B_checkpoint_000.pt',base,torch.optim.Adam(base.parameters(),lr=.0003),
        torch.Generator().manual_seed(11),0,0)
    native,_,_,_=p1._load_base(cfg,'B',0)
    policy=p1.EpisodePolicy(native,cfg,'B')
    torch.save(dict(arm='B',update=0,worker=policy.worker_state()),tmp_path/'B_track2_000.pt')
    packed=launch.checkpoint_weights(cfg,'B',0)
    restored=launch.load_policy(packed,cfg)
    assert restored.layer.base.v2v_width == 48
    for name,value in native.state_dict().items():
        assert torch.equal(restored.layer.base.state_dict()[name],value)
    for name,value in policy.trigger.state_dict().items():
        assert torch.equal(restored.trigger.state_dict()[name],value)


def test_gate_environment_rerun_needs_bound_void_and_distinct_charge_ids(tmp_path):
    from .launch_qualification import gate_job_manifest
    from .launch_training import write
    from .exact_grade import _seal
    from .protocol import digest
    cfg=dict(run_root=str(tmp_path))
    pairs=[(dict(verification_id='same_full_window',sim_s=720),None)]
    first=gate_job_manifest(cfg,pairs)
    cfg['gate_attempt']=2
    with pytest.raises(FileNotFoundError):gate_job_manifest(cfg,pairs)
    write(tmp_path/'GATE_VOID.json',_seal(dict(reason_class='environment_failure',
        new_config_sha256=digest(cfg)),'receipt_sha256'))
    second=gate_job_manifest(cfg,pairs)
    assert first[0]['id']!=second[0]['id'] and first[0]['sim_s']==second[0]['sim_s']==720
    with pytest.raises(ValueError,match='bind'):
        gate_job_manifest(dict(cfg,another_runtime=True),pairs)
    with pytest.raises(ValueError,match='at most one'):
        gate_job_manifest(dict(cfg,gate_attempt=3),pairs)


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


def test_owner_symmetric_augmentation_off_preserves_every_other_pin():
    from copy import deepcopy
    cfg = rc.runtime_config({}, root='/home/ubuntu/test_clean1',
        reference_root='/home/ubuntu/reference', source_root='/home/ubuntu/source',
        common_checkpoint_sha256='a'*64, track2=True)
    cfg['launch_augmentation_decision'] = rc.AUGMENTATION_OFF_DECISION
    cfg['plan_augment']['enabled'] = False
    assert rc.check_pins(cfg)
    bad = deepcopy(cfg)
    bad['plan_augment']['arms'] = ['R']
    with pytest.raises(ValueError, match='plan_augment'): rc.check_pins(bad)
    bad = deepcopy(cfg)
    bad['merge_features']['enabled'] = False
    with pytest.raises(ValueError, match='merge_features'): rc.check_pins(bad)


def test_disabled_augmentation_refuses_children_in_either_live_bank(tmp_path):
    from . import launch_training as training
    from .exact_grade import _seal
    from .protocol import digest
    cfg = dict(run_root=str(tmp_path), plan_augment=dict(enabled=False),
        launch_augmentation_decision=rc.AUGMENTATION_OFF_DECISION)
    training.write(tmp_path/'AUGMENTATION_DISABLED.json', _seal(dict(
        config_sha256=digest(cfg), admitted=dict(B=0, R=0)), 'receipt_sha256'))
    assert training.augmentation_receipt(cfg)['admitted'] == dict(B=0, R=0)
    for arm in ('B', 'R'):
        target=tmp_path/'replay_bank'/arm/'decisions'/'child.json'
        target.parent.mkdir(parents=True)
        target.write_text('{}')
        with pytest.raises(ValueError, match='diagnostic augmented'):
            training.augmentation_receipt(cfg)
        target.unlink()


def test_production_collection_installs_registered_scope_before_guard(monkeypatch):
    from types import SimpleNamespace
    from . import track2_backend as p1, accel_protocol
    monkeypatch.setattr(p1, '_approved', lambda cfg: None)
    installed=[]
    monkeypatch.setattr(accel_protocol, 'install_seed_scope', lambda: installed.append(True))
    def guard(world, role):
        assert installed == [True] and role == 'train'
        raise RuntimeError('guard reached after scope setup')
    monkeypatch.setattr(accel_protocol, 'guard', guard)
    runtime=SimpleNamespace(cfg=dict(worlds_by_update={'1':[dict(seed=986000)]}),arm='A',update=1)
    with pytest.raises(RuntimeError, match='after scope setup'):
        p1.ArmRuntime.collect(runtime)


def test_cross_process_receipts_are_not_visible_during_serialization_and_never_overwrite(tmp_path, monkeypatch):
    import json
    from . import launch_training as training
    target=tmp_path/'A_UPDATE_001.json'
    original=json.dump
    def serialise(value,stream,**kwargs):
        assert not target.exists(), 'consumer must never observe a partial receipt'
        return original(value,stream,**kwargs)
    monkeypatch.setattr(training.json,'dump',serialise)
    training.write(target,dict(complete=True))
    assert json.loads(target.read_text())==dict(complete=True)
    monkeypatch.setattr(training.json,'dump',original)
    with pytest.raises(FileExistsError):training.write(target,dict(complete=False))
    assert json.loads(target.read_text())==dict(complete=True)


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


@pytest.mark.parametrize('completed', ['charged', 'approved', 'prepared'])
def test_unfinished_preparation_recovery_refuses_live_or_approved_roots(tmp_path, monkeypatch, completed):
    from . import launch_preparation as preparation
    root = tmp_path/'clean1'
    rc.init_root(root, track2=True)
    if completed == 'charged':
        (root/'CHARGES.jsonl').write_text('{"id":"already_spent","sim_s":1}\n')
    else:
        marker = 'OWNER_APPROVAL.json' if completed == 'approved' else 'PREPARED.json'
        (root/marker).write_text('{}')
    original = preparation.read
    def receipts(path):
        if str(path).endswith('S0_READY.json'):
            return dict(pass_=True)
        if str(path).endswith('PARITY_PASS.json'):
            return dict(pass_=True, charged_sim_s=960)
        return original(path)
    monkeypatch.setattr(preparation, 'read', receipts)
    with pytest.raises(ValueError, match='unapproved zero-charge'):
        preparation.prepare(tmp_path/'source', root=root, resume=True)


def test_staging_discloses_only_caught_nonselected_missing_snapshot_attempts(tmp_path):
    from types import SimpleNamespace
    from .launch_preparation import successful_reader_inputs
    present = tmp_path/'selected.pt'
    present.write_bytes(b'opaque input')
    missing = tmp_path/'unselected_pending.json.gz'
    recorder = SimpleNamespace(paths={present, missing}, lexical=set())
    assert successful_reader_inputs(recorder, {present}, {missing}) == {present}
    with pytest.raises(ValueError, match='required real-reader'):
        successful_reader_inputs(recorder, {present}, set())
    with pytest.raises(ValueError, match='selected/present'):
        successful_reader_inputs(recorder, {missing}, {missing})
    with pytest.raises(ValueError, match='selected/present'):
        successful_reader_inputs(recorder, set(), {present})


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_legacy_plan_overlay_and_t300_direct_sensing_produce_two_real_distinct_variants(tmp_path, monkeypatch, arm):
    from contextlib import contextmanager
    from copy import deepcopy
    from types import SimpleNamespace
    from . import launch_augmentation as augmentation, launch_production as production
    from .test_plan_augment_cpu import source
    raw = source(arm)
    plan = raw['confirmation']['per_seed_v1']['examples'][0]['plan']
    # Real legacy per_seed_v1 overlays lack an embedded plan, unlike native
    # all8 examples. The production reader's verified chosen plan is the source.
    example = dict(example_id='legacy#s1', admitted=True, cooperative=True,
        tape=dict(branches_dir=str(tmp_path/'tapes/branches'), verification_id='legacy', branch=dict(branch='cand1')))
    parent = dict(verification_id='legacy', example_id='legacy#s1', seed_index=1)
    snapshot = dict(raw['verification']['snapshot_identity'], t0=300)
    snapshot['episode'].update(role='train', world=dict(seed=986026))
    frame = dict(time_s=300, vehicles=deepcopy(raw['context']['rows']))
    # There is deliberately no t299 frame; own t300 sensing is causal. A far
    # public row must be clipped before reaching the perturbation operator.
    frame['vehicles'].append(dict(frame['vehicles'][0], vid='far', s_m=10000., unrelated='excluded'))
    cfg = rc.runtime_config({}, root=tmp_path/'clean1', reference_root=tmp_path/'old',
        source_root=tmp_path/'source', common_checkpoint_sha256='a'*64, track2=True)
    cfg.update(training_seed=42, launch_optimization=deepcopy(production.OPTIMIZATION),
        augmentation_proposals=deepcopy(augmentation.PROPOSAL_SETTINGS))
    fake = SimpleNamespace(confirmations=lambda *a: {'legacy':dict(per_seed_v1=dict(examples=[example]))},
        bank_decisions=lambda *a, **k: [dict(verification_id='legacy', chosen_plan=plan)])
    @contextmanager
    def view(config):
        yield fake
    monkeypatch.setattr(augmentation.p1, '_bank_view', view)
    monkeypatch.setattr(augmentation.rb, 'confirmations', fake.confirmations)
    monkeypatch.setattr(augmentation.rb, 'bank_decisions', fake.bank_decisions)
    monkeypatch.setattr(augmentation.ps, 'examples', lambda *a: [parent])
    monkeypatch.setattr(augmentation.ps, 'load_snapshot', lambda *a: snapshot)
    monkeypatch.setattr(augmentation.bank_v2v, 'trace_frames', lambda *a: {300:frame})
    context = augmentation.proposal_context(plan, {300:frame}, 300)
    assert 'far' not in {r['vid'] for r in context['rows']}
    assert all(r['observed_at_s'] == 300 for r in context['rows'])
    result = augmentation.proposals(cfg, arm)
    assert len(result) == 2 and result[0]['source_id'] == result[1]['source_id'] == 'legacy#s1'
    assert result[0]['plan']['profile'] != result[1]['plan']['profile']
    assert all(r['window'] == dict(start_s=300, end_s=1200, truncated=False) for r in result)
    assert all(r['arm'] == arm for r in result)
    oversized = dict(plan, members=['a','b','c','d','e','f'])
    assert augmentation.proposal_context(oversized, {300:frame}, 300) is None
    assert augmentation.proposal_context(plan, {301:dict(frame,time_s=301)}, 300) is None
    before = {r['snapshot_path']:__import__('pathlib').Path(r['snapshot_path']).read_bytes() for r in result}
    assert augmentation.proposals(cfg, arm) == result
    assert all(__import__('pathlib').Path(path).read_bytes() == data for path,data in before.items())
    from .sumo_verify import snapshot_sha
    from .protocol import digest
    from .exact_grade import _seal
    label=augmentation.moves.from_plan(plan)
    parent_source=dict(row=parent,plan=plan,parameters=dict(onset_s=label.onset,hold_s=label.hold,depth=label.depth),
        context=context,snapshot_sha256=snapshot_sha(snapshot),source_plan_sha256=digest(plan))
    monkeypatch.setattr(augmentation,'full_bank_sources',lambda config,source_arm:[parent_source])
    previous=__import__('pathlib').Path(cfg['run_root'])/'augmentation_sources'/f'PROPOSALS_{arm}.json'
    augmentation.write(__import__('pathlib').Path(cfg['run_root'])/'AUGMENTATION_RESUME.json',_seal(dict(
        previous_proposal_batches={arm:dict(file_sha256=rc.sha(previous))}),'receipt_sha256'))
    extended=augmentation.proposals(dict(cfg,augmentation_recipe_version=2),arm)
    assert len(extended)==4 and extended[:2]==result
    prefix=__import__('pathlib').Path(cfg['run_root'])/'augmentation_sources'/f'PROPOSALS_{arm}_v2.json'
    (__import__('pathlib').Path(cfg['run_root'])/'AUGMENTATION_RESUME.json').write_text(__import__('json').dumps(_seal(dict(
        previous_proposal_batches={arm:dict(path=str(prefix),file_sha256=rc.sha(prefix))}),'receipt_sha256')))
    continued=augmentation.proposals(dict(cfg,augmentation_recipe_version=3),arm)
    assert len(continued)==6 and continued[:4]==extended
    assert all(__import__('pathlib').Path(path).read_bytes()==data for path,data in before.items())
    snapshot['policy'] = dict(state_dict_sha256='changed-source')
    __import__('torch').save(snapshot, result[0]['snapshot_path'])
    with pytest.raises(ValueError, match='snapshot changed'):
        augmentation.proposals(cfg, arm)


def test_all8_charge_ids_distinguish_same_native_id_across_crns_and_windows(tmp_path):
    from .augmentation_charges import job_id, receipt_ids
    first = dict(t2c_artifact_root=str(tmp_path/'sumo_verification/branches'), snapshot_sha256='a'*64,
        verification_id='same-native-proposal', branch='policy', window=dict(start_s=300,end_s=1200))
    second = dict(first, t2c_artifact_root=str(tmp_path/'confirm_seed2_all8/branches'), snapshot_sha256='b'*64)
    extended = dict(first, t2c_artifact_root=str(tmp_path/'confirm_extended_all8/branches'))
    assert len({job_id(first), job_id(second), job_id(extended)}) == 3
    receipt = dict(snapshot_sha256=first['snapshot_sha256'], window=first['window'],
        verification_id=first['verification_id'], branches=[dict(branch='policy')])
    path = tmp_path/'sumo_verification/verifications'/(first['verification_id']+'.json')
    assert receipt_ids(path, receipt) == [job_id(first)]
    assert job_id(dict(first,window=dict(start_s=300,end_s=900))) != job_id(first)


def test_namespaced_augmented_receipt_requires_exact_directory_snapshot_and_amounts(tmp_path, per_seed_bank):
    import json
    from tools.coopmwm import practice_starts as ps, practice_weighting as actual_pw
    from tools.coopmwm.augmentation_charges import receipt_ids
    from .test_practice_weighting_cpu import augment
    from .test_practice_starts_cpu import bank_with_snapshots
    from .exact_grade import _seal
    bank_with_snapshots(tmp_path, vids=('v0',))
    row, manifest = augment(tmp_path)
    path = tmp_path/'replay_bank/B/sumo_verification/verifications/aug1.json'
    receipt = json.loads(path.read_text())
    row['sumo_evidence'].update(charge_scheme='native_directory_v2',
        receipt_sha256=receipt['receipt_sha256'], charge_ids=receipt_ids(path, receipt))
    manifest['prelaunch']['ledger_root'] = str(tmp_path)
    manifest = _seal(dict(manifest,rows=[row]), 'practice_augmented_sha256')
    target = tmp_path/'replay_bank/B/PRACTICE_AUGMENTED.json'
    target.write_text(json.dumps(manifest))
    charges = [dict(id=i,phase=manifest['prelaunch']['phase'],sim_s=1200) for i in row['sumo_evidence']['charge_ids']]
    (tmp_path/'CHARGES.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in charges))
    rows, _ = actual_pw.augmented_examples(tmp_path, 'B', ps.examples(tmp_path,'B'))
    assert len(rows) == 1
    charges[0]['sim_s'] -= 1
    (tmp_path/'CHARGES.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in charges))
    with pytest.raises(ValueError, match='branch charges'):
        actual_pw.augmented_examples(tmp_path,'B',ps.examples(tmp_path,'B'))


def test_smoke_bank_aliases_read_parent_evidence_without_a_second_ledger(tmp_path, per_seed_bank):
    import json
    from tools.coopmwm import practice_starts as ps, practice_weighting as actual_pw
    from .launch_preparation import readonly_bank_view
    from .test_practice_weighting_cpu import augment
    from .test_practice_starts_cpu import bank_with_snapshots
    from .exact_grade import _seal
    bank_with_snapshots(tmp_path, vids=('v0',))
    row, manifest = augment(tmp_path)
    manifest = _seal(dict(manifest,prelaunch=dict(manifest['prelaunch'],ledger_root=str(tmp_path))),
                     'practice_augmented_sha256')
    (tmp_path/'replay_bank/B/PRACTICE_AUGMENTED.json').write_text(json.dumps(manifest))
    smoke = tmp_path/'smoke'
    try:
        readonly_bank_view(tmp_path/'replay_bank/B', smoke/'replay_bank/B')
    except OSError as error:
        if __import__('os').name == 'nt':
            pytest.skip('file symlink privilege is exercised by the cpuV qualification')
        raise error
    rows, _ = actual_pw.augmented_examples(smoke,'B',ps.examples(smoke,'B'))
    assert len(rows) == 1 and not (smoke/'CHARGES.jsonl').exists()


def test_full_training_bank_sources_are_bound_but_not_promoted_to_direct_practice(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from copy import deepcopy
    from types import SimpleNamespace
    import json
    from . import launch_augmentation as aug, launch_production as production
    from . import practice_starts as ps, practice_weighting as pw_local
    from .test_plan_augment_cpu import source, first_receipt
    from .exact_grade import _seal
    root = tmp_path/'clean1';root.mkdir();(root/'AUTHORITY.md').write_text('own experiment\n')
    cfg = rc.runtime_config({}, root=root, reference_root=tmp_path/'old', source_root=tmp_path/'source',
        common_checkpoint_sha256='a'*64, track2=True)
    cfg.update(training_seed=42, launch_optimization=deepcopy(production.OPTIMIZATION),
        augmentation_recipe_version=2, augmentation_proposals=deepcopy(aug.PROPOSAL_SETTINGS),
        augmentation_prelaunch=dict(allocation_sim_s=143280, phase='augment_verify', provenance='fixture',ledger_root=str(root)))
    bases, plans, receipts, examples, frames = {}, {}, {}, {}, {}
    for arm, seed in (('B',986026),('R',986023)):
        raw = source(arm)
        plan = raw['confirmation']['per_seed_v1']['examples'][0]['plan']
        base = deepcopy(raw['verification']['snapshot_identity']);base['t0']=300
        base['episode'].update(role='train',world=dict(seed=seed))
        bases[arm],plans[arm] = base,plan
        receipts[arm] = first_receipt(plan,base)
        examples[arm] = dict(example_id='imported_'+arm+'#s1',seed_index=1,admitted=True,cooperative=True)
        frames[arm] = {300:dict(time_s=300,vehicles=raw['context']['rows'])}
    def bank_pairs(root_arg,arm,*args,example_consumer,**kwargs):
        assert kwargs['imitation_target_space']=='effective' and kwargs['imitation_bank_filter']=='cooperative_only'
        example_consumer(receipt=dict(receipts[arm],_policy_version='u3'),example=examples[arm],
            plan=plans[arm],pairs=[({},1.)],trace=arm)
        return [],{}
    fake = SimpleNamespace(bank_pairs=bank_pairs)
    @contextmanager
    def view(config):yield fake
    monkeypatch.setattr(aug.p1,'_bank_view',view)
    monkeypatch.setattr(aug.rb,'_decision_snapshot',lambda root,arm,*a:bases[arm])
    monkeypatch.setattr(aug.rb,'confirmations',lambda root,arm:{receipts[arm]['verification_id']:
        dict(per_seed_v1=dict(examples=[examples[arm]]))})
    monkeypatch.setattr(aug.bank_v2v,'trace_frames',lambda trace:frames[trace])
    slates={arm:aug.proposals(cfg,arm) for arm in ('B','R')}
    assert all(len(rows)==4 for rows in slates.values())
    assert all(len({__import__('json').dumps(p['plan'],sort_keys=True) for p in rows})==4 for rows in slates.values())
    for arm in ('B','R'):
        assert ps.examples(root,arm)==[]  # parent-only imports never become practice starts
        binding=json.loads((root/f'AUGMENTATION_PARENTS_{arm}.json').read_text())
        assert binding['coverage']==[dict(example_id=examples[arm]['example_id'],members=2)]
        parent=pw_local._bound_training_parent(root,arm,examples[arm]['example_id'],cfg['augmentation_prelaunch'])
        assert parent['example_id']==examples[arm]['example_id']
        assert all(p['parent_example_id']==parent['example_id'] for p in slates[arm])
    quote=aug.slate_quote(cfg,slates)
    assert quote['per_arm']==4 and quote['total_upper_bound_sim_s']==8*14*1200<=143280
    wrong=json.loads((root/'AUGMENTATION_PARENTS_B.json').read_text())
    wrong=_seal(dict(wrong,arm='R'),'parent_bank_sha256')
    (root/'AUGMENTATION_PARENTS_B.json').write_text(json.dumps(wrong))
    with (root/'AUTHORITY.md').open('a') as stream:stream.write('augmentationParentBankSHA'+wrong['parent_bank_sha256'])
    with pytest.raises(ValueError,match='foreign'):
        pw_local._bound_training_parent(root,'B',examples['B']['example_id'],cfg['augmentation_prelaunch'])


def test_none_move_label_does_not_discard_a_nonzero_positive_native_template():
    import numpy as np
    from . import launch_augmentation as aug
    from .test_plan_augment_cpu import source
    raw=source();plan=raw['confirmation']['per_seed_v1']['examples'][0]['plan']
    plan=dict(plan,duration_s=30,profile=[[.1]*30,[.1]*30],perturbation_profile=[[.1]*30,[.1]*30])
    assert aug.moves.from_plan(plan).family=='none'
    parameters=aug.template_parameters(plan)
    assert .3<=parameters['depth']<=1. and 20<=parameters['hold_s']<=90
    context=dict(ego_id='a',t_s=300,rows=raw['context']['rows'],policy_traj={'a':[0.]*30,'b':[0.]*30})
    settings=dict(parameter_fraction=.01,partner_swap_probability=0.,variety=dict(aug.plan_augment.variety.VARIETY_REGISTERED),bounds=(-3.,1.2))
    changed,_=aug.plan_augment._perturb(plan,parameters,context,settings,np.random.default_rng(7))
    assert changed['duration_s']==30 and np.asarray(changed['profile']).min()>=0
    assert np.asarray(changed['profile']).max()>0


def test_qualified_augmentation_adoption_requires_registered_exact_input_binding(tmp_path):
    from copy import deepcopy
    from . import launch_training as training, launch_augmentation as aug, launch_production as production
    from .exact_grade import _seal
    from .protocol import digest
    root=tmp_path/'run';rc.init_root(root,track2=True)
    cfg=rc.runtime_config({},root=root,reference_root=tmp_path/'old',source_root=tmp_path/'source',
        common_checkpoint_sha256='a'*64,track2=True)
    cfg.update(launch_optimization=deepcopy(production.OPTIMIZATION),augmentation_accounting_scheme='native_directory_v2')
    ready=_seal(dict(config_sha256='earlier-source',admitted=dict(B=1,R=1)),'receipt_sha256')
    training.write(root/'AUGMENTATION_READY.json',ready)
    proof=root/'immutable_measurement.json';training.write(proof,dict(measured='unchanged'))
    adoption=_seal(dict(new_config_sha256=digest(cfg),augmentation_file_sha256=rc.sha(root/'AUGMENTATION_READY.json'),
        immutable_inputs={str(proof):rc.sha(proof)}),'receipt_sha256')
    training.write(root/'AUGMENTATION_REBIND.json',adoption)
    with pytest.raises(ValueError,match='another approved runtime'):aug.run(cfg)
    with (root/'AUTHORITY.md').open('a') as stream:stream.write('augmentationRebindSHA'+adoption['receipt_sha256'])
    assert aug.run(cfg)==ready
    proof.write_text('{"measured":"changed"}')
    with pytest.raises(ValueError,match='another approved runtime'):aug.run(cfg)
