"""BOOST-PRE warm start: CPU contracts with fakes (no SUMO, no remote, no real rows)."""
from copy import deepcopy
import io
import json
from types import SimpleNamespace

import pytest
import torch

from . import coop_warmstart as cw
from . import track2_backend as tb
from .v2v_intent import IntentPolicy


@pytest.fixture(autouse=True)
def cpu_threads(monkeypatch):
    from . import exploration
    monkeypatch.setattr(exploration, 'regime', lambda rows: 'ordinary')
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def cfg(tmp_path):
    return dict(track2_backend=True, run_root=str(tmp_path), training_seed=123,
        v2v=True, v2v_dim=48, v2v_intent=True, merge_features=True, host='cpuV',
        layer_config=dict(hidden_dim=8, learning_rate=.0003, kl_budget=.045,
            clip_ratio=.2, max_grad_norm=.5), trigger_candidates=4, trigger_partner_slots=5,
        decision_interval_s=60, trigger_lr=.0003, layer_epochs=1,
        trigger_ppo=dict(clip_ratio=.2, value_weight=.5, entropy_weight=.01, max_grad_norm=.5))


def base():
    torch.manual_seed(7)
    return IntentPolicy()


SMALL = dict(steps=6, batch_size=4, heldout_fraction=.34)


def world(seed, demand=2800, penetration=.4):
    return dict(map='wz3to2', cell=f'D{demand}_p{int(penetration*100)}', demand=demand,
                penetration=penetration, seed=seed)


def positive(arm, seed, members, *, lanes=(0, 1), t0=450, family='closing_pair', vid='v', synergy=900.):
    return dict(schema=cw.POSITIVE_SCHEMA, arm=arm, run_root='/abs/root', verification_id=f'{vid}{seed}{arm}',
                plan_sha256='0'*64, seed=1, situation_id=f's{seed}', world=world(seed), t0=t0,
                family=family, members=list(members), member_lanes=list(lanes), duration_s=60,
                synergy_veh_s={'1': synergy}, synergy_lower95={'1': synergy/2},
                v2v_range=dict(max_dist_t0_m=20.), situation=dict(approach_speed_t0=15.),
                move=dict(family=family, members=[dict(id=v, lane=lane, params=dict(v_w=22., hold_s=60, dmax=.8))
                                                  for v, lane in zip(members, lanes)]))


def negative(seed, members, *, stage='screen', reason='group not better than base on r0 (LB95(T)<=0 or T<1)',
             arm='B', t0=450, family='abreast_wall'):
    return dict(schema='newmap2-negative-v1', arm=arm, situation_id=f'n{seed}', world=world(seed), t0=t0,
                family=family, members=list(members), member_lanes=[0, 1], params=dict(v_w=15., hold_s=90, dmax=1.),
                admitted=False, status='rejected', stage=stage, reason=reason,
                v2v_range=dict(max_dist_t0_m=25.))


def write(path, rows):
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    return path


def rows_files(tmp_path):
    seeds = [989200+i for i in range(8)]
    b = [positive('B', s, (f'b{s}a', f'b{s}b')) for s in seeds]
    b += [dict(r, seed=2) for r in b[:2]]   # CRN replicate rows of one example collapse
    r = [positive('R', s, (f'r{s}a', f'r{s}b')) for s in seeds[:3]]
    n = [negative(s, (f'n{s}a', f'n{s}b')) for s in seeds]
    n.append(negative(seeds[0], ('x1', 'x2'), reason='not the best-T variant of its family at this moment'))
    return (write(tmp_path/'MOVES_B.jsonl', b), write(tmp_path/'MOVES_R.jsonl', r),
            write(tmp_path/'NEGATIVES.jsonl', n))


def dry_run(cfg, tmp_path, out='out', **kwargs):
    b, r, n = rows_files(tmp_path)
    return cw.run(cfg=cfg, base=base(), out_dir=tmp_path/out, mode='dry', positives=dict(B=[b], R=[r]),
                  negatives=[n], training=dict(SMALL, **kwargs.pop('training', {})), **kwargs)


def registration_cfg(cfg, report):
    return dict(cfg, coop_warmstart=dict(report['coop_warmstart'], allow_dry=True))


# ------------------------------------------------------------------ OFF = byte identical

def _clean3_inline(cfg, arm):
    torch.manual_seed(cfg['training_seed'])
    policy = tb.EpisodePolicy(base(), cfg, arm)
    return policy, dict(arm=arm, update=0, worker=policy.worker_state(),
                        layer_optimizer=policy.layer.optimizer.state_dict(),
                        trigger_optimizer=policy.trigger_optimizer.state_dict())


def _bytes(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


@pytest.mark.parametrize('off', [None, False, 'absent'])
@pytest.mark.parametrize('arm', ['B', 'R'])
def test_off_sidecar_is_byte_identical_to_the_clean3_inline_payload(cfg, arm, off):
    policy, expected = _clean3_inline(cfg, arm)
    before = _bytes(policy.worker_state())
    config = dict(cfg) if off == 'absent' else dict(cfg, coop_warmstart=off)
    assert cw.settings(config) is None and cw.apply(config, arm, policy) is None
    assert _bytes(cw.sidecar_state(config, arm, SimpleNamespace(policy=policy))) == _bytes(expected)
    assert _bytes(policy.worker_state()) == before


def test_runtime_modules_never_reference_the_warm_start():
    import pathlib
    root = pathlib.Path(cw.__file__).parent
    for name in ('track2_backend.py', 'coop_layer.py', 'trigger_policy.py', 'accel_plant.py'):
        assert 'coop_warmstart' not in (root/name).read_text(encoding='utf-8')   # prepare-time opt-in only


# ------------------------------------------------------------------ equal effort

def test_B_and_R_get_identical_effort_on_unequal_data(cfg, tmp_path, monkeypatch):
    calls = []
    real = cw._record_ce
    monkeypatch.setattr(cw, '_record_ce', lambda policy, record: calls.append(policy.arm) or real(policy, record))
    steps = []
    original = torch.optim.Adam.step

    def step(self, *args, **kwargs):
        steps.append(len(self.param_groups[0]['params']))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, 'step', step)
    report = dry_run(cfg, tmp_path)
    b, r = report['arms']['B'], report['arms']['R']
    assert b['positive_records'] > r['positive_records'] > 0   # unequal data ...
    assert report['effort_equal'] and b['effort'] == r['effort']   # ... identical effort
    assert b['effort']['optimizer_steps'] == SMALL['steps'] and len(steps) == 2*SMALL['steps']
    assert calls.count('B') == calls.count('R') == SMALL['steps']*SMALL['batch_size']
    assert b['effort']['loss'] == r['effort']['loss'] and b['effort']['learning_rate'] == r['effort']['learning_rate']
    # One shared negatives set for both arms.
    assert b['negative_records'] == r['negative_records'] == report['negatives']['records']
    payloads = {arm: torch.load(report['arms'][arm]['checkpoint'], weights_only=False) for arm in 'BR'}
    assert payloads['B']['effort'] == payloads['R']['effort']
    assert payloads['B']['initial_layer_sha256'] == payloads['R']['initial_layer_sha256']
    assert payloads['B']['negatives_sha256'] == payloads['R']['negatives_sha256']


def test_per_arm_training_keys_are_refused():
    with pytest.raises(ValueError, match='per-arm'):
        cw.training_settings(dict(steps_B=10))
    with pytest.raises(ValueError):
        cw.training_settings(dict(steps=0))


def test_apply_refuses_unequal_effort_wrong_arm_sha_and_dry(cfg, tmp_path):
    report = dry_run(cfg, tmp_path)
    good = registration_cfg(cfg, report)
    with pytest.raises(ValueError, match='plumbing only'):
        cw.load(dict(cfg, coop_warmstart=report['coop_warmstart']), 'B')   # dry needs allow_dry
    swapped = deepcopy(good)
    swapped['coop_warmstart']['B'], swapped['coop_warmstart']['R'] = (good['coop_warmstart']['R'],
                                                                      good['coop_warmstart']['B'])
    with pytest.raises(ValueError, match='own-arm'):
        cw.load(swapped, 'B')
    wrong = deepcopy(good)
    wrong['coop_warmstart']['R']['sha256'] = 'f'*64
    with pytest.raises(ValueError, match='sha256 mismatch'):
        cw.load(wrong, 'B')
    other = dry_run(cfg, tmp_path, out='other', training=dict(steps=7))
    mixed = deepcopy(good)
    mixed['coop_warmstart']['R'] = other['coop_warmstart']['R']
    with pytest.raises(ValueError, match='rows_sha256|step count|effort'):
        cw.load(mixed, 'B')
    with pytest.raises(ValueError, match='A never'):
        cw.load(good, 'A')


# ------------------------------------------------------------------ MAPPO untouched

def test_frozen_mappo_residual_and_value_are_untouched_and_trigger_moves(cfg, tmp_path):
    original = base()
    reference = cw.state_sha256(original)
    report = dry_run(cfg, tmp_path)
    assert cw.state_sha256(original) == reference
    initial = cw.build_policy(base(), cfg, 'B')
    for arm in 'BR':
        payload = torch.load(report['arms'][arm]['checkpoint'], weights_only=False)
        layer, trigger = payload['worker']['layer'], payload['worker']['trigger']
        for name, value in original.state_dict().items():
            assert torch.equal(layer['base.'+name], value) and torch.equal(trigger['base_policy.'+name], value)
        for name, value in initial.layer.state_dict().items():
            if not name.startswith('base.'):
                assert torch.equal(layer[name], value)   # residual and legacy gain unchanged
        for name, value in initial.trigger.trigger_value.state_dict().items():
            assert torch.equal(trigger['trigger_value.'+name], value)
        changed = [n for n, v in initial.trigger.trigger_head.state_dict().items()
                   if not torch.equal(trigger['trigger_head.'+n], v)]
        assert changed
        assert report['arms'][arm]['base_unchanged'] and payload['base_sha256'] == reference
        assert set(report['arms'][arm]['effort']['trained_parameters']) == {
            'trigger_head.'+n for n, _ in initial.trigger.trigger_head.named_parameters()}


def test_apply_restores_into_a_fresh_runtime_and_refuses_another_base(cfg, tmp_path):
    report = dry_run(cfg, tmp_path)
    good = registration_cfg(cfg, report)
    for arm in 'BR':
        policy, _ = _clean3_inline(cfg, arm)
        state = cw.sidecar_state(good, arm, SimpleNamespace(policy=policy))
        payload = torch.load(report['arms'][arm]['checkpoint'], weights_only=False)
        for part in ('layer', 'trigger'):
            for name, value in payload['worker'][part].items():
                assert torch.equal(state['worker'][part][name], value)
        assert state['layer_optimizer']['state'] == {} and state['trigger_optimizer']['state'] == {}
        assert state['coop_warmstart']['steps'] == SMALL['steps']
    torch.manual_seed(cfg['training_seed'])
    other = tb.EpisodePolicy(IntentPolicy(), cfg, 'B')   # a different MAPPO u0
    with pytest.raises(ValueError, match='frozen base'):
        cw.apply(good, 'B', other)


def test_apply_refuses_shape_mismatch(cfg, tmp_path):
    report = dry_run(cfg, tmp_path)
    good = registration_cfg(cfg, report)
    wide = dict(cfg, trigger_partner_slots=6)
    torch.manual_seed(cfg['training_seed'])
    policy = tb.EpisodePolicy(base(), wide, 'B')
    with pytest.raises(ValueError, match='shapes differ'):
        cw.apply(dict(good, trigger_partner_slots=6), 'B', policy)


# ------------------------------------------------------------------ observable-only inputs

def _records(cfg, row, frames, *, target=0, kind='negative'):
    policy = cw.build_policy(base(), cfg, 'B')
    out, _ = cw.frame_records(policy, row, frames, target=target, kind=kind, seed=cw.world_seed(row),
                              dry=True, slate_seed=5)
    return out


def _tensors(records):
    return [(r['batch'].descriptors, r['batch'].mask, r['batch'].actions,
             json.dumps(r['batch'].context, sort_keys=True, default=str)) for r in records]


def _same(a, b):
    assert len(a) == len(b) and a
    for x, y in zip(a, b):
        assert torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) and torch.equal(x[2], y[2]) and x[3] == y[3]


def test_hidden_frame_fields_never_reach_the_trigger(cfg):
    row = negative(989201, ('c1', 'c2'))
    frames = cw.dry_frames(row)
    leaky = {t: [dict(r, tau=1.7, vtype='cav_typical', driver_imperfection=.3, accel_truth=2.6)
                 for r in rows] for t, rows in frames.items()}
    _same(_tensors(_records(cfg, row, frames)), _tensors(_records(cfg, row, leaky)))
    with pytest.raises(ValueError, match='whitelist'):
        cw.whitelist([dict(vid='c1', s_m=0., v_mps=1., lane=0, is_cav=True)])


def test_row_labels_and_outcomes_never_enter_tensors(cfg):
    row = negative(989201, ('c1', 'c2'))
    frames = cw.dry_frames(row)
    relabelled = dict(row, world=world(989201, 3200, .6), cell='D3200_p60', advantage_veh_s=-5e3,
                      adv_lower95=-9e3, synergy_veh_s={'1': -1.}, safety=dict(collisions=0))
    _same(_tensors(_records(cfg, row, frames)), _tensors(_records(cfg, relabelled, frames)))
    for record in _records(cfg, row, frames):
        context = record['batch'].context
        assert set(context) <= {'members', 'x', 'rsu', 'anchors', 'decision_s', 'observed_at_s', 'trigger_v2v'}
        assert context['rsu'] == [0.]*len(context['rsu'])   # no RSU/global channel


def test_eval_range_seed_refuses_the_run_and_other_rows_are_filtered(cfg, tmp_path):
    b, r, n = rows_files(tmp_path)
    bad = write(tmp_path/'BAD.jsonl', [negative(989600, ('h1', 'h2'))])
    with pytest.raises(ValueError, match='HELD-OUT'):
        cw.run(cfg=cfg, base=base(), out_dir=tmp_path/'o', mode='dry', positives=dict(B=[b], R=[r]),
               negatives=[n, bad], training=SMALL)
    assert cw.admission(negative(986050, ('a', 'b'))) == 'not_train_seed'
    assert cw.admission(negative(964100, ('a', 'b'))) == 'not_train_seed'
    assert cw.admission(dict(negative(989201, ('a', 'b')), world=world(989201, 3200, .1))) == 'not_training_cell'
    assert cw.admission(dict(negative(989201, ('a', 'b')), world=dict(world(989201), map='merge'))) == 'other_map'
    assert cw.admission(negative(989201, ('a', 'b'))) is None


def test_restricted_paths_are_refused(cfg, tmp_path):
    b, r, n = rows_files(tmp_path)
    for name in ('holdout', 'heldout_rows', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth', 'sealed'):
        folder = tmp_path/name
        folder.mkdir()
        sneaky = write(folder/'NEG.jsonl', [negative(989201, ('a', 'b'))])
        with pytest.raises(ValueError, match='restricted'):
            cw.run(cfg=cfg, base=base(), out_dir=tmp_path/f'o_{name}', mode='dry',
                   positives=dict(B=[b], R=[r]), negatives=[sneaky], training=SMALL)


def test_each_arm_reads_only_its_own_rows(cfg, tmp_path):
    b, r, n = rows_files(tmp_path)
    with pytest.raises(ValueError, match='own-arm'):
        cw.run(cfg=cfg, base=base(), out_dir=tmp_path/'o', mode='dry', positives=dict(B=[b], R=[b]),
               negatives=[n], training=SMALL)


# ------------------------------------------------------------------ negatives rule and split

def test_negative_rule_uses_failed_evaluations_only():
    assert cw.do_not_act(negative(1, ('a', 'b')))[0]
    assert cw.do_not_act(negative(1, ('a', 'b'), stage='solos', reason='r0 synergy not admitted'))[0]
    assert cw.do_not_act(negative(1, ('a', 'b'), stage='r2', reason='replicate 2 main-window test failed'))[0]
    for stage, reason in (('screen', 'not the best-T variant of its family at this moment'),
                          ('r0_not_picked', 'r0 synergy candidate, but another variant'),
                          ('screen', 'not among the top-2 T random proposals at this moment')):
        assert not cw.do_not_act(negative(1, ('a', 'b'), stage=stage, reason=reason))[0]
    kinds = dict(schema='coopkinds-negative-v1', admitted=False)
    assert cw.do_not_act(dict(kinds, status='screen_fail_LB95T'))[0]
    assert cw.do_not_act(dict(kinds, status='r0_no_synergy'))[0]
    assert not cw.do_not_act(dict(kinds, status='r0_only_budget'))[0]
    assert not cw.do_not_act(dict(negative(1, ('a', 'b')), admitted=True))[0]


def test_split_is_by_seed_common_and_reported(cfg, tmp_path):
    report = dry_run(cfg, tmp_path)
    split = report['split']
    assert split['heldout'] and split['train'] and not set(split['heldout']) & set(split['train'])
    assert cw.split_assignment([3, 1, 2, 4], salt='s', fraction=.5) == cw.split_assignment(
        [4, 2, 1, 3], salt='s', fraction=.5)
    for arm in 'BR':
        after = report['arms'][arm]['after']
        assert after['heldout']['all']['n'] > 0 and after['heldout']['all']['accuracy'] is not None
        assert report['arms'][arm]['split_records']['heldout'] > 0
    assert report['negatives']['labels']['outranked_or_unevaluated:screen'] == 1


def test_crn_replicate_rows_collapse_to_one_example(cfg, tmp_path):
    report = dry_run(cfg, tmp_path)
    # 8 own B situations x 2 members (+2 replicate rows that collapse) -> 16 positive records.
    assert report['arms']['B']['positive_rows'] == 10 and report['arms']['B']['positive_records'] == 16


# ------------------------------------------------------------------ bank mode (fake reader)

def test_bank_mode_uses_the_registered_reader_own_rows_and_shared_frames(cfg, tmp_path, monkeypatch):
    from . import bank_breadth, bank_v2v, moves
    seeds = [989200+i for i in range(6)]
    rows = {arm: [positive(arm, s, (f'{arm}{s}a', f'{arm}{s}b'), vid=arm) for s in seeds] for arm in 'BR'}
    files = {arm: write(tmp_path/f'CLEAN4_MOVES_{arm}.jsonl', rows[arm]) for arm in 'BR'}
    frames = {arm: {(cw.world_seed(r), r['t0']): cw.dry_frames(r) for r in rows[arm]} for arm in 'BR'}
    # A shared negative at a B-verified situation (frames from the B bank trace) and one
    # without any frame (counted, never imputed).
    shared = negative(seeds[1], ('B989201a', 'B989201b'), family='wall')
    shared['move'] = dict(family='wall', members=list(shared['members']), onset=0, hold=60, depth=.8)
    neg = write(tmp_path/'NEG.jsonl', [shared, negative(989210, ('q1', 'q2'), family='wall')])
    visited = []

    def visit(config, arm, consume):
        assert config['mined_moves'].endswith(f'WARMSTART_MINED_{arm}.jsonl') and config['group_lane_moves']
        mined = [json.loads(line) for line in open(config['mined_moves'], encoding='utf-8')]
        assert {r['arm'] for r in mined} == {arm}
        for r in mined+[dict(mined[0], verification_id='inherited_merge_bank')]:
            visited.append((arm, r['verification_id']))
            consume(cfg=config, receipt=dict(verification_id=r['verification_id'], t0=r['t0']),
                    example=None, plan=dict(row=r), pairs=[], trace=(arm, cw.world_seed(r), r['t0']))

    def trace_frames(trace):
        arm, seed, t0 = trace
        got = frames[arm][(seed, t0)]
        return {t: dict(vehicles=v) for t, v in got.items()}

    def observable_contexts(policy, plan, pairs, frames_by_t, *, t0, certification=None):
        row = plan['row']
        reader = cw.observer(policy)
        for t in (t0-1, t0):
            reader.observe(frames_by_t[t]['vehicles'], t)
        ego = sorted(row['members'])[0]
        context = cw._ego_context(reader, ego, t0)
        yield context, moves.Move('wall', tuple(row['members']), 0, 60, .8), reader, 1., t0

    monkeypatch.setattr(bank_breadth, 'visit', visit)
    monkeypatch.setattr(bank_breadth, 'observable_contexts', observable_contexts)
    monkeypatch.setattr(bank_v2v, 'trace_frames', trace_frames)
    report = cw.run(cfg=dict(cfg, group_lane_moves=True), base=base(), out_dir=tmp_path/'bank', mode='bank',
                    positives=dict(B=[files['B']], R=[files['R']]), negatives=[neg], training=SMALL)
    assert not report['dry'] and report['effort_equal']
    for arm in 'BR':
        assert report['arms'][arm]['positive_records'] == len(seeds)
        assert report['arms'][arm]['positive_exclusions'] == {'not_an_admitted_own_arm_row': 1}
        assert report['arms'][arm]['negative_records'] == 2   # one record per observing member
    assert report['negatives']['reasons'] == {'no_observable_frame': 1}
    assert {a for a, _ in visited} == {'B', 'R'}
    with pytest.raises(ValueError, match='group_lane_moves'):
        cw.run(cfg=cfg, base=base(), out_dir=tmp_path/'bank2', mode='bank',
               positives=dict(B=[files['B']], R=[files['R']]), negatives=[neg], training=SMALL)


def test_existing_checkpoints_are_never_overwritten(cfg, tmp_path):
    dry_run(cfg, tmp_path)
    with pytest.raises(ValueError, match='already exist'):
        dry_run(cfg, tmp_path)
