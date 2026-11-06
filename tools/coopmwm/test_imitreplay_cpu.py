"""S-IMITREPLAY: CPU contracts with fakes (no SUMO, no remote, no real rows)."""
import gzip
import json
from pathlib import Path
import types

import pytest
import torch

from . import imitreplay as ir

MAP_SPEC = str(Path(__file__).resolve().parents[2]/'assets'/'research'/'newmap_candidates'/'wz3to2'/'MAP_SPEC.json')


def world(seed, demand=3200, penetration=.6):
    return dict(map='wz3to2', cell=f'D{demand}_p{int(penetration*100)}', demand=demand, penetration=penetration,
                seed=seed)


def row(arm, seed, t0, family, members, rep=1, **kw):
    return dict(dict(schema='coopmwm-mined-move-v1', arm=arm, world=world(seed, **kw.pop('w', {})), t0=t0,
                     family=family, members=list(members), seed=rep, situation_id=f's{seed}_{t0}',
                     duration_s=60, synergy_veh_s={str(rep): 900.}, advantage_veh_s={str(rep): 1200.}), **kw)


def write(path, rows):
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    return path


@pytest.fixture
def files(tmp_path):
    b = [row('B', 989201, 450, 'closing_pair', ('a', 'b'), rep) for rep in (1, 2, 3)]
    b += [row('B', 989205, 800, 'abreast_wall', ('c', 'd', 'e'), 1, w=dict(demand=2800, penetration=.4))]
    b += [row('B', 989007, 800, 'B_lane_balance', ('f', 'g'))]
    b += [row('B', 989650, 450, 'closing_pair', ('h', 'i'))]            # HELD-OUT: refused
    b += [row('B', 989351, 450, 'abreast_wall', ('j', 'k'), w=dict(penetration=.3))]  # unregistered cell
    r = [row('R', 989201, 450, 'R_random', ('x', 'y'))]
    return write(tmp_path/'B.jsonl', b), write(tmp_path/'R.jsonl', r)


def test_moments_group_replicates_and_refuse_heldout_and_cells(files):
    ms, ex = ir.moments([files[0]], 'B')
    keys = sorted((m['world']['seed'], m['t0'], m['family'], tuple(m['replicate_seeds'])) for m in ms)
    assert keys == [(989007, 800, 'lane_balance', (1,)), (989201, 450, 'closing_pair', (1, 2, 3)),
                    (989205, 800, 'abreast_wall', (1,))]
    reasons = ' '.join(e['reason'] for e in ex)
    assert 'not an allowed TRAIN seed' in reasons and len(ex) == 2
    only = {('D3200_p60', 989201, 450, 'closing_pair')}
    ms2, ex2 = ir.moments([files[0]], 'B', only=only)
    assert [m['world']['seed'] for m in ms2] == [989201]


def test_jobs_order_pairs_and_needs(files):
    by = dict(B=ir._order(ir.moments([files[0]], 'B')[0]), R=ir.moments([files[1]], 'R')[0])
    groups = ir.build_jobs(by)
    stages = [g['stage'] for g in groups]
    assert stages == sorted(stages, key=ir.STAGES.index)
    r0 = [g for g in groups if g['stage'] == 'r0_B']
    assert r0[0]['jobs'][0]['world']['cell_key'] == 'D3200_p60'           # canonical cell first
    assert all([j['kind'] for j in g['jobs']] == ['base', 'coop'] for g in r0)
    assert all(j['decode'] == 'greedy' for g in r0 for j in g['jobs'] if j['kind'] == 'coop')
    dith = [g for g in groups if g['stage'] == 'dither_B']
    assert len(dith) == 2*3 and all(j['stop'] == min(1200, j['t0']+420) for g in dith for j in g['jobs'])
    assert all(set(j['needs']) == {f'r0_B_{ir.world_key(j["world"])}_base', f'r0_B_{ir.world_key(j["world"])}_coop'}
               for g in dith for j in g['jobs'])
    ids = [j['id'] for g in groups for j in g['jobs']]
    assert len(ids) == len(set(ids))
    assert any(g['stage'] == 'r0_R' for g in groups) and any(g['stage'] == 'sampled_B' for g in groups)


def test_plan_and_ledger_cap(tmp_path, files):
    root = tmp_path/'root'
    ir.init(root, cap=4800)
    cfg = tmp_path/'cfg.json'
    cfg.write_text('{"track2_backend": true}', encoding='utf-8')
    u0 = tmp_path/'u0.pt'
    torch.save({'policy': {'actor.0.weight': torch.zeros(4, 40)}}, u0)
    side = tmp_path/'side.pt'
    torch.save(dict(arm='B', worker={}), side)
    body = ir.plan(root, config=cfg, u0=u0, sidecars=dict(B=side), rows=dict(B=[files[0]]))
    assert (root/'PLAN.json').exists() and body['planned_sim_s']['r0_B'] == 3*2*1200
    with pytest.raises(ValueError):
        ir.plan(root, config=cfg, u0=u0, sidecars=dict(B=side), rows=dict(B=[files[0]]))
    assert ir.charge(root, [('a', 1200, 'r0_B'), ('b', 1200, 'r0_B')])
    assert ir.charge(root, [('c', 2400, 'r0_B')])
    assert not ir.charge(root, [('d', 1, 'r0_B')])                       # cap
    with pytest.raises(ValueError):
        ir.charge(root, [('a', 1, 'r0_B')])                              # never re-run
    assert ir.spent(root) == 4800
    with pytest.raises(ValueError):
        ir.init(tmp_path/'x', cap=200000)


def _event(t, *, picks=(), active=(), sessions=(), reward=-1.0, rows=()):
    s = {}
    if picks:
        tables, actions, members, accepted = [], [], [], {}
        for ego, move, ok in picks:
            tables.append([None, move])
            actions.append(1)
            members.append(ego)
            accepted[ego] = ok
        s['track2_trigger'] = dict(context=dict(members=members), tables=tables, actions=actions,
                                   accepted=accepted, proposals=dict(selected_non_none=len(picks)))
    s['track2'] = dict(active_moves=[dict(move=m, start=st, until=st+60) for m, st in active],
                       lane_drop=dict(protocol=dict(sessions=list(sessions), joins=[])))
    return dict(time_s=t, sample=s, rows=list(rows), reward=reward)


def test_summarize_steps_and_types():
    ld = dict(family='lane_drop_initiate', kind='closing_pair', members=['a'], wall_id='W1', onset=0, hold=60)
    join = dict(family='lane_drop_join', kind='closing_pair', members=['b'], wall_id='W1')
    steps = [_event(449, rows=[dict(vid='a', s_m=-300., v_mps=20., lane=2, is_cav=True)]),
             _event(450, picks=[('a', ld, True)], active=[(ld, 450)],
                    sessions=[dict(wall_id='W1', initiator='a', kind='closing_pair', members=1, released=False)]),
             _event(451, picks=[('b', join, True)], active=[(ld, 450), (join, 451)],
                    sessions=[dict(wall_id='W1', initiator='a', kind='closing_pair', members=2, released=False)])]
    out = ir.summarize_steps(steps, capture_times=(450,))
    assert [p['type'] for d in out['decisions'] for p in d['picks']] == ['closing_pair', 'closing_pair']
    assert [e['type'] for e in out['executed']] == ['closing_pair', 'closing_pair']
    assert out['sessions'][0]['max_members'] == 2 and '449' in out['captures']
    assert ir.move_type(dict(family='lane_balance_initiate')) == 'lane_balance'
    assert ir.move_type(dict(family='wall_join')) == 'v2v_wall'
    assert ir.norm_family('B_zipper') == 'zipper_order' and ir.norm_family('R_random') == 'R_random'


def _summary(delays, *, start=300, decisions=(), executed=(), sessions=()):
    series, cum = [], 0.
    for i, d in enumerate(delays):
        cum += d
        series.append([start+i, cum, cum, 0., 0., 0., 0., 0.])
    return dict(series_keys=list(ir.SERIES_KEYS), series=series, decisions=list(decisions),
                executed=list(executed), sessions=list(sessions), captures={}, dither_log=[])


def _moment(family='closing_pair', t0=450):
    return dict(id='B_m', arm='B', world=dict(cell_key='D3200_p60', seed=989201), t0=t0, family=family,
                members=['a', 'b'], duration_s=60, source='x')


def test_links_and_sign_convention():
    base = _summary([10.]*900)
    ex = [dict(family='lane_drop_initiate', type='closing_pair', role='initiate', members=['a'], session='W1',
               start=452, until=512, first_t=452, last_t=512),
          dict(family='lane_drop_join', type='closing_pair', role='join', members=['b'], session='W1',
               start=453, until=512, first_t=453, last_t=512)]
    better = _summary([10.]*152+[8.]*748, executed=ex)
    a = ir.assess_moment(_moment(), base, better)
    assert a['link'] == 'reproduced' and a['dJ']['move'] == pytest.approx(-2.*178) and a['group_members'] == 2
    assert a['first_divergence_s'] == 452
    worse = _summary([10.]*152+[11.]*748, executed=ex)
    assert ir.assess_moment(_moment(), base, worse)['link'] == 'no_gain'
    assert ir.assess_moment(_moment(), base, _summary([10.]*900))['link'] == 'no_fire'
    assert ir.assess_moment(_moment('abreast_wall'), base, better)['link'] == 'wrong_move'
    solo = _summary([10.]*152+[8.]*748, executed=ex[:1])
    assert ir.assess_moment(_moment(), base, solo)['link'] == 'no_assembly'
    late = [dict(e, start=e['start']+40, first_t=e['first_t']+40) for e in ex]
    assert ir.assess_moment(_moment(), base, _summary([10.]*900, executed=late))['link'] == 'no_fire'
    ft = ir.false_triggers([_moment()], base, _summary([10.]*900, executed=late))
    assert len(ft) == 2 and ft[0]['dJ_local'] == 0.
    r = dict(_moment('R_random'), arm='R')
    assert ir.assess_moment(r, base, better)['type_match']


def test_recorder_dither_overrides_one_cav_for_two_seconds():
    class Ctl:
        def apply(self, rows, commands, now):
            return {'z': dict(requested_raw=.3, acceleration_residual=.1)}
    ctl = Ctl()
    rec = ir.Recorder(dither=dict(vid='d', t0=450, a=-1.0, seconds=2))
    rec(ctl, {}, {})
    cmds = {'d': .5, 'z': .2}
    assert 'd' not in ctl.apply([], cmds, 449)
    for t in (450, 451):
        out = ctl.apply([], cmds, t)
        assert out['d']['requested_raw'] == -1.0 and out['z']['requested_raw'] == .3
    assert 'd' not in ctl.apply([], cmds, 452)
    assert [x['t'] for x in rec.dither_log if 'present' in x] == [450, 451]


def test_greedy_patch_forces_deterministic_and_restores(monkeypatch):
    from . import feasible_proposals as fp
    seen = []
    monkeypatch.setattr(fp, 'decide_and_start', lambda *a, **k: seen.append(k.get('deterministic')))
    original = fp.decide_and_start
    with ir.Greedy():
        fp.decide_and_start(None, deterministic=False)
    assert seen == [True] and fp.decide_and_start is original


def test_resolve_dither_rule(tmp_path):
    root = tmp_path
    m = dict(_moment(), id='B_x', world=dict(cell_key='D3200_p60', seed=989201))
    rows = [dict(vid='a', s_m=0., v_mps=20., lane=1, is_cav=True),           # verified member
            dict(vid='n1', s_m=-50., v_mps=20., lane=1, is_cav=True),
            dict(vid='n2', s_m=-400., v_mps=20., lane=0, is_cav=True),
            dict(vid='h', s_m=-10., v_mps=20., lane=0, is_cav=False),        # HDV never
            dict(vid='busy', s_m=-20., v_mps=20., lane=0, is_cav=True)]      # r0 coop move member
    for kind, executed in (('base', []), ('coop', [dict(members=['busy'], start=440, until=500, first_t=440,
                                                            last_t=500)])):
        d = root/'jobs'/f'r0_B_D3200_p60_s989201_{kind}'
        d.mkdir(parents=True)
        with gzip.open(d/'SUMMARY.json.gz', 'wt') as f:
            json.dump(dict(captures={'450': rows}, executed=executed), f)
    r1 = ir.resolve_dither(root, m, 'r1', drop_s=75.44)
    assert r1['vid'] == 'n1' and 'busy' in r1['excluded']
    r2 = ir.resolve_dither(root, m, 'r2', drop_s=75.44)
    assert r2['vid'] in ('n1', 'n2')
    assert ir.resolve_dither(root, m, 'r2', drop_s=75.44) == r2               # frozen once written


def test_worker_with_fake_plant(tmp_path, monkeypatch, files):
    """worker(): precharge check, Plan A job fields + track2 fields, greedy patch active only in the coop
    run, per-second totals via the cf.tick wrapper, compact SUMMARY + DONE."""
    from tools.coopfree import backend as cfb
    from . import accel_plant, feasible_proposals as fp, clean4_verify as c4v
    monkeypatch.setenv('COOPMWM_MAP_SPEC', MAP_SPEC)
    monkeypatch.setenv('COOPMWM_G5_RUNTIME', '1')
    root = tmp_path/'root'
    ir.init(root)
    cfg = tmp_path/'cfg.json'
    cfg.write_text('{"track2_backend": true}', encoding='utf-8')
    u0 = tmp_path/'u0.pt'
    torch.save({'policy': {'actor.0.weight': torch.zeros(4, 40)}}, u0)
    side = tmp_path/'side.pt'
    torch.save(dict(arm='B', worker={'schema': 'w'}), side)
    body = ir.plan(root, config=cfg, u0=u0, sidecars=dict(B=side), rows=dict(B=[files[0]]))
    g = next(g for g in body['groups'] if g['stage'] == 'r0_B')
    seen = {}

    def fake_tick(b, controller=None, trace=None):
        b.totals['team_delay_vs'] += 5.
        b.env.now += 1.
        return []

    monkeypatch.setattr(cfb, 'tick', fake_tick)

    def fake_episode(job, controller_hook=None):
        seen[job['id']] = dict(job, greedy=fp.decide_and_start.__name__ == 'decide_and_start'
                               and fp.decide_and_start.__qualname__.startswith('Greedy'))
        ctl = types.SimpleNamespace(events=[], apply=lambda rows, c, now: {})
        controller_hook(ctl, job, {})
        b = types.SimpleNamespace(env=types.SimpleNamespace(now=300.), totals=dict(team_delay_vs=0.))
        steps = []
        for t in range(300, job['stop']):
            cfb.tick(b, ctl, None)
            steps.append(dict(time_s=t, sample={}, rows=[dict(vid='q', s_m=1., v_mps=2., lane=1, is_cav=True)],
                              reward=-5.))
        return dict(steps=steps, totals=dict(b.totals), actual_sim_s=job['stop']-300)

    monkeypatch.setattr(accel_plant, 'episode', fake_episode)
    for j in g['jobs']:
        with pytest.raises(ValueError):
            ir.worker(root, j['id'])                                       # not precharged
    assert ir.charge(root, [(j['id'], j['sim_s'], j['stage']) for j in g['jobs']])
    for j in g['jobs']:
        s = ir.worker(root, j['id'])
        assert s['error'] is None and len(s['series']) == 900 and s['totals']['team_delay_vs'] == 4500.
    base_id, coop_id = (j['id'] for j in g['jobs'])
    w = seen[base_id]['world']
    assert seen[base_id]['noise_seed'] == seen[coop_id]['noise_seed'] == c4v.noise_seed(w, 'B')
    assert not seen[base_id].get('track2_backend') and seen[coop_id]['track2_backend']
    assert seen[coop_id]['track2_state'] == {'schema': 'w'} and seen[coop_id]['greedy'] and not seen[base_id]['greedy']
    assert seen[base_id]['warm_history'] and seen[base_id]['b_replay'] and seen[base_id]['stop'] == 1200
    assert cfb.tick is fake_tick and (root/'jobs'/coop_id/'DONE.json').exists()
    rep = ir.analyze(root)
    assert rep['links']['B'] == {'no_fire': 1} or rep['links']['B'].get('no_fire', 0) >= 1


def test_episode_components_accepts_a_warm_started_worker(tmp_path):
    """The coop job's track2 fields restore into a fresh EpisodePolicy exactly as accel_plant does."""
    from . import track2_backend as tb, acceleration
    from .v2v_intent import IntentPolicy
    cfg = dict(track2_backend=True, run_root=str(tmp_path), training_seed=123,
               v2v=True, v2v_dim=48, v2v_intent=True, merge_features=True, host='cpuV',
               layer_config=dict(hidden_dim=8, learning_rate=.0003, kl_budget=.045, clip_ratio=.2, max_grad_norm=.5),
               trigger_candidates=4, trigger_partner_slots=5, decision_interval_s=60, trigger_lr=.0003,
               layer_epochs=1, trigger_ppo=dict(clip_ratio=.2, value_weight=.5, entropy_weight=.01, max_grad_norm=.5),
               track2_reward=dict(weight=.5, provenance='test'))
    torch.manual_seed(7)
    base = IntentPolicy()
    trained = tb.EpisodePolicy(base, cfg, 'B')
    with torch.no_grad():
        for p in trained.trigger.parameters():
            if p.requires_grad:
                p.add_(.01)
    state = trained.worker_state()
    job = dict(track2_backend=True, track2_config=cfg, arm='B', track2_state=state, noise_seed=5)
    policy, ctl = tb.episode_components(job, base, acceleration.Controller)
    assert ir.state_equal(policy.trigger.state_dict(), trained.trigger.state_dict())
