"""CPU tests for the W3 wall-fit head (fakes only: no SUMO, no WM checkpoint, no GPU, no ssh).

Record formats are taken from the producing code (tools/mining/probe.py + probe_wm.py at
c4/coop-probe 5e6eb9e4, tools/coopmwm/verify_perseed.confirm_all8, probe_native.extract); the
fixtures below are synthetic and never read ledger files.
"""
import copy
import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.wallfit import dataset as ds
from tools.wallfit import evaluate as ev
from tools.wallfit import features as ft
from tools.wallfit import guard
from tools.wallfit import rank_api
from tools.wallfit import synthetic

ROOT = Path(__file__).resolve().parents[2]
FAST = dict(epochs=200, ensemble=2)


# ------------------------------------------------------------------ fake ledger fixtures

def _rows(seed_tag='1'):
    rows = []
    for i, s in enumerate((-260., -220., -180., -140., -100., -60., -20., 20., 60.)):
        rows.append(dict(vid=f'mainline.{100 + i}', s_m=s, v_mps=12. + i, lane=1, is_cav=i % 2 == 0, length_m=5.))
    for i, s in enumerate((-240., -150., -70.)):
        rows.append(dict(vid=f'mainline.{200 + i}', s_m=s, v_mps=20., lane=0, is_cav=False, length_m=5.))
    for i, s in enumerate((70., 62., 54., 46.)):
        rows.append(dict(vid=f'ramp.{10 + i}', s_m=s, v_mps=.5, lane=2, is_cav=i == 0, length_m=5.))
    return rows


WALL = ['mainline.104', 'mainline.102', 'mainline.100']
WALL_PLAN = dict(id='g_wall_mainline.104_mainline.102_mainline.100_o0_h30_d1', family='wall', members=WALL,
                 duration_s=30, origin='mining_op2', probe=dict(onset=0, hold=30, depth=1.0))


def _probe_row(sid, seed, k=1, salt=None, synergy=336.7, status='admitted', tag='wall_lead_s12', setting='wall',
               tail=120, plan=WALL_PLAN):
    Jp, Jg = 15000., 15000. - 490.
    solos = {f'solo_{m}_o0_h30_d1': Jp - a for m, a in zip(plan['members'], (0., 153.6, -190.))}
    return dict(setting=setting, state_id=sid, world=dict(cell=10, demand=2400, penetration=.4, repeat=1, seed=seed),
                arm='R', t0=455, seed_index=k, salt=salt, verification_id='sv_x', status='accepted',
                window=dict(start_s=455, end_s=605, tail_s=tail), tag=tag, group_id=plan['id'], family=plan['family'],
                members=plan['members'], n_members=len(plan['members']), J_policy=Jp, J_group=Jg, J_solos=solos,
                group_adv=Jp - Jg, best_solo=None, best_solo_adv=153.6, synergy=synergy,
                synergy_lb95=synergy - 200., adv_lb95=100., synergy_status=status)


def _context(sid, seed, t0=455):
    return dict(state_id=sid, world=dict(cell=10, demand=2400, penetration=.4, repeat=1, seed=seed), arm='R', t0=t0,
                group=['mainline.104', 'ramp.10'], pressure=[1, 2, 3], rows=_rows(), cavs=WALL, setting='wall')


def _query(sid, seed, t0=455):
    rows = _rows()
    record = dict(state_id=sid, world_id=seed, time_s=t0, rows=rows, group=WALL[:2],
                  commands={v: 0. for v in WALL}, policy_residuals={v: 0. for v in WALL}, history=[])
    q = dict(record=record, ego=WALL[0], sensing_packets={}, source_sha256='0' * 64, base_continuation={})
    acts = [dict(id='policy'), dict(id='MAPPO'),
            dict(id='W0', wm_move=dict(family='ramp_assist', members=WALL, onset=0, hold=30, depth=1.),
                 wm_active_members=WALL)]
    for m in WALL:
        acts.append(dict(id=f'S_{m}_o0_h30_d1', wm_move=dict(family='ramp_assist', members=WALL, onset=0, hold=30,
                                                             depth=1.), wm_active_members=[m]))
    groups = dict(W0=dict(plan_id=WALL_PLAN['id'], solos=[f'S_{m}_o0_h30_d1' for m in WALL]))
    return dict(query=q, actions=acts, groups=groups, H=120, ctx_dir='x', plans=[WALL_PLAN['id']], dropped=[])


def _all8(sid_pad, seed, vid_hash='bd791969'):
    vid = f'sv_{sid_pad}_{vid_hash}'
    ex = []
    for k, (adm, coop, syn) in enumerate(((True, 'cooperative', 336.7), (False, None, None), (True, 'not_cooperative', -10.)), 1):
        e = dict(seed_index=k, candidate_index=0, plan=WALL_PLAN, plan_id=WALL_PLAN['id'], members=WALL, admitted=adm,
                 status='admitted' if adm else 'rejected', advantage_veh_s=400. if adm else -20.,
                 lower95_one_sided_veh_s=50. if adm else -5., tape=dict(window=dict(tail_s=120)))
        if adm:
            e.update(cooperative=coop == 'cooperative', cooperative_status=coop,
                     synergy=dict(pooled_total_veh_s=syn, lower95_one_sided_veh_s=syn - 150.),
                     solo_advantages_veh_s=[0., 63.3, -100.])
        ex.append(e)
    return vid, dict(verification_id=vid, arm='B', t0=455, per_seed_v1=dict(examples=ex))


def make_probe_root(tmp, seed=986061):
    """A cpuV-like coop_probe root: results, jobs, collect contexts, wmq/wmrank, native bank, flat examples."""
    r = Path(tmp) / 'coopprobe'
    lead = f'op2_collect01_R_train_{seed}_t455'
    scr = f'cp_wall_R_train_{seed + 2}_t300'
    (r / 'results').mkdir(parents=True)
    rows = [_probe_row(lead, seed, 1), _probe_row(lead, seed, 2, salt=f'coopprobe#{lead}#r2', synergy=-274.2,
                                                    status='not_admitted'),
            dict(state_id=lead, tag='broken', error='boom', status='error')]
    scr_plan = dict(WALL_PLAN, id='g_wall_mainline.104_mainline.102_mainline.100_o0_h20_d0.6',
                    probe=dict(onset=0, hold=20, depth=.6), duration_s=20)
    rows.append(_probe_row(scr, seed + 2, 1, synergy=-0.05, status='not_admitted', tag=f'wall_scr_{scr}', plan=scr_plan))
    (r / 'results' / 'wall.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    (r / 'results' / 'lowcoop.jsonl').write_text(json.dumps(_probe_row(scr, seed + 2, 1, setting='lowcoop',
                                                                       tag='low', plan=scr_plan)) + '\n')
    (r / 'results' / 'wall_ext.jsonl').write_text(json.dumps(_probe_row(lead, seed, 1, setting='wall_ext', tail=600,
                                                                        synergy=7619.9)) + '\n')
    (r / 'results' / 'SUMMARY.json').write_text('{}')
    for sid, s in ((lead, seed), (scr, seed + 2)):
        d = r / 'collect' / sid.rsplit('_t', 1)[0] / f'ctx_t{sid.rsplit("_t", 1)[1]}'
        d.mkdir(parents=True)
        (d / 'CONTEXT.json').write_text(json.dumps(_context(sid, s, int(sid.rsplit('_t', 1)[1]))))
    (r / 'jobs').mkdir()
    (r / 'jobs' / f'auto_wall_scr_{scr}.json').write_text(json.dumps([dict(
        ctx_dir=f'collect/{scr.rsplit("_t", 1)[0]}/ctx_t300', plans=[scr_plan], seeds=[1], tag=f'wall_scr_{scr}',
        wm=dict(mean_synergy=3.2, saving=9.1, best_solo_saving=1.7, plan_id=scr_plan['id']))]))
    (r / 'wmq').mkdir()
    (r / 'wmrank').mkdir()
    (r / 'wmq' / f'wall__{lead}__E.query.json').write_text(json.dumps(_query(lead, seed)))
    (r / 'wmrank' / f'wall__{lead}__E.rank.json').write_text(json.dumps(dict(ranked=[dict(
        group='W0', plan_id=WALL_PLAN['id'], wm_mean_synergy=-21.6, wm_synergy_draws=[-20., -23.2],
        wm_saving_vs_policy=5., wm_best_solo_saving=26.6)], H=120, K=8, n_actions=5)))
    vid, rec = _all8(f'op2_collect01_R_train_{seed}_t0455', seed)
    bank = r / 'minebank' / 'bank' / 'B' / 'confirmations_all8'
    bank.mkdir(parents=True)
    (bank / f'{vid}.json').write_text(json.dumps(rec))
    flat = [dict(arm='B', verification_id=vid, t0=455, plan_id=WALL_PLAN['id'], family='wall', members=WALL, seed=1,
                 admitted=True, cooperative=True, status='cooperative', advantage_veh_s=400., synergy_veh_s=336.7,
                 synergy_lower95=73.6)]
    (r / 'minebank' / 'MINED_EXAMPLES.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in flat))
    return r, lead, scr, vid


# ------------------------------------------------------------------ guard: forbidden seeds

@pytest.mark.parametrize('seed', [964000, 964100, 964199, 962000, 962555, 962999, 977000, 977999, 989600, 969420, 0])
def test_guard_refuses_forbidden_and_non_train_seeds(seed):
    with pytest.raises(guard.ForbiddenSeedError):
        guard.check_seed(seed)


def test_guard_train_blocks_match_registered_protocol():
    from tools.coopmwm import protocol
    registered = {w['seed'] for i in range(len(protocol.TRAIN_STARTS)) for w in protocol.worlds('train', i)}
    allowed = {s for lo, hi in guard.TRAIN_SEED_BLOCKS for s in range(lo, hi + 1)}
    assert registered == allowed
    for s in (986000, 986061, 986071, 987035, 988071):
        assert guard.check_seed(s) == s
    assert not allowed & {s for lo, hi in guard.FORBIDDEN_SEED_RANGES for s in range(lo, hi + 1)}


def test_guard_path_tokens_are_refused_without_opening():
    for bad in ('/x/holdout/a.json', 'E:/y/HELDOUT_eval.json', '/z/cal_split/q', '/c/chunk5/x', '/d/j4b.json',
                '/e/inc1/x', '/f/sealed/x', '/g/hetdata_truth.json', '/h/vehicle_params.json', '/i/vType.xml'):
        assert not guard.path_allowed(bad)
        with pytest.raises(guard.ForbiddenPathError):
            guard.check_path(bad)
    assert guard.path_allowed('/home/ubuntu/coopprobe20261106/results/wall.jsonl')


def test_walk_prunes_forbidden_dirs_before_listing(tmp_path, monkeypatch):
    # A harmless stand-in token (the real tokens are never materialised on disk by tests).
    monkeypatch.setattr(guard, 'FORBIDDEN_PATH_TOKENS', guard.FORBIDDEN_PATH_TOKENS + ('zzblocked',))
    r, lead, scr, vid = make_probe_root(tmp_path)
    bad = r / 'zzblocked_dir' / 'results'
    bad.mkdir(parents=True)
    (bad / 'wall.jsonl').write_text(json.dumps(_probe_row('cp_wall_R_train_964010_t300', 964010)) + '\n')
    rec = ds.build([r], tmp_path / 'out')         # would raise ForbiddenSeedError if the file were read
    assert rec['skipped_forbidden_paths'] == 1
    assert rec['pairs'] >= 2


@pytest.mark.parametrize('where', ['results', 'all8', 'flat', 'context', 'query'])
def test_dataset_refuses_forbidden_seeds(tmp_path, where):
    r, lead, scr, vid = make_probe_root(tmp_path)
    bad_seed, bad_sid = 964010, 'cp_wall_R_train_964010_t300'
    if where == 'results':
        with (r / 'results' / 'wall.jsonl').open('a') as f:
            f.write(json.dumps(_probe_row(bad_sid, bad_seed)) + '\n')
    elif where == 'all8':
        v, rec = _all8('cp_wall_R_train_962123_t0300', 962123, 'aaaaaaaa')
        (r / 'minebank' / 'bank' / 'B' / 'confirmations_all8' / f'{v}.json').write_text(json.dumps(rec))
    elif where == 'flat':
        with (r / 'minebank' / 'MINED_EXAMPLES.jsonl').open('a') as f:
            f.write(json.dumps(dict(verification_id='sv_cp_wall_R_train_977001_t0300_bbbbbbbb', seed=1, t0=300,
                                    plan_id=WALL_PLAN['id'], family='wall', members=WALL, admitted=True,
                                    cooperative=True, synergy_veh_s=5.)) + '\n')
    elif where == 'context':
        d = r / 'collect' / 'x' / 'ctx_t300'
        d.mkdir(parents=True)
        (d / 'CONTEXT.json').write_text(json.dumps(_context(bad_sid, bad_seed, 300)))
    else:
        (r / 'wmq' / f'wall__{bad_sid}__E.query.json').write_text(json.dumps(_query(bad_sid, bad_seed, 300)))
    out = tmp_path / 'out'
    with pytest.raises(guard.ForbiddenSeedError):
        ds.build([r], out)
    assert not (out / 'LABELS.jsonl').exists()


def test_load_refuses_forbidden_seed_in_labels(tmp_path):
    pairs, _ = synthetic.synthetic_pairs(n_worlds=2, per_world=2, latent_dim=0)
    bad = copy.deepcopy(pairs[0])
    bad['world_seed'] = 964001
    p = synthetic.write_labels(pairs + [bad], tmp_path / 'L.jsonl')
    with pytest.raises(guard.ForbiddenSeedError):
        ds.load_pairs([p])
    with pytest.raises(guard.ForbiddenSeedError):
        ds.split_by_seed([986000, 977001])


# ------------------------------------------------------------------ dataset build (fake ledgers)

def test_build_joins_labels_situations_and_wm(tmp_path):
    r, lead, scr, vid = make_probe_root(tmp_path)
    rec = ds.build([r], tmp_path / 'out')
    pairs = {p['state_id'] + '|' + p['move']['family'] + str(p['move']['hold']): p
             for p in ds.load_pairs([tmp_path / 'out' / 'LABELS.jsonl'])}
    assert rec['skips']['excluded_setting_lowcoop'] == 1 and rec['skips']['tail_s_600'] == 1
    assert rec['skips']['probe_row_without_outcome'] == 1
    assert rec['duplicate_conflicts'] == 0 and rec['skips']['duplicate_seed_rows'] == 2   # crn1 from all8+flat+probe
    lp = pairs[lead + '|wall30']
    seeds = {s['seed_id']: s for s in lp['seeds']}
    assert set(seeds) == {'crn1', f'salt:coopprobe#{lead}#r2', f'all8:{vid}:2', f'all8:{vid}:3'}
    assert seeds['crn1']['source'] == 'all8'                  # native record wins the shared CRN seed
    assert seeds[f'all8:{vid}:2']['admitted'] is False and seeds[f'all8:{vid}:2']['synergy'] is None
    lab = lp['labels']
    assert lab['n_measured'] == 3 and lab['admitted_any'] is True
    assert lab['synergy'] == pytest.approx((336.7 - 274.2 - 10.) / 3)
    assert lab['admitted_rate'] == pytest.approx(1 / 4)
    # situation from the WM query; WM prediction from the probe rank (plan family 'wall' despite the
    # probe's 'ramp_assist' wm_move label)
    assert lp['situation']['source'] == 'query' and lp['situation']['query_id'] == f'wall__{lead}__E'
    assert lp['wm']['mean_synergy'] == pytest.approx(-21.6) and lp['wm']['horizon_s'] == 120
    sp = pairs[scr + '|wall20']
    assert sp['situation']['source'] == 'context'
    assert sp['wm']['source'] == 'job' and sp['wm']['mean_synergy'] == pytest.approx(3.2)
    assert sp['labels']['admitted_any'] is False
    with gzip.open(tmp_path / 'out' / 'QUERIES.jsonl.gz', 'rt') as f:
        qs = [json.loads(x) for x in f]
    assert [q['query_id'] for q in qs] == [f'wall__{lead}__E']
    assert rec['world_seeds'] == [986061, 986063]
    assert rec['new_sumo_sim_s'] == 0


def test_query_builder_rebuilds_queries_for_context_only_pairs(tmp_path):
    r, lead, scr, vid = make_probe_root(tmp_path)
    calls = []

    def builder(ctx_dir, owners):
        calls.append((Path(ctx_dir).name, list(owners)))
        return _query(scr, 986063, 300)['query']
    rec = ds.build([r], tmp_path / 'out', query_builder=builder)
    pairs = {p['state_id']: p for p in ds.load_pairs([tmp_path / 'out' / 'LABELS.jsonl'])}
    assert calls == [('ctx_t300', ['mainline.104', 'mainline.102'])]
    assert pairs[scr]['situation']['source'] == 'query'
    assert pairs[scr]['situation']['query_id'] == f'built__{scr}__mainline.104+mainline.102'
    assert rec['queries'] == 2 and rec['pairs_with_query'] == 2

    def broken(ctx_dir, owners):
        raise RuntimeError('tape missing')
    rec = ds.build([r], tmp_path / 'out2', query_builder=broken)
    assert rec['skips']['query_build_error'] == 1 and rec['pairs_with_query'] == 1

    def forbidden(ctx_dir, owners):
        q = _query(scr, 986063, 300)['query']
        q['record']['world_id'] = 964050
        return q
    with pytest.raises(guard.ForbiddenSeedError):
        ds.build([r], tmp_path / 'out3', query_builder=forbidden)


def test_latent_binding_env_is_applied(tmp_path, monkeypatch):
    from tools.wallfit import latent as lt
    monkeypatch.setenv('WALLFIT_TEST_VAR', 'before')   # undo restores the original (absent) state
    b = tmp_path / 'GPU_BINDING.json'
    b.write_text(json.dumps(dict(worker_env=dict(variables=dict(WALLFIT_TEST_VAR='7')))))
    assert lt.apply_binding(str(b)) == {'WALLFIT_TEST_VAR': '7'}
    import os
    assert os.environ['WALLFIT_TEST_VAR'] == '7'
    assert lt.apply_binding(None) == {}


def test_dedup_keeps_native_admission_and_fills_measured_synergy():
    mv = ft.move_from_plan(WALL_PLAN)
    native = ds._seed_row(source='all8', state_id='s_R_train_986068_t455', world_seed=986068, move=mv, seed_id='crn1',
                          seed_index=1, group_adv=87.8, admitted=False, status='not_admitted_vs_policy')
    probe = ds._seed_row(source='probe', state_id='s_R_train_986068_t455', world_seed=986068, move=mv, seed_id='crn1',
                         seed_index=1, group_adv=87.8, synergy=15.2, synergy_lb95=4.06, admitted=True,
                         J_policy=100., status='admitted')
    other = dict(probe, seed_id='salt:x', synergy=-3., admitted=False)
    rows, conflicts = ds._dedup([probe, other, native])
    by = {r['seed_id']: r for r in rows}
    assert conflicts == 0 and len(rows) == 2
    assert by['crn1']['source'] == 'all8' and by['crn1']['admitted'] is False
    assert by['crn1']['synergy'] == 15.2 and by['crn1']['synergy_lb95'] == 4.06 and by['crn1']['J_policy'] == 100.
    assert by['crn1']['filled_from'] == ['probe']
    lab = ds.aggregate(rows)
    assert lab['n_measured'] == 2 and lab['admitted_any'] is False


def test_parse_and_move_keys_match_between_plans_and_wm_actions():
    q = _query('s_R_train_986000_t300', 986000)
    a = next(x for x in q['actions'] if x['id'] == 'W0')
    assert ft.move_key(ft.move_from_action(a, WALL_PLAN['id'])) == ft.move_key(ft.move_from_plan(WALL_PLAN))
    g5 = ft.parse_plan_id('a_yield_lane_mainline.132_ramp.26_o29_h20_d0.3')
    assert g5 == dict(family='yield_lane', members=['mainline.132', 'ramp.26'], onset=29, hold=20, depth=.3)
    assert ft.parse_plan_id('lead_wall_mainline.191_mainline.176_o0_h30_d1')['family'] == 'wall'
    assert ds.state_id_from_vid('sv_cp_hp_R_train_986067_t0340_9b768543') == ('cp_hp_R_train_986067_t340', 340)
    for x in (-500., -0.3, 0., 7., 336.7):
        assert ft.inv_slog(ft.slog(x)) == pytest.approx(x, abs=1e-6)


# ------------------------------------------------------------------ split by seed

def test_split_is_by_world_seed(tmp_path):
    pairs, latents = synthetic.synthetic_pairs(n_worlds=12, per_world=4)
    seeds = [p['world_seed'] for p in pairs]
    s1 = ds.split_by_seed(seeds, folds=5)
    s2 = ds.split_by_seed(list(reversed(seeds)), folds=5)
    assert s1 == s2                                            # deterministic, order-free, outcome-blind
    assert set(s1['test_seeds']).isdisjoint(s1['train_seeds'])
    assert set(s1['test_seeds']) | set(s1['train_seeds']) == set(seeds)
    assert s1['test_seeds'] == sorted(int(k) for k, f in s1['fold_of'].items() if f == 0)
    assert len(s1['test_seeds']) >= 2
    # labels never influence the split
    shuffled = copy.deepcopy(pairs)
    for p in shuffled:
        p['labels']['synergy'] = -p['labels']['synergy']
    assert ds.split_by_seed([p['world_seed'] for p in shuffled], folds=5) == s1
    lab = synthetic.write_labels(pairs, tmp_path / 'L.jsonl')
    res = ev.run([lab], tmp_path / 'run', config=FAST)
    preds = [json.loads(x) for x in (tmp_path / 'run' / 'PREDICTIONS.jsonl').read_text().splitlines()]
    role = {}
    for p in preds:
        assert role.setdefault(p['world_seed'], p['role']) == p['role']
        assert (p['role'] == 'test') == ('head_split' in p)
    from tools.wallfit import head as hd
    b = hd.load(tmp_path / 'run' / 'HEAD_SPLIT.pt')
    assert set(b['train_seeds']).isdisjoint(res['split']['test_seeds'])
    assert res['split_eval']['head_all']['n'] == sum(p['world_seed'] in res['split']['test_seeds'] for p in pairs)


# ------------------------------------------------------------------ head training

def test_head_trains_on_synthetic_data_and_improves_rank_correlation():
    from tools.wallfit import head as hd
    pairs, latents = synthetic.synthetic_pairs(n_worlds=12, per_world=8, seed=3)
    split = ds.split_by_seed([p['world_seed'] for p in pairs], folds=4)
    test_seeds = set(split['test_seeds'])
    train = [p for p in pairs if p['world_seed'] not in test_seeds]
    test = [p for p in pairs if p['world_seed'] in test_seeds]
    syn = [p['labels']['synergy'] for p in test]
    wm_rho = ev.spearman([ev.wm_score(p) for p in test], syn)
    init = hd.untrained(latent_dim=ft.latent_row_dim(next(iter(latents.values()))))
    init_rho = ev.spearman([x['score'] for x in hd.predict(init, test, latents)], syn)
    assert init_rho == pytest.approx(wm_rho)                   # zero-init head == the current WM ranking
    b = hd.fit(train, latents, FAST)
    assert all(np.isfinite(b['final_losses']))
    scores = [x['score'] for x in hd.predict(b, test, latents)]
    rho = ev.spearman(scores, syn)
    assert wm_rho < 0 and rho > 0.5 and rho > wm_rho + 0.5
    cmp = ev.compare(test, scores)
    assert cmp['head_common']['precision_at_3'] >= cmp['wm_common']['precision_at_3']
    assert cmp['head_common']['precision_at_3'] > cmp['head_common']['base_rate']


def test_metrics_precision_ties_and_spearman():
    assert ev.precision_at_k([3., 2., 1.], [1., 0., 1.], 1) == 1.
    assert ev.precision_at_k([1., 1., 1., 0.], [1., 0., 0., 1.], 1) == pytest.approx(1 / 3)
    assert ev.precision_at_k([1., 0.], [0., 1.], 5) == pytest.approx(.5)
    assert ev.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.)
    assert ev.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.)
    assert ev.spearman([1, 1, 1], [1, 2, 3]) is None


# ------------------------------------------------------------------ ranking API

def reference_probe_rank(sc, groups):
    """Verbatim ranking block of tools/mining/probe_wm.py (c4/coop-probe 5e6eb9e4)."""
    def mean(xs):
        return sum(x['team_delay'] for x in xs)/len(xs)
    ranked = []
    for gid, g in groups.items():
        joint = sc[gid]
        paired = [min(sc[s][k]['team_delay'] for s in g['solos'])-joint[k]['team_delay']
                  for k in range(len(joint))]
        ranked.append(dict(group=gid, plan_id=g['plan_id'], wm_mean_synergy=sum(paired)/len(paired),
                           wm_synergy_draws=paired, wm_saving_vs_policy=mean(sc['policy'])-mean(joint),
                           wm_best_solo_saving=max(mean(sc['policy'])-mean(sc[s]) for s in g['solos'])))
    ranked.sort(key=lambda r: (-r['wm_mean_synergy'], -r['wm_saving_vs_policy']))
    return ranked


def _probe_call(seed=0, n_groups=4, K=8):
    """A probe_wm-shaped call: d (query/actions/groups/H) and sc = res['scores'][str(H)]."""
    rng = np.random.default_rng(seed)
    d = _query('op2_collect01_R_train_986061_t455', 986061)
    acts, groups = [dict(id='policy'), dict(id='MAPPO')], {}
    mains = [f'mainline.{100 + i}' for i in range(0, 9, 2)]
    for g in range(n_groups):
        members = mains[:3 + g % 2] if g % 3 else mains[1:4]
        hold, depth = (20, .6) if g % 2 else (30, 1.)
        gid = f'W{g}'
        acts.append(dict(id=gid, wm_move=dict(family='ramp_assist', members=members, onset=0, hold=hold, depth=depth),
                         wm_active_members=members))
        sids = []
        for m in members:
            sid = f'S_{m}_o0_h{hold}_d{depth:g}'
            if sid not in [a['id'] for a in acts]:
                acts.append(dict(id=sid, wm_move=dict(family='ramp_assist', members=members, onset=0, hold=hold,
                                                      depth=depth), wm_active_members=[m]))
            sids.append(sid)
        groups[gid] = dict(plan_id=f'g_wall_{"_".join(members)}_o0_h{hold}_d{depth:g}', solos=sids)
    d.update(actions=acts, groups=groups)
    sc = {a['id']: [dict(draw_id=k, team_delay=float(1000. + rng.normal(0, 50))) for k in range(K)] for a in acts}
    return d, sc


@pytest.fixture(scope='module')
def trained_head():
    from tools.wallfit import head as hd
    pairs, latents = synthetic.synthetic_pairs(n_worlds=6, per_world=6, seed=5, latent_dim=0)
    return hd.fit(pairs, None, dict(FAST, ensemble=1, epochs=100))


def test_default_off_keeps_probe_ranking_unchanged(monkeypatch):
    monkeypatch.delenv(rank_api.ENV, raising=False)
    assert not rank_api.enabled()
    for seed in range(5):
        d, sc = _probe_call(seed)
        ref = reference_probe_rank(sc, d['groups'])
        got = rank_api.rank_groups(sc, d['groups'], query=d['query'], actions=d['actions'], H=d['H'])
        assert got == ref and json.dumps(got) == json.dumps(ref)
        assert sorted(ref, key=rank_api.sort_key) == sorted(ref, key=lambda r: (-r['wm_mean_synergy'],
                                                                                 -r['wm_saving_vs_policy']))
    monkeypatch.setenv(rank_api.ENV, '')
    assert not rank_api.enabled()
    d, sc = _probe_call(9)
    assert rank_api.rank_groups(sc, d['groups']) == reference_probe_rank(sc, d['groups'])
    rows = [dict(family='yield_lane', members=['mainline.1', 'ramp.2'], wm_mean_synergy=1., wm_saving_vs_policy=0.)]
    assert rank_api.annotate_rows(rows, d['query']) is rows


def test_default_off_path_imports_no_torch():
    code = ('import sys, json; from tools.wallfit import rank_api; '
            'sc={"policy":[{"team_delay":1.0}],"W0":[{"team_delay":0.5}],"S0":[{"team_delay":0.9}]}; '
            'g={"W0":{"plan_id":"g_wall_mainline.1_mainline.2_o0_h30_d1","solos":["S0"]}}; '
            'r=rank_api.rank_groups(sc,g); print(json.dumps(dict(torch="torch" in sys.modules, n=len(r), '
            'head="tools.wallfit.head" in sys.modules)))')
    env = {k: v for k, v in __import__('os').environ.items() if k != rank_api.ENV}
    out = subprocess.run([sys.executable, '-c', code], cwd=str(ROOT), env=env, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == dict(torch=False, n=1, head=False)


def test_api_compatible_with_probe_ranking_call(trained_head, tmp_path, monkeypatch):
    from tools.wallfit import head as hd
    d, sc = _probe_call(2)
    ref = reference_probe_rank(sc, d['groups'])
    on = rank_api.rank_groups(sc, d['groups'], query=d['query'], actions=d['actions'], H=d['H'], head=trained_head)
    assert sorted(r['group'] for r in on) == sorted(r['group'] for r in ref)
    by = {r['group']: r for r in ref}
    for r in on:
        assert {k: r[k] for k in by[r['group']]} == by[r['group']]      # every WM field kept, unchanged
        assert r['ranker'] == 'wallfit_head' and np.isfinite(r['head_score']) and 0. <= r['head_admit_prob'] <= 1.
        assert r['head_latent_used'] is False
    assert on == sorted(on, key=rank_api.sort_key)
    json.dumps(on)                                                      # probe_wm writes it as JSON
    # enabled through the environment with a saved head file (the operator path)
    path = tmp_path / 'HEAD.pt'
    hd.save(trained_head, path)
    monkeypatch.setenv(rank_api.ENV, str(path))
    assert rank_api.enabled()
    env_on = rank_api.rank_groups(sc, d['groups'], query=d['query'], actions=d['actions'], H=d['H'])
    assert [r['group'] for r in env_on] == [r['group'] for r in on]
    assert [r['head_score'] for r in env_on] == pytest.approx([r['head_score'] for r in on])
    # mining lane rows (wm_lane_worker2.rank shape) are annotated and re-sorted, values preserved
    rows = [dict(family='yield_lane', members=['mainline.104', 'ramp.10'], onset=0, hold=20, depth=.3,
                 wm_action_id=f'a{i}', wm_mean_synergy=float(i), wm_saving_vs_policy=0.) for i in range(3)]
    ann = rank_api.annotate_rows(rows, d['query'], H=300)
    assert sorted(r['wm_action_id'] for r in ann) == ['a0', 'a1', 'a2']
    assert all('head_score' in r for r in ann) and 'head_score' not in rows[0]


def test_head_with_latent_from_fake_runtime():
    """latent.extract on a fake gpuA runtime: slot order == record rows, no weight change."""
    import torch
    from tools.wallfit import latent as lt

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def encode_drivers(self, state, **hist):
            k = state['s_m'].shape[1]
            emb = state['s_m'][..., None].repeat(1, 1, 4) * self.w
            return dict(embedding=emb, values=torch.zeros(1, k, 9), uncertainty=torch.ones(1, k, 9)), emb, None

    class Source:
        def __init__(self, rows):
            self.state = dict(s_m=torch.tensor([[r['s_m'] for r in rows] + [0., 0.]]))
            self.rollout_kwargs = dict(driver_history=torch.zeros(1, len(rows) + 2, 10, 6), unused=1)

    class Memory:
        def __init__(self, rec):
            self.source = Source(rec['rows'])

    class RT:
        model = Model()
    d, sc = _probe_call(1)
    seen = {}

    def request_memory(rt, record, binding):
        seen['binding'] = binding
        return Memory(record)
    before = RT.model.w.detach().clone()
    lat = lt.extract(RT, d['query'], request_memory=request_memory)
    assert seen['binding'] == {'memory_mode': 'warm'}
    assert lat['vids'] == [str(r['vid']) for r in d['query']['record']['rows']]
    assert tuple(lat['embedding'].shape) == (len(lat['vids']), 4) and not lat['embedding'].requires_grad
    assert float(lat['embedding'][0, 0]) == pytest.approx(d['query']['record']['rows'][0]['s_m'])
    assert torch.equal(RT.model.w.detach(), before)
    # the latent pools into fixed-width features for every move
    mv = ft.normalize_move(dict(family='wall', members=WALL, onset=0, hold=30, depth=1.))
    np_lat = {k: (v.numpy() if hasattr(v, 'numpy') else v) for k, v in lat.items()}
    vec, mask = ft.latent_features(np_lat, d['query']['record']['rows'], mv, ft.latent_row_dim(np_lat))
    assert mask == 1. and vec.shape == (2 * 22,)


def test_merge_of_two_label_files_dedups_seed_rows(tmp_path):
    pairs, _ = synthetic.synthetic_pairs(n_worlds=3, per_world=2, latent_dim=0)
    a = synthetic.write_labels(pairs, tmp_path / 'A.jsonl')
    extra = copy.deepcopy(pairs[0])
    new = dict(extra['seeds'][0], seed_id='all8:sv_x:2', seed_index=2, synergy=999., source='all8')
    extra['seeds'] = extra['seeds'] + [new]
    b = synthetic.write_labels([extra], tmp_path / 'B.jsonl')
    merged = {p['pair_id']: p for p in ds.load_pairs([a, b])}
    assert len(merged) == len(pairs)
    m = merged[pairs[0]['pair_id']]
    assert len(m['seeds']) == len(pairs[0]['seeds']) + 1
    assert m['labels']['n_measured'] == len(pairs[0]['seeds']) + 1
