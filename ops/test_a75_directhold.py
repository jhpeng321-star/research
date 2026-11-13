"""CPU-only tests for the amendment-75 D1 driver (no simulator): the seeded draw, the pinned candidates, that the
CURRENT registered executor (oracle_plan.StrategyCore / bank_plans) honours every drawn value, the bank rule, the
reference pins and the drop-raw safety rules."""
import gzip
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a75_directhold as D  # noqa: E402
import nm_finaleval as N  # noqa: E402
from tools.coopmwm import a37_cellbank as C  # noqa: E402
from tools.coopmwm import oracle_plan as O  # noqa: E402


# ------------------------------------------------------------------------------------------------ draw

def test_draw_is_deterministic_and_stratified():
    a, b = D.draw_strategies(), D.draw_strategies()
    assert a == b and len(a) == 36 and len({O.strategy_key(s) for s in a}) == 36
    holds = [s['roles'][0]['hold_s'] for s in a]
    sizes = [len(s['roles']) for s in a]
    assert {h: holds.count(h) for h in D.HOLDS} == {10: 8, 20: 8, 30: 7, 45: 7, 60: 6}
    assert max(holds.count(h) for h in D.HOLDS) <= 9 and min(holds.count(h) for h in D.HOLDS) >= 6
    assert {n: sizes.count(n) for n in D.SIZES} == {2: 12, 3: 12, 4: 12}
    for h in D.HOLDS:                      # every hold_s class has every size at least twice
        assert all(sum(1 for s in a if s['roles'][0]['hold_s'] == h and len(s['roles']) == n) >= 2 for n in D.SIZES)
    assert D.draw_strategies(seed=1) != a


def test_drawn_values_are_in_the_grid_and_follow_the_layout_rules():
    for s in D.draw_strategies():
        roles = s['roles']
        assert s['name'] == f'd1_{O.strategy_key(s)}' and s['slots'] == 1
        assert s['gate'] is None and s['start_gate'] is None and s['phases'] is None
        assert s['zone'] in O.ZONES and s['cooldown_s'] in D.COOLDOWNS and isinstance(s['spread_m'], float)
        assert len({r['hold_s'] for r in roles}) == 1 and roles[0]['hold_s'] in D.HOLDS
        assert len({(r['lane'], r['rank']) for r in roles}) == len(roles) and min(r['onset_s'] for r in roles) == 0
        for r in roles:
            assert r['v_w'] in D.V_W and r['dmax'] in D.DMAX and r['onset_s'] in D.ONSETS and r['lc_target'] is None
            assert r['lane'] in (0, 1, 2) and r['rank'] in (0, 1) and r['v_w'] != 15.0
            assert isinstance(r['v_w'], float) and isinstance(r['hold_s'], int)
        if s['trigger'] == 'wall':
            assert len(roles) <= 3 and all(r['rank'] == 0 for r in roles) and s['spread_m'] == 200.0
            assert len({r['lane'] for r in roles}) == len(roles)
        else:
            assert s['trigger'] == 'lanes' and s['spread_m'] in (200.0, 300.0)


def test_pinned_candidates_are_the_draw():
    cands = D.candidates()
    assert [c['strategy'] for c in cands] == D.draw_strategies()
    assert [c['draw_rank'] for c in cands] == list(range(1, 37))
    doc = N.read(D.CANDS_FILE)
    assert doc['doc_sha256'] == D.CANDS_PIN.read_text().split()[0]
    assert doc['candidates_by_hold_s'] == {'10': 8, '20': 8, '30': 7, '45': 7, '60': 6}
    assert doc['candidates_by_size'] == {'2': 12, '3': 12, '4': 12}


def test_edited_candidates_are_refused(tmp_path, monkeypatch):
    doc = json.loads(D.CANDS_FILE.read_text(encoding='utf-8'))
    doc['candidates'][0]['strategy']['roles'][0]['v_w'] = 13.0
    p = tmp_path/'c.json'
    p.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(N.Blocked):
        D.candidates(p)
    from tools.coopmwm import a42_bigbank as B
    doc = B.seal({k: v for k, v in doc.items() if k != 'doc_sha256'})     # re-sealed + re-pinned: still not the draw
    p.write_text(json.dumps(doc), encoding='utf-8')
    pin = tmp_path/'pin'
    pin.write_text(doc['doc_sha256'])
    monkeypatch.setattr(D, 'CANDS_PIN', pin)
    with pytest.raises(N.Blocked, match='seeded draw'):
        D.candidates(p)


# ------------------------------------------------------------------------------------------------ executor honours it

def test_registered_bank_executor_accepts_every_plan_unchanged():
    cands = D.candidates()
    doc = C.bank_doc([dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=0.0) for c in cands], D.CELL)
    plans, refused = O.bank_plans(doc, D.CELL)
    assert not refused and len(plans) == 36
    spec = {p['family']: p['spec'] for p in plans}
    for c in cands:                        # the spec the executor rebuilds from the bank is the drawn plan, field by field
        assert spec[c['family']] == c['strategy'] and C.executor_refusal(c['strategy']) is None
        one, ref = O.bank_plans(C.restrict(doc, c['family']), D.CELL)
        assert [p['family'] for p in one] == [c['family']] and not ref


def _rows(v=25.0):
    """Two CAVs per lane, 50 m apart, inside every zone (x = 1600 / 1550 m), constant speed."""
    out = []
    for sumo_lane in (0, 1, 2):
        for k, x in enumerate((1600.0, 1550.0)):
            out.append(dict(vid=f'c{sumo_lane}{k}', s_m=x-O.S_OFF, v_mps=v, lane=O.wm_lane_of(sumo_lane), is_cav=True))
    return out


@pytest.mark.parametrize('idx', range(36))
def test_strategy_core_honours_hold_onset_speed_brake_cooldown(idx):
    s = D.draw_strategies()[idx]
    assert all(lo <= 1550.0 and 1600.0 <= hi for lo, hi in [O.ZONES[s['zone']]])
    core = O.StrategyCore(s, warmup_s=300)
    rows = _rows()
    cmds = {r['vid']: 0.0 for r in rows}
    for t in range(290, 700):
        acc, lanes = core.step(rows, cmds, t)
        assert t >= 300 or (not acc and not lanes)
    assert core.acts and core.acts[0]['t'] == 300
    roles = s['roles']
    span = max(r['onset_s']+r['hold_s'] for r in roles)
    # activation times: a new session starts when the group is free again and the cooldown has passed
    step = max(s['cooldown_s'], span)
    assert [a['t'] for a in core.acts] == list(range(300, 700, step))
    first = core.acts[0]
    assert len(first['members']) == len(roles) == len(set(first['members']))
    for role, vid, lane in zip(roles, first['members'], first['sumo_lanes']):
        assert lane == role['lane'] and vid == f'c{role["lane"]}{0 if s["trigger"] == "wall" else role["rank"]}'
        mine = [(t, a) for t, v, a in core.issued if v == vid and t < 300+step]
        want = max(-role['dmax'], min(1.0, role['v_w']-25.0))
        # issued exactly during [onset, onset + hold) of the session, with the role's own target speed / brake cap
        assert [t for t, _ in mine] == list(range(300+role['onset_s'], 300+role['onset_s']+role['hold_s']))
        assert all(a == round(want, 4) for _, a in mine)
        assert first['release'][vid] == [role['onset_s']+role['hold_s'], 'hold_end']


def test_hold_10_and_60_differ_only_in_duration():
    s = next(x for x in D.draw_strategies() if x['roles'][0]['hold_s'] == 10)
    lens = {}
    for h in (10, 60):
        t_ = json.loads(json.dumps(s))
        for r in t_['roles']:
            r['hold_s'] = h
        core = O.StrategyCore(t_, warmup_s=300)
        rows = _rows()
        for t in range(300, 400):
            core.step(rows, {r['vid']: 0.0 for r in rows}, t)
        vid = core.acts[0]['members'][0]
        lens[h] = sum(1 for t, v, _ in core.issued if v == vid and t < 300+t_['roles'][0]['onset_s']+h)
    assert lens == {10: 10, 60: 60}


# ------------------------------------------------------------------------------------------------ bank rule

def _row(i, hold, hi, ok=True, per=None, key=None):
    return dict(family=f'f{i}', key=key or f'k{i:02}', draw_rank=i, hold_s=hold, PASS=ok, ci95=[hi-50.0, hi],
                per_seed=per or {'1': float(i), '2': 0.0})


def test_bank_choice_best_per_hold_class_at_most_four():
    rows = [_row(1, 10, -5.0), _row(2, 10, -9.0), _row(3, 20, -1.0), _row(4, 30, -20.0), _row(5, 45, -3.0),
            _row(6, 60, -7.0), _row(7, 60, -30.0, ok=False), _row(8, 20, 4.0, ok=False)]
    chosen, dup, table = D.bank_choice(rows)
    assert chosen == ['f4', 'f2', 'f6', 'f5'] and not dup          # the weakest class best (20: f3, -1) is left out
    assert table['10']['best'] == 'f2' and table['10']['tested'] == 2 and table['20']['in_bank'] is False
    assert table['60']['admitted'] == 1 and table['60']['delay_test_pass'] == 2 and table['30']['in_bank'] is True


def test_bank_choice_duplicates_and_empty():
    same = {'1': -3.0, '2': 4.0}
    rows = [_row(1, 10, -5.0, per=same), _row(2, 10, -9.0, per=same), _row(3, 20, 2.0, ok=False)]
    chosen, dup, table = D.bank_choice(rows)
    assert dup == {'f2': 'f1'} and chosen == ['f1'] and table['10']['duplicates'] == 1
    assert D.bank_choice([_row(1, 10, 3.0, ok=False)])[0] == []
    with pytest.raises(ValueError):
        D.bank_choice([_row(1, 90, -1.0)])
    a, b = _row(1, 10, -5.0, key='kb'), _row(2, 10, -5.0, key='ka')    # tie on the bound: strategy key
    assert D.bank_choice([a, b])[0] == ['f2']


# ------------------------------------------------------------------------------------------------ reference pins

def test_ref_pins_are_sealed_and_complete():
    p = D.ref_pins()
    assert p['a68v_mappo60c']['n'] == 24 and len(p['a68v_mappo60c']['reports']) == 24
    assert sorted(p['a68v_mappo60c']['reports']) == [f'a68v_mappo60c_{i:04}' for i in range(24)]
    for k in ('a70c_mappo60c', 'a70c_duo_hold_60c'):
        assert p[k]['n'] == 96 and set(p[k]['reports']) == set(p[k]['vehicle_logs']) and len(p[k]['reports']) == 96
    assert p['a68v_mappo60c']['wave_receipt_sha256'] == [
        '1a6d31ddb386f0130ee05f755d73992a8716e8f45bfa2d26a30b337d5ca0e0a0']


def test_pinned_reports_refuse_a_changed_file(tmp_path):
    job = 'a68v_mappo60c_0000'
    d = tmp_path/'evaluation'/job
    d.mkdir(parents=True)
    (d/'vehicles_report.json').write_text('{}')
    pin = D.ref_pins()['a68v_mappo60c']
    rec = dict(job_id=job, vehicle_log=dict(sha256=pin['vehicle_logs'][job]))
    with pytest.raises(N.Blocked, match='pinned'):
        D.pinned_reports(tmp_path, {'w': rec}, ['w'], pin)


# ------------------------------------------------------------------------------------------------ drop raw

def _episode(root, job, seed=992500, wave='A75V_x'):
    d = root/'evaluation'/job
    d.mkdir(parents=True)
    vl = N._fake_vehicle_log(d/N.VEHICLE_LOG, seed, 'U0')
    raw = gzip.compress(json.dumps(dict(job=job, steps=list(range(50)))).encode())
    (d/'RESULT.json.gz').write_bytes(raw)
    import hashlib
    (d/'RESULT.json').write_text(json.dumps(dict(encoding='coopmwm-lossless-gzip-v1', file='RESULT.json.gz',
                                                sha256=hashlib.sha256(raw).hexdigest(), uncompressed_bytes=1)))
    (d/'observable.jsonl.gz').write_bytes(gzip.compress(b'{"t": 1}\n'))
    (d/'CLOSE.json').write_text('{"actual_sim_s":1200.0,"charged_sim_s":1200}')
    (d/'scenarios').mkdir()
    (d/'scenarios'/'x.rou.xml').write_text('<routes/>')
    return dict(job_id=job, world_id=f'w{seed}', world=dict(seed=seed), vehicle_log=vl)


@pytest.fixture
def droproot(tmp_path, monkeypatch):
    monkeypatch.setattr(N, '_FAKE_LOGS', {})
    root = tmp_path/'root'
    recs = {'A75V_x': [_episode(root, f'a75v_x_{i:04}', 992500+i) for i in range(3)]}
    state = dict(cov=3)
    monkeypatch.setattr(N, 'published', lambda r, name, specs, check: (recs[name][:state['cov']], {}, state['cov']))
    monkeypatch.setattr(N, '_REPORTS', {})
    monkeypatch.setattr(N, '_FAKE_LOGS', {})
    return root, recs, state, [('A75V_x', 'episode', [None]*3)]


def test_drop_raw_removes_only_raw_files_after_the_report_is_verified(droproot):
    root, recs, state, waves = droproot
    state['cov'] = 2                                         # the third episode is not published yet: untouched
    out = D.drop_raw(root, waves, None)
    assert out['dropped'] == 2 and out['bytes'] > 0
    for i, r in enumerate(recs['A75V_x']):
        d = root/'evaluation'/r['job_id']
        raw = [(d/f).exists() for f in D.RAW_FILES]
        assert raw == [i == 2]*3
        for keep in ('RESULT.json', 'CLOSE.json', 'scenarios/x.rou.xml'):
            assert (d/keep).is_file()
        if i < 2:
            rec = N.verify(N.read(d/D.DROP_FILE), 'receipt_sha256', 'x')
            assert [x['file'] for x in rec['removed']] == list(D.RAW_FILES)
            assert rec['removed'][1]['sha256'] == r['vehicle_log']['sha256'] == rec['vehicle_log_sha256_sealed']
            assert rec['removed'][0]['sha256'] == N.read(d/'RESULT.json')['sha256']
            c = N.read(d/'vehicles_report.json')
            assert c['log_sha256'] == r['vehicle_log']['sha256'] and c['version'] == N.REPORT_VERSION
            from a32_ablation import report_of
            assert report_of(root, r) == c['report'] and c['report']['ok']     # admission still gets its report
    assert len((root/D.DROP_MANIFEST).read_text().splitlines()) == 2
    assert D.drop_raw(root, waves, None) == dict(dropped=0, bytes=0)          # idempotent
    state['cov'] = 3
    assert D.drop_raw(root, waves, None)['dropped'] == 1


def test_drop_raw_keeps_everything_when_the_log_differs_from_the_sealed_record(droproot):
    root, recs, state, waves = droproot
    d = root/'evaluation'/recs['A75V_x'][1]['job_id']
    (d/N.VEHICLE_LOG).write_bytes((d/N.VEHICLE_LOG).read_bytes()+b' ')
    with pytest.raises(N.Blocked):
        D.drop_raw(root, waves, None)
    for r in recs['A75V_x']:
        assert all((root/'evaluation'/r['job_id']/f).is_file() for f in D.RAW_FILES)
        assert not (root/'evaluation'/r['job_id']/D.DROP_FILE).exists()


def test_drop_raw_keeps_everything_when_result_pointer_mismatches(droproot):
    root, recs, state, waves = droproot
    d = root/'evaluation'/recs['A75V_x'][0]['job_id']
    (d/'RESULT.json.gz').write_bytes(b'other')
    with pytest.raises(N.Blocked, match='RESULT.json'):
        D.drop_raw(root, waves, None)
    assert all((d/f).is_file() for f in D.RAW_FILES) and not (d/D.DROP_FILE).exists()


def test_drop_raw_refuses_other_episodes(droproot, monkeypatch):
    root, recs, state, waves = droproot
    for bad in ('a75i_mappo60c_0000', 'a68v_mappo60c_0000', 'a71v_x_0000'):
        r = _episode(root, bad)
        monkeypatch.setattr(N, 'published', lambda *a, r=r: ([r], {}, 1))
        with pytest.raises(N.Blocked, match='not an A75'):
            D.drop_raw(root, waves, None)
        assert all((root/'evaluation'/bad/f).is_file() for f in D.RAW_FILES)
    r = dict(recs['A75V_x'][0], work_dir='../../outside')
    monkeypatch.setattr(N, 'published', lambda *a: ([r], {}, 1))
    with pytest.raises(N.Blocked, match='not an A75'):
        D.drop_raw(root, waves, None)


def test_caps_and_stage_names():
    assert D.VERIFY_CAP == 36*24*1200 == 1_036_800 and D.CONFIRM_CAP == 96*1200 and D.IDENT_CAP == 4*1200
    assert D.CAP_SIM_S == 1_156_800 and D.STAGE_PREFIX == dict(ident='a75i_', verify='a75v_', confirm='a75c_')
    assert set(D.RAW_FILES) == {'RESULT.json.gz', 'vehicles.jsonl.gz', 'observable.jsonl.gz'}
