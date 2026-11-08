"""DEMOX demo extractor (tools/coopmwm/demo_extract.py, ops/demo_extract_host.py): CPU-only, zero SUMO.

Covers seed-range assertions, banned-path guards, admitted-pair filtering (wm + rule plans, source tags),
paired-effect signs and weights, the act/none tape choice, member-row labelling, tape input reconstruction, the
demo_bc-compatible JSON + npz round trip and the host tape reducer. Featurizer parity against recorded training
records needs a run root and tapes; it is run by the extractor (``parity_recorded`` / ``parity_strip``) and
recorded in the demo manifest, not here.
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ops import demo_extract_host as host  # noqa: E402
from tools.coopmwm import demo_bc  # noqa: E402
from tools.coopmwm import demo_extract as dx  # noqa: E402

FREE = {'cong': 'free', 'lane_diff': 'L2notfaster'}
CONG = {'cong': 'congested', 'lane_diff': 'L2faster'}


def _guidance():
    pairs = [dict(family='mut_a', bin=FREE, admitted=True, plan_source='wm'),
             dict(family='mut_a', bin=CONG, admitted=False, plan_source='wm'),
             dict(family='duo12_v20', bin=FREE, admitted=True, plan_source='rule'),
             dict(family='wall_g_solo2', bin=FREE, admitted=True, plan_source='rule'),
             dict(family='mut_b', bin=FREE, admitted=False, plan_source='wm')]
    return dict(situational=dict(pairs=pairs))


def _onset(arm, seed, t, b=FREE, root='wmv_r3_D2800', holder='v1'):
    return dict(root=root, cell='D2800_p20', seed=seed, arm=arm, t=t, holder=holder, mode='privileged', bin=b)


def _bank_row(family, seed, t, diff, members=('v1', 'v2'), root='/root/x/wmv_r3_D2800'):
    return dict(root=root, cell='D2800_p20', seed=seed, family=family, t=t, diff_vs_incumbent=diff,
                members=list(members))


def _write_lines(path, rows):
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    return str(path)


def _trigger(families_by_row):
    members = [f'v{i+1}' for i in range(len(families_by_row))]
    tables, mask, desc = [], [], []
    for fams in families_by_row:
        row = [None]+[dict(family='free_program_initiate', template=dict(source=dict(variant=f))) for f in fams]
        row += [None]*(3-len(row))
        tables.append(row)
        mask.append([e is not None or j == 0 for j, e in enumerate(row)])
        desc.append([[float(j)]*2 for j in range(3)])
    n = len(members)
    return dict(context=dict(members=members, x=[[0.]*2]*n), tables=tables, mask=mask, descriptors=desc,
                actions=[0]*n, logp=[0.]*n, values=[0.]*n)


def test_seed_ranges_and_heldout():
    assert dx.assert_verification_seeds([989950, 989999, 991000, 991175])
    with pytest.raises(ValueError, match='held-out'):
        dx.assert_verification_seeds([989974, 989650])
    with pytest.raises(ValueError, match='verification ranges'):
        dx.assert_verification_seeds([989949])
    with pytest.raises(ValueError, match='verification ranges'):
        dx.assert_verification_seeds([992000])


@pytest.mark.parametrize('bad', ['/x/sealed/a', 'E:/HoldOut/b', '/heldout/c', '/a/cal_split/x', '/chunk5/t',
                                 '/j4b/x', '/inc1/y', '/w/hetdata_truth.json', '/w/driver_params.json',
                                 '/w/vType_truth.xml'])
def test_banned_paths(bad):
    with pytest.raises(ValueError, match='banned'):
        dx.guard_path(bad)
    with pytest.raises(ValueError, match='banned'):
        host.guard(bad)


def test_plan_activations_admitted_only_with_source_tags(tmp_path):
    onsets = [_onset('mut_a', 991164, 380), _onset('mut_a', 991165, 500, b=CONG), _onset('duo12_v20', 991166, 400),
              _onset('wall_g_solo2', 991166, 410), _onset('mut_b', 991167, 300), _onset('mut_a', 991168, 600)]
    bank = [_bank_row('mut_a', 991164, 380, -100.), _bank_row('mut_a', 991164, 900, -100.),
            _bank_row('duo12_v20', 991166, 400, 50.), _bank_row('wall_g_solo2', 991166, 410, None)]
    effects, members = dx.read_bank([_write_lines(tmp_path/'BANK.jsonl', bank)])
    acts, skipped = dx.plan_activations(_guidance(), onsets, effects, members)
    assert [(a['family'], a['seed'], a['source'], a['diff']) for a in acts] == [
        ('mut_a', 991164, 'wm', -100.), ('duo12_v20', 991166, 'rule', 50.)]
    assert acts[0]['members'] == ['v1', 'v2']
    assert sorted(s['reason'] for s in skipped) == ['incumbent has no paired effect vs itself',
                                                    'no per-seed bank effect']


def test_inconsistent_seed_effect_rejected(tmp_path):
    bank = [_bank_row('mut_a', 991164, 380, -100.), _bank_row('mut_a', 991164, 900, -90.)]
    with pytest.raises(ValueError, match='inconsistent'):
        dx.read_bank([_write_lines(tmp_path/'BANK.jsonl', bank)])


def test_panel_seed_activation_rejected(tmp_path):
    effects, members = dx.read_bank([_write_lines(tmp_path/'B.jsonl', [_bank_row('mut_a', 989650, 380, -1.)])])
    with pytest.raises(ValueError, match='held-out'):
        dx.plan_activations(_guidance(), [_onset('mut_a', 989650, 380)], effects, members)


def _acts():
    base = dict(root='wmv_r3_D2800', cell='D2800_p20', source='wm', bin=FREE, holder='v1')
    return [dict(base, seed=991164, family='mut_a', t=380, diff=-200., members=['v1', 'v2']),
            dict(base, seed=991165, family='mut_a', t=420, diff=-50., members=['v1']),
            dict(base, seed=991166, family='mut_a', t=500, diff=80., members=['v2']),
            dict(base, seed=991167, family='duo12_v20', source='rule', t=300, diff=40., members=['v1'])]


def test_work_list_tapes():
    work = dx.work_list(_acts())
    jobs = {w['job']: (w['until'], w['ticks'][0], w['ticks'][-1], len(w['ticks'])) for w in work.values()}
    assert jobs == {'D2800_p20_s991164_mut_a': (380, 320, 380, 61), 'D2800_p20_s991165_mut_a': (420, 360, 420, 61),
                    'D2800_p20_s991166_wall_g_solo2': (500, 440, 500, 61), 'D2800_p20_s991166_u0': (500, 440, 500, 61),
                    'D2800_p20_s991167_wall_g_solo2': (300, 300, 300, 1), 'D2800_p20_s991167_u0': (300, 300, 300, 1)}
    inc = [dict(_acts()[2], family='wall_g_solo2')]
    assert [w['job'] for w in dx.work_list(inc).values()] == ['D2800_p20_s991166_u0']


def test_build_items_labels_and_weights():
    acts = _acts()
    trig_plan = _trigger([['mut_a'], ['mut_a', 'x']])
    trig_ref = _trigger([['mut_a', 'duo12_v20'], ['mut_a']])
    triggers = {('wmv_r3_D2800__D2800_p20_s991164_mut_a', 380): trig_plan,
                ('wmv_r3_D2800__D2800_p20_s991165_mut_a', 420): _trigger([['x'], ['y']]),
                ('wmv_r3_D2800__D2800_p20_s991166_wall_g_solo2', 500): trig_ref,
                ('wmv_r3_D2800__D2800_p20_s991166_u0', 500): None,
                ('wmv_r3_D2800__D2800_p20_s991167_wall_g_solo2', 300): trig_ref,
                ('wmv_r3_D2800__D2800_p20_s991167_u0', 300): trig_ref}
    items, skipped = dx.build_items(acts, triggers)
    got = [(x['kind'], x['meta']['seed'], x['row'], x['slot'], round(x['weight'], 6), x['meta']['source'])
           for x in items]
    # helped 991164: both members offered -> weight 200/200 split over 2 rows; hurt: harm/max harm / 2 tapes / rows
    assert got == [('act', 991164, 0, 1, 0.5, 'wm'), ('act', 991164, 1, 1, 0.5, 'wm'),
                   ('none', 991166, 1, 1, 0.5, 'wm'),
                   ('none', 991167, 0, 2, 0.25, 'rule'), ('none', 991167, 0, 2, 0.25, 'rule')]
    assert all(0 < x['weight'] <= 1 for x in items)
    reasons = sorted(s['reason'] for s in skipped)
    assert reasons == ['no trigger decision in the onset window',
                       'plan not offered to any member row in the onset window']
    s = dx.summary(items)
    assert (s['rows'], s['act'], s['none'], s['plans'], s['sources']) == (5, 2, 3, ['duo12_v20', 'mut_a'],
                                                                          ['rule', 'wm'])


def test_build_items_latest_offering_decision_in_window():
    acts = _acts()[:1]
    key = 'wmv_r3_D2800__D2800_p20_s991164_mut_a'
    triggers = {(key, 300): _trigger([['mut_a'], ['x']]), (key, 340): _trigger([['mut_a'], ['mut_a']]),
                (key, 370): _trigger([['x'], ['y']]), (key, 381): _trigger([['mut_a'], ['mut_a']])}
    items, _ = dx.build_items(acts, triggers)
    assert [(x['meta']['decision_t'], x['meta']['onset_offset_s'], x['row']) for x in items] == [(340, -40, 0),
                                                                                                  (340, -40, 1)]
    items, skipped = dx.build_items(acts, triggers, lookback=5)
    assert not items and skipped[0]['reason'] == 'no trigger decision in the onset window'


def test_tape_inputs_commands():
    line = dict(time_s=301.0, actions=[dict(vid='c1', mappo_raw=1.5)],
                vehicles=[dict(vid='c1', s_m=1., v_mps=2., lane=0, is_cav=True, length_m=5., issued_acceleration_mps2=.3,
                               policy_mean_acceleration_mps2=.3, controller='setpoint'),
                          dict(vid='c2', s_m=5., v_mps=2., lane=1, is_cav=True, length_m=5., issued_acceleration_mps2=.7,
                               policy_mean_acceleration_mps2=.7, controller='policy_mean'),
                          dict(vid='h1', s_m=9., v_mps=3., lane=1, is_cav=False, length_m=5.)])
    rows, commands, issued = dx.tape_inputs(line)
    assert rows[0] == dict(vid='c1', s_m=1., v_mps=2., lane=0, is_cav=True, length_m=5.)
    assert commands == {'c1': 1.5, 'c2': .7}
    assert issued is line
    assert dx.tape_inputs(line, strip=False)[1] == {'c1': 1.5, 'c2': 0.}


def test_outputs_roundtrip_demo_bc_and_npz(tmp_path):
    items, _ = dx.build_items(_acts()[:1], {('wmv_r3_D2800__D2800_p20_s991164_mut_a', 380):
                                            _trigger([['mut_a'], ['mut_a']])})
    man = dx.write_outputs(items, [dict(reason='r')], tmp_path/'demo', dict(note='test'))
    raw = (tmp_path/'demo'/'DEMO.json').read_bytes()
    assert b'\r\n' not in raw and man['json']['sha256'] == hashlib.sha256(raw).hexdigest()
    doc = demo_bc.load(tmp_path/'demo'/'DEMO.json', man['json']['sha256'])
    assert doc['schema'] == demo_bc.SCHEMA and doc['rule'] == demo_bc.RULE
    for x in doc['items']:
        assert {'kind', 'weight', 'row', 'slot', 'trigger', 'meta'} <= set(x)
        assert {'job', 'seed', 't', 'cell', 'family', 'source'} <= set(x['meta'])
    back = dx.load_npz(tmp_path/'demo'/'DEMO.npz', man['npz']['sha256'])
    assert [(x['kind'], x['row'], x['slot'], x['weight'], x['trigger']) for x in back] == \
        [(x['kind'], x['row'], x['slot'], x['weight'], x['trigger']) for x in items]
    m = json.loads((tmp_path/'demo'/'DEMO.manifest.json').read_text(encoding='utf-8'))
    assert m['summary']['rows'] == 2 and m['skipped_reasons'] == {'r': 1}


def test_host_reducer(tmp_path):
    tape = tmp_path/'job'/'plant'
    tape.mkdir(parents=True)
    with gzip.open(tape/'observable.jsonl.gz', 'wt', encoding='utf-8') as fh:
        for t in range(300, 310):
            fh.write(json.dumps(dict(time_s=float(t), vehicles=[], actions=[]))+'\n')
    req = tmp_path/'req.json'
    req.write_text(json.dumps(dict(jobs=[dict(key='k1', tape=str(tape/'observable.jsonl.gz'), until=304)])))
    assert host.main(['--request', str(req), '--out', str(tmp_path/'out')]) == 0
    man = json.loads((tmp_path/'out'/'HOST_MANIFEST.json').read_text())
    assert man['receipts'][0]['lines'] == 5 and man['receipts'][0]['last_t'] == 304
    seconds = dx.tape_seconds(tmp_path/'out'/'k1.jsonl.gz')
    assert sorted(seconds) == [300, 301, 302, 303, 304]
    with pytest.raises(SystemExit):
        host.main(['--request', str(req), '--out', str(tmp_path/'out')])


def test_compare_records_rng_entries_not_compared():
    rec = _trigger([['mut_a', 'r1'], ['r2']])
    got = _trigger([['mut_a', 'r9'], ['mut_a']])
    out = dx.compare_records(got, rec, ['mut_a'])
    assert out['ok'] and out['slot0_equal'] == '2/2' and out['bank_entries_both'] == 1
    assert out['bank_entries_one_side_rng'] == 1 and not out['full_tables']
    bad = json.loads(json.dumps(got))
    bad['context']['x'][0] = [1., 0.]
    assert not dx.compare_records(bad, rec, ['mut_a'])['ok']
    bad = json.loads(json.dumps(got))
    bad['descriptors'][0][1] = [9., 9.]
    assert not dx.compare_records(bad, rec, ['mut_a'])['ok']
