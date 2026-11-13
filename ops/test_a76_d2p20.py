"""A76 (A76-D2P20) driver, zero SUMO: constants / caps, seed blocks + runner table patch, the unchanged A75-D2
candidates and their D2800_p20 banks (executor accepts all 36 / the six examples / the hold fixture), comparator
parameter (env guard, executor-field rebinding only, write-once speed-only binding), mechanics verdict logic."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import a76_d2p20 as M  # noqa: E402
import a75_d2forms as D2  # noqa: E402
from tools.coopmwm import a75_holdforms as H  # noqa: E402

U0 = M.FROZEN_U0_SHA256
OTHER = 'ab' * 32


def test_caps_and_constants():
    assert (M.MECH_CAP, M.VERIFY_CAP, M.CONFIRM_CAP, M.CAP_SIM_S) == (24_000, 1_065_600, 230_400, 1_320_000)
    assert M.VERIFY_CAP - M.VERIFY_N*M.HORIZON == 36*24*1200 == 1_036_800
    assert M.CONFIRM_CAP - M.CONFIRM_N*M.HORIZON == 115_200
    assert M.CELL == 'D2800_p20' and M.A75_SHA256 == D2.A75_SHA256
    assert M.BOOTSTRAP == dict(reps=10000, level=.95, seed=20261104, method='paired_percentile')
    assert [w for w, _, _ in M.FORM_WAVES] and len({w for w, _, _ in M.FORM_WAVES}) == 7
    assert [f for _, f, _ in M.FORM_WAVES] == [w for _, w, _ in D2.FORM_WAVES]


def test_cap_check():
    assert M.cap_check(0, 0, 24_000, 'mech')
    with pytest.raises(SystemExit):
        M.cap_check(1_200, 1_200, 24_000, 'mech')
    with pytest.raises(SystemExit):
        M.cap_check(0, M.CAP_SIM_S, 1_200, 'confirm')


def test_seed_blocks_fresh_and_not_forbidden():
    for w, (lo, hi, end) in M.BLOCKS.items():
        assert lo % 100 == 0 and end == lo+99 and lo <= hi <= end
        assert not any(lo <= b1 and end >= b0 for b0, b1 in M.FORBIDDEN)
    assert M.BLOCKS['m'][1]-M.BLOCKS['m'][0]+1 == M.MECH_WORLDS
    assert M.BLOCKS['v'][1]-M.BLOCKS['v'][0]+1 == M.VERIFY_N
    assert M.BLOCKS['c'][1]-M.BLOCKS['c'][0]+1 == M.CONFIRM_N
    table = ((997100, 997199), (997200, 997299), (997300, 997399))
    assert M.runner_block('v', table) == (997200, 997223, 997200, 997299)
    with pytest.raises(ValueError):
        M.runner_block('v', ((997100, 997199), (997250, 997299), (997300, 997399)))
    rows = [(lo, hi, lo) for lo, hi in table] + [(992400, 992499, 992400)]
    assert M.check_blocks_free(table, rows)
    with pytest.raises(ValueError):
        M.check_blocks_free(table, rows + [(997150, 997160, 997150)])


def test_patch_tables_a76(tmp_path):
    sys.path.insert(0, str(HERE/'a76'/'host'))
    import patch_tables_a76 as P
    src = ("A44_MECH_RANGE = (992400, 992499)\n"
           "X = ((1, 2, 3),) + (\n                       (A44_MECH_RANGE[0], A44_MECH_RANGE[1], A44_MECH_RANGE[0]),)\n")
    out = P.runner(src)
    assert 'A76_RANGES = ((997100, 997199), (997200, 997299), (997300, 997399))' in out
    assert out.count('for lo, hi in A76_RANGES') == 1 and P.runner(out) == out
    ns = {}
    exec(out, ns)
    assert ns['X'][-3:] == ((997100, 997199, 997100), (997200, 997299, 997200), (997300, 997399, 997300))
    with pytest.raises(SystemExit):
        P.runner(src + "Y = (997350, 997360)\n")
    with pytest.raises(SystemExit):
        P.runner(out.replace('(997300, 997399))\n', '(997300, 997398))\n', 1))


def test_candidates_are_the_registered_a75d2_draw_and_run_at_p20():
    c = M.candidates()
    assert len(c) == 36 and [x['draw_rank'] for x in c] == list(range(1, 37))
    doc = M.cands_bank_doc(c)
    assert {r['regime'] for r in doc['rows']} == {'D2800_p20'}
    got, refused = H.a75_bank_plans(doc, 'D2800_p20')
    assert len(got) == 36 and not any(refused.values())
    # the same strategies, unchanged (only the bank cell differs from the A75-D2 bank)
    d60 = H.bank_doc([dict(family=x['family'], strategy=x['strategy'], lb95_veh_s=0.0) for x in c], 'D2800_p60')
    assert [r['family'] for r in d60['rows']] == [r['family'] for r in doc['rows']]
    assert H.a75_bank_plans(doc, 'D2800_p60')[0] == []


def test_examples_bank_rebuilt_for_p20():
    strat, bank = M.examples_bank20()
    got, refused = H.a75_bank_plans(bank, 'D2800_p20')
    assert sorted(p['family'] for p in got) == sorted(D2.EXAMPLE_FAMILIES) and not any(refused.values())
    assert set(strat) == set(D2.EXAMPLE_FAMILIES)


def test_hold_fixture_bank():
    h = M.hold_fixture()
    assert h['name'] == 'a44_fix_hold' and all(r['lc_target'] is None for r in h['roles'])
    doc = M.ident_bank_doc()
    assert {r['regime'] for r in doc['rows']} == {'D2800_p20'}
    got, refused = H.a75_bank_plans(doc, 'D2800_p20')
    assert [p['family'] for p in got] == ['a44_fix_hold'] and not any(refused.values())


def test_comp_env_guard():
    with pytest.raises(SystemExit):
        M.comp_env({})
    with pytest.raises(SystemExit):
        M.comp_env({M.COMP_SHA_ENV: 'd9042cc5'})
    assert M.comp_env({M.COMP_SHA_ENV: U0}) == (None, U0)
    assert M.comp_env({M.COMP_SHA_ENV: OTHER, M.COMP_ENV: '/x/p.pt'}) == ('/x/p.pt', OTHER)


def _spec(kind='a37', sha=U0, job='j'):
    ck = dict(kind=kind, executor_sha256=sha, executor_path='/u0.pt', binding_sha256='b0', binding_path='/B0.json',
              comparator_kind='u0_transferred_d9042cc5', label='u0', a37_arm='A37_PLAN', residual_sha256='r')
    return dict(job_id=job, arm='A37_PLAN', checkpoint=ck, world=dict(seed=1))


def test_with_exe_only_the_six_executor_fields():
    comp = dict(rebind=True, kind='a76_comparator', label='new', path='/new.pt', sha256=OTHER, binding_sha256='bb',
                binding_path='/root/a76p20x/root/BOUND_A76_COMP.json')
    s = [_spec(), _spec(kind='u0')]
    out = M.with_exe(s, comp, 'a76v_x')
    assert [o['job_id'] for o in out] == ['a76v_x_0000', 'a76v_x_0001']
    for a, b in zip(s, out):
        changed = {k for k in set(a['checkpoint']) | set(b['checkpoint'])
                   if a['checkpoint'].get(k) != b['checkpoint'].get(k)}
        assert changed == set(M.EXEC_FIELDS)
        assert b['checkpoint']['executor_sha256'] == OTHER and a['checkpoint']['executor_sha256'] == U0
        assert {k: v for k, v in a.items() if k not in ('checkpoint', 'job_id')} == \
            {k: v for k, v in b.items() if k not in ('checkpoint', 'job_id')}
    same = M.with_exe(s, dict(comp, rebind=False), 't')
    assert [x['checkpoint'] for x in same] == [x['checkpoint'] for x in s]
    with pytest.raises(ValueError):
        M.with_exe([_spec(sha=OTHER)], comp, 't')
    with pytest.raises(ValueError):
        M.with_exe([_spec(kind='a75')], comp, 't')


def test_bind_comp_speed_only_write_once(tmp_path, monkeypatch):
    N = M._mods()[0]
    pol = tmp_path/'policy.pt'
    pol.write_bytes(b'fake export')
    sha = N.file_sha(pol)
    root = tmp_path/'root'
    root.mkdir()
    meta = dict(format='c4_dreamer/3', trainer='mappo', iteration=7, lane_control='policy', lane_head=True)
    monkeypatch.setattr(N, 'export_meta', lambda p: dict(meta))
    with pytest.raises(N.Blocked):
        M.bind_comp(root, pol, sha, 'reg line')                  # lane head refused by default
    with pytest.raises(N.Blocked):
        M.bind_comp(root, pol, U0, 'reg line')                   # U0 needs no binding
    with pytest.raises(N.Blocked):
        M.bind_comp(root, pol, OTHER, 'reg line')                # file does not hash to the pin
    meta.update(lane_control='native', lane_head=False)
    v = M.bind_comp(root, pol, sha, 'reg line')
    assert v['speed_only'] and v['sha256'] == sha and v['amendment'] == M.AMENDMENT
    assert N.verify(N.read(root/M.BOUND_COMP), 'binding_sha256', 'x')['path'] == str(pol)
    with pytest.raises(N.Blocked):
        M.bind_comp(root, pol, sha, 'reg line')                  # write-once
    root2 = tmp_path/'root2'
    root2.mkdir()
    (root2/'WAVE_A76V_COMP.json').write_text('{}')
    with pytest.raises(N.Blocked):
        M.bind_comp(root2, pol, sha, 'reg line')                 # no binding after a wave exists


def _irow(world, act=1, ok=True):
    r37 = dict(world_id=world, J_veh_s=10.0, safety=dict(collisions=0, teleports=0), executor_sha256_loaded=U0,
               a37=dict(activations=act, activations_by_family={'a44_fix_hold': act}, issued=3, acts=[1]))
    r75 = dict(r37, a75=dict(activations=act, activations_by_family={'a44_fix_hold': act}, issued=3, acts=[1],
                             extended=[], holdforms=[]))
    if not ok:
        r75['J_veh_s'] = 11.0
    return M.ident_row(r37, r75, 'abc', 'abc', U0)


def test_mech_verdict():
    ident = [_irow('w0'), _irow('w1', act=0)]
    assert all(r['PASS'] for r in ident)
    forms = {'A76M_F1_SHORT': [dict(world_id='w0', exercised=True, errors=[]),
                               dict(world_id='w1', exercised=False, errors=[])]}
    safety = [('A76M_F1_SHORT', 'w0', dict(collisions=0, teleports=0))]
    v, fails, ex = M.mech_verdict(ident, forms, safety)
    assert v == 'PASS' and not fails and ex == {'A76M_F1_SHORT': 1}
    assert M.mech_verdict([_irow('w0', act=0), _irow('w1', act=0)], forms, safety)[0] == 'FAIL'   # vacuous
    assert M.mech_verdict([_irow('w0'), _irow('w1', ok=False)], forms, safety)[0] == 'FAIL'
    none = {'A76M_F1_SHORT': [dict(world_id='w0', exercised=False, errors=[])]}
    assert M.mech_verdict(ident, none, safety)[0] == 'FAIL'                                        # nothing ran
    bad = {'A76M_F1_SHORT': [dict(world_id='w0', exercised=True, errors=['w: release odd'])]}
    assert M.mech_verdict(ident, bad, safety)[0] == 'FAIL'
    crash = safety + [('A76I_A37', 'w0', dict(collisions=1, teleports=0))]
    assert M.mech_verdict(ident, forms, crash)[0] == 'FAIL'


def test_split_errors_on_real_check_messages():
    assert M.split_errors(D2.check_short(dict(activations=0, acts=[]), dict(onset_s=0, hold_s=10))) == (False, [])
    assert M.split_errors(['no activation', 'x: release odd']) == (False, ['x: release odd'])
    assert M.split_errors([]) == (True, [])


def test_bank_rule_reused_unchanged():
    rows = [dict(family=f'p{i}', key=f'k{i}', draw_rank=i, form=f, PASS=True, ci95=[-9, -float(i)],
                 per_seed={1: float(i)}) for i, f in enumerate(D2.FORMS, start=1)]
    chosen, dup, table = D2.bank_choice(rows)
    assert chosen == ['p6', 'p5', 'p4', 'p3'] and not dup and sum(t['in_bank'] for t in table.values()) == 4


def test_drop_clone_uses_a76_prefixes():
    X = M.dropper()
    assert X.PREFIX == M.PREFIX and X.DROP_MANIFEST == 'A76_DROPPED_RAW.jsonl' and X.AMENDMENT == '76'
    assert D2.PREFIX['v'] == 'a75d2v_'             # the A75-D2 module itself is untouched
    assert X.pointer_ok is not D2.pointer_ok


def test_driver_text_rules_mention_the_registered_numbers():
    assert '997200-997223' in M.ADMIT_RULE and '997300-997395' in M.PRIMARY_RULE
    assert json.dumps(M.BLOCKS)
