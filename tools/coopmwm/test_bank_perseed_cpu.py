"""CPU tests: bank rule per_seed_v1 (owner 2026-11-03 21:45) - every (plan, seed) verification is
its own bank example; imitation pairs come from THAT seed's plan-branch tape.

Builds a real on-disk bank (sealed first-seed receipts, replication and solo receipts, branch
RESULT.json tapes with per-second team_delay_vs fingerprints AND one-second plan steps) and reads
it through replay_bank.confirmations / bank_pairs with bank_statistics.install(rb, 'per_seed_v1')."""
import copy
import json

import numpy as np
import pytest
import torch

from tools.coopmwm import bank_imitation as bi
from tools.coopmwm import bank_statistics as bs
from tools.coopmwm import exact_grade as xg
from tools.coopmwm import replay_bank as rb
from tools.coopmwm.protocol import digest
from tools.coopmwm.test_bank_boot_cpu import N, T0, T_END, noisy, tape
from tools.coopmwm.test_bank_imitation_cpu import Policy

D = 20
MEMBERS = ['c1', 'c2']
SAFE = dict(collisions=0., teleports=0., hard_brake=0., ttc_below_1p5_events=0.)


def plan_tape(increments, seed_tag, members=MEMBERS):
    out = tape(increments)
    out['steps'] = [dict(time_s=T0+i, sample=dict(
        context=dict(members=list(members)), action=[[.1*seed_tag]]*len(members),
        behaviour=dict(executed=True, seed_tag=seed_tag),
        plan_executions=[dict(plan_id='p', proposal_members=list(members))])) for i in range(D)]
    return out


def write_receipt(root_dir, vid, pol, plan, arm, members=MEMBERS):
    branches = []
    for name, t in (('policy', pol), ('cand1', plan)):
        d = root_dir/'branches'/vid/name
        d.mkdir(parents=True, exist_ok=True)
        (d/'RESULT.json').write_text(json.dumps(t))
        branches.append(dict(branch=name, status='graded', result_sha256=digest(t),
                             J_veh_s=float.fromhex(t['final']['metrics']['team_delay_vs'])-1000.))
    J = [b['J_veh_s'] for b in branches]
    r = dict(verification_id=vid, status='accepted', episode=dict(arm=arm, id='e', world=dict(seed=986066)),
             t0=T0, k=1, window=dict(start_s=T0, end_s=T_END), branches=branches, chosen=0,
             J_policy_veh_s=J[0], J_candidates_veh_s=[J[1]], grade_veh_s=J[0]-J[1],
             chosen_plan=dict(id='p', members=list(members), duration_s=D), timing=dict(finished_unix=1.))
    r = xg._seal(r, 'receipt_sha256')
    (root_dir/'verifications').mkdir(parents=True, exist_ok=True)
    (root_dir/'verifications'/(vid+'.json')).write_text(json.dumps(r))
    return r


def make(root, vid, gain1, gain2, *, syn1=None, syn2=None, arm='B', checks=None, deltas=None,
         members=MEMBERS, cooperation=True, seed=0):
    """gain_k: per-second plan advantage on seed k; syn_k: group minus best solo on seed k."""
    bank = rb.bank_root(root, arm)
    rng = np.random.default_rng(seed)
    rows = []
    for k, (gain, root_dir) in enumerate(((gain1, bank/'sumo_verification'), (gain2, bank/'confirm_replication')), 1):
        base = 2.+rng.random(N)
        rows.append((write_receipt(root_dir, vid, tape(base), plan_tape(base-gain, k, members), arm, members),
                     base, gain))
    first, rep = rows[0][0], rows[1][0]
    c = dict(verification_id=vid, t0=T0, policy_version='u3', admitted=True,
             checks=checks or dict(replication=True, no_later_harm=True, safety=True),
             replication=dict(verification_id=vid, status='accepted', receipt_sha256=rep['receipt_sha256']),
             first_seed_advantage_veh_s=first['grade_veh_s'], replication_advantage_veh_s=rep['grade_veh_s'],
             mean_replicated_advantage_veh_s=(first['grade_veh_s']+rep['grade_veh_s'])/2,
             safety_deltas=deltas or dict(replication=dict(SAFE), extended=dict(SAFE)))
    if cooperation:
        co = dict(group_size=len(members), solo=[], complete=True, cooperative=True, cooperation_gain_veh_s=1.)
        for key, syn, (_, base, gain), d in (('solo', syn1, rows[0], bank/'confirm_solo'),
                                             ('replication_solo', syn2, rows[1], bank/'confirm_solo_replication')):
            if syn is None:
                continue
            co[key] = []
            for m in range(len(members)):
                svid = f'{vid}__solo_c{m}_{key}'
                worse = syn if m == 0 else np.abs(syn)*2+1.
                r = write_receipt(d, svid, tape(base), tape(base-gain+worse), arm, members)
                co[key].append(dict(member=f'c{m}', verification_id=svid, status='accepted',
                                    receipt_sha256=r['receipt_sha256']))
        if not co['solo']:
            co.pop('solo')
            co['solo'] = []
        c['cooperation'] = co
    (bank/'confirmations').mkdir(parents=True, exist_ok=True)
    (bank/'confirmations'/(vid+'.json')).write_text(json.dumps(xg._seal(c, 'confirmation_sha256')))
    return c


def base():
    """The un-wrapped reader (another test module may have installed a rule process-wide)."""
    c = rb.confirmations
    while hasattr(c, 'original'):
        c = c.original
    return c


@pytest.fixture
def ps(monkeypatch):
    monkeypatch.setattr(rb, 'confirmations', base())   # restored after each test
    bs._CACHE.clear()
    bs.install(rb, bank_rule='per_seed_v1')
    return rb.confirmations


def win():
    return noisy(3., .5, 101)


def lose():
    return noisy(.05, 5., 102)      # positive total (first-seed positive) but lower bound < 0


def seeds_of(pairs):
    return sorted({s['behaviour']['seed_tag'] for s, _ in pairs})


def test_plan_winning_only_on_seed2_contributes_one_seed2_example(tmp_path, ps):
    make(tmp_path, 'only2', lose(), win())
    c = ps(tmp_path, 'B')['only2']
    ex = {e['seed_index']: e for e in c['per_seed_v1']['examples']}
    assert c['admitted'] and c['per_seed_v1']['admitted_seeds'] == [2]
    assert not ex[1]['admitted'] and 'lower bound' in ex[1]['reason'] and ex[2]['admitted']
    assert ex[2]['later_harm'] == 'not_measured' and ex[2]['tape']['branches_dir'].endswith('branches')
    assert 'confirm_replication' in ex[2]['tape']['branches_dir']
    assert ex[1]['tape']['intent_entry'] != ex[2]['tape']['intent_entry']
    pairs, s = rb.bank_pairs(tmp_path, 'B', 12345)
    assert len(s['entries']) == 1 and s['entries'][0]['seed_index'] == 2
    assert s['entries'][0]['seed_advantage_veh_s'] == c['replication_advantage_veh_s']
    assert len(pairs) == D and seeds_of(pairs) == [2] and all(w > 0 for _, w in pairs)
    assert s['counts']['per_seed_v1']['by_seed'] == {'1': 0, '2': 1}
    assert c['legacy_admitted'] is True


def test_plan_winning_on_both_contributes_two_examples(tmp_path, ps):
    make(tmp_path, 'both', win(), noisy(2., .5, 103))
    pairs, s = rb.bank_pairs(tmp_path, 'B', 12345)
    assert [e['seed_index'] for e in s['entries']] == [1, 2]
    assert len(pairs) == 2*D and seeds_of(pairs) == [1, 2]
    assert s['counts']['per_seed_v1']['plans_used'] == 1 and s['counts']['used_admitted'] == 2


def test_within_seed_cooperation_label_and_replicated_flag(tmp_path, ps):
    make(tmp_path, 'c1only', win(), win(), syn1=noisy(3., .5, 104), syn2=noisy(0., 1., 105))
    make(tmp_path, 'crep', win(), win(), syn1=noisy(3., .5, 106), syn2=noisy(2., .5, 107))
    conf = ps(tmp_path, 'B')
    a = {e['seed_index']: e for e in conf['c1only']['per_seed_v1']['examples']}
    assert a[1]['cooperative'] is True and a[1]['cooperative_status'] == 'cooperative'
    assert a[2]['cooperative'] is False and a[2]['cooperative_status'] == 'not_cooperative'
    assert conf['c1only']['cooperation']['cooperative_status'] == 'seed1_only'
    assert conf['c1only']['per_seed_v1']['replicated'] is False
    assert conf['crep']['per_seed_v1']['replicated'] is True
    assert conf['crep']['cooperation']['cooperative_status'] == 'replicated'
    pairs, s = rb.bank_pairs(tmp_path, 'B', 12345)
    coop = [e for e in s['entries'] if e['cooperative']]
    assert len(coop) == 3 and {(e['verification_id'], e['seed_index']) for e in coop} == {
        ('c1only', 1), ('crep', 1), ('crep', 2)}
    # weights are advantage-based and identical whether or not the example is cooperative/replicated
    assert all(e['weight'] == 1. for e in s['entries'])
    assert s['exposure']['cooperative_entries'] == 3 and s['exposure']['non_cooperative_entries'] == 1
    assert s['exposure']['cooperative_share_of_samples'] >= .5


def test_no_solos_admitted_not_cooperative(tmp_path, ps):
    make(tmp_path, 'nosolo', win(), win(), syn1=noisy(3., .5, 108))       # seed-2 solos absent
    make(tmp_path, 'nocoop', win(), win(), cooperation=False)               # no cooperation record
    conf = ps(tmp_path, 'B')
    a = {e['seed_index']: e for e in conf['nosolo']['per_seed_v1']['examples']}
    assert a[1]['cooperative'] is True and a[2]['admitted'] and a[2]['cooperative_status'] == 'no_solos'
    b = conf['nocoop']['per_seed_v1']['examples']
    assert all(e['admitted'] and e['cooperative_status'] == 'no_solos' and e['cooperative'] is False for e in b)
    single = make(tmp_path, 'single', win(), win(), members=['c1'], cooperation=False)
    assert single
    bs._CACHE.clear()
    assert all(e['cooperative_status'] == 'single_cav'
               for e in ps(tmp_path, 'B')['single']['per_seed_v1']['examples'])


def test_safety_and_later_harm_per_seed(tmp_path, ps):
    bad = dict(SAFE, collisions=1.)
    make(tmp_path, 'rep_unsafe', win(), win(), deltas=dict(replication=bad, extended=dict(SAFE)))
    make(tmp_path, 'ext_unsafe', win(), win(), deltas=dict(replication=dict(SAFE), extended=bad))
    make(tmp_path, 'brake_offset', win(), win(),
         deltas=dict(replication=dict(SAFE, hard_brake=2., ttc_below_1p5_events=-3.), extended=dict(SAFE)))
    make(tmp_path, 'harm', win(), win(), checks=dict(replication=True, no_later_harm=False, safety=True))
    conf = ps(tmp_path, 'B')

    def seeds(vid):
        return conf[vid]['per_seed_v1']['admitted_seeds']
    assert seeds('rep_unsafe') == [1] and seeds('ext_unsafe') == [2] and seeds('brake_offset') == [1, 2]
    assert seeds('harm') == [2]
    ex = {e['seed_index']: e for e in conf['harm']['per_seed_v1']['examples']}
    assert 'later-harm' in ex[1]['reason'] and ex[2]['later_harm'] == 'not_measured'
    ex = {e['seed_index']: e for e in conf['rep_unsafe']['per_seed_v1']['examples']}
    assert 'safety' in ex[2]['reason'] and ex[2]['safety_source'] == 'replication_window_seed2'


def test_bootstrap_identical_to_boot_v2_method_per_seed(tmp_path, ps):
    make(tmp_path, 'm', win(), noisy(2., .5, 109))
    ex = ps(tmp_path, 'B')['m']['per_seed_v1']['examples']
    for e in ex:
        a = e['admission']
        assert a['B'] == bs.BOOT_B and a['rng_seed'] == bs.BOOT_RNG_SEED and a['unit_source'] == bs.UNIT_PER_STEP
        assert a['per_seed'][0]['block_len'] == bs.BLOCK_S and e['n_units'] == N
    assert abs(ex[0]['total_veh_s']-3.*N) < 1e-6 and abs(ex[1]['total_veh_s']-2.*N) < 1e-6


def test_B_and_R_identical(tmp_path, ps):
    for arm in ('B', 'R'):
        make(tmp_path, 'x', lose(), win(), arm=arm, syn1=noisy(3., .5, 110), syn2=noisy(3., .5, 111))
    b, r = ps(tmp_path, 'B')['x'], ps(tmp_path, 'R')['x']
    strip = lambda c: [{k: v for k, v in e.items() if k != 'tape'} for e in c['per_seed_v1']['examples']]
    assert strip(b) == strip(r)
    pb, sb = rb.bank_pairs(tmp_path, 'B', 12345)
    pr, sr = rb.bank_pairs(tmp_path, 'R', 12345)
    assert [e['seed_index'] for e in sb['entries']] == [e['seed_index'] for e in sr['entries']] == [2]
    assert sb['counts']['per_seed_v1'] == sr['counts']['per_seed_v1']


def test_data_missing_per_example_and_all_missing_raises(tmp_path, ps):
    make(tmp_path, 'half', win(), win())
    p = rb.bank_root(tmp_path, 'B')/'confirm_replication'/'branches'/'half'/'cand1'/'RESULT.json'
    p.unlink()
    c = ps(tmp_path, 'B')['half']
    ex = {e['seed_index']: e for e in c['per_seed_v1']['examples']}
    assert ex[1]['admitted'] and ex[2]['status'] == 'data_missing' and c['admitted']
    bs._CACHE.clear()
    (rb.bank_root(tmp_path, 'B')/'sumo_verification'/'branches'/'half'/'cand1'/'RESULT.json').unlink()
    with pytest.raises(bs.BankDataMissing):
        ps(tmp_path, 'B')


def test_boot_v2_and_t1727_unchanged_and_never_stacked(tmp_path, monkeypatch):
    make(tmp_path, 'only2', lose(), win(), cooperation=False)
    monkeypatch.setattr(rb, 'confirmations', base())
    bs._CACHE.clear()
    bs.install(rb)                                   # boot_v2 default
    c = rb.confirmations(tmp_path, 'B')['only2']
    assert c['bank_rule'] == 'boot_v2' and 'per_seed_v1' not in c and c['admitted']   # pooled rule
    with pytest.raises(ValueError, match='refusing to stack'):
        bs.install(rb, bank_rule='per_seed_v1')
    pairs, s = rb.bank_pairs(tmp_path, 'B', 12345)       # boot_v2: ONE entry, first-seed tape
    assert seeds_of(pairs) == [1] and len(pairs) == D and 'per_seed_v1' not in s['counts']
    assert 'seed_index' not in s['entries'][0]
    monkeypatch.setattr(rb, 'confirmations', rb.confirmations.original)
    bs.install(rb, bank_rule='t1727')
    assert 'replicated_advantage_ci95' in rb.confirmations(tmp_path, 'B')['only2']
    with pytest.raises(ValueError):
        bs.install(rb, bank_rule='per_seed_v1')


def test_end_to_end_bank_pairs_to_run_separate(tmp_path, ps):
    make(tmp_path, 'only2', lose(), win())
    make(tmp_path, 'both', win(), win(), syn1=noisy(3., .5, 112), syn2=noisy(2., .5, 113))
    make(tmp_path, 'plain', win(), win(), cooperation=False)
    pairs, s = rb.bank_pairs(tmp_path, 'B', 12345)
    assert [(e['verification_id'], e['seed_index']) for e in s['entries']] == [
        ('both', 1), ('both', 2), ('only2', 2), ('plain', 1), ('plain', 2)]
    # non-coop 3 examples (3D pairs), cooperative 2 per-seed examples (2D) oversampled to 50%
    assert s['exposure']['cooperative_entries'] == 2 and s['exposure']['duplicated_pairs'] == D
    assert len(pairs) == 6*D and s['exposure']['cooperative_share_of_samples'] == .5
    p = Policy()
    opt = torch.optim.Adam(p.parameters(), lr=.05)
    out = bi.run_separate(p, opt, pairs, torch.Generator().manual_seed(3), copy.deepcopy(p))
    assert out['pairs_total'] == 6*D and out['epochs_completed'] >= 1 and out['pass1_effective']
    assert out['action_error_after_mps2'] < out['action_error_before_mps2']


def test_report_and_preview_cli(tmp_path, monkeypatch):
    make(tmp_path, 'only2', lose(), win())
    make(tmp_path, 'both', win(), win(), syn1=noisy(3., .5, 114), syn2=noisy(0., 1., 115))
    monkeypatch.setattr(rb, 'confirmations', base())
    bs._CACHE.clear()
    out = tmp_path/'PREVIEW.json'
    bs.main(['preview', '--root', str(tmp_path), '--arm', 'B', '--out', str(out), '--rule', 'per_seed_v1'])
    v = json.loads(out.read_text())
    assert v['bank_rule'] == 'per_seed_v1' and v['examples_admitted'] == 3
    assert v['admitted_by_seed'] == {'1': 1, '2': 2} and v['plans_contributing'] == {'0': 0, '1': 1, '2': 1}
    assert v['cooperative_examples'] == 1 and v['cooperative_by_seed'] == {'1': 1, '2': 0}
    assert v['by_policy_version'] == {'u3': dict(admitted=3, cooperative=1)}
    assert v['admitted_later_harm_not_measured'] == 2 and v['replicated_cooperative_plans'] == 0
    assert all('lower95_veh_s' in r for r in v['entries'])
    assert not hasattr(rb.confirmations, 'bank_rule')          # preview never installs a rule
    with pytest.raises(SystemExit):
        bs.main(['preview', '--root', str(tmp_path), '--arm', 'B', '--out', str(out), '--rule', 'per_seed_v1'])


def test_live_config_key_selects_rule(tmp_path, monkeypatch):
    from tools.coopmwm import v2v_launch3 as launch
    from tools.coopmwm import v2v_live
    seen = []
    monkeypatch.setattr(bs, 'install', lambda rb_, bank_rule='boot_v2', sources=None: seen.append(bank_rule))
    monkeypatch.setattr(bi, 'run', bi.run)
    v2v_live.hooks(tmp_path, dict(bank_statistics='per_seed_v1'))
    v2v_live.hooks(tmp_path, dict(bank_statistics='boot_v2'))
    v2v_live.hooks(tmp_path, dict())
    assert seen == ['per_seed_v1', 'boot_v2', 'boot_v2']
    assert launch.BANK_OPTIONS['bank_statistics'] == 'boot_v2'
    assert set(launch.BANK_RULES_ALLOWED) == {'boot_v2', 'per_seed_v1'}
