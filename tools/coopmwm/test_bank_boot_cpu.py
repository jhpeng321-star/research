"""CPU tests: bank rule v2 (boot_v2) - per-unit paired bootstrap admission and synergy.

Builds a real on-disk bank (sealed first-seed receipts, replication and solo receipts and
branch RESULT.json tapes with per-second team_delay_vs fingerprints) and reads it through
replay_bank.confirmations with bank_statistics.install(rb) (the V2V path)."""
import json

import numpy as np
import pytest

from tools.coopmwm import bank_statistics as bs
from tools.coopmwm import exact_grade as xg
from tools.coopmwm import replay_bank as rb
from tools.coopmwm.protocol import digest

T0, N = 400, 200
T_END = T0+N


def fp(t, cum):
    return dict(t_s=float(t), metrics=dict(team_delay_vs=float(cum).hex(), collisions=(0.).hex()))


def tape(increments, per_vehicle=None, drop_metrics=False):
    cum = np.concatenate([[1000.], 1000.+np.cumsum(increments)])
    prefixes = {str(T0+i): fp(T0+i, cum[i]) for i in range(len(increments))}
    out = dict(prefixes=prefixes, final=fp(T_END, cum[-1]))
    if drop_metrics:
        out = dict(prefixes={k: dict(t_s=v['t_s']) for k, v in prefixes.items()}, final=dict(t_s=float(T_END)))
    if per_vehicle is not None:
        out[bs.PER_VEHICLE_KEY] = per_vehicle
    return out


def write_receipt(root_dir, vid, pol, plan, arm='B'):
    """Sealed sumo_verify-like receipt + two branch tapes; returns the receipt."""
    branches = []
    for name, t in (('policy', pol), ('cand1', plan)):
        d = root_dir/'branches'/vid/name
        d.mkdir(parents=True, exist_ok=True)
        (d/'RESULT.json').write_text(json.dumps(t))
        branches.append(dict(branch=name, status='graded', result_sha256=digest(t)))
    def J(t):
        return (float.fromhex(t['final']['metrics']['team_delay_vs'])-1000.) if 'metrics' in t['final'] else 0.
    r = dict(verification_id=vid, status='accepted', episode=dict(arm=arm, id='e'), t0=T0,
             window=dict(start_s=T0, end_s=T_END), branches=branches, chosen=0,
             J_policy_veh_s=J(pol), J_candidates_veh_s=[J(plan)], grade_veh_s=J(pol)-J(plan),
             chosen_plan=dict(id='p', members=['c1', 'c2']), timing=dict(finished_unix=1.))
    r = xg._seal(r, 'receipt_sha256')
    (root_dir/'verifications').mkdir(parents=True, exist_ok=True)
    (root_dir/'verifications'/(vid+'.json')).write_text(json.dumps(r))
    return r


def make_entry(root, vid, gain1, gain2, *, synergy1=None, synergy2=None, n_members=2,
               checks=None, per_vehicle=False, drop_metrics=False, seed=0):
    """gain_s: per-second (or per-vehicle) plan advantage arrays; synergy_s: group - best solo."""
    bank = rb.bank_root(root, 'B')
    rng = np.random.default_rng(seed)
    rows = []
    for k, (gain, root_dir) in enumerate(((gain1, bank/'sumo_verification'), (gain2, bank/'confirm_replication'))):
        base = 2.+rng.random(N)
        if per_vehicle:
            pv_pol = {f'v{i}': 50. for i in range(len(gain))}
            pv_plan = {f'v{i}': 50.-g for i, g in enumerate(gain)}
            pol, plan = tape(base, pv_pol), tape(base-np.sum(gain)/N, pv_plan)
        else:
            pol, plan = tape(base, drop_metrics=drop_metrics), tape(base-gain, drop_metrics=drop_metrics)
        rows.append((write_receipt(root_dir, vid, pol, plan), base, gain))
    first, rep = rows[0][0], rows[1][0]
    co = None
    if synergy1 is not None:
        co = dict(group_size=n_members, solo=[], complete=True, cooperative=True, cooperation_gain_veh_s=1.)
        for key, syn, (_, base, gain), d in (('solo', synergy1, rows[0], bank/'confirm_solo'),
                                             ('replication_solo', synergy2, rows[1], bank/'confirm_solo_replication')):
            if syn is None:
                continue
            co[key] = []
            for m in range(n_members):
                svid = f'{vid}__solo_c{m}_{key}'
                # member m solo: worse than the group by `syn` (best member) or by 2*syn+1 (others)
                worse = syn if m == 0 else np.abs(syn)*2+1.
                r = write_receipt(d, svid, tape(base), tape(base-gain+worse))
                co[key].append(dict(member=f'c{m}', verification_id=svid, status='accepted',
                                    receipt_sha256=r['receipt_sha256']))
    c = dict(verification_id=vid, t0=T0, policy_version='unknown', admitted=True,
             checks=checks or dict(replication=True, no_later_harm=True, safety=True),
             replication=dict(verification_id=vid, status='accepted', receipt_sha256=rep['receipt_sha256']),
             first_seed_advantage_veh_s=first['grade_veh_s'], replication_advantage_veh_s=rep['grade_veh_s'],
             mean_replicated_advantage_veh_s=(first['grade_veh_s']+rep['grade_veh_s'])/2)
    if co is not None:
        c['cooperation'] = co
    (bank/'confirmations').mkdir(parents=True, exist_ok=True)
    (bank/'confirmations'/(vid+'.json')).write_text(json.dumps(xg._seal(c, 'confirmation_sha256')))
    return c


@pytest.fixture
def v2(monkeypatch):
    monkeypatch.setattr(rb, 'confirmations', rb.confirmations)   # restored after each test
    bs._CACHE.clear()
    bs.install(rb)
    return rb.confirmations


def noisy(mean, sd, seed):
    x = np.random.default_rng(seed).normal(0., sd, N)
    return x-x.mean()+mean


def test_zero_gain_dropped(tmp_path, v2):
    make_entry(tmp_path, 'zero', np.zeros(N), np.zeros(N))
    c = v2(tmp_path, 'B')['zero']
    assert c['legacy_admitted'] and not c['admitted']
    assert 'not positive' in c['bank_rule_v2']['reason']


def test_noisy_dropped_and_clear_positive_kept(tmp_path, v2):
    make_entry(tmp_path, 'noisy', noisy(.05, 5., 1), noisy(.05, 5., 2))
    make_entry(tmp_path, 'clear', noisy(2., .5, 3), noisy(1.5, .5, 4))
    conf = v2(tmp_path, 'B')
    n, c = conf['noisy']['bank_rule_v2'], conf['clear']['bank_rule_v2']
    assert not conf['noisy']['admitted'] and 'lower bound' in n['reason']
    assert all(m > 0 for m in n['per_seed_means_veh_s']) and n['admission']['lower95_one_sided_veh_s'] < 0
    assert conf['clear']['admitted'] and c['admission']['lower95_one_sided_veh_s'] > 0
    assert c['n_units'] == [N, N] and c['admission']['unit_source'] == bs.UNIT_PER_STEP
    assert c['admission']['B'] == 2000 and c['admission']['rng_seed'] == bs.BOOT_RNG_SEED
    assert abs(c['admission']['pooled_total_veh_s']-(2.*N+1.5*N)) < 1e-6   # exact grades reconcile


def test_one_seed_negative_dropped(tmp_path, v2):
    make_entry(tmp_path, 'oneneg', noisy(5., .5, 5), noisy(-.1, .2, 6))
    c = v2(tmp_path, 'B')['oneneg']
    assert c['bank_rule_v2']['admission']['lower95_one_sided_veh_s'] > 0   # pooled alone would admit
    assert not c['admitted'] and 'every seed' in c['bank_rule_v2']['reason']


def test_later_harm_and_safety_retained(tmp_path, v2):
    make_entry(tmp_path, 'harm', noisy(2., .5, 7), noisy(2., .5, 8),
               checks=dict(replication=True, no_later_harm=False, safety=True))
    make_entry(tmp_path, 'unsafe', noisy(2., .5, 9), noisy(2., .5, 10),
               checks=dict(replication=True, no_later_harm=True, safety=False))
    conf = v2(tmp_path, 'B')
    assert not conf['harm']['admitted'] and 'later-harm' in conf['harm']['bank_rule_v2']['reason']
    assert not conf['unsafe']['admitted'] and 'safety' in conf['unsafe']['bank_rule_v2']['reason']


def test_synergy_near_zero_not_cooperative(tmp_path, v2):
    make_entry(tmp_path, 'syn0', noisy(2., .5, 11), noisy(2., .5, 12),
               synergy1=noisy(0., 1., 13), synergy2=noisy(0., 1., 14))
    c = v2(tmp_path, 'B')['syn0']
    assert c['admitted'] and c['cooperation']['cooperative'] is False
    assert c['cooperation']['cooperative_status'] == 'not_cooperative'
    assert c['cooperation']['legacy_cooperative'] is True


def test_replicated_synergy_cooperative(tmp_path, v2):
    make_entry(tmp_path, 'synrep', noisy(5., .5, 15), noisy(5., .5, 16),
               synergy1=noisy(3., .5, 17), synergy2=noisy(2., .5, 18), n_members=3)
    co = v2(tmp_path, 'B')['synrep']['cooperation']
    assert co['cooperative'] is True and co['cooperative_status'] == 'replicated'
    assert [b['member'] for b in co['synergy_v2']['best_solo_per_seed']] == ['c0', 'c0']
    assert co['synergy_v2']['lower95_one_sided_veh_s'] > 0 and not co['needs_second_seed_solos']


def test_single_seed_synergy_path(tmp_path, v2):
    make_entry(tmp_path, 'syn1', noisy(5., .5, 19), noisy(5., .5, 20), synergy1=noisy(3., .5, 21))
    make_entry(tmp_path, 'syn1neg', noisy(5., .5, 22), noisy(5., .5, 23), synergy1=noisy(-1., .5, 24))
    conf = v2(tmp_path, 'B')
    co = conf['syn1']['cooperation']
    assert conf['syn1']['admitted'] and co['cooperative'] is False
    assert co['cooperative_status'] == 'single_seed' and co['needs_second_seed_solos']
    assert conf['syn1neg']['cooperation']['cooperative_status'] == 'not_cooperative'
    rep = bs.report(rb, tmp_path, 'B')
    assert rep['needs_second_seed_solos'] == ['syn1'] and rep['single_seed'] == 1
    assert rep['cooperative_replicated'] == 0 and rep['admitted'] == 2


def test_boot_merge_checks_only_boot_schema_and_allows_missing_seed2_solos(tmp_path, v2):
    c=make_entry(tmp_path,'merge',noisy(5.,.1,50),noisy(5.,.1,51),synergy1=noisy(3.,.1,52))
    assert 'replication_cooperation_gain_veh_s' not in c['cooperation']
    bs.check_keys([c],bank_rule='boot_v2')
    assert v2(tmp_path,'B')['merge']['cooperation']['cooperative_status']=='single_seed'
    c['checks'].pop('safety')
    with pytest.raises(KeyError,match='checks.safety'):bs.check_keys([c],bank_rule='boot_v2')


def test_per_vehicle_units_used_when_stored(tmp_path, v2):
    g = np.concatenate([np.full(40, 3.), np.zeros(10)])
    make_entry(tmp_path, 'pv', g, g, per_vehicle=True)
    a = v2(tmp_path, 'B')['pv']['bank_rule_v2']
    assert a['admission']['unit_source'] == bs.UNIT_PER_VEHICLE and a['n_units'] == [50, 50]
    assert v2(tmp_path, 'B')['pv']['admitted']


def test_missing_keys_raise_and_partial_missing_rejects_one(tmp_path, v2):
    make_entry(tmp_path, 'nodata', noisy(2., .5, 25), noisy(2., .5, 26), drop_metrics=True)
    with pytest.raises(bs.BankDataMissing):
        v2(tmp_path, 'B')
    make_entry(tmp_path, 'good', noisy(2., .5, 27), noisy(2., .5, 28))
    conf = v2(tmp_path, 'B')
    assert conf['good']['admitted'] and not conf['nodata']['admitted']
    assert 'per-unit data missing' in conf['nodata']['bank_rule_v2']['reason']


def test_legacy_unchanged_when_not_installed(tmp_path, monkeypatch):
    make_entry(tmp_path, 'zero', np.zeros(N), np.zeros(N))
    legacy = rb.confirmations(tmp_path, 'B')['zero']
    assert legacy['admitted'] is True and 'bank_rule_v2' not in legacy
    monkeypatch.setattr(rb, 'confirmations', rb.confirmations)
    bs.install(rb, bank_rule='t1727')
    assert rb.confirmations.statistical_1727 and 'replicated_advantage_ci95' in rb.confirmations(tmp_path, 'B')['zero']
    with pytest.raises(ValueError):
        bs.install(rb)          # never stack two rules in one process
    with pytest.raises(ValueError):
        bs.install(rb, bank_rule='nope')


def test_bootstrap_deterministic_and_block_mode():
    x = [noisy(1., 3., 30), noisy(1., 3., 31)]
    a, b = bs.paired_bootstrap(x, block=30), bs.paired_bootstrap(x, block=30)
    assert a == b and a['per_seed'][0]['block_len'] == 30
    zero = bs.paired_bootstrap([np.zeros(10)])
    assert zero['lower95_one_sided_veh_s'] == 0.


def test_legacy_common_top8_window_is_clipped_to_recorded_solo_window():
    group=dict(start_s=300,end_s=720,tail_s=120)
    solo=dict(start_s=300,end_s=570,tail_s=120)
    assert bs.paired_window(group,solo)['end_s']==570
    assert group['end_s']==720
    with pytest.raises(ValueError,match='start times'):
        bs.paired_window(group,dict(solo,start_s=301))
