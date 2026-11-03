"""CPU tests: verify_mode='all8_both_seeds' (owner 2026-11-03 22:05). No SUMO: a deterministic
per-second plant writes exact team_delay_vs fingerprints whose effect depends on (plan, seed)."""
import gzip
import json
import math
import time

import pytest
import torch

from tools.coopmwm import replay_bank as rb
from tools.coopmwm import sumo_verify as sv
from tools.coopmwm import verify_perseed as vp
from tools.coopmwm import exact_grade as xg
from tools.coopmwm.protocol import digest
from tools.coopmwm.storage import ledger
from tools.coopnative.storage import canonical
from tools.coopmwm.test_replay_bank_confirm_cpu import TapeRunner, bank_budget, fp, plan, snapshot  # noqa: F401

PHASE = 'bank_verification_ext'


def seed_of(snap):
    return 2 if 'replication_salt' in snap['rng'] else 1


class SeedPlant:
    """Per-second plant. gains[(plan_id, seed)] = delay reduction per controlled second (veh-s/s);
    solo plans ('<id>__solo_<m>') use gains[(id, seed, 'solo')] (default 0). Writes RESULT.json
    (per-second fingerprints + plan steps), the observable trace and the branch spec file."""

    def __init__(self, work, gains):
        self.work, self.gains, self.admits = work, gains, []

    def admit(self, specs):
        self.admits.append([s['branch'] for s in specs]);return None

    def rate(self, spec, snap, t):
        p = spec['plan'];k = seed_of(snap)
        noise = 0.05*math.sin(1.3*t+k)
        base = 2.0+0.4*math.sin(0.21*t+k)+noise
        if p is None or t >= spec['window']['start_s']+int(p['duration_s']):
            return base
        if '__solo_' in p['id']:
            g = self.gains.get((p['id'].split('__solo_')[0], k, 'solo'), 0.)
        else:
            g = self.gains.get((p['id'], k), -0.3)
        return base-g+0.02*math.cos(0.7*t+len(p['id']))

    def run(self, spec, snap):
        w = snap['episode']['world']['seed'];t0, t_end = spec['window']['start_s'], spec['window']['end_s']
        bdir = self.work/'branches'/spec['verification_id'];d = bdir/spec['branch'];d.mkdir(parents=True, exist_ok=True)
        sp = bdir/f'{spec["branch"]}.spec.json'
        if not sp.exists():
            sp.write_text(json.dumps(spec))
        row = dict(time_s=t0, vehicles=[dict(lane='m1', s_m=10., v_mps=8., length_m=5.),
                                        dict(lane='m1', s_m=0., v_mps=10., length_m=5.)])
        with gzip.open(d/'observable.jsonl.gz', 'wt', encoding='utf-8') as f:
            f.write(canonical(row)+'\n')
        prefixes = {str(t): fp(t, 3.*w+0.5*t*t/100.) for t in range(300, t0+1)}
        cum = 3.*w+0.5*t0*t0/100.
        for t in range(t0, t_end):
            cum += self.rate(spec, snap, t)
            if t+1 < t_end:
                prefixes[str(t+1)] = fp(t+1, cum)
        final = fp(t_end, cum)
        p = spec['plan']
        steps = ([dict(time_s=t, sample=dict(plan_executions=[dict(plan_id=p['id'], proposal_members=list(p['members']))],
                                             absolute_accels={v: 0.5 for v in p['members']}))
                  for t in range(t0, t0+int(p['duration_s']))] if p is not None else [])
        tape = dict(prefixes=prefixes, final=final, steps=steps)
        from tools.coopmwm.accel_storage import write as awrite
        awrite(d/'RESULT.json', tape)
        return dict(prefixes=prefixes, final=final, actual_sim_s=t_end-300, result_sha256=digest(tape))


def setup(tmp_path, gains, k=8, t0=600):
    root = tmp_path;bank = rb.bank_root(root, 'B');snap = snapshot(t0)
    plans = [plan(f'p{i+1}', 0.1) for i in range(k)]
    plant = SeedPlant(bank/'sumo_verification', gains)
    r = sv.verify(snap, plans, plant, sv.local_charge(tmp_path/'first'), 'wm', root=bank/'sumo_verification')
    (bank/'snapshots').mkdir(parents=True, exist_ok=True)
    torch.save(snap, bank/'snapshots'/'s1.pt')
    (bank/'decisions').mkdir(exist_ok=True)
    (bank/'decisions'/'s1.json').write_text(json.dumps(dict(state_id='s1', verification_id=r['verification_id'],
                                                            status=r['status'], policy_update=5)))
    plants = []

    def runner_for(work):
        p = SeedPlant(work, gains);plants.append(p);return p
    return root, snap, plans, r, runner_for, plants


def run_all8(root, runner_for, health=None):
    calls = []
    out = rb.confirm_pending(root, 'B', runner_for, str(root), PHASE, verify_mode=vp.ALL8)
    return out, calls


def record(root):
    recs = vp.records(root, 'B');assert len(recs) == 1;return next(iter(recs.values()))


def ex(c, i, k):
    return next(e for e in c['per_seed_v1']['examples'] if e['candidate_index'] == i and e['seed_index'] == k)


# ---------------------------------------------------------------- tests

def test_off_path_identical(tmp_path, monkeypatch):
    """verify_mode None / 'legacy' / absent -> the legacy confirmation path, identical bytes."""
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    outs = []
    for name, kw in (('absent', {}), ('none', dict(verify_mode=None)), ('legacy', dict(verify_mode='legacy'))):
        root = tmp_path/name;root.mkdir()
        _budget(root)
        bank = rb.bank_root(root, 'B');snap = snapshot(420)
        runner = TapeRunner(bank/'sumo_verification')
        r = sv.verify(snap, [plan('a', 0.5), plan('b', 0.1)], runner, sv.local_charge(root/'c'), 'wm',
                      root=bank/'sumo_verification')
        (bank/'snapshots').mkdir();torch.save(snap, bank/'snapshots'/'s1.pt')
        (bank/'decisions').mkdir()
        (bank/'decisions'/'s1.json').write_text(json.dumps(dict(state_id='s1', verification_id=r['verification_id'],
                                                                status=r['status'], policy_update=5)))
        res = rb.confirm_pending(root, 'B', lambda w: TapeRunner(w), str(root), PHASE, **kw)
        assert res == [True]
        assert not (bank/vp.RECORDS).exists() and not (bank/vp.SEED2_ROOT).exists()
        conf = (bank/'confirmations'/(r['verification_id']+'.json')).read_bytes()
        rep = sorted((bank/'confirm_replication'/'verifications').glob('*.json'))
        outs.append((conf, [p.read_bytes() for p in rep],
                     [{k: v for k, v in x.items() if k != 'id'} for x in ledger(root)]))
    assert outs[0] == outs[1] == outs[2]
    assert vp.mode(None) == vp.mode('') == vp.LEGACY
    with pytest.raises(ValueError):
        vp.mode('all8')


def _budget(root):
    from tools.coopmwm.warm_protocol import budget
    b = budget();b['total'] += 2600000;b['allocated'] += 2600000
    b['phases'] = dict(b['phases'], bank_verification_ext=2600000)
    (root/'B_REPLAY_PHASE_RECONCILIATION_0627.json').write_text(json.dumps(
        dict(budget=b, amendment_sha256='test-bank-budget')))
    (root/'AUTHORITY.md').write_text('test-bank-budget')


def test_both_seeds_get_all8(tmp_path):
    gains = {('p1', 1): 0.6, ('p1', 2): 0.6}
    root, snap, plans, r, runner_for, plants = setup(tmp_path, gains)
    assert r['status'] == 'accepted'
    run_all8(root, runner_for)
    c = record(root)
    s2 = json.loads((rb.bank_root(root, 'B')/vp.SEED2_ROOT/'verifications'/(c['seed2']['verification_id']+'.json')).read_text())
    assert s2['k'] == 8 and len(s2['branches']) == 9 and s2['status'] in ('accepted', 'rejected')
    assert s2['candidate_plan_sha256'] == r['candidate_plan_sha256']
    assert s2['snapshot_sha256'] == sv.snapshot_sha(rb.replicate(snap, r['verification_id']))
    assert s2['snapshot_sha256'] != r['snapshot_sha256']
    assert len(c['per_seed_v1']['examples']) == 16
    assert {(e['seed_index'], e['candidate_index']) for e in c['per_seed_v1']['examples']} == \
        {(k, i) for k in (1, 2) for i in range(8)}
    assert c['winners'] == {'1': ['p1'], '2': ['p1']}
    assert all(e['status'] == 'rejected' for e in c['per_seed_v1']['examples'] if e['plan_id'] != 'p1')


def test_seed2_only_winner_admitted_from_seed2_tape(tmp_path):
    """First seed rejected (no candidate helps); p5 helps only on seed 2 -> admitted from seed 2."""
    gains = {('p5', 2): 0.7}
    root, snap, plans, r, runner_for, plants = setup(tmp_path, gains)
    assert r['status'] == 'rejected'
    assert [x['verification_id'] for x in vp.pending(root, 'B')] == [r['verification_id']]
    res = run_all8(root, runner_for)[0]
    assert res == [True]
    c = record(root)
    assert c['first_seed_status'] == 'rejected' and c['winners'] == {'1': [], '2': ['p5']}
    e = ex(c, 4, 2)
    # MAIN 2026-11-03 (launch-final): seed-2 later harm is MEASURED on seed 2's own CRN
    assert e['admitted'] and e['later_harm'] == 'pass' and e['plan']['id'] == 'p5'
    assert e['safety_source'] == 'extended_window_seed2' and c['extended_seed2']['candidate_indices'] == [4]
    assert e['tape']['branches_dir'].replace('\\', '/').endswith(vp.SEED2_ROOT+'/branches')
    assert e['tape']['intent_entry'] == f'{r["verification_id"]}/seed2/cand5'
    assert not ex(c, 4, 1)['admitted']
    rows = vp.bank_examples(root, 'B')
    assert [x['example_id'] for x in rows] == [f'{r["verification_id"]}#c5#s2']
    tape, entry = vp.imitation_entry(rows[0], c['t0'], 1.)
    pairs = xg.imitation_pairs(tape, entry)
    assert len(pairs) == 120
    # the tape IS the seed-2 branch tape of p5 (not the first-seed one)
    s2 = json.loads((rb.bank_root(root, 'B')/vp.SEED2_ROOT/'verifications'/(r['verification_id']+'.json')).read_text())
    assert digest(tape) == s2['branches'][5]['result_sha256'] != r['branches'][5]['result_sha256']
    assert vp.pending(root, 'B') == []


def test_different_winners_per_seed_and_within_seed_cooperation(tmp_path):
    gains = {('p1', 1): 0.6, ('p2', 1): 0.5, ('p3', 2): 0.6,
             ('p1', 1, 'solo'): 0.1,            # p1 seed 1: group beats every solo -> cooperative
             ('p2', 1, 'solo'): 0.5,            # p2 seed 1: a solo does as well -> not cooperative
             ('p3', 2, 'solo'): 0.1}            # p3 seed 2: cooperative on seed 2
    root, snap, plans, r, runner_for, plants = setup(tmp_path, gains)
    run_all8(root, runner_for)
    c = record(root)
    assert c['winners'] == {'1': ['p1', 'p2'], '2': ['p3']} and c['distinct_winners_across_seeds']
    assert ex(c, 0, 1)['later_harm'] == 'pass' and ex(c, 0, 1)['safety_source'] == 'extended_window_seed1'
    assert ex(c, 0, 1)['cooperative'] is True and ex(c, 0, 1)['cooperative_status'] == 'cooperative'
    assert ex(c, 1, 1)['cooperative'] is False and ex(c, 1, 1)['cooperative_status'] == 'not_cooperative'
    assert ex(c, 2, 2)['cooperative'] is True and ex(c, 2, 2)['solo_policy_matches_reference']
    assert ex(c, 0, 2)['cooperative_status'] == 'not_admitted'
    assert c['per_seed_v1']['cooperative_seeds'] == [1, 2] and c['per_seed_v1']['replicated_plans'] == []
    kinds = sorted(x['kind'] for x in c['charges'])
    assert kinds == sorted([vp.KIND_SEED2, vp.KIND_EXT, vp.KIND_EXT2, vp.KIND_SOLO[1], vp.KIND_SOLO[1],
                            vp.KIND_SOLO[2]])
    bank = rb.bank_root(root, 'B')
    assert len(list((bank/vp.SOLO_ROOTS[1]/'verifications').glob('*.json'))) == 2
    assert len(list((bank/vp.SOLO_ROOTS[2]/'verifications').glob('*.json'))) == 1


def test_charges_recorded_through_existing_hooks(tmp_path, monkeypatch):
    gains = {('p1', 1): 0.6, ('p3', 2): 0.6}
    root, snap, plans, r, runner_for, plants = setup(tmp_path, gains)
    health = []
    import tools.coopmwm.warm_health as wh
    monkeypatch.setattr(wh, 'assert_running', lambda rt, arm: health.append((str(rt), arm)))
    seen = []
    original = rb._confirm_charge

    def spy(run_root, phase, kind):
        inner = original(run_root, phase, kind)
        def charge(vid, specs):
            seen.append(kind);return inner(vid, specs)
        return charge
    monkeypatch.setattr(rb, '_confirm_charge', spy)
    run_all8(root, runner_for)
    c = record(root)
    rows = [x for x in ledger(root) if x['phase'] == PHASE]
    by_kind = {}
    for x in rows:
        k = next(k for k in (vp.KIND_SEED2, vp.KIND_EXT, vp.KIND_EXT2, vp.KIND_SOLO[1], vp.KIND_SOLO[2])
                 if x['id'].startswith(k+'_'))
        by_kind[k] = by_kind.get(k, 0)+x['sim_s']
    assert sum(by_kind.values()) == c['charged_sim_s'] > 0
    assert sorted(seen) == sorted(x['kind'] for x in c['charges'])
    assert len(health) == len(c['charges']) == 5               # assert_running before every verification
    assert sum(len(a) for p in plants for a in p.admits) == sum(x['n_branches'] for x in c['charges'])
    q = vp.quote_decision(600, [120]*8, [['c1', 'c2']]*8, {1: [0], 2: [2]})
    assert q['all8_confirmation_sim_s'] == c['charged_sim_s']   # quote == actual charges


def test_builder_precharge_ids_match_live_hook(tmp_path):
    from tools.coopmwm import bank_builder as bb
    gains = {('p1', 1): 0.6}
    root, snap, plans, r, runner_for, plants = setup(tmp_path, gains)
    jobs = bb.quoted_jobs_all8(snap, plans, r['verification_id'], sv.build_branches, rb.replicate)
    entries = {j['id']: dict(j, phase=bb.PHASE) for j in jobs}
    hook = bb.all8_charge(rb._confirm_charge, entries)(str(root), bb.PHASE, vp.KIND_SEED2)
    specs = sv.build_branches(rb.replicate(snap, r['verification_id']), plans)
    hook(specs[0]['verification_id'], specs)        # every seed-2 branch is covered by the precharge
    assert len(jobs) == 9 and sum(j['sim_s'] for j in jobs) == vp.quote_decision(600, [120]*8, [['a']]*8)['all8_seed2_sim_s']


def test_per_seed_v1_bank_pairs_integration(tmp_path):
    """Stub of the per_seed_v1 bank_pairs example loop (c4/bank-perseed, uncommitted at build time):
    every admitted example -> its own seed's tape -> imitation pairs with its own plan."""
    gains = {('p1', 1): 0.6, ('p3', 2): 0.6, ('p1', 1, 'solo'): 0.1}
    root, snap, plans, r, runner_for, plants = setup(tmp_path, gains)
    run_all8(root, runner_for)
    conf = vp.overlay({}, root, 'B')
    c = conf[r['verification_id']]
    assert c['bank_rule'] == 'per_seed_v1' and c['admitted'] and c['mean_replicated_advantage_veh_s'] > 0
    n = 0
    for e in c['per_seed_v1']['examples']:
        for key in ('seed_index', 'example_id', 'admitted', 'status', 'reason', 'cooperative', 'cooperative_status'):
            assert key in e
        if not e['admitted']:
            continue
        t = e['tape']
        for key in ('branches_dir', 'verification_id', 'branch', 'window', 'intent_entry'):
            assert key in t
        assert e['advantage_veh_s'] > 0 and e['lower95_one_sided_veh_s'] > 0 and 'later_harm' in e
        tape, entry = vp.imitation_entry(e, c['t0'], xg.weight(e['advantage_veh_s'], None, xg.WEIGHT_CLIP))
        pairs = xg.imitation_pairs(tape, entry)
        assert len(pairs) == e['duration_s'] and all(p[0]['behaviour']['proposal_members'] == e['members'] for p in pairs)
        n += 1
    assert n == 2 and {e['seed_index'] for e in vp.bank_examples(root, 'B')} == {1, 2}


def test_quote():
    q = vp.standard_quote()['t0_600']
    assert q['first_seed_negative_seed2_none']['multiplier_total'] == 2.0
    assert q['one_winner_each_seed']['all8_total_sim_s'] > q['one_winner_each_seed']['legacy_total_sim_s']


# ---------------------------------------------------------------- common-window cooperation label

def _tape(t0, t_end, rate):
    cum, prefixes = 100., {}
    for t in range(t0, t_end):
        prefixes[str(t)] = fp(t, cum)
        cum += rate(t)
    return dict(prefixes=prefixes, final=fp(t_end, cum), steps=[]), cum


def _base(t):
    return 2.0+0.4*math.sin(0.21*t)


def _solo_fixture(tmp_path, solo_end, group_end=720, t0=600, solo_ref_bump=None):
    """Group policy/plan tapes over [t0, group_end]; one solo verification over [t0, solo_end]."""
    from tools.coopmwm.accel_storage import write as awrite
    ref, J_ref = _tape(t0, group_end, _base)
    plan, _ = _tape(t0, group_end, lambda t: _base(t)-0.6)
    bump = (lambda t: 0.5 if (solo_ref_bump is not None and t == solo_ref_bump) else 0.)
    sref, J_sp = _tape(t0, solo_end, lambda t: _base(t)+bump(t))
    ssolo, J_ss = _tape(t0, solo_end, lambda t: _base(t)-0.1)
    vid = 'solo_v'
    sworse, J_sw = _tape(t0, solo_end, lambda t: _base(t)-0.02)
    for b, tape in (('policy', sref), ('cand1', ssolo), ('cand2', sworse)):
        d = tmp_path/'branches'/vid/b;d.mkdir(parents=True);awrite(d/'RESULT.json', tape)
    solo = dict(verification_id=vid, receipt_sha256='x', status='accepted', J_policy_veh_s=J_sp-100.,
                J_candidates_veh_s=[J_ss-100., J_sw-100.], window=dict(start_s=t0, end_s=solo_end),
                branches=[dict(branch='policy'), dict(branch='cand1'), dict(branch='cand2')])
    window = dict(start_s=t0, end_s=group_end)
    return plan, ref, window, solo, J_ref-100.


def test_shorter_solo_window_identical_reference_is_cooperative_on_common_window(tmp_path):
    plan, ref, window, solo, J_ref = _solo_fixture(tmp_path, solo_end=660)
    out = vp._cooperation(1, dict(members=['a', 'b']), plan, window, solo, tmp_path, J_ref, True, reference_tape=ref)
    assert solo['J_policy_veh_s'] != J_ref
    assert out['cooperative'] is True and out['cooperative_status'] == 'cooperative'
    assert out['solo_policy_matches_reference'] is False
    assert out['solo_policy_matches_reference_on_common_window'] is True
    assert out['synergy_window_used']['end_s'] == 660 and out['synergy']['paired_window']['used']['end_s'] == 660


def test_shorter_solo_window_differing_reference_is_mismatch(tmp_path):
    plan, ref, window, solo, J_ref = _solo_fixture(tmp_path, solo_end=660, solo_ref_bump=630)
    out = vp._cooperation(1, dict(members=['a', 'b']), plan, window, solo, tmp_path, J_ref, True, reference_tape=ref)
    assert out['cooperative'] is False and out['cooperative_status'] == 'reference_mismatch'
    assert out['solo_policy_matches_reference_on_common_window'] is False
    # a difference AFTER the common window is irrelevant
    plan, ref, window, solo, J_ref = _solo_fixture(tmp_path/'late', solo_end=660)
    ref['prefixes']['700'] = fp(700, 1.)
    out = vp._cooperation(1, dict(members=['a', 'b']), plan, window, solo, tmp_path/'late', J_ref, True,
                          reference_tape=ref)
    assert out['cooperative_status'] == 'cooperative'
    # no reference tape available -> conservative mismatch
    plan, ref, window, solo, J_ref = _solo_fixture(tmp_path/'none', solo_end=660)
    out = vp._cooperation(1, dict(members=['a', 'b']), plan, window, solo, tmp_path/'none', J_ref, True)
    assert out['cooperative_status'] == 'reference_mismatch'


def test_equal_windows_unchanged_keys(tmp_path):
    plan, ref, window, solo, J_ref = _solo_fixture(tmp_path, solo_end=720)
    assert solo['J_policy_veh_s'] == J_ref
    a = vp._cooperation(1, dict(members=['a', 'b']), plan, window, solo, tmp_path, J_ref, True)
    b = vp._cooperation(1, dict(members=['a', 'b']), plan, window, solo, tmp_path, J_ref, True, reference_tape=ref)
    assert a == b and a['solo_policy_matches_reference'] is True
    assert set(a) == {'cooperative', 'cooperative_status', 'statistical_reason', 'synergy', 'solo_advantages_veh_s',
                      'solo_verification_id', 'solo_receipt_sha256', 'solo_policy_matches_reference'}
    assert a['cooperative_status'] == 'cooperative' and 'common window' not in a['statistical_reason']


def test_reader_relabels_stored_reference_mismatch(tmp_path):
    from tools.coopmwm.accel_storage import write as awrite
    bank = tmp_path/'bank';t0 = 600
    plan, ref, window, solo, J_ref = _solo_fixture(bank/vp.SOLO_ROOTS[1], solo_end=660)
    sroot = bank/vp.SOLO_ROOTS[1];(sroot/'verifications').mkdir()
    solo.pop('receipt_sha256');solo = xg._seal(solo, 'receipt_sha256')
    (sroot/'verifications'/'solo_v.json').write_text(json.dumps(solo))
    gdir = tmp_path/'first';gv = 'grp_v'
    for b, tape in (('policy', ref), ('cand1', plan)):
        d = gdir/'branches'/gv/b;d.mkdir(parents=True);awrite(d/'RESULT.json', tape)
    grec = xg._seal(dict(verification_id=gv, status='accepted', J_policy_veh_s=J_ref,
                         branches=[dict(branch='policy'), dict(branch='cand1')], window=window), 'receipt_sha256')
    (gdir/'verifications').mkdir();(gdir/'verifications'/(gv+'.json')).write_text(json.dumps(grec))
    stale = dict(seed_index=1, example_id=gv+'#s1', plan_id='p1', candidate_index=0, admitted=True, status='admitted',
                 cooperative=False, cooperative_status='reference_mismatch', solo_verification_id='solo_v',
                 solo_receipt_sha256=solo['receipt_sha256'],
                 tape=dict(branches_dir=str(gdir/'branches'), verification_id=gv, branch=dict(branch='cand1'),
                           window=window))
    rec = dict(verification_id=gv, candidates=[dict(index=0, plan_id='p1', members=['a', 'b'])],
               per_seed_v1=dict(examples=[stale], cooperative_seeds=[], replicated=False, replicated_plans=[]),
               cooperation=dict(cooperative=False, cooperative_seeds=[], cooperative_examples=0, replicated=False))
    out = vp.relabel_reference_mismatch(rec, bank)
    e = out['per_seed_v1']['examples'][0]
    assert e['cooperative'] is True and e['cooperative_status'] == 'cooperative' and e['reader_relabel']
    assert out['per_seed_v1']['cooperative_seeds'] == [1] and out['cooperation']['cooperative'] is True
    assert rec['per_seed_v1']['examples'][0]['cooperative_status'] == 'reference_mismatch'   # input untouched
    # nothing to relabel -> same object; unreadable solo -> stored label kept
    ok = dict(rec, per_seed_v1=dict(rec['per_seed_v1'], examples=[dict(stale, cooperative_status='not_cooperative')]))
    assert vp.relabel_reference_mismatch(ok, bank) is ok
    assert vp.relabel_reference_mismatch(rec, tmp_path/'nowhere') is rec
