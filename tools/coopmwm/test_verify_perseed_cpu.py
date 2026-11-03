"""CPU tests: verify_mode='all8_both_seeds' (owner 2026-11-03 22:05). No SUMO: a deterministic
per-second plant writes exact team_delay_vs fingerprints whose effect depends on (plan, seed)."""
import gzip
import json
import math
import re
import time

import pytest
import torch

from tools.coopmwm import bank_statistics as bs
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
    """CRN seed index of a snapshot: 1 = first seed, 2 = replicate(salt=vid),
    k >= 3 = extra seed (salt '<vid>#seed<k>' ['#x<n>' only in the eval-collision case])."""
    salt = snap['rng'].get('replication_salt')
    if salt is None:
        return 1
    if '#seed' in salt:
        return int(salt.split('#seed', 1)[1].split('#', 1)[0])
    return 2


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


def setup(tmp_path, gains, k=8, t0=600, arm='B'):
    root = tmp_path;bank = rb.bank_root(root, arm);snap = snapshot(t0)
    snap['episode'] = dict(snap['episode'], id=f'warm_u005_{arm}_7', arm=arm)
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


class ExtraPlant(SeedPlant):
    """SeedPlant whose per-(plan, seed) ``later_harm`` veh-s/s loss starts only AFTER the
    ordinary window (t0+D+TAIL_S): only an extended-window verification can see it."""

    def __init__(self, work, gains, later_harm=None):
        super().__init__(work, gains)
        self.later_harm = later_harm or {}

    def rate(self, spec, snap, t):
        r = super().rate(spec, snap, t)
        p = spec['plan']
        if p is not None and '__solo_' not in p['id'] and t >= spec['window']['start_s']+int(p['duration_s'])+sv.TAIL_S:
            r += self.later_harm.get((p['id'], seed_of(snap)), 0.)
        return r


def plants_of(plant_cls, gains, later_harm=None):
    made = []
    def runner_for(work):
        kw = dict(later_harm=later_harm) if plant_cls is ExtraPlant else {}
        p = plant_cls(work, gains, **kw);made.append(p);return p
    return runner_for, made


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


def test_builder_extras_group_precharge_all_seeds_one_transaction(tmp_path):
    """bank_builder.extras_charge: the FIRST extra-seed group charge pre-reserves EVERY extra
    seed's group verification (cooperative candidates + policy) in ONE transaction; the later
    extra-seed charges only validate their precharged rows (no new ledger rows); extended/solo
    kinds are not intercepted and keep the live hook."""
    from tools.coopmwm import bank_builder as bb
    from tools.coopmwm.accel_storage import reserve
    from tools.coopmwm.warm_protocol import install_accounting
    gains = {('p1', 1): 0.6, ('p1', 2): 0.6, ('p1', 1, 'solo'): 0.1, ('p1', 2, 'solo'): 0.1,
             ('p1', 3): 0.6, ('p1', 3, 'solo'): 0.1}
    root, snap, plans, r, _, _ = setup(tmp_path, gains)
    _budget(root)
    install_accounting(root)
    vid = r['verification_id']
    reserve(root, bb.quoted_jobs_all8(snap, plans, vid, sv.build_branches, rb.replicate), bb.PHASE)
    hook = bb.extras_charge(bb.all8_charge(rb._confirm_charge, {j['id']: j for j in ledger(root)}),
                            snap, vid, 2)
    coop = [plans[0]]                                   # p1 is the only cooperative candidate
    specs3 = sv.build_branches(rb.replicate(snap, vp.extra_seed_salt(vid, 3)), coop)
    hook(str(root), bb.PHASE, 'confirm_seed3_all8')(specs3[0]['verification_id'], specs3)
    rows = [x for x in ledger(root) if x['id'].startswith(('confirm_seed3_all8_', 'confirm_seed4_all8_'))]
    assert len(rows) == 4                               # seeds 3 AND 4 were reserved by that ONE charge
    specs4 = sv.build_branches(rb.replicate(snap, vp.extra_seed_salt(vid, 4)), coop)
    hook(str(root), bb.PHASE, 'confirm_seed4_all8')(specs4[0]['verification_id'], specs4)
    assert len([x for x in ledger(root) if x['id'].startswith('confirm_seed4_all8_')]) == 2   # no new rows
    assert sum(j['sim_s'] for j in hook.extras_state['entries'].values()) == sum(x['sim_s'] for x in rows)
    assert sum(x['sim_s'] for x in rows) == 4*sv.branch_sim_s(min(600+120+sv.TAIL_S, 1200))
    # extended/solo kinds keep the live hook (each reserves its own kind-prefixed rows)
    spec = sv.build_branches(snap, [plans[0]])[0]
    hook(str(root), bb.PHASE, vp.KIND_EXT)(spec['verification_id'], [spec])
    assert any(x['id'].startswith(vp.KIND_EXT+'_') for x in ledger(root))
    assert not any(x['id'].startswith('confirm_seed5') for x in ledger(root))


def test_builder_extras_group_precharge_budget_block_is_all_or_nothing(tmp_path):
    """A budget shortfall at the first extra-seed group charge raises BudgetBlocked and leaves NO
    extra-seed rows in the ledger (never partial extras); nothing is cached, so a decision with
    a sufficient budget would reserve every extra seed's group instead."""
    from tools.coopmwm import bank_builder as bb
    from tools.coopmwm.sumo_verify import BudgetBlocked
    from tools.coopmwm.warm_protocol import budget as warm_budget, install_accounting
    gains = {('p1', 1): 0.6, ('p1', 2): 0.6, ('p1', 1, 'solo'): 0.1, ('p1', 2, 'solo'): 0.1,
             ('p1', 3): 0.6, ('p1', 3, 'solo'): 0.1}
    root, snap, plans, r, _, _ = setup(tmp_path, gains)
    b = warm_budget();b['total'] += 1000;b['allocated'] += 1000
    b['phases'] = dict(b['phases'], bank_verification_ext=1000)
    (root/'B_REPLAY_PHASE_RECONCILIATION_0627.json').write_text(json.dumps(
        dict(budget=b, amendment_sha256='test-bank-budget')))
    (root/'AUTHORITY.md').write_text('test-bank-budget')
    install_accounting(root)
    vid = r['verification_id']
    hook = bb.extras_charge(bb.all8_charge(rb._confirm_charge, {}), snap, vid, 2)
    specs3 = sv.build_branches(rb.replicate(snap, vp.extra_seed_salt(vid, 3)), [plans[0]])
    with pytest.raises(BudgetBlocked):
        hook(str(root), bb.PHASE, 'confirm_seed3_all8')(specs3[0]['verification_id'], specs3)
    assert not [x for x in ledger(root) if x['id'].startswith(('confirm_seed3_all8_', 'confirm_seed4_all8_'))]
    assert not hook.extras_state                        # all-or-nothing: nothing reserved, nothing cached


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


# ---------------------------------------------------------------- BUILD option extra_seeds

EXTRA_SEEDS4 = dict(extra_seeds=4)


def test_extra_seed_salts_deterministic_distinct_and_never_evaluation_seeds():
    """extra_seed_salt: pure deterministic function of (vid, k); distinct across seeds and vids;
    the replicate() generator seed it implies never lands in an evaluation seed range."""
    from tools.coopmwm.protocol import digest
    assert vp.extra_seed_indices(4) == [3, 4, 5, 6] and vp.extra_seed_indices(0) == []
    assert vp.extra_seed_indices(None) == []
    with pytest.raises(ValueError):
        vp.extra_seed_indices(-1)
    with pytest.raises(ValueError):
        vp.extra_seed_salt('v', 2)
    s3 = vp.extra_seed_salt('vid1', 3)
    assert s3 == vp.extra_seed_salt('vid1', 3) == 'vid1#seed3'      # deterministic
    assert len({vp.extra_seed_salt('vid1', k) for k in range(3, 30)}) == 27
    assert all(vp.extra_seed_salt('vid1', k) != 'vid1' for k in range(3, 8))   # seed-2 salt distinct
    assert vp.extra_seed_salt('vid1', 3) != vp.extra_seed_salt('vid2', 3)
    lo_hi = list(vp.EVAL_SEED_RANGES)
    assert lo_hi == [(962000, 962999), (964000, 964199), (977000, 977999)]

    def gen_seed(salt):
        return int(digest(dict(replication=salt))[:15], 16)
    for vid in ('a', 'vid1', 'x'*40):
        for k in range(3, 40):
            salt = vp.extra_seed_salt(vid, k)
            g = gen_seed(salt)
            assert not any(lo <= g <= hi for lo, hi in lo_hi), (vid, k, salt, g)
            assert rb.replicate(snapshot(420), salt)['rng']['replication_salt'] == salt
    states = {k: rb.replicate(snapshot(420), vp.extra_seed_salt('vid1', k))['generator_state'].numpy().tobytes()
              for k in range(3, 8)}
    assert len(set(states.values())) == 5                            # distinct CRN streams
    assert states[3] != rb.replicate(snapshot(420), 'vid1')['generator_state'].numpy().tobytes()


def test_extra_seeds_zero_is_byte_identical(tmp_path, monkeypatch):
    """extra_seeds absent == extra_seeds 0: byte-identical records, receipts and charges; no
    extra-seed verification roots are created and the record has no extra keys."""
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    gains = {('p1', 1): 0.6, ('p3', 2): 0.6, ('p1', 1, 'solo'): 0.1}     # cooperative example on seed 1
    outs = []
    for name, kw in (('absent', {}), ('zero', dict(extra_seeds=0))):
        root = tmp_path/name;root.mkdir()
        _budget(root)                       # sub-root budget, like test_off_path_identical
        _, _, _, r, runner_for, _ = setup(root, gains)
        rb.confirm_pending(root, 'B', runner_for, str(root), PHASE, verify_mode=vp.ALL8, **kw)
        bank = rb.bank_root(root, 'B')
        recs = sorted(p.relative_to(bank).as_posix()+':'+p.read_text(encoding='utf-8')
                      for p in bank.rglob('verifications/*.json'))
        ledger_rows = [{k: v for k, v in x.items() if k != 'id'} for x in ledger(root)]
        rec = json.loads((bank/vp.RECORDS/(r['verification_id']+'.json')).read_text(encoding='utf-8'))
        assert 'extra' not in rec and 'extra_seeds' not in rec
        assert not (bank/vp.seed_kinds(3)[0]).exists()
        # the record embeds the bank's absolute branches_dir/root (JSON-escaped) and its own
        # seal digest over that path; identical modulo both
        rec_text = (bank/vp.RECORDS/(r['verification_id']+'.json')).read_text(encoding='utf-8')
        rec_text = rec_text.replace(json.dumps(str(bank))[1:-1], '<BANK>')
        rec_text = re.sub(r'"confirmation_sha256":"[0-9a-f]{64}"', '"confirmation_sha256":"<SEAL>"', rec_text)
        outs.append((recs, ledger_rows, rec_text))
    assert outs[0] == outs[1]
    q_off = vp.quote_decision(600, [120]*8, [['c1', 'c2']]*8, {1: [0], 2: [0]})
    assert not any(k.startswith('extra') or k == 'cooperative' for k in q_off)


def test_extra_seeds_independent_admission_own_solos_and_charges(tmp_path, monkeypatch):
    """extra_seeds=4 (E2E): p1 is cooperative on seeds 1/2 -> verified on seeds 3..6. Seed 3
    admits (and is cooperative), seed 4 rejects on its own LB, seed 5 fails ITS OWN later-harm
    window, seed 6 admits but its solo does as well (not cooperative). Every extra seed is its
    own batched verification + extended window + solos, charged through the existing hooks;
    per_seed_v1 consumes the appended seed_index>=3 examples unchanged."""
    gains = {('p1', 1): 0.6, ('p1', 2): 0.6, ('p1', 1, 'solo'): 0.1, ('p1', 2, 'solo'): 0.1,
             ('p1', 3): 0.6, ('p1', 3, 'solo'): 0.1, ('p1', 5): 0.6, ('p1', 6): 0.6, ('p1', 6, 'solo'): 0.6,
             ('p3', 2): 0.6, ('p3', 2, 'solo'): 0.6}
    root, snap, plans, r, _, _ = setup(tmp_path, gains)
    runner_for, _ = plants_of(ExtraPlant, gains, {('p1', 5): 0.5})
    health, seen_kinds = [], []
    import tools.coopmwm.warm_health as wh
    monkeypatch.setattr(wh, 'assert_running', lambda rt, arm: health.append((str(rt), arm)))
    original = rb._confirm_charge

    def spy(run_root, phase, kind):
        inner = original(run_root, phase, kind)
        def charge(vid, specs):
            seen_kinds.append(kind);return inner(vid, specs)
        return charge
    monkeypatch.setattr(rb, '_confirm_charge', spy)
    assert rb.confirm_pending(root, 'B', runner_for, str(root), PHASE, verify_mode=vp.ALL8,
                              **EXTRA_SEEDS4) == [True]
    bank = rb.bank_root(root, 'B')
    c = vp.records(root, 'B')[r['verification_id']]
    assert c['extra_seeds'] == 4 and sorted(c['extra']) == ['3', '4', '5', '6']
    for k in range(3, 7):
        x = c['extra'][str(k)]
        assert x['salt'] == vp.extra_seed_salt(r['verification_id'], k)
        assert x['kind'] == f'confirm_seed{k}_all8' and x['later_harm_measured'] is True
        assert x['candidates'] == [0]
        assert sv.snapshot_sha(rb.replicate(snap, x['salt'])) == x['snapshot_sha256'] != c['seed2']['snapshot_sha256']
    # independent per-seed admission of the SAME plan p1 (candidate 0)
    ex = {e['seed_index']: e for e in c['per_seed_v1']['examples'] if e['candidate_index'] == 0}
    assert sorted(ex) == [1, 2, 3, 4, 5, 6]
    assert ex[3]['admitted'] and ex[3]['example_id'] == f'{r["verification_id"]}#c1#s3'
    assert ex[4]['status'] == 'rejected' and 'seed-4 one-sided 95% lower bound' in ex[4]['reason']
    assert ex[5]['later_harm'] == 'fail' and 'seed-5 later-harm check failed' in ex[5]['reason']
    assert ex[5]['extended_advantage_veh_s'] < 0 and ex[5]['safety_source'] == 'extended_window_seed5'
    assert ex[6]['admitted'] and ex[6]['cooperative'] is False and ex[6]['cooperative_status'] == 'not_cooperative'
    assert ex[3]['cooperative'] is True and ex[2]['cooperative'] is True
    assert ex[3]['tape']['branches_dir'].replace('\\', '/').endswith('confirm_seed3_all8/branches')
    assert ex[3]['tape']['intent_entry'] == f'{r["verification_id"]}/seed3/cand1'
    assert not (bank/'confirm_extended_seed4_all8').exists()
    assert c['extra']['4']['extended'] is None and c['extra']['5']['extended']['candidate_indices'] == [0]
    assert c['extra']['3']['extended']['candidate_indices'] == [0]
    assert c['winners']['3'] == ['p1'] and c['winners']['4'] == [] and c['winners']['6'] == ['p1']
    assert sorted(c['winners']) == ['1', '2', '3', '4', '5', '6']
    assert c['per_seed_v1']['admitted_seeds'] == [1, 2, 3, 6]
    assert c['per_seed_v1']['cooperative_seeds'] == [1, 2, 3]
    assert c['per_seed_v1']['replicated'] is True and c['per_seed_v1']['replicated_plans'] == ['p1']
    # solos per extra seed only for its ADMITTED examples, on that seed's own root and CRN
    for k in (3, 6):
        sroot = bank/f'confirm_solo_all8_s{k}'
        receipts = list((sroot/'verifications').glob('*.json'))
        assert len(receipts) == 1 and c['extra'][str(k)]['solo_root'] == str(sroot)
        solo = json.loads(receipts[0].read_text(encoding='utf-8'))
        assert sv.snapshot_sha(rb.replicate(snap, c['extra'][str(k)]['salt'])) == solo['snapshot_sha256']
    assert not (bank/'confirm_solo_all8_s4').exists() and not (bank/'confirm_solo_all8_s5').exists()
    # charges: every extra verification went through the SAME hook, ledger agrees with the record
    assert len(health) == len(c['charges']) == 15
    kinds = sorted(seen_kinds)
    assert kinds.count('confirm_seed3_all8') == 1 and kinds.count('confirm_extended_seed4_all8') == 0
    assert all(f'confirm_seed{k}_all8' in kinds for k in range(3, 7))
    assert kinds.count('confirm_extended_seed5_all8') == 1
    assert kinds.count('confirm_solo_all8_s3') == 1 and kinds.count('confirm_solo_all8_s6') == 1
    assert sorted(seen_kinds) == sorted(x['kind'] for x in c['charges'])
    rows = [x for x in ledger(root) if x['phase'] == PHASE]
    assert sum(x['sim_s'] for x in rows) == c['charged_sim_s']
    extra_kinds = [k for k in set(seen_kinds) if k.startswith(('confirm_seed3', 'confirm_seed4', 'confirm_seed5',
                                                              'confirm_seed6', 'confirm_extended_seed3',
                                                              'confirm_extended_seed5', 'confirm_extended_seed6',
                                                              'confirm_solo_all8_s3', 'confirm_solo_all8_s6'))]
    extra_rows = [x for x in rows if any(x['id'].startswith(k+'_') for k in extra_kinds)]
    extra_charged = sum(x['charged_sim_s'] for x in c['charges'] if x['kind'] in extra_kinds)
    assert extra_charged == sum(x['sim_s'] for x in extra_rows) > 0
    q = vp.quote_decision(600, [120]*8, [['c1', 'c2']]*8, {1: [0], 2: [0]}, extra_seeds=4, cooperative=[0])
    assert q['extra_seeds_sim_s'] >= extra_charged          # quote's upper-bound assumption
    # per_seed_v1 consumption unchanged: overlay/bank_examples/imitation_entry on the seed-3 tape
    conf = vp.overlay({}, root, 'B')
    assert {e['seed_index'] for e in conf[r['verification_id']]['per_seed_v1']['examples']} == {1, 2, 3, 4, 5, 6}
    rows2 = vp.bank_examples(root, 'B')
    s3 = [x for x in rows2 if x['seed_index'] == 3]
    assert len(s3) == 1 and s3[0]['example_id'] == f'{r["verification_id"]}#c1#s3'
    tape, entry = vp.imitation_entry(s3[0], conf[r['verification_id']]['t0'], 1.)
    assert len(xg.imitation_pairs(tape, entry)) == 120
    assert s3[0]['tape']['branch']['branch'] == 'cand1'


def test_report_per_seed_consumes_rejected_extra_examples(tmp_path, monkeypatch):
    """A rejected extra-seed example carries the SAME not_admitted cooperative fields as the
    seeds-1/2 loop sets, so bank_statistics.report_per_seed reads a record with rejected extra
    examples (here: seed 4 fails its own lower bound) without KeyError."""
    gains = {('p1', 1): 0.6, ('p1', 2): 0.6, ('p1', 1, 'solo'): 0.1, ('p1', 2, 'solo'): 0.1,
             ('p1', 3): 0.6, ('p1', 3, 'solo'): 0.1}
    root, snap, plans, r, runner_for, _ = setup(tmp_path, gains)
    assert rb.confirm_pending(root, 'B', runner_for, str(root), PHASE, verify_mode=vp.ALL8,
                              extra_seeds=2) == [True]
    c = vp.records(root, 'B')[r['verification_id']]
    ex = {e['seed_index']: e for e in c['per_seed_v1']['examples'] if e['candidate_index'] == 0}
    assert ex[3]['admitted'] and ex[3]['cooperative'] is True and ex[3]['cooperative_status'] == 'cooperative'
    assert ex[4]['status'] == 'rejected' and 'seed-4 one-sided 95% lower bound' in ex[4]['reason']
    assert ex[4]['cooperative'] is False and ex[4]['cooperative_status'] == 'not_admitted'
    unwrapped = rb.confirmations
    while hasattr(unwrapped, 'original'):
        unwrapped = unwrapped.original
    monkeypatch.setattr(rb, 'confirmations', unwrapped)
    bs._CACHE.clear()
    bs.install(rb, bank_rule='per_seed_v1')
    rep = bs.report_per_seed(rb, root, 'B')                  # no KeyError on the rejected extra example
    rows = {x['example_id']: x for x in rep['entries']}
    for k in (1, 2, 3, 4):
        row = rows[ex[k]['example_id']]
        assert row['admitted'] == ex[k]['admitted']
        assert row['cooperative'] == ex[k]['cooperative']
        assert row['cooperative_status'] == ex[k]['cooperative_status']
    assert rows[ex[4]['example_id']]['cooperative_status'] == 'not_admitted'
    assert rep['cooperative_examples'] == 3 and rep['cooperative_by_seed'] == {'1': 1, '2': 1}


def test_extra_seeds_quote_keys_and_cli():
    """Quote: extra-seed cost keys only with extra_seeds>0; CLI 'extra-seeds' prints n=4 rows."""
    D, M = [120]*8, [['c1', 'c2']]*8
    q = vp.quote_decision(600, D, M, {1: [0], 2: [0]}, extra_seeds=4, cooperative=[0])
    one = sv.branch_sim_s(min(600+120+sv.TAIL_S, 1200))
    ext = sv.branch_sim_s(min(600+120+rb.EXTENDED_TAIL_S, 1200))
    assert q['extra_seeds'] == 4 and q['cooperative'] == [0]
    assert q['extra_seeds_group_sim_s'] == 4*2*one
    assert q['extra_seeds_extended_sim_s'] == 4*2*ext
    assert q['extra_seeds_solo_sim_s'] == 4*3*one
    assert q['extra_seeds_sim_s'] == (q['extra_seeds_group_sim_s']+q['extra_seeds_extended_sim_s']
                                      + q['extra_seeds_solo_sim_s'])
    assert q['all8_total_with_extra_sim_s'] == q['all8_total_sim_s']+q['extra_seeds_sim_s']
    assert q['multiplier_total_with_extra'] > q['multiplier_total']
    no_coop = vp.quote_decision(600, D, M, {1: [], 2: []}, extra_seeds=4)
    assert no_coop['extra_seeds_sim_s'] == 0 and no_coop['cooperative'] == []
    import io, sys
    buf = io.StringIO();old = sys.stdout
    sys.stdout = buf
    try:
        vp.main(['extra-seeds'])
    finally:
        sys.stdout = old
    out = json.loads(buf.getvalue())
    row = out['t0_600']['one_cooperative_winner']
    assert row['extra_seeds'] == 4 and row['extra_seeds_sim_s'] == q['extra_seeds_sim_s']


def test_extra_seeds_on_arm_R(tmp_path):
    """Same BUILD option on the R bank: salts, admission and examples are arm-agnostic."""
    gains = {('p1', 1): 0.6, ('p1', 2): 0.6, ('p1', 1, 'solo'): 0.1, ('p1', 2, 'solo'): 0.1, ('p1', 3): 0.6,
             ('p1', 3, 'solo'): 0.1, ('p1', 4): 0.6, ('p1', 4, 'solo'): 0.6}
    root, snap, plans, r, runner_for, _ = setup(tmp_path, gains, arm='R')
    assert rb.confirm_pending(root, 'R', runner_for, str(root), PHASE, verify_mode=vp.ALL8,
                              extra_seeds=2) == [True]
    c = vp.records(root, 'R')[r['verification_id']]
    assert c['arm'] == 'R' and sorted(c['extra']) == ['3', '4']
    ex = {e['seed_index']: e for e in c['per_seed_v1']['examples'] if e['candidate_index'] == 0}
    assert ex[3]['admitted'] and ex[3]['cooperative'] is True and ex[4]['admitted']
    assert ex[4]['cooperative'] is False and ex[4]['cooperative_status'] == 'not_cooperative'
    assert {e['seed_index'] for e in vp.bank_examples(root, 'R')} == {1, 2, 3, 4}
    assert (rb.bank_root(root, 'R')/'confirm_solo_all8_s3').exists()


def test_reader_relabels_stored_reference_mismatch_on_extra_seed(tmp_path):
    """The reader-side relabel resolves seed_index>=3 solos via seed_kinds (not SOLO_ROOTS)."""
    from tools.coopmwm.accel_storage import write as awrite
    bank = tmp_path/'bank'
    sroot = bank/vp.seed_kinds(3)[2]
    plan, ref, window, solo, J_ref = _solo_fixture(sroot, solo_end=660)
    (sroot/'verifications').mkdir()
    solo.pop('receipt_sha256');solo = xg._seal(solo, 'receipt_sha256')
    (sroot/'verifications'/'solo_v.json').write_text(json.dumps(solo))
    gdir = tmp_path/'seed3';gv = 'grp_v3'
    for b, tape in (('policy', ref), ('cand1', plan)):
        d = gdir/'branches'/gv/b;d.mkdir(parents=True);awrite(d/'RESULT.json', tape)
    grec = xg._seal(dict(verification_id=gv, status='accepted', J_policy_veh_s=J_ref,
                         branches=[dict(branch='policy'), dict(branch='cand1')], window=window), 'receipt_sha256')
    (gdir/'verifications').mkdir();(gdir/'verifications'/(gv+'.json')).write_text(json.dumps(grec))
    stale = dict(seed_index=3, example_id=gv+'#c1#s3', plan_id='p1', candidate_index=0, admitted=True,
                 status='admitted', cooperative=False, cooperative_status='reference_mismatch',
                 solo_verification_id='solo_v', solo_receipt_sha256=solo['receipt_sha256'],
                 tape=dict(branches_dir=str(gdir/'branches'), verification_id=gv, branch=dict(branch='cand1'),
                           window=window))
    rec = dict(verification_id=gv, candidates=[dict(index=0, plan_id='p1', members=['a', 'b'])],
               per_seed_v1=dict(examples=[stale], cooperative_seeds=[], replicated=False, replicated_plans=[]),
               cooperation=dict(cooperative=False, cooperative_seeds=[], cooperative_examples=0, replicated=False))
    out = vp.relabel_reference_mismatch(rec, bank)
    e = out['per_seed_v1']['examples'][0]
    assert e['cooperative'] is True and e['cooperative_status'] == 'cooperative'
    assert out['per_seed_v1']['cooperative_seeds'] == [3]
    assert out['per_seed_v1']['replicated'] is False        # an extra seed alone never flips the legacy flag
