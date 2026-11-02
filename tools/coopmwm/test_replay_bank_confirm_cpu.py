"""CPU regression: the confirmation stage's (a) replication and (b) extended-window branches
must BOTH run. They share the verification_id (episode/t0/candidate digest), so their ledger
charge ids are kind-prefixed and land in the bank_verification_ext phase (MAIN 2026-11-02 22:32);
unqualified ids hit the durable duplicate-charge guard and the second branch is budget_blocked."""
import gzip
import json

from tools.coopmwm import replay_bank as rb
from tools.coopmwm import sumo_verify as sv
from tools.coopmwm.storage import ledger
from tools.coopnative.storage import canonical


def fp(t, delay):
    hexf = lambda x: float(x).hex()
    return dict(t_s=t, metrics=dict(team_delay_vs=hexf(delay), arrived=hexf(t), collisions=hexf(0.),
                                    teleports=hexf(0.), ttc_below_3_vs=hexf(0.), hard_brake=hexf(0.)))


def snapshot(t0=420):
    return dict(episode=dict(id='warm_u005_B_7', world=dict(seed=7), arm='B', update=5, role='train'),
                policy=dict(state_dict_sha256='p'), sumo=dict(runtime='rt', checkpoint='ck', warm_history=True,
                                                             accel_bounds=[-3, 2], noise_seed=11),
                t0=t0, end_s=1200, prefixes={str(t): fp(t, 3.*7+0.5*t*t/100.) for t in range(300, t0+1)},
                samples={}, rng=dict(generator_state_sha256='g'), committed_plans=[])


def plan(pid, gain):
    return dict(id=pid, members=['c1', 'c2'], duration_s=120, profile=[[gain]*120 for _ in range(2)],
                saving_vs_mappo=gain, saving_vs_policy=gain)


class FakeRunner:
    """Deterministic J(plan) = J_policy - 10*mean(profile); writes the observable trace rows
    (two same-lane vehicles, closing TTC 2.0 s -> no TTC<1.5 events) the safety check reads."""

    def __init__(self, work, J_policy=500.):
        self.work, self.J_policy = work, J_policy

    def J(self, p):
        if p is None:
            return self.J_policy
        prof = [x for row in p['profile'] for x in row]
        return self.J_policy-10.*sum(prof)/len(prof)

    def run(self, spec, snap):
        w = snap['episode']['world']['seed'];t0, t_end = spec['window']['start_s'], spec['window']['end_s']
        d = self.work/'branches'/spec['verification_id']/spec['branch'];d.mkdir(parents=True, exist_ok=True)
        row = dict(time_s=t0, vehicles=[dict(lane='m1', s_m=10., v_mps=8., length_m=5.),
                                        dict(lane='m1', s_m=0., v_mps=10., length_m=5.)])
        with gzip.open(d/'observable.jsonl.gz', 'wt', encoding='utf-8') as f:
            f.write(canonical(row)+'\n')
        prefixes = {str(t): fp(t, 3.*w+0.5*t*t/100.) for t in range(300, t0+1)}
        return dict(prefixes=prefixes, final=fp(t_end, 3.*w+0.5*t0*t0/100.+self.J(spec['plan'])),
                    actual_sim_s=t_end-300)


def test_replication_and_extended_branches_both_complete_and_charge_distinct(tmp_path):
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    log = []
    def first_charge(vid, specs):
        log.extend(s['branch'] for s in specs)
    first = sv.verify(snapshot(), [plan('a', 1.), plan('b', 4.), plan('c', -2.)], FakeRunner(bank/'sumo_verification'),
                      first_charge, root=bank/'sumo_verification')
    assert first['status'] == 'accepted' and first['grade_veh_s'] > 0
    receipt = {k: v for k, v in first.items() if not k.startswith('_')}
    out = rb.confirm(root, 'B', receipt, snapshot(), lambda work: FakeRunner(work), str(root),
                     'bank_verification_ext', workers=2, version='u4')
    # THE regression: before the fix the extended branch was budget_blocked (duplicate charge
    # of the replication's ids: same episode/t0/candidate -> same verification_id).
    assert out['replication']['status'] == 'accepted', out.get('reason')
    assert out['extended']['status'] == 'accepted', out.get('reason')
    assert out['checks'] == dict(replication=True, no_later_harm=True, safety=True)
    assert out['admitted'] is True
    assert out['mean_replicated_advantage_veh_s'] == (receipt['grade_veh_s']+out['replication_advantage_veh_s'])/2
    assert out['safety_deltas']['replication']['ttc_below_1p5_events'] == 0.
    # extended window = min(episode end, t0+D+600)
    assert out['extended']['window']['tail_s'] == 600 and out['extended']['window']['end_s'] == 420+120+600
    # sealed confirmation record for the first-seed decision
    saved = json.loads((bank/'confirmations'/f'{receipt["verification_id"]}.json').read_text())
    from tools.coopmwm.exact_grade import verify_sealed
    verify_sealed(saved, 'confirmation_sha256');assert saved['admitted'] is True
    # charges: bank_verification_ext phase, ids kind-prefixed (replication vs extended never collide)
    entries = [e for e in ledger(root) if e['id'].startswith('confirm_')]
    assert {e['id'].split('_')[0]+'_'+e['id'].split('_')[1] for e in entries} == {'confirm_replication', 'confirm_extended'}
    assert all(e['phase'] == 'bank_verification_ext' for e in entries) and len(entries) == 4
    # the first-seed decision used the stub charge (log only): nothing unprefixed reached the ledger
    assert {e['id'] for e in ledger(root)} == {e['id'] for e in entries}


def test_quality_counts_admitted_after_confirmation(tmp_path):
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    first = sv.verify(snapshot(), [plan('a', 4.)], FakeRunner(bank/'sumo_verification'),
                      lambda vid, specs: None, root=bank/'sumo_verification')
    receipt = {k: v for k, v in first.items() if not k.startswith('_')}
    rb.confirm(root, 'B', receipt, snapshot(), lambda work: FakeRunner(work), str(root),
               'bank_verification_ext', workers=2, version='u4')
    q = rb.bank_quality(root, 'B')
    assert q['decisions'] == 1 and q['first_seed_positives'] == 1
    assert q['confirmed'] == 1 and q['pending_confirmation'] == 0
    assert q['admitted'] == 1 and q['confirmation_rate'] == 1.0
    assert q['mean_admitted_advantage_veh_s'] is not None and q['by_policy_version'] == {'u4': 1}
    conf = rb.confirmations(root, 'B')
    assert len(conf) == 1 and next(iter(conf.values()))['admitted'] is True
