"""CPU regression: the confirmation stage's (a) replication and (b) extended-window branches
must BOTH run. They share the verification_id (episode/t0/candidate digest), so their ledger
charge ids are kind-prefixed and land in the bank_verification_ext phase (MAIN 2026-11-02 22:32);
unqualified ids hit the durable duplicate-charge guard and the second branch is budget_blocked.

MAIN 2026-11-03 00:01: every ADMITTED multi-CAV entry also runs per-member SOLO branches
(cooperation/synergy check, kind confirm_solo, same phase); single-CAV plans are
cooperative=False by definition. The imitation batch oversamples cooperative entries to 50%
of the imitation samples with unchanged weights; BANK_QUALITY.json carries the synergy twins.
"""
import gzip
import json
import pytest

from tools.coopmwm import replay_bank as rb
from tools.coopmwm import sumo_verify as sv
from tools.coopmwm.storage import ledger
from tools.coopnative.storage import canonical


@pytest.fixture(autouse=True)
def bank_budget(tmp_path):
    """Use the later bank budget through an authorised additive binding, not by
    changing the historical acceleration/resume quote shared by other tests."""
    from .warm_protocol import budget
    b = budget()
    b['total'] += 2600000
    b['allocated'] += 2600000
    b['phases'] = dict(b['phases'], bank_verification_ext=2600000)
    (tmp_path/'B_REPLAY_PHASE_RECONCILIATION_0627.json').write_text(json.dumps(
        dict(budget=b, amendment_sha256='test-bank-budget')))
    (tmp_path/'AUTHORITY.md').write_text('test-bank-budget')


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


def plan(pid, gain, members=('c1', 'c2')):
    return dict(id=pid, members=list(members), duration_s=120,
                profile=[[gain]*120 for _ in members],
                saving_vs_mappo=gain, saving_vs_policy=gain)


def duo_plan(pid, g1, g2):
    return dict(id=pid, members=['c1', 'c2'], duration_s=120, profile=[[g1]*120, [g2]*120],
                saving_vs_mappo=(g1+g2)/2, saving_vs_policy=(g1+g2)/2)


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


class TapeRunner(FakeRunner):
    """Adds the chosen branch's RESULT.json one-second plan tape (for bank_pairs); the returned
    result_sha256 identifies the tape, exactly like the real plant runner. ``cooperative`` switches
    J to the sum-profile (members help each other) plant."""

    def __init__(self, work, write_result=True, cooperative=False):
        super().__init__(work);self.write_result = write_result;self.cooperative = cooperative

    def J(self, p):
        if p is None:
            return self.J_policy
        if self.cooperative:
            return self.J_policy-float(sum(x for row in p['profile'] for x in row))
        prof = [x for row in p['profile'] for x in row]
        return self.J_policy-10.*sum(prof)/len(prof)

    def run(self, spec, snap):
        r = super().run(spec, snap)
        p = spec['plan']
        if self.write_result and p is not None:
            from tools.coopmwm.accel_storage import write as awrite
            from tools.coopmwm.protocol import digest
            tape = dict(steps=[dict(time_s=t, sample=dict(plan_executions=[dict(plan_id=p['id'],
                proposal_members=list(p['members']))], absolute_accels={v: 0.5 for v in p['members']}))
                for t in range(spec['window']['start_s'], spec['window']['start_s']+p['duration_s'])])
            awrite(self.work/'branches'/spec['verification_id']/spec['branch']/'RESULT.json', tape)
            r['result_sha256'] = digest(tape)
        return r


class SumRunner(FakeRunner):
    """Cooperative plant: J(plan) = J_policy - sum(profile) (the members HELP EACH OTHER:
    the group beats every member-only execution)."""

    def J(self, p):
        if p is None:
            return self.J_policy
        return self.J_policy-float(sum(x for row in p['profile'] for x in row))


class AmdRunner(FakeRunner):
    """MAIN 2026-11-03 04:26 amendment plant: the candidate branch carries extra window
    safety totals (collisions/hard_brake) and TTC<1.5 s trace events (a front row faster
    than its follower scores TTC 0 events), reproducing the real 986016-style pattern."""

    def __init__(self, work, J_policy=500., collisions=0., hard_brake=0., ttc_events=0, cooperative=False):
        super().__init__(work, J_policy)
        self.collisions, self.hard_brake, self.ttc_events = collisions, hard_brake, ttc_events
        self.cooperative = cooperative

    def J(self, p):
        if p is None:
            return self.J_policy
        if self.cooperative:
            return self.J_policy-float(sum(x for row in p['profile'] for x in row))
        prof = [x for row in p['profile'] for x in row]
        return self.J_policy-10.*sum(prof)/len(prof)

    def run(self, spec, snap):
        cand = spec['branch'] != 'policy'
        w = snap['episode']['world']['seed'];t0, t_end = spec['window']['start_s'], spec['window']['end_s']
        d = self.work/'branches'/spec['verification_id']/spec['branch'];d.mkdir(parents=True, exist_ok=True)
        hexf = lambda x: float(x).hex()
        row = dict(time_s=t0, vehicles=[dict(lane='m1', s_m=10., v_mps=8., length_m=5.),
                                        dict(lane='m1', s_m=0., v_mps=12., length_m=5.)])
        events = 0 if cand else self.ttc_events      # the plan REMOVES TTC events vs 'policy continues'
        with gzip.open(d/'observable.jsonl.gz', 'wt', encoding='utf-8') as f:
            for _ in range(events):
                f.write(canonical(row)+'\n')
        def fp2(t, delay):
            return dict(t_s=t, metrics=dict(team_delay_vs=hexf(delay), arrived=hexf(t),
                                            collisions=hexf(self.collisions if cand else 0.),
                                            teleports=hexf(0.), ttc_below_3_vs=hexf(0.),
                                            hard_brake=hexf(self.hard_brake if cand else 0.)))
        prefixes = {str(t): fp(t, 3.*w+0.5*t*t/100.) for t in range(300, t0+1)}   # CRN: identical prefix tape
        return dict(prefixes=prefixes, final=fp2(t_end, 3.*w+0.5*t0*t0/100.+self.J(spec['plan'])),
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
    # MAIN 00:01 cooperation: 2-CAV plan -> one SOLO branch per member on the first seed. This
    # plant makes every solo exactly as good as the group (J depends on the profile MEAN), so
    # the gain is 0 and the entry is tagged cooperative=False.
    co = out['cooperation']
    assert co['group_size'] == 2 and co['complete'] is True and co['cooperative'] is False
    assert co['cooperation_gain_veh_s'] == 0. and len(co['solo']) == 2
    assert co['solo_advantages_veh_s'] == [out['first_seed_advantage_veh_s']]*2
    assert all(s['status'] == 'accepted' and s['verification_id'] != receipt['verification_id'] for s in co['solo'])
    # sealed confirmation record for the first-seed decision
    saved = json.loads((bank/'confirmations'/f'{receipt["verification_id"]}.json').read_text())
    from tools.coopmwm.exact_grade import verify_sealed
    verify_sealed(saved, 'confirmation_sha256');assert saved['admitted'] is True
    verify_sealed(saved, 'confirmation_sha256');assert saved['cooperation']['cooperative'] is False
    # charges: bank_verification_ext phase, ids kind-prefixed (replication / extended / solo never collide)
    entries = [e for e in ledger(root) if e['id'].startswith('confirm_')]
    assert {e['id'].split('_')[0]+'_'+e['id'].split('_')[1] for e in entries} == \
        {'confirm_replication', 'confirm_extended', 'confirm_solo'}
    assert all(e['phase'] == 'bank_verification_ext' for e in entries) and len(entries) == 8
    # the first-seed decision used the stub charge (log only): nothing unprefixed reached the ledger
    assert {e['id'] for e in ledger(root)} == {e['id'] for e in entries}


def test_cooperative_plan_beats_every_solo(tmp_path):
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    first = sv.verify(snapshot(), [duo_plan('d', 1., .5)], SumRunner(bank/'sumo_verification'),
                      lambda vid, specs: None, root=bank/'sumo_verification')
    assert first['status'] == 'accepted' and first['grade_veh_s'] == 180.   # sum 1.5*120
    receipt = {k: v for k, v in first.items() if not k.startswith('_')}
    out = rb.confirm(root, 'B', receipt, snapshot(), lambda work: SumRunner(work), str(root),
                     'bank_verification_ext', workers=2, version='u4')
    assert out['admitted'] is True
    co = out['cooperation']
    assert co['cooperative'] is True and co['complete'] is True
    assert co['solo_advantages_veh_s'] == [120., 60.]       # c1 alone 120, c2 alone 60
    assert co['best_solo_member'] == 'c1' and co['best_solo_advantage_veh_s'] == 120.
    assert co['cooperation_gain_veh_s'] == 60.              # 180 - 120
    saved = json.loads((bank/'confirmations'/f'{receipt["verification_id"]}.json').read_text())
    assert saved['cooperation']['cooperative'] is True
    q = rb.bank_quality(root, 'B')
    assert q['cooperative'] == 1 and q['cooperation_undecided'] == 0
    assert q['mean_cooperation_gain_veh_s'] == 60.
    twins = q['synergy_twins']
    assert len(twins) == 1 and twins[0]['cooperative'] is True
    assert twins[0]['first_seed_advantage_veh_s'] == 180. and twins[0]['best_solo_advantage_veh_s'] == 120.


def test_single_cav_plan_is_never_cooperative_and_runs_no_solo(tmp_path):
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    first = sv.verify(snapshot(), [plan('s', 1., members=('c1',))], FakeRunner(bank/'sumo_verification'),
                      lambda vid, specs: None, root=bank/'sumo_verification')
    receipt = {k: v for k, v in first.items() if not k.startswith('_')}
    out = rb.confirm(root, 'B', receipt, snapshot(), lambda work: FakeRunner(work), str(root),
                     'bank_verification_ext', workers=2, version='u4')
    assert out['admitted'] is True
    co = out['cooperation']
    assert co['group_size'] == 1 and co['cooperative'] is False and co['solo'] == []
    kinds = {e['id'].split('_')[0]+'_'+e['id'].split('_')[1] for e in ledger(root) if e['id'].startswith('confirm_')}
    assert kinds == {'confirm_replication', 'confirm_extended'}      # no confirm_solo branches at all


def _confirmed_decision(root, bank, t0, runner, pid, gain, members=('c1', 'c2')):
    first = sv.verify(snapshot(t0), [plan(pid, gain, members)], runner(bank/'sumo_verification'),
                      lambda vid, specs: None, root=bank/'sumo_verification')
    assert first['status'] == 'accepted', first.get('reason')
    receipt = {k: v for k, v in first.items() if not k.startswith('_')}
    out = rb.confirm(root, 'B', receipt, snapshot(t0), lambda work: runner(work), str(root),
                     'bank_verification_ext', workers=2, version='u4')
    assert out['admitted'] is True, out
    return out


def test_exposure_oversamples_cooperative_entries_to_half_with_unchanged_weights(tmp_path):
    """2 non-cooperative + 1 cooperative admitted entry: imitation samples must end at a 50%
    cooperative share (pair duplication only) and every weight stays exactly as before."""
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    r = lambda work: TapeRunner(work)
    _confirmed_decision(root, bank, 420, r, 'n1', 1.)                   # mean-J plant: gain 0 -> False
    _confirmed_decision(root, bank, 460, r, 'n2', 1.)                   # False
    rc = lambda work: TapeRunner(work, cooperative=True)
    first = sv.verify(snapshot(440), [duo_plan('coop', 1., .5)], rc(bank/'sumo_verification'),
                      lambda vid, specs: None, root=bank/'sumo_verification')
    receipt = {k: v for k, v in first.items() if not k.startswith('_')}
    out = rb.confirm(root, 'B', receipt, snapshot(440), rc, str(root),
                     'bank_verification_ext', workers=2, version='u4')
    assert out['admitted'] and out['cooperation']['cooperative'] is True
    pairs, s = rb.bank_pairs(root, 'B', 12345)
    assert s['counts']['used'] == 3 and s['counts']['cooperative'] == 1
    assert s['exposure']['cooperative_entries'] == 1 and s['exposure']['non_cooperative_entries'] == 2
    assert s['exposure']['cooperative_samples_raw'] == 120 and s['exposure']['non_cooperative_samples'] == 240
    assert s['exposure']['duplicated_pairs'] == 120
    assert len(pairs) == 480 and s['exposure']['cooperative_share_of_samples'] == 0.5
    assert len({id(p) for p in pairs}) == 360          # 240 unique others + 120 unique coop, duplicated not re-run
    assert all(w == 1. for _, w in pairs)              # weights unchanged (uniform pre-freeze, beta=1)
    assert [e['cooperative'] for e in s['entries']] == [False, False, True]  # first-seed finish order
    # mean gain over ALL admitted entries with a recorded gain (0-gain non-cooperative included)
    assert s['exposure']['mean_cooperation_gain_veh_s'] == 20.
    assert s['exposure']['mean_cooperation_gain_ci95'] == [0., 60.]
    q = rb.bank_quality(root, 'B')
    assert q['cooperative'] == 1 and q['mean_cooperation_gain_veh_s'] == 20.
    assert len(q['synergy_twins']) == 3 and sum(t['cooperative'] is True for t in q['synergy_twins']) == 1


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


# ---------------------------------------------------------------- MAIN 2026-11-03 04:26 amendment

def test_safety_total_events_offset_admitted_and_old_rule_disclosed(tmp_path):
    """The 986016 pattern: hard_brake +3 but TTC<1.5 s events -92. The NEW rule (total
    safety-critical events must not increase) admits; the OLD verdict (no metric may
    increase) is False and its count is disclosed in BANK_QUALITY."""
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    r = lambda work: AmdRunner(work, collisions=0., hard_brake=3., ttc_events=92)
    first = sv.verify(snapshot(), [plan('a', 1.)], r(bank/'sumo_verification'),
                      lambda vid, specs: None, root=bank/'sumo_verification')
    assert first['status'] == 'accepted'
    receipt = {k: v for k, v in first.items() if not k.startswith('_')}
    out = rb.confirm(root, 'B', receipt, snapshot(), r, str(root),
                     'bank_verification_ext', workers=2, version='u4')
    assert out['safety_deltas']['replication']['hard_brake'] == 3.
    assert out['safety_deltas']['replication']['ttc_below_1p5_events'] == -92.
    assert out['checks']['safety'] is True and out['safety_old_rule_ok'] is False
    assert out['admitted'] is True
    saved = json.loads((bank/'confirmations'/f'{receipt["verification_id"]}.json').read_text())
    assert saved['safety_old_rule_ok'] is False and saved['admitted'] is True
    q = rb.bank_quality(root, 'B')
    assert q['rejections']['safety'] == 0 and q['rejections']['safety_old_rule'] == 1
    assert q['admitted'] == 1


def test_collision_increase_rejected_under_both_rules(tmp_path):
    """Hard constraint: a collision increase is never offset - rejected by the NEW rule and
    by the OLD rule alike; the entry is not admitted and feeds no imitation."""
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    r = lambda work: AmdRunner(work, collisions=1.)
    first = sv.verify(snapshot(), [plan('a', 1.)], r(bank/'sumo_verification'),
                      lambda vid, specs: None, root=bank/'sumo_verification')
    receipt = {k: v for k, v in first.items() if not k.startswith('_')}
    out = rb.confirm(root, 'B', receipt, snapshot(), r, str(root),
                     'bank_verification_ext', workers=2, version='u4')
    assert out['checks']['safety'] is False and out['safety_old_rule_ok'] is False
    assert out['admitted'] is False
    # the plan moves 2 CAVs and passes replication + later-harm, so its synergy check still
    # runs as a diagnostic (this mean-J plant makes every solo equal: gain 0, not cooperative)
    assert out['cooperation']['complete'] is True and out['cooperation']['cooperative'] is False
    q = rb.bank_quality(root, 'B')
    assert q['rejections']['safety'] == 1 and q['rejections']['safety_old_rule'] == 1
    assert q['admitted'] == 0 and q['synergy_diagnostic'] == 1 and q['cooperative'] == 0
    twins = q['synergy_twins']
    assert len(twins) == 1 and twins[0]['admitted'] is False and twins[0]['cooperative'] is False


def test_synergy_diagnostic_published_when_safety_rejects(tmp_path):
    """MAIN 04:26: a decision passing replication + later-harm but rejected by safety still
    runs its SOLO branches as a DIAGNOSTIC and publishes its synergy twin row (tagged
    not admitted); it never enters imitation, so the admitted cooperative count is 0."""
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    r = lambda work: AmdRunner(work, collisions=1., cooperative=True)
    first = sv.verify(snapshot(), [duo_plan('d', 1., .5)], r(bank/'sumo_verification'),
                      lambda vid, specs: None, root=bank/'sumo_verification')
    assert first['status'] == 'accepted' and first['grade_veh_s'] == 180.
    receipt = {k: v for k, v in first.items() if not k.startswith('_')}
    out = rb.confirm(root, 'B', receipt, snapshot(), r, str(root),
                     'bank_verification_ext', workers=2, version='u4')
    assert out['checks']['replication'] is True and out['checks']['no_later_harm'] is True
    assert out['checks']['safety'] is False and out['admitted'] is False
    co = out['cooperation']
    assert co['complete'] is True and co['cooperative'] is True
    assert co['solo_advantages_veh_s'] == [120., 60.] and co['cooperation_gain_veh_s'] == 60.
    q = rb.bank_quality(root, 'B')
    assert q['admitted'] == 0 and q['cooperative'] == 0 and q['synergy_diagnostic'] == 1
    twins = q['synergy_twins']
    assert len(twins) == 1 and twins[0]['admitted'] is False and twins[0]['cooperative'] is True
    assert twins[0]['safety_old_rule_ok'] is False and twins[0]['best_solo_advantage_veh_s'] == 120.
    pairs, s = rb.bank_pairs(root, 'B', 12345)
    assert pairs == [] and s['counts']['used'] == 1 and s['counts']['not_admitted'] == 1
