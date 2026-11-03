"""CPU tests for the exact gate runner (synthetic banks and synthetic job outputs, no SUMO):
the offline fixed-update fields and checks, job planning with the registered caps, output
consumption (executor graded receipts and raw .result.json regrades through
sumo_verify.run_branch), and the PASS/FAIL verdict algebra."""
import json

import pytest
import torch

from tools.coopmwm import coopfix_exact_gate as ceg
from tools.coopmwm import policy_synergy as ps, sumo_verify as sv
from tools.coopmwm.accel_learner import save as ckpt_save
from tools.coopmwm.acceleration import Policy, encode
from tools.coopmwm.exact_grade import verify_sealed
from tools.coopmwm.v2v_obs import V2VConfig, v2v_features
from .test_replay_bank_confirm_cpu import snapshot, plan as group_plan, fp, SumRunner

MANDATORY_COPY = tuple(ceg.MANDATORY_STATES)
T_DF = 12.706204736432095      # the registered df=1 multiplier (bank_statistics.replicated_ci)


@pytest.fixture(autouse=True)
def bank_budget(tmp_path):
    """Same authorised additive binding as the confirmation-stage tests (never the shared
    historical quote)."""
    from tools.coopmwm.warm_protocol import budget
    b = budget()
    b['total'] += 2600000
    b['allocated'] += 2600000
    b['phases'] = dict(b['phases'], bank_verification_ext=2600000)
    (tmp_path/'B_REPLAY_PHASE_RECONCILIATION_0627.json').write_text(json.dumps(
        dict(budget=b, amendment_sha256='test-bank-budget')))
    (tmp_path/'AUTHORITY.md').write_text('test-bank-budget')


@pytest.fixture(autouse=True)
def mandatory_states():
    caps = (ceg.MAX_GROUP, ceg.ISOLATED_STATE_CAP_SIM_S, ceg.MAX_WINDOW_END_S)
    ceg.MANDATORY_STATES = (986066, 986048, 986003)
    yield
    ceg.MANDATORY_STATES = MANDATORY_COPY
    ceg.MAX_GROUP, ceg.ISOLATED_STATE_CAP_SIM_S, ceg.MAX_WINDOW_END_S = caps


# ---------------------------------------------------------------- offline fixed update

def synth_rows():
    return [dict(vid='cav0', s_m=-30., v_mps=8., lane=0, is_cav=True, length_m=5.),
            dict(vid='cav1', s_m=-12., v_mps=9.5, lane=0, is_cav=True, length_m=5.),
            dict(vid='cav2', s_m=25., v_mps=10., lane=1, is_cav=True, length_m=5.),
            dict(vid='hdv9', s_m=-50., v_mps=7., lane=2, is_cav=False, length_m=5.)]


def blocks_for(state, members, present=True):
    cfg = V2VConfig(v2v=True)
    table = {row['vid']: row for row in state}
    out = {}
    for vid in members:
        row = [float(t) for t in v2v_features(table, vid, {'cav0': .4, 'cav1': -1.2, 'cav2': 0.}, cfg)]
        if not present:                       # carry the block, no partner present
            row = [0.]*len(row)
        out[vid] = row
    return out


def pair(members=('cav0', 'cav1'), blocks=None, action=(.3, .5, .1)):
    context = encode(synth_rows())
    if blocks is not None:
        context['v2v'] = blocks
    return dict(context=context, action=[[a] for a in action],
                behaviour=dict(proposal_members=list(members), executed=True))


def fixture_checkpoint(path):
    torch.set_num_threads(1)
    torch.manual_seed(47)
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    mean, value = policy(encode(synth_rows()))
    (mean.sum()+value.sum()).backward()
    optimizer.step()
    ckpt_save(path, policy, optimizer, torch.Generator().manual_seed(9), 0, {'charged': 1})
    return policy


def test_fixed_update_offline_fields_widened_imitation_and_intent_accounting(tmp_path):
    ckpt = tmp_path/'B_checkpoint_000.pt'
    fixture_checkpoint(ckpt)
    present = blocks_for(synth_rows(), ['cav0', 'cav1', 'cav2'])
    absent = blocks_for(synth_rows(), ['cav0', 'cav1', 'cav2'], present=False)
    pairs = [(pair(blocks=present), 1.), (pair(blocks=present), 1.), (pair(blocks=absent), 1.),
             (pair(members=('cav0',)), 1.)]
    summary = dict(exposure=dict(cooperative_samples_raw=1))
    state, offline = ceg.fixed_update(ckpt, pairs, summary, coefficient=.01, passes=5, v2v=True)
    # the post-update policy is the widened one; the partner-intent columns moved
    assert int(state['actor.0.weight'].shape[1]) == 29+4+24
    assert offline['partner_intent_weights']['moved'] is True
    assert offline['v2v_dim'] == 24 and offline['passes']['requested'] == 5
    # cooperative pairs: the structural rule + dedup through the exposure record
    assert offline['cooperative_pairs']['count'] == 1 and offline['cooperative_pairs']['raw'] == 1
    # intent pairs: only samples with a present partner bit
    assert offline['intent_pairs']['count'] == 2
    assert set(offline['checks']) == {'full_pass', 'kl_within_limit', 'cooperative_error_falls',
                                      'partner_intent_weights_moved'}
    assert isinstance(offline['checks']['cooperative_error_falls'], bool)
    assert offline['cooperative_pairs']['clusters'] == 1   # 1 cooperative pair -> no CI, disclosed
    # flag off: no widening, no weight movement, base state width
    plain_state, plain = ceg.fixed_update(ckpt, pairs, summary, coefficient=.01, passes=5, v2v=False)
    assert int(plain_state['actor.0.weight'].shape[1]) == 29+4
    assert plain['v2v_dim'] == 0 and plain['partner_intent_weights']['moved'] is False
    assert plain['intent_pairs']['count'] == 2


def test_paired_fall_ci_and_offline_checks_algebra():
    mean, ci, n = ceg.paired_fall_ci([1., 1., 1.], [.5, .6, .4])
    assert abs(mean+.5) < 1e-12 and ci[1] < 0 and n == 3
    mean, ci, n = ceg.paired_fall_ci([1.], [2.])
    assert mean == 1. and ci is None and n == 1
    def offline(ci95, moved, completed=5, kl=.001):
        return dict(passes=dict(completed=completed,
                                stop_reason='all registered imitation epochs complete'),
                    kl=kl, kl_limit=.045, cooperative_pairs=dict(change_ci95=ci95),
                    partner_intent_weights=dict(moved=moved))
    assert all(ceg.offline_checks(offline([-2., -1.], True)).values())
    assert not ceg.offline_checks(offline([-2., 0.], True))['cooperative_error_falls']
    assert not ceg.offline_checks(offline([-2., -1.], False))['partner_intent_weights_moved']
    assert not ceg.offline_checks(offline([-2., -1.], True, kl=.9))['kl_within_limit']
    assert not ceg.offline_checks(offline([-2., -1.], True, completed=0))['full_pass']


def test_paired_seed_ci_matches_the_registered_estimators():
    out = ceg.paired_seed_ci([14., 15.5], student_t=True)
    assert out['mean'] == 14.75 and out['bootstrap_ci95'] == [14., 15.5]
    assert out['student_t_ci95'] == pytest.approx([14.75 - T_DF*.75, 14.75 + T_DF*.75])
    single = ceg.paired_seed_ci([3.])
    assert single['bootstrap_ci95'] is None and single['student_t_ci95'] is None


# ---------------------------------------------------------------- synthetic cooperative bank

def rb_bank(root):
    from tools.coopmwm import replay_bank as rb
    bank = rb.bank_root(root, 'B')
    bank.mkdir(parents=True, exist_ok=True)
    return bank


def rb_confirm(root, receipt, snap):
    from tools.coopmwm import replay_bank as rb
    out = rb.confirm(root, 'B', receipt, snap, lambda work: SumRunner(work), str(root),
                     'bank_verification_ext', workers=2, version='u3')
    replicated_solo_overlay(rb.bank_root(root, 'B'), receipt['verification_id'])
    return out


def replicated_solo_overlay(bank, vid):
    """The MAIN17:27 replicated-synergy record the bank v2 rule reads: the deterministic
    SumRunner plant makes the second seed identical, so the overlay carries the same seed-2
    cooperation gain (written through the registered safety0426 overlay path)."""
    from tools.coopmwm import bank_safety
    from tools.coopmwm.exact_grade import _seal, _write_new
    original = json.loads((bank/'confirmations'/f'{vid}.json').read_text())
    out = bank_safety.reevaluate(original)
    out['cooperation']['replication_cooperation_gain_veh_s'] = \
        out['cooperation']['cooperation_gain_veh_s']
    _write_new(bank/'confirmations_safety0426'/f'{vid}.json', _seal(out, 'confirmation_sha256'))


def coop_state(root, bank, seed, pid, gain, members, t0=420):
    snap = snapshot(t0)
    snap['episode']['world']['seed'] = seed
    snap['prefixes'] = {str(t): fp(t, 3.*seed+.5*t*t/100.) for t in range(300, t0+1)}
    first = sv.verify(snap, [group_plan(pid, gain, members)],
                      SumRunner(bank/'sumo_verification'), lambda vid, specs: None,
                      root=bank/'sumo_verification')
    assert first['status'] == 'accepted', first.get('reason')
    receipt = {k: v for k, v in first.items() if not k.startswith('_')}
    out = rb_confirm(root, receipt, snap)
    assert out['admitted'] is True and out['cooperation']['cooperative'] is True, out
    link_and_snapshot(bank, snap, receipt['verification_id'])   # the rb.collect decisions layout
    return first


def link_and_snapshot(bank, snap, vid):
    """The decision link + decision-time snapshot the bank reader resolves
    (replay_bank.collect layout: snapshots/<state_id>.pt, decisions/<state_id>.json)."""
    state_id = f'st_{snap["episode"]["world"]["seed"]}'
    (bank/'snapshots').mkdir(parents=True, exist_ok=True)
    with (bank/'snapshots'/f'{state_id}.pt').open('xb') as f:
        torch.save(snap, f)
    (bank/'decisions').mkdir(parents=True, exist_ok=True)
    (bank/'decisions'/f'{state_id}.json').write_text(json.dumps(
        dict(state_id=state_id, verification_id=vid, status='accepted')))


SIX = tuple(f'c{i}' for i in range(6))


@pytest.fixture
def gate_bank(tmp_path):
    root = tmp_path
    bank = rb_bank(root)
    coop_state(root, bank, 986066, 'six', 1., SIX)
    coop_state(root, bank, 986048, 'duo48', 1., ('a1', 'a2'))
    coop_state(root, bank, 986003, 'duo03', 1., ('b1', 'b2'))
    return root, bank


def offline_fixture():
    offline = dict(gate=ceg.GATE, passes=dict(completed=5, requested=5, optimizer_steps=9,
                                              pairs_seen=12,
                                              stop_reason='all registered imitation epochs complete'),
                   kl=.001, kl_limit=.045, v2v=True, v2v_dim=24,
                   cooperative_pairs=dict(change_ci95=[-2., -1.]),
                   partner_intent_weights=dict(moved=True))
    offline['checks'] = ceg.offline_checks(offline)
    return offline


# ---------------------------------------------------------------- planning

def test_plan_jobs_quotes_and_sealed_specs(gate_bank, tmp_path):
    root, bank = gate_bank
    torch.manual_seed(3)
    new, old = Policy().state_dict(), Policy().state_dict()
    control = tmp_path/'control'
    record = ceg.plan(root, 'B', control, new, old, offline=offline_fixture())
    verify_sealed(record, 'gate_plan_sha256')
    assert record['mandatory_states'] == [986066, 986048, 986003]
    assert {s['seed'] for s in record['states']} == {986066, 986048, 986003}
    six = next(s for s in record['states'] if s['seed'] == 986066)
    assert six['group'] == list(SIX) and len(six['jobs']) == 14      # 8 isolated + 6 second-seed
    assert len(six['isolated_quote']['branch_ids']) == 8             # all + 6 solos + background
    assert len(six['seed2_solo_quote']['branch_ids']) == 6
    end_s = six['window']['end_s']
    assert six['isolated_quote']['sim_s'] == 8*end_s <= ceg.ISOLATED_STATE_CAP_SIM_S
    assert six['seed2_solo_quote']['sim_s'] == 6*end_s
    assert record['totals']['jobs'] == 14 + 6 + 6                   # six-CAV + two duos
    assert record['totals']['total_sim_s'] == sum(j['sim_s'] for s in record['states']
                                                  for j in s['jobs'])
    assert record['new_state_sha256'] == sv._state_sha(new)
    # every job carries a sealed existing-format spec and its snapshot; ids never collide
    ids = []
    for state in record['states']:
        for job in state['jobs']:
            spec = json.loads((control/'jobs'/f'{job["verification_id"]}.{job["branch"]}.spec.json')
                              .read_text())
            verify_sealed(spec, 'branch_sha256')
            assert spec['branch_sha256'] == job['spec_sha256']
            assert spec['sim_s'] == job['sim_s']
            snap = torch.load(control/'jobs'/f'{job["verification_id"]}.SNAPSHOT.pt',
                              weights_only=False)
            assert sv.snapshot_sha(snap) == job['snapshot_sha256']
            assert job['seed'] == (2 if '_seed2_' in job['verification_id'] else 1)
            ids.append(job['verification_id'])
    assert len(set(ids)) == len(ids)
    branches = sorted(j['branch'] for j in six['jobs'])
    assert branches == ['all', 'background'] + sorted([f'solo{i}' for i in range(6)]*2)
    # the isolated membership: group-only NEW for 'all', exactly one member NEW per solo
    for job in six['jobs']:
        spec = json.loads((control/'jobs'/f'{job["verification_id"]}.{job["branch"]}.spec.json')
                          .read_text())
        snap = torch.load(control/'jobs'/f'{job["verification_id"]}.SNAPSHOT.pt', weights_only=False)
        cfg = snap['policy_synergy']
        assert cfg['background_old'] is (job['branch'] != 'background')
        assert len(cfg['new_members']) == (6 if job['branch'] == 'all'
                                           else 1 if job['branch'].startswith('solo') else 0)
    # re-planning is idempotent (the sealed plan is returned as-is)
    assert ceg.plan(root, 'B', control, new, old, offline=offline_fixture()) == record


def test_plan_requires_certified_mandatory_states_and_registered_caps(gate_bank, tmp_path):
    root, bank = gate_bank
    new, old = Policy().state_dict(), Policy().state_dict()
    ceg.MANDATORY_STATES = (986066, 986999)
    with pytest.raises(ValueError, match='986999'):
        ceg.plan(root, 'B', tmp_path/'c1', new, old, offline=offline_fixture())
    ceg.MANDATORY_STATES = (986066, 986048, 986003)
    ceg.ISOLATED_STATE_CAP_SIM_S = 100
    with pytest.raises(ValueError, match='exceeds'):
        ceg.plan(root, 'B', tmp_path/'c2', new, old, offline=offline_fixture())
    ceg.ISOLATED_STATE_CAP_SIM_S = 9600
    ceg.MAX_GROUP = 2
    with pytest.raises(ValueError, match='group size'):
        ceg.plan(root, 'B', tmp_path/'c3', new, old, offline=offline_fixture())


# ---------------------------------------------------------------- consumption and verdict

def graded(control, job, j):
    path = control/'branches'/job['verification_id']/f'{job["branch"]}.graded.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(branch=job['branch'], index=0, status='graded',
                                    J_veh_s=float(j), window=None,
                                    branch_sha256=job['spec_sha256'])))
    return path


def write_receipts(control, record, deltas):
    """Executor-style graded receipts: per state {('branch', seed): delta} (seed-1 branches
    may use the bare 'branch' key) relative to the state's J_u3 reference
    (delta = J_ref - J_branch)."""
    for state in record['states']:
        d = deltas[state['seed']]
        for job in state['jobs']:
            key = (job['branch'], job['seed'])
            delta = d.get(key, d.get(job['branch']) if job['seed'] == 1 else None)
            if delta is not None:
                graded(control, job, state['j_ref_veh_s'] - delta)


def duo_deltas(all_gain, s0, s1, s0b, s1b, background=0.):
    return {'all': all_gain, 'background': background, ('solo0', 1): s0, ('solo1', 1): s1,
            ('solo0', 2): s0b, ('solo1', 2): s1b}


PASS_DUOS = {986048: duo_deltas(10., 5., 2., 4., 3.), 986003: duo_deltas(10., 5., 2., 4., 3.)}


def test_collect_pass_verdict_algebra_and_cis(gate_bank, tmp_path):
    root, bank = gate_bank
    control = tmp_path/'control'
    record = ceg.plan(root, 'B', control, Policy().state_dict(), Policy().state_dict(),
                      offline=offline_fixture())
    write_receipts(control, record, {
        986066: {'all': 20., 'background': 1.,
                 **{(f'solo{i}', 1): 5.+.2*i for i in range(6)},
                 **{(f'solo{i}', 2): 4.+.1*i for i in range(6)}},
        **PASS_DUOS})
    report = ceg.collect(control, record)
    verify_sealed(json.loads((control/ceg.GATE_FILE).read_text()), 'gate_report_sha256')
    states = {s['seed']: s for s in report['states']}
    six = states[986066]
    assert six['group_only_gain_veh_s'] == 20.
    assert six['best_solo_member'] == 'c5' and six['best_solo_gain_veh_s'] == pytest.approx(5.25)
    # synergy seeds: seed1 20-6=14, seed2 20-4.5=15.5
    assert six['synergy_veh_s'] == pytest.approx(14.75)
    assert six['synergy_ci95'] == pytest.approx([14., 15.5])
    assert six['synergy_student_t_ci95'][0] == pytest.approx(14.75 - T_DF*.75)
    assert six['background_only_delta_veh_s'] == 1.
    pv = six['per_vehicle']['c1']                      # solo1 deltas: 5.2 / 4.1
    assert pv['delta_veh_s'] == pytest.approx(4.65) and pv['seeds_graded'] == 2
    assert pv['ci95'] == pytest.approx([4.1, 5.2])
    assert pv['student_t_ci95'][0] == pytest.approx(4.65 - T_DF*.55)
    duo = states[986048]
    assert duo['complete'] is True and duo['synergy_veh_s'] == pytest.approx(5.5)
    assert duo['synergy_ci95'] == pytest.approx([5., 6.])
    assert report['offline_checks'] == record['offline']['checks']
    assert report['verdict'] == 'PASS'                 # only 986066 is verdict-binding
    assert report['gate_state_synergy_lower_bound'] == 14.
    assert report['reasons'] == []


def test_collect_fails_on_negative_gate_state_synergy_or_offline(gate_bank, tmp_path):
    root, bank = gate_bank
    new, old = Policy().state_dict(), Policy().state_dict()
    control = tmp_path/'c1'
    record = ceg.plan(root, 'B', control, new, old, offline=offline_fixture())
    write_receipts(control, record, {
        986066: {'all': 20., 'background': 0.,
                 **{(f'solo{i}', 1): 22. for i in range(6)},
                 **{(f'solo{i}', 2): 21. for i in range(6)}},
        **PASS_DUOS})
    report = ceg.collect(control, record)
    assert report['verdict'] == 'FAIL'
    assert report['gate_state_synergy_lower_bound'] < 0
    assert 'gate_state_synergy_ci_lower_gt_0' in report['reasons']
    # offline checks failing fails the gate even with positive synergy
    offline = offline_fixture()
    offline['cooperative_pairs']['change_ci95'] = [-1., 0.5]
    offline['checks'] = ceg.offline_checks(offline)
    control2 = tmp_path/'c2'
    record2 = ceg.plan(root, 'B', control2, new, old, offline=offline)
    write_receipts(control2, record2, {
        986066: {'all': 20., 'background': 0.,
                 **{(f'solo{i}', 1): 5. for i in range(6)},
                 **{(f'solo{i}', 2): 4. for i in range(6)}},
        **PASS_DUOS})
    report2 = ceg.collect(control2, record2)
    assert report2['verdict'] == 'FAIL' and report2['reasons'] == ['offline_checks']
    assert report2['gate_state_synergy_lower_bound'] == 15.


def test_incomplete_receipts_are_disclosed_never_imputed(gate_bank, tmp_path):
    root, bank = gate_bank
    control = tmp_path/'control'
    record = ceg.plan(root, 'B', control, Policy().state_dict(), Policy().state_dict(),
                      offline=offline_fixture())
    report = ceg.collect(control, record)              # no outputs at all
    states = {s['seed']: s for s in report['states']}
    assert states[986066]['status'] == 'incomplete'
    assert states[986066]['group_only_gain_veh_s'] is None
    assert states[986066]['reason'].startswith('the group-only branch is ungraded')
    assert report['verdict'] == 'FAIL'
    assert set(report['reasons']) >= {'gate_state_complete', 'gate_state_synergy_ci_lower_gt_0'}
    # a single ungraded second-seed solo keeps the gate state incomplete (2-seed replication)
    control2 = tmp_path/'control2'
    record2 = ceg.plan(root, 'B', control2, Policy().state_dict(), Policy().state_dict(),
                       offline=offline_fixture())
    write_receipts(control2, record2, {
        986066: {'all': 20., 'background': 0.,
                 **{(f'solo{i}', 1): 5. for i in range(6)},
                 **{(f'solo{i}', 2): 4. for i in range(5)}},
        **PASS_DUOS})
    report2 = ceg.collect(control2, record2)
    states2 = {s['seed']: s for s in report2['states']}
    assert states2[986066]['complete'] is False
    assert 'seed2/solo5' in states2[986066]['ungraded_branches']
    assert report2['verdict'] == 'FAIL' and 'gate_state_complete' in report2['reasons']


def test_receipts_from_raw_results_are_regraded_without_sumo(gate_bank, tmp_path):
    root, bank = gate_bank
    control = tmp_path/'control'
    record = ceg.plan(root, 'B', control, Policy().state_dict(), Policy().state_dict(),
                      offline=offline_fixture())
    state = next(s for s in record['states'] if s['seed'] == 986048)
    j_ref, t0, end_s = state['j_ref_veh_s'], state['window']['start_s'], state['window']['end_s']
    seed = 986048
    base = 3.*seed + .5*t0*t0/100.                     # the snapshot fixture prefix formula
    at_end = 3.*seed + .5*end_s*end_s/100.
    job = next(j for j in state['jobs'] if j['branch'] == 'all')
    snap = torch.load(control/'jobs'/f'{job["verification_id"]}.SNAPSHOT.pt', weights_only=False)
    # raw BranchRunner-layout result with a team-delay total worth a 10 veh-s group-only gain
    raw = dict(prefixes=dict(snap['prefixes']),
               final=fp(end_s, at_end + (j_ref - 10.) - (at_end - base)), actual_sim_s=end_s)
    path = control/'branches'/job['verification_id']/'all.result.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw))
    for other in state['jobs']:
        if other['branch'] != 'all':
            key = (other['branch'], other['seed'])
            table = {('background', 1): 0., ('solo0', 1): 5., ('solo1', 1): 2.,
                     ('solo0', 2): 4., ('solo1', 2): 3.}
            if key in table:
                graded(control, other, j_ref - table[key])
    report = ceg.collect(control, record)
    duo = next(s for s in report['states'] if s['seed'] == 986048)
    assert duo['group_only_gain_veh_s'] == 10.         # regraded through sumo_verify.run_branch
    assert duo['complete'] is True
    assert report['verdict'] == 'FAIL'                 # the gate state itself was never run


def test_execute_writes_graded_receipts_through_the_registered_runner(gate_bank, tmp_path,
                                                                      monkeypatch):
    root, bank = gate_bank
    control = tmp_path/'control'
    record = ceg.plan(root, 'B', control, Policy().state_dict(), Policy().state_dict(),
                      offline=offline_fixture())
    ran = []

    class FakeBranchRunner:
        def __init__(self, work, workers=1):
            self.work = work

        def prepare(self, specs, snapshot):
            for spec in specs:
                d = self.work/spec['verification_id']
                d.mkdir(parents=True, exist_ok=True)
                with (d/'SNAPSHOT.pt').open('xb') as f:
                    torch.save(snapshot, f)

    def fake_run_branch(spec, snap, runner):
        ran.append((spec['verification_id'], spec['branch']))
        return dict(branch=spec['branch'], index=0, status='graded',
                    J_veh_s=float(spec['window']['end_s']), window=None,
                    branch_sha256=spec['branch_sha256'])

    monkeypatch.setattr(ps, 'BranchRunner', FakeBranchRunner)
    monkeypatch.setattr(sv, 'run_branch', fake_run_branch)
    quotes = ceg.execute(control)
    assert len(ran) == record['totals']['jobs']
    assert quotes == [dict(verification_id=s['verification_id'],
                           branch_ids=[j['verification_id'] for j in s['jobs']],
                           sim_s=s['isolated_quote']['sim_s']+s['seed2_solo_quote']['sim_s'],
                           phase='bank_verification_ext') for s in record['states']]
    for state in record['states']:
        for job in state['jobs']:
            path = control/'branches'/job['verification_id']/f'{job["branch"]}.graded.json'
            assert json.loads(path.read_text())['status'] == 'graded'
    # the execute receipts feed collect end to end
    report = ceg.collect(control, record)
    assert all(s['status'] == 'graded' for s in report['states'])
