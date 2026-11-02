"""Verifier CRN replay and first-window equivalence; never launches SUMO."""
import copy

import pytest
import torch

from tools.coopmwm import exact_grade as xg, sumo_verify as sv
from tools.coopmwm.protocol import digest
from tools.coopmwm.test_sumo_verify_cpu import FakePlant, fp, plan, snapshot


@pytest.mark.parametrize('fault', [None, 'boot', 'pid', 'argv', 'source', 'exiting'])
def test_beta_handover_controls_only_receipt_bound_supervisor(tmp_path, fault):
    from tools.coopmwm.verification_beta_ops import identity
    proc=tmp_path/'proc';boot=proc/'sys/kernel/random/boot_id'
    boot.parent.mkdir(parents=True);boot.write_text('boot')
    d=dict(pid=123,start_ticks='456',boot_id='boot',root='/run',source='/source',component='R')
    p=proc/'123';p.mkdir()
    st=['0']*22;st[2]='S';st[21]='789' if fault=='pid' else '456'
    (p/'stat').write_text(' '.join(st))
    args=['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.incident_recovery','/run','--component','R']
    if fault=='argv':args[-1]='B'
    (p/'cmdline').write_bytes(b'\0'.join(x.encode() for x in args)+b'\0')
    if fault=='exiting':(p/'cmdline').write_bytes(b'')
    env='PYTHONPATH='+('/other' if fault=='source' else '/source')+'/pytools:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime'
    (p/'environ').write_bytes(env.encode()+b'\0')
    if fault=='boot':d['boot_id']='other'
    if fault=='exiting':assert identity(d,proc)=='exiting'
    elif fault:
        with pytest.raises(AssertionError):identity(d,proc)
    else:assert identity(d,proc)=='S'


def test_beta_transition_uses_original_nested_operational_deadline(tmp_path):
    from tools.coopmwm.verification_beta_ops import resource_deadline
    from tools.coopmwm.storage import write
    write(tmp_path/'EXACT_OPERATIONAL_BINDING.json',dict(gpu_deadline=1793614022.4598482))
    assert resource_deadline(tmp_path,{})==1793614022.4598482
    assert resource_deadline(tmp_path,dict(gpu_deadline=123.))==123.


def test_resource_extension_preserves_scientific_config_and_requires_authority(tmp_path):
    from tools.coopmwm.verification_beta_ops import extended_config,resource_deadline,RESOURCE_DEADLINE
    from tools.coopmwm.storage import write
    from datetime import datetime,timezone,timedelta
    assert RESOURCE_DEADLINE==datetime(2026,11,5,23,59,tzinfo=timezone(timedelta(hours=8))).timestamp()
    original=dict(gpu_deadline=123.,operational=dict(gpu_deadline=123.,K=4,H=300),
        endpoints=[dict(deadline_unix=123.,root='/unchanged')],beta=1.,budget=9500000)
    changed=extended_config(original)
    assert original['gpu_deadline']==123.
    assert changed==dict(gpu_deadline=RESOURCE_DEADLINE,
        operational=dict(gpu_deadline=RESOURCE_DEADLINE,K=4,H=300),
        endpoints=[dict(deadline_unix=RESOURCE_DEADLINE,root='/unchanged')],beta=1.,budget=9500000)
    write(tmp_path/'RESOURCE_EXTENSION_1634.json',dict(gpu_deadline=RESOURCE_DEADLINE,amendment_sha256='approved'))
    (tmp_path/'AUTHORITY.md').write_text('')
    with pytest.raises(AssertionError):resource_deadline(tmp_path,original)
    (tmp_path/'AUTHORITY.md').write_text('approved')
    assert resource_deadline(tmp_path,original)==RESOURCE_DEADLINE


def test_resource_config_replacement_is_atomic_and_retains_original_snapshot(tmp_path):
    from tools.coopmwm.verification_beta_ops import replace_resource_config
    from tools.coopmwm.storage import write,read
    old=dict(gpu_deadline=123.,H=300)
    write(tmp_path/'ORIGINAL.json',old);write(tmp_path/'BINDING.json',old)
    replace_resource_config(tmp_path/'BINDING.json',dict(gpu_deadline=456.,H=300))
    assert read(tmp_path/'ORIGINAL.json')==old
    assert read(tmp_path/'BINDING.json')==dict(gpu_deadline=456.,H=300)
    assert not (tmp_path/'BINDING.json.deadline1634.publishing').exists()


@pytest.mark.parametrize('arm', ['B', 'R'])
@pytest.mark.parametrize('accepted', [False, True])
def test_verification_trust_replay_includes_rejection(arm, accepted):
    p = dict(accepted=accepted, sumo_trust_drawn=True)
    if arm == 'R':
        p['random_match'] = dict(key='ticket')
    step = dict(sample=dict(behaviour=dict(eligible_query=True)), proposal=p)
    actual = torch.Generator().manual_seed(321)
    torch.rand((), generator=actual)  # query admission
    if arm == 'R':
        torch.randint(2**31, (), generator=actual)  # original matched sampler
    torch.rand((), generator=actual)  # pre-verification trust draw
    replay = torch.Generator().manual_seed(321)
    for call in xg.guide_draws(step, arm):
        xg.issue(replay, call)
    assert torch.equal(actual.get_state(), replay.get_state())


def test_legacy_decline_has_no_trust_or_random_sampler_draw():
    step = dict(sample=dict(behaviour=dict(eligible_query=True)), proposal=dict(accepted=False))
    assert xg.guide_draws(step, 'R') == ['rand']


def continuation():
    snap = snapshot(t0=400)
    c = plan('winner', gain=2.)
    live = dict(id=snap['episode']['id'], prefixes={str(t): fp(t, t*2.) for t in range(400, 411)},
                final=fp(1200, 2400.))

    class WithFingerprints(FakePlant):
        def run(self, spec, state):
            r = super().run(spec, state)
            r['continuation_prefix_sha256'] = {t: digest(v) for t, v in live['prefixes'].items()}
            return r

    verification = sv.verify(snap, [c], WithFingerprints(), lambda *args: None)
    live['steps'] = [dict(time_s=400, proposal=dict(plan=c), sample=dict(behaviour=dict(
        executed=True, plan_start=True, sumo_verification=dict(receipt_sha256=verification['receipt_sha256']))))]
    return verification, live


def test_chosen_branch_matches_live_to_next_query_window():
    r, live = continuation()
    gate = sv.check_live_continuation(r, live)
    assert gate['pass_'] and gate['checked_to_s'] == 410 and gate['fingerprints'] == 11
    assert gate['new_sim_s'] == 0
    # Later queries may change traffic: no overclaim about their trajectories.
    live['prefixes']['411'] = fp(411, -1.)
    assert sv.check_live_continuation(r, live) == gate


@pytest.mark.parametrize('fault',[None,'missing','different','plan_started'])
def test_rejected_verification_checks_policy_branch_without_outcome_selection(fault):
    r,live=continuation()
    r.update(status='rejected',chosen=None,chosen_plan=None)
    r=sv._seal(r,'receipt_sha256')
    step=live['steps'][0];step['proposal']['accepted']=False
    b=step['sample']['behaviour'];b.update(executed=False,plan_start=False,
        sumo_verification=dict(receipt_sha256=r['receipt_sha256']))
    if fault=='missing':live['prefixes'].pop('405')
    if fault=='different':live['prefixes']['410']=fp(410,-1.)
    if fault=='plan_started':b['plan_start']=True
    if fault:
        with pytest.raises((ValueError,sv.PrefixMismatch)):sv.check_live_continuation(r,live)
    else:
        gate=sv.check_live_continuation(r,live)
        assert gate['pass_'] and gate['decision_status']=='rejected' and gate['continuation_branch']=='policy'


@pytest.mark.parametrize('fault', ['missing', 'different', 'wrong_plan', 'wrong_receipt', 'not_executed'])
def test_smoke_rejects_incomplete_or_nonidentical_live_continuation(fault):
    r, live = continuation()
    live = copy.deepcopy(live)
    b = live['steps'][0]['sample']['behaviour']
    if fault == 'missing':
        live['prefixes'].pop('405')
    elif fault == 'different':
        live['prefixes']['410'] = fp(410, 820.000000001)
    elif fault == 'wrong_plan':
        live['steps'][0]['proposal']['plan']['id'] = 'another'
    elif fault == 'wrong_receipt':
        b['sumo_verification']['receipt_sha256'] = 'another'
    else:
        b['executed'] = False
    with pytest.raises((sv.PrefixMismatch, ValueError)):
        sv.check_live_continuation(r, live)


@pytest.fixture(params=[False, True], ids=['legacy-trust', 'verified-beta1-zero-trust'])
def verified_episode(tmp_path, monkeypatch, request):
    """Actual loop/controllers against the deterministic toy plant; never SUMO."""
    from tools.coopmwm import accel_plant, plan_client
    from tools.coopmwm.accel_storage import read
    from tools.coopmwm.test_exact_grade_cpu import WORLD, fake_sumo, policy

    class FourCandidates:
        def __init__(self, generator):
            self.generator = generator

        def __call__(self, rows, commands, now, sample):
            if now != 320:
                return None
            from tools.coopmwm.mppi_search import group_components
            members = group_components(rows)[0]['members']
            torch.rand((), generator=self.generator)  # one admitted query
            topk = [dict(id=f'P320_{i}', members=members, duration_s=30,
                         knot_s=10, family='MPPI', knots=[[a]*4 for _ in members])
                    for i, a in enumerate((-.5, .5, 1.5, 2.6))]
            return dict(accepted=True, option='MPPI', plan=topk[0], topk=topk,
                        behaviour=dict(route='wm', eligible_query=True, regime='ordinary',
                                       sample_probability=1., executed=False))

    ident = 'warm_u001_B_986001'
    root = tmp_path/'verify'
    charges = []
    monkeypatch.setattr(plan_client, 'guide', lambda job, g: FourCandidates(g))
    monkeypatch.setattr(sv, 'SubprocessBranchRunner', lambda work, **kw: sv.LocalBranchRunner(work))
    monkeypatch.setattr(sv, 'accel_charge', lambda *a, **kw: lambda vid, specs: charges.append(specs))
    job = dict(id=ident, path=str(tmp_path/'jobs'/ident), world=WORLD, role='train', arm='B',
               update=1, training_seed=20261101,
               weights=policy().state_dict(), runtime='', checkpoint='', noise_seed=11,
               deterministic=False, sim_s=520, stop=520, queue_root=str(tmp_path),
               accel_bounds=[-3., 2.6], gpu_deadline=0, guided=False, beta=1., warm_history=False,
                sumo_verify=dict(root=str(root), run_root=str(tmp_path), workers=1))
    if request.param:
        job['beta']=0.
        job['sumo_verify']['guide_beta']=1.
    with fake_sumo():
        live = accel_plant.episode(job)
    receipts = [read(p) for p in (root/'verifications').glob('*.json')]
    assert len(receipts) == 1 and receipts[0]['status'] == 'accepted'
    return live, receipts[0], root, charges


def test_actual_episode_loop_and_controller_match_verified_branch(verified_episode):
    live, receipt, _, charges = verified_episode
    assert len(charges) == 1 and len(charges[0]) == 5
    assert all(b['status'] == 'graded' for b in receipt['branches'])
    assert all(b['audit']['crn_verified_t0'] for b in receipt['branches'])
    assert sv.check_live_continuation(receipt, live)['checked_to_s'] == 330
    step=next(s for s in live['steps'] if s['time_s']==320)
    assert step['sample']['behaviour']['beta']==1.


def test_registered_smoke_replays_prefix_then_verifies_five_branches(verified_episode,tmp_path,monkeypatch):
    from tools.coopmwm import accel_plant
    from tools.coopmwm.test_exact_grade_cpu import fake_sumo,policy
    from tools.coopmwm.verification_smoke import controller_class
    from tools.coopmwm.accel_storage import read
    original,_,_,_=verified_episode
    proposal=next(s['proposal'] for s in original['steps'] if s['time_s']==320)
    monkeypatch.setattr(accel_plant,'Controller',controller_class(original,320,proposal))
    root=tmp_path/'smoke'
    job=dict(id='real_smoke_fixture',path=str(tmp_path/'smoke_job'),world=original['world'],
        role='train',arm='B',weights=policy().state_dict(),runtime='',checkpoint='',noise_seed=11,
        deterministic=False,sim_s=330,stop=330,queue_root=str(tmp_path),accel_bounds=[-3.,2.6],
        gpu_deadline=0,guided=False,beta=1.,warm_history=False,
        branch=dict(time_s=320,takeover_s=320,revert=[],prefixes=original['prefixes']),
        sumo_verify=dict(root=str(root),run_root=str(tmp_path),workers=1))
    with fake_sumo():live=accel_plant.episode(job)
    receipts=[read(p) for p in (root/'verifications').glob('*.json')]
    assert len(receipts)==1 and len(receipts[0]['branches'])==5
    assert sv.check_live_continuation(receipts[0],live)['checked_to_s']==330


def grade_meta(live):
    return dict(run_root='run', result_path='run/RESULT.json', update=1, arm='B', training_seed=20261101,
                checkpoint_path='checkpoint.pt', checkpoint_sha256='fixture', checkpoint_update=0,
                noise_seed=11)


def test_verified_grade_reuse_is_exact_and_never_runs_or_charges_again(verified_episode, tmp_path):
    from tools.coopmwm import verification_grades as vg
    live, source, root, _ = verified_episode
    vg.record_smoke(root, source, live)
    q = xg.GradeQueue(tmp_path/'grades', charge=lambda *args: pytest.fail('no duplicate branch charge'))
    ids = vg.enqueue_episode(q, live, grade_meta(live))
    assert len(ids) == 1 and q.pending() == []
    r = q.receipt(ids[0])
    xg.verify_sealed(r, 'receipt_sha256')
    assert r['grade_veh_s'] == source['grade_veh_s'] > 0
    assert r['charged_sim_s'] == r['actual_sim_s'] == 0 and r['no_fresh_branch']
    assert r['J_without_veh_s']-r['J_with_veh_s'] == r['grade_veh_s']
    assert r['verification_source']['receipt_sha256'] == source['receipt_sha256']
    assert vg.enqueue_episode(q, live, grade_meta(live)) == ids
    assert q.receipt(ids[0]) == r and len(q.receipts()) == 1
    T = q.frozen_temperature('B', 20261101)
    weights = xg.imitation_weights(q.receipts(), 'B', T, 20261101)
    entry = next(iter(weights['weights'].values()))
    assert xg.imitation_pairs(live, entry)  # usable by existing exact-only learner


def test_verified_grade_reuse_requires_real_smoke(verified_episode, tmp_path):
    from tools.coopmwm import verification_grades as vg
    live, _, _, _ = verified_episode
    q = xg.GradeQueue(tmp_path/'grades')
    with pytest.raises(FileNotFoundError):
        vg.enqueue_episode(q, live, grade_meta(live))
    assert q.ids() == [] and q.receipts() == []


def test_concurrent_verified_grade_importers_adopt_one_receipt(verified_episode, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from tools.coopmwm import verification_grades as vg
    live, source, root, _ = verified_episode
    vg.record_smoke(root, source, live)
    q = xg.GradeQueue(tmp_path/'grades', charge=lambda *args: pytest.fail('no duplicate charge'))
    with ThreadPoolExecutor(max_workers=4) as workers:
        ids = list(workers.map(lambda _: vg.enqueue_episode(q, live, grade_meta(live)), range(8)))
    assert all(value == ids[0] for value in ids)
    assert len(q.receipts()) == len(q.ids()) == 1 and not q.pending()


def test_importer_rechecks_a_receipt_published_after_its_first_read(verified_episode, tmp_path, monkeypatch):
    from tools.coopmwm import verification_grades as vg
    live, source, root, _ = verified_episode
    vg.record_smoke(root, source, live)
    q = xg.GradeQueue(tmp_path/'grades')
    ids = vg.enqueue_episode(q, live, grade_meta(live))
    original = q.receipt
    calls = []

    def initially_unpublished(gid):
        calls.append(gid)
        return None if len(calls) == 1 else original(gid)

    monkeypatch.setattr(q, 'receipt', initially_unpublished)
    assert vg.enqueue_episode(q, live, grade_meta(live)) == ids
    assert len(calls) >= 2 and len(q.receipts()) == 1


def test_verified_grade_reuse_rejects_wrong_source_identity(verified_episode, tmp_path):
    from tools.coopmwm import verification_grades as vg
    live, source, root, _ = verified_episode
    vg.record_smoke(root, source, live)
    bad = copy.deepcopy(live)
    step = next(s for s in bad['steps'] if s['time_s'] == 320)
    step['sample']['behaviour']['sumo_verification']['receipt_sha256'] = 'wrong'
    q = xg.GradeQueue(tmp_path/'grades')
    with pytest.raises(ValueError, match='source SHA'):
        vg.enqueue_episode(q, bad, grade_meta(bad))
    assert q.ids() == [] and q.receipts() == []


def test_truncated_verified_plan_queues_executed_portion_not_full_verification(verified_episode, tmp_path):
    from tools.coopmwm import verification_grades as vg
    live, source, root, _ = verified_episode
    vg.record_smoke(root, source, live)
    truncated = copy.deepcopy(live)
    step = next(s for s in truncated['steps'] if s['time_s'] == 330)
    step['sample']['plan_truncations'] = [dict(plan_id=source['chosen_plan']['id'], start_s=320)]
    q = xg.GradeQueue(tmp_path/'grades')
    ids = vg.enqueue_episode(q, truncated, grade_meta(truncated))
    assert len(ids) == 1 and q.pending() == ids and q.receipts() == []
    assert q.spec(ids[0])['plan']['duration_s'] == 10
    assert q.spec(ids[0])['window']['end_s'] == 450
