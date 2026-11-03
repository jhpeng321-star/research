"""Scientific regressions: member intervention, CRN, branch reuse, exact algebra."""
import copy
import json
import torch
import pytest
from . import policy_synergy as ps, sumo_verify as sv
from .exact_grade import _seal
from .test_replay_bank_confirm_cpu import snapshot, plan


class DrawPolicy:
    def __init__(self, mean, scale):self.mean,self.scale=mean,scale
    def sample(self, context, generator, deterministic=False, memory=None):
        shared=torch.randn((),generator=generator)
        noise=torch.randn(len(context['members']),generator=generator)+shared
        actions=[[(self.mean+self.scale*float(n))] for n in noise]
        return dict(action=copy.deepcopy(actions),own_action=copy.deepcopy(actions)),dict(zip(context['members'],noise.tolist()))


def test_mixed_policy_changes_only_reverted_members_and_preserves_rng():
    ctx=dict(members=['a','b','outside'])
    new,old=DrawPolicy(2.,.5),DrawPolicy(-1.,1.)
    a=torch.Generator().manual_seed(21);b=torch.Generator().manual_seed(21)
    c=torch.Generator().manual_seed(21)
    expected_new,mn=new.sample(ctx,a);expected_old,mo=old.sample(ctx,b)
    actual,m=ps.MixedPolicy(new,old,['b']).sample(ctx,c)
    assert actual['action']==[expected_new['action'][0],expected_old['action'][1],expected_new['action'][2]]
    assert actual['own_action']==actual['action']
    assert m==dict(a=mn['a'],b=mo['b'],outside=mn['outside'])
    assert torch.equal(a.get_state(),b.get_state()) and torch.equal(a.get_state(),c.get_state())


def test_equal_policies_exactly_reduce_to_unmixed_sampling():
    p=DrawPolicy(.3,.4);ctx=dict(members=['a','b','x'])
    a=torch.Generator().manual_seed(41);b=torch.Generator().manual_seed(41)
    x,m=p.sample(ctx,a);y,n=ps.MixedPolicy(p,p,['a','b']).sample(ctx,b)
    y.pop('policy_synergy_diagnostic_only')
    assert x==y and m==n and torch.equal(a.get_state(),b.get_state())


def fixture():
    s=snapshot(420);s['episode']['world']['seed']=986048
    s['generator_state']=torch.Generator().manual_seed(3).get_state()
    p=plan('cooperative',4.)
    r=dict(verification_id='saved',chosen_plan=p,proposer='wm',
           window=dict(start_s=420,end_s=840,tail_s=120,metric='team_delay_vs',truncated=False))
    return s,r,{'w':torch.ones(2)},{'w':torch.zeros(2)}


def test_all_and_solos_use_original_slate_window_unique_bound_identities():
    s,r,new,old=fixture();before=copy.deepcopy(r)
    pairs=ps.specs_for(s,r,8,'B',new,old)
    assert len(pairs)==len(r['chosen_plan']['members'])+1
    ids=[];hashes=[]
    for spec,snap in pairs:
        sv.verify_sealed(spec,'branch_sha256');ids.append(spec['verification_id']);hashes.append(spec['snapshot_sha256'])
        assert spec['window']==r['window'] and spec['sim_s']==840 and spec['plan'] is None
        assert spec['snapshot_sha256']==sv.snapshot_sha(snap)
        assert snap['policy_synergy']['intervention_start_s']==421
        assert snap['policy_synergy']['non_group'].startswith('new policy')
        assert snap['samples'] is s['samples'] and torch.equal(snap['generator_state'],s['generator_state'])
    assert len(set(ids))==len(ids) and len(set(hashes))==len(ids)
    assert r==before
    assert pairs[0][1]['policy_synergy']['new_members']==r['chosen_plan']['members']
    assert all(len(snap['policy_synergy']['new_members'])==1 for _,snap in pairs[1:])
    assert ps.specs_for(s,r,9,'B',new,old)[0][0]['verification_id']!=ids[0]
    assert ps.specs_for(s,r,8,'R',new,old)[0][0]['verification_id']!=ids[0]


def test_unregistered_state_refused():
    s,r,new,old=fixture();s['episode']['world']['seed']=985000
    with pytest.raises(ValueError,match='TRAIN'):ps.specs_for(s,r,8,'B',new,old)


def test_exact_policy_synergy_algebra_and_missing_branch_exclusion():
    results=[dict(status='graded',J_veh_s=j) for j in [70.,90.,85.]]
    out=ps.metrics(100.,results,['a','b'])
    assert out['delta_all_new_veh_s']==30. and out['delta_only_member_new_veh_s']=={'a':10.,'b':15.}
    assert out['policy_cooperation_gain_veh_s']==15. and out['positive_policy_cooperation']
    results[0]['J_veh_s']=95.
    assert ps.metrics(100.,results,['a','b'])['policy_cooperation_gain_veh_s']==-10.
    results[1]=dict(status='failed',J_veh_s=None)
    assert ps.metrics(100.,results,['a','b'])['policy_cooperation_gain_veh_s'] is None


def test_adopt_only_exact_checkpoint_and_window_sealed_lesson():
    s,r,new,old=fixture();ns=dict(s,weights=new,policy=dict(state_dict_sha256=sv._state_sha(new)))
    e=dict(verification_id=r['verification_id'],status='graded',J_new_policy_veh_s=7.,
           new_policy_branch=dict(window=r['window'],snapshot_sha256=sv.snapshot_sha(ns)))
    lesson=_seal(dict(entries=[e]),'lesson_check_sha256')
    assert ps.adopted_all(lesson,r,s,new)['J_veh_s']==7.
    assert ps.adopted_all(lesson,r,s,old) is None
    changed=copy.deepcopy(r);changed['window']['end_s']=720
    assert ps.adopted_all(lesson,changed,s,new) is None
    lesson['entries'][0]['J_new_policy_veh_s']=0.
    with pytest.raises(ValueError):ps.adopted_all(lesson,r,s,new)


def test_interrupted_claim_and_complete_output_are_never_replayed(tmp_path):
    base=tmp_path/'control/u008_B';base.mkdir(parents=True)
    (base/'CLAIM.json').write_text('{}')
    assert ps.run_update(tmp_path,'B',8,dict(control_root=str(tmp_path/'control')))['status']=='interrupted_claim_not_replayed'
    (tmp_path/'POLICY_SYNERGY_u008_B.json').write_text(json.dumps(dict(status='complete')))
    assert ps.run_update(tmp_path,'B',8,dict(control_root=str(tmp_path/'control')))==dict(status='complete')


def test_real_controller_keeps_prefix_t0_then_member_specific_current_policy():
    from .acceleration import Policy
    from tools.coopnative.policy import encode
    from .test_acceleration_cpu import rows
    old,new=Policy(),Policy()
    with torch.no_grad():
        old.actor[-1].bias.fill_(-.2);new.actor[-1].bias.fill_(.7)
    rs=rows();ctx=encode(rs);members=ctx['members'];assert len(members)>=2
    g=torch.Generator().manual_seed(82)
    sample,mem=old.sample(ctx,g)
    snap=dict(t0=300,samples={'300':sample},committed_plans=[],generator_state=g.get_state().clone(),
        rng=dict(generator_state_sha256=__import__('hashlib').sha256(g.get_state().numpy().tobytes()).hexdigest(),memory_after_t0=mem),
        u3_weights=old.state_dict(),policy_synergy=dict(group=members,new_members=[members[0]],
            u3_state_sha256=sv._state_sha(old.state_dict()),new_state_sha256=sv._state_sha(new.state_dict())))
    controller=ps.controller_class()(new,torch.Generator(),snap,None)
    commands={m:0. for m in members}
    controller.apply(rs,commands,300)
    assert controller.events[-1]['sample']['action']==sample['action']
    expected=ps.MixedPolicy(new,old,members[1:])
    gen=torch.Generator().set_state(controller.generator.get_state())
    result,memory=expected.sample(ctx,gen,memory=controller.memory)
    controller.apply(rs,commands,301)
    assert controller.events[-1]['sample']['action']==result['action']
    assert controller.memory==memory and torch.equal(controller.generator.get_state(),gen.get_state())


@pytest.mark.parametrize('later_source',[False,True])
def test_complete_branch_set_precharged_once_and_later_snapshot_gets_u3_reference(tmp_path,monkeypatch,later_source):
    from . import replay_start, warm_health
    monkeypatch.setattr(warm_health,'assert_running',lambda *a:None)
    s,r,new,old=fixture();s['weights']={'w':torch.full((2,),2.)} if later_source else old
    r.update(status='accepted',branches=[dict(J_veh_s=100.)])
    bank=ps.rb.bank_root(tmp_path,'B');(bank/'decisions').mkdir(parents=True)
    (tmp_path/'B_checkpoint_008.pt').write_bytes(b'new');(tmp_path/'B_checkpoint_003.pt').write_bytes(b'old')
    monkeypatch.setattr(ps.torch,'load',lambda path,**kw:dict(policy=new if '008' in str(path) else old))
    monkeypatch.setattr(replay_start,'accounting',lambda cfg:None)
    monkeypatch.setattr(ps.rb,'bank_decisions',lambda *a,**kw:[r])
    monkeypatch.setattr(ps.rb,'confirmations',lambda *a:{'saved':dict(admitted=True,cooperation=dict(cooperative=True,cooperation_gain_veh_s=17.))})
    monkeypatch.setattr(ps.rb,'_decision_snapshot',lambda *a:s)
    calls=[]
    def charge(vid,specs):
        calls.append(('charge',len(specs),sum(x['sim_s'] for x in specs)))
    monkeypatch.setattr(ps.rb,'_confirm_charge',lambda *a:charge)
    class Runner:
        def __init__(self,*a,**kw):pass
        def admit(self,specs):return None
        def prepare(self,*a):pass
    monkeypatch.setattr(ps,'BranchRunner',Runner)
    def branch(spec,snap,runner):
        assert calls[0][0]=='charge'
        calls.append(('run',spec['branch']))
        return dict(status='graded',J_veh_s={'all':70.,'solo0':90.,'solo1':85.,'u3_reference':100.}[spec['branch']])
    monkeypatch.setattr(ps.sv,'run_branch',branch)
    cfg=dict(control_root=str(tmp_path/'control'),collection_root='unused')
    out=ps.run_update(tmp_path,'B',8,cfg)
    assert out['status']=='complete' and out['entries'][0]['policy_cooperation_gain_veh_s']==15.
    n=4 if later_source else 3
    assert calls[0]==('charge',n,840*n) and len(calls)==n+1
    assert out['entries'][0]['plan_cooperation_gain_veh_s']==17.
    before=list(calls);ps.run_update(tmp_path,'B',8,cfg);assert calls==before
