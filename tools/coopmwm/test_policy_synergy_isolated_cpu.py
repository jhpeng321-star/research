"""Isolate group learning from changes in all background CAVs, including arrivals."""
import json
import torch
import pytest
from . import policy_synergy as ps, replay_start, warm_health
from .exact_grade import _seal
from .protocol import sha
from .test_policy_synergy_cpu import DrawPolicy, fixture, diagnostic_owner


@pytest.mark.parametrize('selected', [['a','b'], ['a'], []])
def test_group_only_mask_preserves_background_old_and_crn(selected):
    ctx=dict(members=['a','outside','b','new_arrival'])
    new,old=DrawPolicy(2.,.5),DrawPolicy(-1.,1.)
    ga=torch.Generator().manual_seed(21);gb=torch.Generator().manual_seed(21)
    gc=torch.Generator().manual_seed(21)
    a,ma=new.sample(ctx,ga);b,mb=old.sample(ctx,gb)
    out,mem=ps.MixedPolicy(new,old,[],new_members_only=selected).sample(ctx,gc)
    expected=[a['action'][i] if m in selected else b['action'][i] for i,m in enumerate(ctx['members'])]
    assert out['action']==out['own_action']==expected
    assert mem=={m:(ma if m in selected else mb)[m] for m in ctx['members']}
    assert torch.equal(ga.get_state(),gc.get_state())


def test_background_only_keeps_group_old_and_background_new():
    ctx=dict(members=['a','b','outside','new_arrival'])
    new,old=DrawPolicy(2.,.5),DrawPolicy(-1.,1.)
    ga=torch.Generator().manual_seed(4);gb=torch.Generator().manual_seed(4)
    gc=torch.Generator().manual_seed(4)
    a,_=new.sample(ctx,ga);b,_=old.sample(ctx,gb)
    out,_=ps.MixedPolicy(new,old,['a','b']).sample(ctx,gc)
    assert out['action']==b['action'][:2]+a['action'][2:]


def test_isolated_specs_bind_masks_original_window_and_distinct_ids():
    s,r,new,old=fixture()
    pairs=ps.specs_for(s,r,9,'B',new,old,isolated=True)
    legacy=ps.specs_for(s,r,9,'B',new,old)
    assert len(pairs)==len(r['chosen_plan']['members'])+2
    assert not {x['verification_id'] for x,_ in pairs}.intersection(x['verification_id'] for x,_ in legacy)
    for spec,snap in pairs:
        assert spec['window']==r['window'] and spec['snapshot_sha256']==ps.sv.snapshot_sha(snap)
        assert snap['policy_synergy']['intervention_start_s']==s['t0']+1
        assert snap['policy_synergy']['background_old']==(spec['branch']!='background')
    assert pairs[-1][1]['policy_synergy']['new_members']==[]


@pytest.mark.parametrize('later_source',[False,True])
@pytest.mark.parametrize('suffix',['_ISOLATED','_ISOLATED_V2'])
def test_isolated_complete_precharge_adopts_old_reference_not_all_new_lesson(tmp_path,monkeypatch,later_source,suffix,diagnostic_owner):
    s,r,new,old=fixture();s['weights']={'w':torch.full((2,),2.)} if later_source else old
    r.update(status='accepted',branches=[dict(J_veh_s=999.)])
    bank=ps.rb.bank_root(tmp_path,'B');(bank/'decisions').mkdir(parents=True)
    cp=tmp_path/'B_checkpoint_009.pt';cp.write_bytes(b'new')
    (tmp_path/'B_checkpoint_003.pt').write_bytes(b'old')
    legacy=_seal(dict(entries=[dict(verification_id='saved',status='graded',
        group=r['chosen_plan']['members'],window=r['window'],J_u3_veh_s=100.)],
        checkpoint_sha256=sha(cp)), 'policy_synergy_sha256')
    lp=tmp_path/'POLICY_SYNERGY_u009_B.json';lp.write_text(json.dumps(legacy));original=lp.read_bytes()
    monkeypatch.setattr(ps.torch,'load',lambda path,**kw:dict(policy=new if '009' in str(path) else old))
    monkeypatch.setattr(replay_start,'accounting',lambda cfg:None)
    monkeypatch.setattr(warm_health,'assert_running',lambda *a:None)
    monkeypatch.setattr(ps.rb,'bank_decisions',lambda *a,**kw:[r])
    monkeypatch.setattr(ps.rb,'confirmations',lambda *a:{'saved':dict(admitted=True,
        cooperation=dict(cooperative=True,cooperation_gain_veh_s=17.))})
    monkeypatch.setattr(ps.rb,'_decision_snapshot',lambda *a:s)
    calls=[]
    kinds=[]
    def charger(root,phase,kind):
        kinds.append(kind)
        return lambda vid,specs:calls.append(('charge',len(specs),sum(x['sim_s'] for x in specs)))
    monkeypatch.setattr(ps.rb,'_confirm_charge',charger)
    monkeypatch.setattr(ps,'adopted_all',lambda *a:pytest.fail('all-new lesson is not isolated'))
    class Runner:
        def __init__(self,*a,**kw):pass
        def admit(self,specs):return None
        def prepare(self,*a):pass
    monkeypatch.setattr(ps,'BranchRunner',Runner)
    def branch(spec,snap,runner):
        assert calls[0][0]=='charge';calls.append(('run',spec['branch']))
        return dict(status='graded',J_veh_s={'all':70.,'solo0':90.,'solo1':85.,'background':40.}[spec['branch']])
    monkeypatch.setattr(ps.sv,'run_branch',branch)
    cfg=dict(control_root=str(tmp_path/'isolated'),collection_root='unused',isolated=True,output_suffix=suffix)
    out=ps.run_update(tmp_path,'B',9,cfg);row=out['entries'][0]
    assert out['status']=='complete' and row['policy_cooperation_gain_veh_s']==15.
    assert row['background_only_delta_veh_s']==60. and row['J_u3_veh_s']==100.
    assert row['adopted_u3_reference_sha256']==legacy['policy_synergy_sha256']
    assert calls[0]==('charge',4,840*4) and len(calls)==5
    assert kinds==['policy_synergy'+suffix.lower()+'_u009_B']
    assert lp.read_bytes()==original
    combined=json.loads((tmp_path/f'POLICY_SYNERGY_u009_B_WITH{suffix}.json').read_text())
    assert combined['original']==legacy
    before=list(calls);ps.run_update(tmp_path,'B',9,cfg);assert calls==before


def test_isolated_interrupted_claim_is_not_replayed(tmp_path):
    base=tmp_path/'control/u009_B';base.mkdir(parents=True)
    (base/'CLAIM.json').write_text('{}')
    assert ps.run_update(tmp_path,'B',9,dict(control_root=str(tmp_path/'control'),isolated=True))['status']=='interrupted_claim_not_replayed'


def test_real_charge_adapter_separates_original_and_isolated_identity(monkeypatch):
    seen=[]
    monkeypatch.setattr(ps.sv,'accel_charge',lambda root,phase:lambda vid,specs:seen.append(vid))
    for kind in ('policy_synergy_u009_B','policy_synergy_isolated_v2_u009_B'):
        ps.rb._confirm_charge('root','bank_verification_ext',kind)('same_saved_state',[])
    assert len(set(seen))==2
