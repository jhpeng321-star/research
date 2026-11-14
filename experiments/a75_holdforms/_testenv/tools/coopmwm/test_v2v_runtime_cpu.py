"""Live sampling/scoring and spawned-worker contracts, without SUMO."""
import copy
import torch
import pytest
from .acceleration import Policy as BasePolicy, Controller, encode
from .v2v_runtime import Policy, from_weights, controller_class
from .v2v_model import extend_for_v2v
from .v2v_obs import V2V_DIM


def rows():
    from .test_acceleration_cpu import rows as fixture
    result=fixture()
    cavs=[v for v in result if v['is_cav']]
    cavs[1]['s_m']=cavs[0]['s_m']+20.
    return result


def test_zero_init_sampling_scoring_and_noise_are_identical():
    torch.manual_seed(21)
    base=BasePolicy();wide=copy.deepcopy(base);extend_for_v2v(wide,V2V_DIM)
    wide=from_weights(wide.state_dict(),enabled=True)
    context=encode(rows())
    extended=dict(context,v2v={v:[.3]*V2V_DIM for v in context['members']})
    a,_=base.sample(context,torch.Generator().manual_seed(9))
    b,_=wide.sample(extended,torch.Generator().manual_seed(9))
    assert a['action']==b['action'] and a['logp']==b['logp']
    # Widened zero columns can change BLAS summation of the critic by a few
    # float32 ulps on Linux; use the integration's registered 1e-7 value bound.
    assert a['value']==pytest.approx(b['value'],abs=1e-7)
    assert torch.equal(base.score(a)[0],wide.score(b)[0])
    assert type(from_weights(base.state_dict(),enabled=True)) is BasePolicy
    with pytest.raises(ValueError,match='explicit V2V'):
        from_weights(wide.state_dict())


def test_controller_records_partner_features_and_trainable_columns_receive_gradients():
    torch.manual_seed(32)
    policy=Policy()
    controller=controller_class(Controller)(policy,torch.Generator().manual_seed(4),False)
    sensor=rows();commands={r['vid']:0.2 for r in sensor if r['is_cav']}
    controller.apply(sensor,commands,301)
    controller.observe_issued(dict(time_s=301,vehicles=[dict(r,issued_acceleration_mps2=.8)
        for r in sensor]))
    controller.apply(sensor,commands,302)
    sample=controller.events[-1]['sample']
    assert set(sample['context']['v2v'])==set(sample['context']['members'])
    assert all(len(v)==V2V_DIM for v in sample['context']['v2v'].values())
    # Unfreeze the existing zero-output head, then verify PPO's score path
    # reaches the NEW V2V columns (the initial zero head otherwise masks them).
    with torch.no_grad():policy.actor[-1].weight.fill_(.1)
    loss=-policy.score(sample)[0]
    loss.backward()
    assert policy.actor[0].weight.grad[:,29:29+V2V_DIM].abs().sum()>0


def test_bank_and_runtime_features_match_without_current_action_leakage():
    from .bank_v2v import pair_v2v_block
    sensor=rows();members=encode(sensor)['members']
    old=dict(time_s=300,vehicles=[dict(r,issued_acceleration_mps2=.7) for r in sensor])
    now=dict(time_s=301,vehicles=[dict(r,v_mps=r['v_mps']+.4,issued_acceleration_mps2=123.) for r in sensor])
    a,_=pair_v2v_block(now,old,members)
    altered=copy.deepcopy(now)
    for r in altered['vehicles']:r['issued_acceleration_mps2']=-999.
    b,_=pair_v2v_block(altered,old,members)
    assert a==b
    policy=Policy();c=controller_class(Controller)(policy,torch.Generator().manual_seed(2),True)
    c.observe_issued(old)
    c.apply([dict(r,v_mps=r['v_mps']+.4) for r in sensor],{v:.2 for v in members},301)
    assert c.events[-1]['sample']['context']['v2v']==a


def test_missing_partner_payload_fails_and_empty_road_is_supported():
    policy=Policy();ctx=encode(rows())
    with pytest.raises(ValueError,match='partner observations'):policy(ctx)
    mean,value=policy(dict(x=[],rsu=[0.]*12,members=[]))
    assert mean.shape==(0,1) and torch.isfinite(value)


def test_checkpoint_adapter_preserves_optimizer_rng_and_widened_resume(tmp_path):
    from .accel_learner import save
    from .v2v_runtime import load_checkpoint
    base=BasePolicy();opt=torch.optim.Adam(base.parameters(),lr=3e-4)
    generator=torch.Generator().manual_seed(17)
    path=tmp_path/'u0.pt';save(path,base,opt,generator,0,0)
    wide,opt,rng,payload=load_checkpoint(path,config=dict(v2v=True,v2v_dim=V2V_DIM))
    assert isinstance(wide,Policy) and torch.equal(generator.get_state(),rng.get_state())
    assert set(map(id,wide.parameters()))=={id(p) for g in opt.param_groups for p in g['params']}
    context=encode(rows());context['v2v']={v:[.2]*V2V_DIM for v in context['members']}
    loss=wide(context)[1];loss.backward();opt.step()
    next_path=tmp_path/'u1.pt';save(next_path,wide,opt,rng,1,1200)
    restored,optimizer,_,_=load_checkpoint(next_path,config=dict(v2v=True,v2v_dim=V2V_DIM))
    assert torch.equal(restored(context)[1],wide(context)[1])
    assert len(optimizer.state)==len(opt.state)
