"""Saved own-policy R actions/RNG must match before restoring a missed snapshot."""
import copy
import torch
import pytest
from .r_bank_recover import regenerate


def test_saved_episode_uses_lossless_result_reader(tmp_path):
    from . import accel_storage,r_bank_recover
    path=tmp_path/'RESULT.json';result=dict(arm='R',steps=[])
    accel_storage.write(path,result)
    assert r_bank_recover.read(path)==result


class Policy:
    def sample(self,context,generator,deterministic,memory):
        value=torch.randn((1,1),generator=generator).tolist()
        return dict(own_action=value,action=value,previous_noise=[[memory.get('a',0.)]]),{'a':value[0][0]}


def test_saved_own_actions_restore_exact_generator_state():
    p=Policy();g=torch.Generator().manual_seed(42);memory={};steps=[]
    for t in range(300,305):
        s,memory=p.sample({'members':['a']},g,False,memory)
        steps.append(dict(time_s=t,sample=dict(s,context={'members':['a']})))
    restored,m,events,sample=regenerate(p,dict(steps=steps),42,304)
    assert torch.equal(restored.get_state(),g.get_state())
    assert m==memory and len(events)==4 and sample==steps[-1]['sample']


def test_corrupt_saved_actions_or_guidance_never_produces_a_snapshot():
    p=Policy();g=torch.Generator().manual_seed(42)
    s,_=p.sample({},g,False,{})
    result=dict(steps=[dict(time_s=300,sample=dict(s,context={}))])
    bad=copy.deepcopy(result);bad['steps'][0]['sample']['own_action']=[[999.]]
    with pytest.raises(AssertionError):regenerate(p,bad,42,300)
    bad=copy.deepcopy(result);bad['steps'][0]['sample']['behaviour']=dict(executed=True)
    with pytest.raises(AssertionError):regenerate(p,bad,42,300)
