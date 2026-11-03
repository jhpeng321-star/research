import copy
import math
import pytest
import torch
from .bank_imitation import run,joint_kl


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.mean=torch.nn.Parameter(torch.tensor(0.))
        self.log_std=torch.nn.Parameter(torch.tensor([math.log(.5)]));self.critic=torch.nn.Parameter(torch.tensor(3.))
    def forward(self,context):return self.mean.expand(len(context['members']),1),self.critic
    def score(self,sample):raise AssertionError('imitation must not compute plan ratios')


def pairs(n=5):
    sample=dict(context={'members':['a','b']},action=[[.1],[.1]],
        behaviour=dict(executed=True,proposal_members=['a','b']))
    return [(copy.deepcopy(sample),1.) for _ in range(n)]


def test_five_full_passes_cover_all_members_and_reduce_error_without_critic_or_ratios():
    p=Policy();ref=copy.deepcopy(p);opt=torch.optim.Adam(p.parameters(),lr=1e-4)
    out=run(p,opt,pairs(),torch.Generator().manual_seed(7),ref)
    assert out['epochs_completed']==5 and out['pairs_seen']==25
    assert out['member_actions_seen']==50 and out['optimizer_steps']==5
    assert out['action_error_after_mps2']<out['action_error_before_mps2']
    assert out['plan_ratios_computed']==out['critic_steps']==0
    assert float(p.critic)==3.


def test_existing_pre_update_kl_is_the_only_early_stop():
    ref=Policy();p=copy.deepcopy(ref)
    with torch.no_grad():p.mean.fill_(1.)
    opt=torch.optim.Adam(p.parameters(),lr=1e-4)
    out=run(p,opt,pairs(),torch.Generator().manual_seed(7),ref)
    assert out['epochs_completed']==out['optimizer_steps']==0
    assert out['kl']>.045 and 'KL' in out['stop_reason']
    assert out['action_error_after_mps2']==out['action_error_before_mps2']
    assert 'KL' in out['error_not_decreased_explanation']


def test_empty_bank_and_no_shrinking_pass_count():
    p=Policy();opt=torch.optim.Adam(p.parameters())
    out=run(p,opt,[],torch.Generator().manual_seed(7),copy.deepcopy(p))
    assert out['epochs_completed']==0 and out['pairs_total']==0
    assert joint_kl(p,p,[pairs()[0][0]])==0.
    with pytest.raises(ValueError):run(p,opt,pairs(),torch.Generator(),copy.deepcopy(p),passes=4)
