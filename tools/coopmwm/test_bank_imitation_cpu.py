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


def own_pass_bank(n_other=120,n_coop=40):
    def sample(t):
        return dict(context={'members':['a','b']},action=[[t],[t]],
            behaviour=dict(executed=True,proposal_members=['a','b']))
    pairs=[(sample(.05),1.) for _ in range(n_other)]+[(sample(.15),1.) for _ in range(n_coop)]
    return pairs,[False]*n_other+[True]*n_coop


def test_own_pass_guarantees_a_full_first_pass_and_keeps_kl_bounded_when_old_rule_stops_inside_it():
    pairs,flags=own_pass_bank()
    ref=Policy();p=copy.deepcopy(ref);opt=torch.optim.Adam(p.parameters(),lr=.08)
    old=run(p,opt,pairs,torch.Generator().manual_seed(7),ref)
    # the registered rule: the KL stop fires inside the FIRST pass, before one full coverage
    assert old['epochs_completed']==0 and old['pairs_seen']<old['pairs_total']
    assert 'KL' in old['stop_reason']
    p2=Policy();ref2=copy.deepcopy(p2);opt2=torch.optim.Adam(p2.parameters(),lr=.08)
    new=run(p2,opt2,pairs,torch.Generator().manual_seed(7),ref2,imitation_trust='own_pass',cooperative=flags)
    assert new['passes_completed']==new['requested_epochs']==5 and new['epochs_completed']==new['passes_completed']
    assert new['pairs_seen']>=new['pairs_total'] and new['optimizer_steps']>=1
    assert new['kl_after_pass1']<=.045 and new['kl']<=.045
    assert new['kl_reference']=='frozen post-PPO pre-imitation policy (own_pass)'
    assert new['cooperative_pairs']==40
    assert new['action_error_before_cooperative_mps2']==pytest.approx(.15,abs=1e-6)
    assert new['action_error_after_cooperative_mps2']<new['action_error_before_cooperative_mps2']
    assert new['action_error_before_other_mps2']==pytest.approx(.05,abs=1e-6)
    assert new['action_error_after_other_mps2']<new['action_error_before_other_mps2']
    schedule=new['lr_schedule']
    assert schedule['total_halvings']>=1 and 0.<schedule['final_scale']<1.
    assert len(schedule['per_step'])==new['optimizer_steps']
    assert all(step['kl']<=.045 for step in schedule['per_step'])


def test_option_off_reproduces_the_registered_rule_exactly_and_stays_inert():
    p=Policy();ref=copy.deepcopy(p);opt=torch.optim.Adam(p.parameters(),lr=1e-4)
    out=run(p,opt,pairs(),torch.Generator().manual_seed(7),ref)
    p2=Policy();ref2=copy.deepcopy(p2);opt2=torch.optim.Adam(p2.parameters(),lr=1e-4)
    assert run(p2,opt2,pairs(),torch.Generator().manual_seed(7),ref2,1.,5,imitation_trust=None)==out
    p3=Policy();ref3=copy.deepcopy(p3);opt3=torch.optim.Adam(p3.parameters(),lr=1e-4)
    # cooperative flags without the option are inert: the registered rule is byte-identical
    assert run(p3,opt3,pairs(),torch.Generator().manual_seed(7),ref3,1.,5,cooperative=[True]*5)==out
    for key in ('imitation_trust','passes_completed','kl_after_pass1','cooperative_pairs','lr_schedule',
                'action_error_before_cooperative_mps2','action_error_after_cooperative_mps2',
                'action_error_before_other_mps2','action_error_after_other_mps2'):
        assert key not in out
    assert out['kl_reference']=='frozen pre-update policy'
    with pytest.raises(ValueError):run(p,opt,pairs(),torch.Generator(),copy.deepcopy(p),1.,5,imitation_trust='other')
