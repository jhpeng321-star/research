import copy
import torch
from .test_bank_imitation_cpu import Policy,pairs
from .bank_imitation import run_separate
from .bank_statistics import view,replicated_ci


def test_separate_budget_ignores_ppo_kl_and_completes_real_passes():
    old=Policy();p=copy.deepcopy(old)
    with torch.no_grad():p.mean.fill_(1.)
    opt=torch.optim.Adam(p.parameters(),lr=.05)
    result=run_separate(p,opt,pairs(257),torch.Generator().manual_seed(7),old)
    assert result['epochs_completed']>=1
    assert result['pairs_seen']==257*result['epochs_completed']
    assert result['kl']<=.045 and result['backtracks']>0
    assert result['action_error_after_mps2']<result['action_error_before_mps2']
    assert result['critic_steps']==result['plan_ratios_computed']==0
    assert float(p.critic)==3. and opt.param_groups[0]['lr']==.05


def test_student_ci_excludes_unstable_positive_and_no_pseudo_replication():
    c=dict(admitted=True,first_seed_advantage_veh_s=100.,replication_advantage_veh_s=.001,
        extended_advantage_veh_s=10000.,cooperation=dict(complete=True,cooperative=True,cooperation_gain_veh_s=50.))
    original=copy.deepcopy(c);v=view(c)
    assert not v['admitted'] and not v['cooperation']['cooperative']
    assert v['replicated_advantage_ci95'][0]<0 and c==original
    assert replicated_ci(None,1) is None


def test_reproducible_positive_and_replicated_synergy_required():
    c=dict(admitted=True,first_seed_advantage_veh_s=100.,replication_advantage_veh_s=101.,
        cooperation=dict(complete=True,cooperative=True,cooperation_gain_veh_s=30.))
    v=view(c);assert v['admitted'] and not v['cooperation']['cooperative']
    c['cooperation']['replication_cooperation_gain_veh_s']=30.1
    assert view(c)['cooperation']['cooperative']
    c['admitted']=False
    assert not view(c)['admitted']
