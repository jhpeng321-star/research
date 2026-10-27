"""Real actor/encoder/WM/actor-update CPU diagnosis, with an in-memory fix proof.

The candidate is applied only to new objects in this process. No runtime file changes.
"""
# ruff: noqa: E402 -- CUDA visibility and isolated source path precede torch imports.
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import types

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path.home() / 'groupgrad'
sys.path[:0] = [str(ROOT / 'runtime/src'), str(ROOT / 'runtime')]
import numpy as np
import torch
from wms.control import policy as policy_module
from wms.control.coopw import term_objectives
from wms.control.counterfactual import CounterfactualConfig
from wms.control.dreamer import DreamerConfig
from wms.control.native_cav import head_native_accel
from wms.control.obs import build_observations
from wms.control.policy import CavPolicy, RsuGroupEncoder, TeamCritic
from wms.control.policy14 import Policy14Trainer, make_controller_factory
from wms.control.reward import team_reward
from wms.control.safety import safe_accel
from wms.control.start_states import StartBatch
from wms.control.train_cli import GroupSignal
from wms.model import ModelSpec, WmSModel
from wms.physics.config import PhysicsConfig


class Sampler:
    def sample(self, batch_size, device='cpu'):
        def row(values, dtype=torch.float32):
            return torch.tensor([values], dtype=dtype).expand(batch_size,-1).clone().to(device)
        state = dict(s_m=row([-60.,-80.,-100.,-120.,-140.,-160.]),
                     v_mps=row([5.,6.,7.,4.,5.,6.]), a_mps2=row([0.]*6), length_m=row([5.]*6),
                     lane=row([1,0,1,1,0,0],torch.long), present=row([True]*6,torch.bool),
                     is_cav=row([True,True,True,False,False,False],torch.bool))
        return StartBatch(state=state)


def grad_vector(term, params):
    gs = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten().double()
                      for p,g in zip(params,gs)])


def magnitude(params):
    return sum(float(p.grad.detach().double().square().sum())
               for p in params if p.grad is not None)**.5


def make_trainer(candidate=False, seed=101):
    torch.manual_seed(seed)
    model = WmSModel(ModelSpec(rollout_mode='driver_params', lane_model='hazard_v5',
                              merge_decision='first_passage', lane_decision='sample'),
                     physics=PhysicsConfig(secure_gap_veto=True))
    policy = CavPolicy(lane_head=False, residual_on_native=True, accel_max=1.2)
    group = GroupSignal(RsuGroupEncoder())
    trainer = Policy14Trainer([model],policy,group,TeamCritic(),Sampler(),
        observation_builder=build_observations,reward_fn=team_reward,safety=safe_accel,
        controller_factory=make_controller_factory(native_ramp_merge='wm_law',native_lane_change=True),
        config=DreamerConfig(batch_size=1,horizon=3,actor_lr=5e-5,
            gamma=1.,lambda_=1.,entropy_weight=0.,uncertainty_weight=0.,
            checkpoint_every=0,validation_every=0,wm_safety_report_only=True),
        counterfactual=CounterfactualConfig(enabled=True,n_individual=0,n_group=1,n_flip=3,
            native_cav='idm',group_min_size=3,group_max_size=3,max_starts=1,
            curriculum_updates=(0.,1.), w_individual=(8.9,8.9),w_group=(9.9,9.9),
            w_coop=(8.9,8.9),w_team=(10.3,10.3),w_all=(1.,1.)))
    if candidate:
        # Proposed default-OFF native_residual_train_group: both stop-gradient barriers
        # must change. Keep zero head, native plant, frozen local features and lane law.
        source = textwrap.dedent(inspect.getsource(CavPolicy.forward))
        assert source.count('features.detach()') == 1
        namespace = dict(vars(policy_module))
        exec(source.replace('features.detach()',
             'torch.cat((own.detach(), context.detach(), signal), -1)'), namespace)
        policy.forward = types.MethodType(namespace['forward'],policy)
        group.requires_grad_(True)
        trainer.actor_parameters = [p for p in list(policy.parameters())+list(group.parameters())
                                    if p.requires_grad]
        trainer.actor_optimizer = torch.optim.Adam(trainer.actor_parameters,lr=trainer.config.actor_lr)
    return trainer


def run_unit(candidate):
    trainer = make_trainer(candidate)
    rows = []
    original = trainer.actor_objective
    def audited(imagined,scale,weights,**kwargs):
        loss,extra = original(imagined,scale,weights,**kwargs)
        params = trainer.actor_parameters
        terms,_ = term_objectives(imagined,scale,trainer.config.lane_pg_weight,uncertainty=False)
        weighted = {k:-weights[k]*v for k,v in terms.items()}
        vectors = {k:grad_vector(v,params) for k,v in weighted.items()}
        total = grad_vector(loss,params)
        coop = sum((v for k,v in vectors.items() if k!='all'),torch.zeros_like(total))
        rows.append(dict(scale=scale,weights=weights,n_rows=imagined['returns'].shape[-1],
            n_groups=len(imagined['variants'].groups),
            weighted_term_grad_norms={k:float(v.norm()) for k,v in vectors.items()},
            coop_terms_combined_grad_norm=float(coop.norm()),total_objective_grad_norm=float(total.norm()),
            coop_to_total_grad_ratio=float(coop.norm()/total.norm()),
            all_to_total_cosine=float(torch.nn.functional.cosine_similarity(vectors['all'],total,dim=0)),
            first_step_adam_delta_difference_max=float((trainer.config.actor_lr*total/(total.abs()+1e-8)-
                trainer.config.actor_lr*vectors['all']/(vectors['all'].abs()+1e-8)).abs().max()),
            gradient_decomposition_max_error=float((sum(vectors.values())-total).abs().max())))
        return loss,extra
    trainer.actor_objective = audited
    before = {k:v.clone() for k,v in trainer.group_encoder.state_dict().items()}
    for _ in range(2):
        metrics = trainer.update()
        rows[-1].update(group_grad_norm=magnitude(trainer.group_encoder.parameters()),
            residual_grad_norm=magnitude(trainer.policy.residual_head.parameters()),
            actor_grad_norm=metrics.get('actor_grad_norm'), actor_loss=metrics['actor_loss'],
            group_changed=sum(not torch.equal(v,trainer.group_encoder.state_dict()[k])
                              for k,v in before.items()))
    result = dict(candidate=candidate, actor_optimizer_parameters=sum(p.numel() for p in trainer.actor_parameters),
                  group_requires_grad=all(p.requires_grad for p in trainer.group_encoder.parameters()),
                  updates=rows)
    assert all(r['residual_grad_norm']>0 for r in rows)
    assert all(r['gradient_decomposition_max_error']<1e-4 for r in rows)
    if candidate:
        assert rows[0]['group_grad_norm']==0 # zero residual W blocks upstream gradient at update zero
        assert rows[1]['group_grad_norm']>0 and rows[1]['group_changed']>0
    else:
        assert all(r['group_grad_norm']==0 and r['group_changed']==0 for r in rows)
    return result


def stats(x):
    return dict(n=x.numel(),mean_abs=float(x.double().abs().mean()),
                median_abs=float(x.abs().median()),p95_abs=float(x.abs().quantile(.95)),
                max_abs=float(x.abs().max()),mean=float(x.double().mean()))


def physical_residuals():
    # All 26 converted cycle1 TRAIN starts, fixed before examining residuals.
    with np.load(ROOT/'checkpoints/starts.pt',allow_pickle=False) as a:
        state = {k.removeprefix('state__'):torch.from_numpy(a[k].copy())
                 for k in a.files if k.startswith('state__')}
    cav = state['present'].bool() & state['is_cav'].bool() & (state['lane']>=0)
    result = dict(n_starts=state['s_m'].shape[0], cohort='all converted cycle1 starts',
                  starts_sha256=hashlib.sha256((ROOT/'checkpoints/starts.pt').read_bytes()).hexdigest())
    # Do one start at a time to bound memory with padded slots and match row-local execution.
    values = {k:[] for k in ('native_head','c00','c01_on','c01_off','c02','recorded_accel')}
    predictions = {}
    group_ablation = {}
    for label in ('c00','c01_on','c01_off','c02'):
        p = torch.load(ROOT/f'checkpoints/{label}.pt',map_location='cpu',weights_only=False)
        policy = CavPolicy(lane_head=False,residual_on_native=True,accel_max=1.2)
        group = GroupSignal(RsuGroupEncoder())
        policy.load_state_dict(p['policy'])
        group.load_state_dict(p['group_encoder'])
        collected=[]
        ablations=[]
        with torch.no_grad():
            for i in range(state['s_m'].shape[0]):
                st={k:v[i:i+1] for k,v in state.items()}
                obs=build_observations(st)
                signal=group(obs,st)
                residual,_=policy(obs,signal,deterministic=True)
                ablated,_=policy(obs,torch.zeros_like(signal),deterministic=True)
                collected.append(residual[cav[i:i+1]])
                ablations.append((residual-ablated)[cav[i:i+1]])
                if label=='c00':
                    native=head_native_accel(st,path=ROOT/'checkpoints/nativehead.pt')
                    values['native_head'].append(native[cav[i:i+1]])
                    values['recorded_accel'].append(st['a_mps2'][cav[i:i+1]])
        predictions[label]=torch.cat(collected)
        group_ablation[label]=stats(torch.cat(ablations))
        values[label]=collected
    result['stats']={k:stats(torch.cat(v)) for k,v in values.items()}
    result['on_minus_off']=stats(predictions['c01_on']-predictions['c01_off'])
    result['group_signal_zero_ablation_delta']=group_ablation
    return result


def main():
    torch.set_num_threads(1)
    result = dict(device='cpu',num_threads=1,baseline=run_unit(False),candidate=run_unit(True),
                  residuals=physical_residuals())
    receipt={}
    needles=('requires_grad_(False)','features.detach()','actor_parameters =','actor_optimizer =',
             'radius = min','residual = radius','credit_multipliers','loss = -','penalty.detach()',
             'self.base, self.paired','self.native_critic')
    for name in ('policy','dreamer','policy14','counterfactual','coopw','paircrit','longcritic','bridge'):
        local=ROOT/f'runtime/src/wms/control/{name}.py'
        deployed=ROOT/f'deployed/{name}.py'
        a,b=local.read_text(),deployed.read_text()
        receipt[name]=dict(unit_sha256=hashlib.sha256(local.read_bytes()).hexdigest(),
            deployed_sha256=hashlib.sha256(deployed.read_bytes()).hexdigest(),
            full_ast_equal=ast.dump(ast.parse(a))==ast.dump(ast.parse(b)),
            deployed_key_lines=[dict(line=i,text=s.strip()) for i,s in enumerate(b.splitlines(),1)
                                if any(n in s for n in needles)])
    result['source_receipt']=receipt
    (ROOT/'diagnosis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    print('CPU diagnosis PASS')


if __name__=='__main__':
    main()
