"""Register MAIN02:01 learner and baseline-only AWR scale before B/R outcomes."""
import json
from . import ops
from .storage import read,write
from .protocol import checked,sha,REGISTRY
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_awrcpu1'


def register():
    target=ops.EVIDENCE/'MAIN0201_AWR_REGISTRATION.json';assert not target.exists()
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_awrcpu1.json')
    assert cpu['pass_'] and cpu['tests']==133 and cpu['skips']==cpu['sim_s']==0
    data=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib
r=pathlib.Path({ROOT!r});sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.accel_learner import awr_config
for arm in ('B','R'):assert not (r/f'WARM_{{arm}}_DISPATCH.json').exists() and not (r/f'WARM_{{arm}}_STARTED.json').exists()
p=r/'MAPPO_CHECK.json';controls=json.loads(p.read_text())['records']
print(json.dumps(dict(config=awr_config(controls),baseline_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source=str(p))))''',timeout=40))
    value=dict(cpu=cpu,scale=data,new_sumo_sim_s=0,
        own_PPO='mean clipped PPO over nonexecuted steps only; ratio pi_new/pi_old, includes rejected/declined queries; no score/ratio on plan steps',
        AWR='completed plan discounted SUMO return plus frozen rollout critic tail minus start value; positive only; per-member Gaussian NLL, weight min(20,exp(A/T)); coefficient1, no beta',
        critic='same SUMO GAE return, every step; no WM reward',
        fixed_secondary='A/B seed1 update24 retained for sealed-panel evaluation ONLY if additional 24-update seeds are approved later; no present evaluation or checkpoint selection',
        scope='A48/B48/R24 unchanged; A exactu16 Adam/RNG continuation; B/R MAPPOu0')
    write(ops.EVIDENCE/'MAIN0201_AWR_AMENDMENT.json',value);h=sha(ops.EVIDENCE/'MAIN0201_AWR_AMENDMENT.json')
    line=('\nS-COOPMWM MAIN02:01 OWNER LEARNING-UPDATE CORRECTION REGISTERED amendmentSHA'+h+
        ': before any B/R outcome, PPO own chosen/nonexecuted steps only including declined/rejected WM queries, standard pi_new/pi_old; '
        'executed committed plan steps never scored/ratioed for PPO; critic allsteps SUMO team-delay GAE. '
        'Completed-plan A=sum gamma^j SUMOreward/10000+gamma^D frozenV_end-frozenV_start; imitate iff A>0, '
        'weight=min(20,exp(A/T)), coefficient1, no beta. T='+repr(data['config']['temperature'])+
        ' learner units, RMS existing MAPPO TRAIN-check(J900-J600)/10000, baselineSHA'+data['baseline_sha256']+
        '; no A/B/R outcomes used. CPU133PASS/0skips/zeroSUMO. Same update for A/B/R. '
        'A/B seed1 u24 fixed SECONDARY point, evaluated on sealed panel only if extra24-update seeds later approved. '
        'Fixed finalsA/B48,R24, allocation6473440/cap6500000/free26560 unchanged. No new physics in registration.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    binding=dict(config=data['config'],amendment_sha256=h,baseline_sha256=data['baseline_sha256'],cpu=cpu)
    ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({ROOT!r});assert r.is_dir()
for arm in ('B','R'):assert not (r/f'WARM_{{arm}}_DISPATCH.json').exists()
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
with (r/'AWR_BINDING.json').open('x') as f:json.dump(json.loads({json.dumps(binding)!r}),f,sort_keys=True)
print('registered without physics')''',timeout=35)
    write(target,dict(line=line,amendment_sha256=h,binding=binding,new_sumo_sim_s=0))
    print(json.dumps(dict(registered=True,temperature=data['config']['temperature'],amendment_sha256=h,new_sumo_sim_s=0)))


if __name__=='__main__':register()
