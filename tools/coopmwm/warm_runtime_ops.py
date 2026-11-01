"""Bind the corrected CPU8 B/R learner without changing live A's CPU6 source."""
import json
from . import ops
from .storage import read,write
from .protocol import checked,sha,REGISTRY
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_mppicpu8'


def register():
    dest=ops.EVIDENCE/'WARM_BR_CPU8_REGISTRATION.json';assert not dest.exists()
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_mppicpu8.json')
    assert cpu['pass_'] and cpu['tests']==126 and not cpu['skips'] and not cpu['sim_s']
    centre=read(ops.EVIDENCE/'WARM_R_CENTRE_REGISTRATION.json')
    value=dict(cpu=cpu,source_root=SOURCE,hashes=cpu['source_hashes'],R_centre_amendment=centre['amendment_sha256'],
        correction='trust layout derived from all10 option/regime buckets, not hardcoded8; MPPI never aliases own controls',
        A='preserved live CPU6 source; no executed MPPI plans, so unaffected',new_sumo_sim_s=0)
    write(ops.EVIDENCE/'WARM_BR_CPU8_AMENDMENT.json',value);h=sha(ops.EVIDENCE/'WARM_BR_CPU8_AMENDMENT.json')
    line=('\nS-COOPMWM B/R CPU8 ENGINEERING RELEASE SOURCE REGISTERED amendmentSHA'+h+
        ': CPU126PASS/0skips/zeroSUMO; before any B/R update, fix MPPI trust bookkeeping: '
        '10 type/regime buckets and2 own-regime controls use separate dynamic columns (old8-bucket offsets aliased MPPI/controls). '
        'R same-policy-centre correction retained, candidate scores zero, helper after B-u4 pause. '
        'A CPU6 u4/Adam/RNG continuation unaffected. B/R launch only CPU8 source via explicit per-armV2bindings; '
        'GPU numerical planner/encoder/WM unchanged; full warm GPU release still required. Allocation6473440/cap6500000 unchanged.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    binding=dict(source_root=SOURCE,hashes=cpu['source_hashes'],cpu=cpu,amendment_sha256=h)
    result=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({ROOT!r});source=pathlib.Path({SOURCE!r});assert r.is_dir() and source.is_dir()
for arm in ('B','R'):
 assert not (r/f'WARM_{{arm}}_DISPATCH.json').exists() and not (r/f'WARM_{{arm}}_STARTED.json').exists()
 with (r/f'WARM_{{arm}}_RUNTIME_BINDING_V2.json').open('x') as f:json.dump(json.loads({json.dumps(binding)!r}),f,sort_keys=True)
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
print(json.dumps(dict(registered=True,source_root=str(source),new_sumo_sim_s=0)))''',timeout=30))
    write(dest,dict(line=line,amendment_sha256=h,binding=binding));print(json.dumps(result))


if __name__=='__main__':register()
