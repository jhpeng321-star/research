"""Register logging and owner-gated future proposer activation; zero physics."""
import json
from . import ops
from .protocol import checked,REGISTRY,sha
from .storage import write
from .warm_deploy import ROOT


def register():
    target=ops.EVIDENCE/'EXACT_PROPOSER_PREREGISTRATION.json'
    assert not target.exists()
    doc=ops.LOCAL/'docs/paper_results/PLAN_PROPOSER_PREREG.md'
    code=ops.LOCAL/'tools/coopmwm/plan_proposer.py'
    before=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({ROOT!r})
assert not list(r.glob('B_UPDATE_*.json')) and not list(r.glob('R_UPDATE_*.json'))
print(json.dumps(dict(B_R_updates=0,new_sumo_sim_s=0)))''',timeout=30))
    line=('\nS-COOPMWM MAIN0252/0253 LEARNED PROPOSER PRE-REGISTERED docSHA'+sha(doc)+' codeSHA'+sha(code)+
        ': logging-only RecordingEvaluator from first B search, byte-identical scores; no activation now. '
        'B same-arm/seed TRAIN only, exact-positive SUMO bonus only; R never. Minimum200Bsearches/32TRAINworlds; '
        'offline world-bootstrap checks at every4-update boundary throughu40, config frozen in SHA-pinned document. '
        'Activation requires BOTH offline PASS AND explicit OWNER approval recorded by MAIN in REGISTRATION.md; NEVER automatic. '
        'Seeds replace population slots, registered horizons/search cost unchanged; no SUMO spending.\n')
    if line not in checked(REGISTRY).read_text(encoding='utf-8'):
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    ops.ssh.remote('cpuQ',f'''import pathlib
r=pathlib.Path({ROOT!r});assert not list(r.glob('B_UPDATE_*.json'))
if {line!r} not in (r/'AUTHORITY.md').read_text():
 with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
print('registered before B/R updates; zero physics')''',timeout=30)
    write(target,dict(line=line,document_sha256=sha(doc),code_sha256=sha(code),pre_outcome=before,new_sumo_sim_s=0))
    print(json.dumps(dict(registered=True,document_sha256=sha(doc),activation=False,new_sumo_sim_s=0)))


if __name__=='__main__':
    from .warm_transport import retry
    retry(register,lambda:None,lambda event:print(json.dumps(event),flush=True))
