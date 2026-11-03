"""Add MAIN12:26's isolated diagnostic alongside the adopted original sidecar."""
import argparse
import json
from . import ops, replay_start_ops, policy_synergy_ops as parent
from .ls_collect_ops import ROOT, COLLECTION
from .ls_validity_ops import remote
from .protocol import REGISTRY, checked, sha
from .storage import read, write

SOURCE='/home/ubuntu/coopmwm20261103_isosynergycpu2'
BASE='/home/ubuntu/coopmwm20261103_policysynergycpu2'
CONTROL=ROOT+'/policy_synergy_isolated_1226_v2'
REG='POLICY_SYNERGY_ISOLATED_REGISTRATION_1226_V2.json'
BINDING='POLICY_SYNERGY_ISOLATED_BINDING_1226_V2.json'


def qualify():
    names=('policy_synergy','policy_synergy_ops','isolated_synergy_ops','test_policy_synergy_isolated_cpu')
    files={n:checked(ops.LOCAL/f'tools/coopmwm/{n}.py').read_text(encoding='utf-8') for n in names}
    replay_start_ops.BASE=BASE
    print(remote('cpuQ',replay_start_ops.qualification_payload(files,source=SOURCE,
        selection=['test_policy_synergy_isolated_cpu','test_policy_synergy_cpu',
                   'test_replay_bank_lesson_cpu','test_sumo_verify_cpu']),timeout=50))


def cpu_result():
    replay_start_ops.SOURCE=SOURCE;replay_start_ops.cpu_result()


def register():
    p=ops.EVIDENCE/REG
    if not p.exists():
        cpu=read(ops.EVIDENCE/('B_REPLAY_START_CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
        assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
        live=json.loads(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({ROOT!r});b=json.loads((r/'B_REPLAY_EXPOSURE_V2.json').read_text())
print(json.dumps(dict(budget=b['budget'],gpu_deadline=b['gpu_deadline'])))''',timeout=40))
        line=('\nS-COOPMWM MAIN2026-11-03 12:26 ISOLATED POLICY COOPERATION DIAGNOSTIC V2: '
            'V1 isolated charge keys collided with original diagnostic, all blocked '
            'BEFORE any isolated reservation/physics. Retain V1 reports/alerts/source; '
            'retire only verified idle isolated owner3511227. V2 distinct charge/output '
            'namespaces and one-use IDs; no original branch/job replay. '
            'add independent durable sidecar fromu9 throughu24 for BOTH B/R, keep '
            'original3443527 and every original report/branch. Same frozen cooperative '
            'state list as original each update; isolate allgroupNEW vs eachsoloNEW, '
            'all OTHER CAVs u3 (including new arrivals). Also background-only NEW '
            'with group u3. Same original state/prefix/window/CRN, captured t0 retained, '
            'interventiont0+1. Reuse sealed original same-checkpoint u3 reference, '
            'NEVER reuse all-new lesson as isolated. New(n+2)branches precharged '
            'atomically bank_verification_ext, <=9600sim-s/state n<=6/end<=1200; '
            'phase/total guard14.5M unchanged, no implicit transfer; shortage alerts '
            'before physics. One branch/nice19, health guards/no physics retry. '
            'Publish isolated and original jointly as POLICY_SYNERGY_uNNN_ARM_WITH_ISOLATED_V2.json, '
            'original sealed canonical remains immutable. No training/GPU/optimizer '
            'handover or scientific selection; no sealed outcome access.\n')
        write(p,dict(line=line,authority='MAIN2026-11-03 12:26',source_root=SOURCE,
            control_root=CONTROL,collection_root=COLLECTION,cpu=cpu,**live,
            isolated=True,output_suffix='_ISOLATED_V2',first_update=9,final_update=24,phase='bank_verification_ext',new_reserved_sim_s=0))
    v=read(p);h=sha(p);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(v['line']+'amendmentSHA'+h+'\n')
    print(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({ROOT!r});v=json.loads({json.dumps(dict(v,amendment_sha256=h))!r});p=r/{BINDING!r}
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
a=r/'AUTHORITY.md'
if {h!r} not in a.read_text():
 with a.open('a') as f:f.write(v['line']+'amendmentSHA'+{h!r}+chr(10))
print(json.dumps(dict(registered=True,sha256={h!r},new_reserved_sim_s=0)))''',timeout=40))


def setup_parent():
    parent.SOURCE=SOURCE;parent.CONTROL=CONTROL;parent.BINDING=BINDING
    parent.DISPATCH='POLICY_SYNERGY_ISOLATED_DISPATCH_1226_V2.json'
    parent.REPORT_PATTERN='POLICY_SYNERGY_u[0-9][0-9][0-9]_[BR]_ISOLATED_V2.json'
    parent.STATUS_PREFIX='POLICY_SYNERGY_ISOLATED_STATUS'
    parent.COMBINED_PATTERN='POLICY_SYNERGY_u[0-9][0-9][0-9]_[BR]_WITH_ISOLATED_V2.json'


def launch():setup_parent();parent.launch()
def status():setup_parent();return parent.status()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cmd',choices=('qualify','cpu_result','register','launch','status'))
    globals()[p.parse_args().cmd]()
