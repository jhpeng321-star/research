"""MAIN06:27 append-only budget reconciliation; no reservation or job dispatch."""
import argparse
import copy
import json

from . import ops
from .ls_collect_ops import ROOT
from .ls_validity_ops import remote
from .protocol import REGISTRY, checked, digest, sha
from .storage import read, write

CAP = 14_500_000
NAME = 'B_REPLAY_PHASE_RECONCILIATION_0627.json'


def reconcile(budget, charges, superseded, resume=3):
    """Protect every phase and retained failure; expose shortages, never fund them."""
    b = copy.deepcopy(budget)
    assert b['total'] == CAP
    assert sum(b['phases'].values()) == b['allocated'] <= CAP
    assert b['phases']['superseded'] == superseded
    assert 'superseded' not in charges, 'superseded must be counted exactly once'
    charged = dict(charges, superseded=superseded)
    assert all(type(v) is int and v >= 0 for v in charged.values())
    assert not set(charged)-set(b['phases']), 'unrecognised charged phase'
    phases = {p: dict(allocated=a, charged=charged.get(p, 0),
                      remaining=a-charged.get(p, 0))
              for p, a in sorted(b['phases'].items())}
    assert all(v['remaining'] >= 0 for v in phases.values())
    total_charged = sum(charged.values())
    assert total_charged <= CAP
    required = 2*(24-resume)*24*1200
    shortfall = max(0, required-phases['train']['remaining'])
    return dict(budget=b, phases=phases, total_charged_sim_s=total_charged,
        total_remaining_sim_s=CAP-total_charged,
        remaining_PPO=dict(arms=['B', 'R'], resume_update=resume, final_update=24,
            updates=2*(24-resume), episodes_per_update=24, sim_s_per_episode=1200,
            required_sim_s=required, train_phase_shortfall_sim_s=shortfall),
        funding_complete=shortfall == 0, new_reserved_sim_s=0,
        restrictions=dict(sumo_verification_k8='restricted; no transfer to PPO',
            interim_u8_test='restricted interim comparison only',
            bank_verification_ext='bank verification, confirmation, synergy and lesson checks',
            final_test='existing final episode liabilities',
            paired_branch='existing final paired-branch liabilities'),
        no_phase_transfer=True, charges_retained=True)


def register():
    path = ops.EVIDENCE/NAME
    if path.exists():
        value = read(path)
    else:
        snapshot = json.loads(remote('cpuQ', f'''from pathlib import Path
import json,sys,time,hashlib,collections
r=Path({ROOT!r});sys.path.insert(0,'/home/ubuntu/coopmwm20261103_bankcpu2/pytools')
from tools.coopmwm.accel_storage import ledger,SUPERSEDED_CHARGE
p=r/'BANK_VERIFICATION_CAP_2232.json';v=json.loads(p.read_text());counts=collections.Counter()
entries=ledger(r)
for e in entries:counts[e['phase']]+=e['sim_s']
print(json.dumps(dict(unix=time.time(),budget=v['budget'],charges=dict(counts),superseded=SUPERSEDED_CHARGE,parent_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),ledger_sha256=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest())))''', timeout=40))
        value = reconcile(snapshot['budget'], snapshot['charges'], snapshot['superseded'])
        value.update(snapshot=snapshot, authority='MAIN2026-11-03 06:27',
            brief_sha256=sha(ops.LOCAL.parent/'_sol/coopmwm_prompt.md'),
            operational_cap_authority=CAP, supersedes='fresh6M resume guard',
            live_guard_changed=False, new_sumo_sim_s=0)
        write(path, value)
    h = sha(path)
    line = ('\nS-COOPMWM MAIN2026-11-03 06:27 B-REPLAY PHASE RECONCILIATION '
        'amendmentSHA'+h+': operational authority14500000 supersedes fresh6Mguard. '
        'Existing phase allocations and restricted uses retained append-only; '
        f'charged{value["total_charged_sim_s"]}; totalremaining{value["total_remaining_sim_s"]}. '
        f'Remaining B/Ru4-u24 PPO1209600; trainphaseavailable{value["phases"]["train"]["remaining"]}; '
        f'trainphaseshortfall{value["remaining_PPO"]["train_phase_shortfall_sim_s"]}. '
        'No phase transfer, reservation, job dispatch or sealed access; report exact '
        'phase shortage to MAIN before full-run funding/launch. Bank confirmation, '
        'lesson checks and protected final/interim liabilities remain in their phases.\n')
    registry = checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a', encoding='utf-8') as f:
            f.write(line)
    binding = dict(value, amendment_sha256=h, line=line)
    result = remote('cpuQ', f'''from pathlib import Path
import json
r=Path({ROOT!r});p=r/{NAME!r};v=json.loads({json.dumps(binding)!r})
if p.exists():assert json.loads(p.read_text())==v,'adopt original reconciliation'
else:
 assert r.is_dir()
 with p.open('x') as f:json.dump(v,f,sort_keys=True)
authority=r/'AUTHORITY.md'
if {h!r} not in authority.read_text():
 with authority.open('a') as f:f.write({line!r})
print(json.dumps(dict(registered=True,amendment_sha256={h!r},new_reserved_sim_s=0)))''', timeout=40)
    print(result)
    print(json.dumps(dict(amendment_sha256=h,
        total_charged_sim_s=value['total_charged_sim_s'],
        total_remaining_sim_s=value['total_remaining_sim_s'],
        train_phase=value['phases']['train'], remaining_PPO=value['remaining_PPO'])))


def transfer():
    """MAIN07:06 exact authorized transfer; preserve total and all old records."""
    name='B_REPLAY_PHASE_TRANSFER_0706.json';path=ops.EVIDENCE/name
    if path.exists():
        value=read(path)
    else:
        parent=read(ops.EVIDENCE/NAME)
        snapshot=json.loads(remote('cpuQ',f'''from pathlib import Path
import sys,json,time,collections
r=Path({ROOT!r});sys.path.insert(0,'/home/ubuntu/coopmwm20261103_bankcpu2/pytools')
from tools.coopmwm.accel_storage import ledger,SUPERSEDED_CHARGE
counts=collections.Counter()
for x in ledger(r):counts[x['phase']]+=x['sim_s']
print(json.dumps(dict(unix=time.time(),charges=dict(counts),superseded=SUPERSEDED_CHARGE)))''',timeout=40))
        b=copy.deepcopy(parent['budget'])
        checks=min(144000,b['phases']['checks']-snapshot['charges'].get('checks',0))
        remainder=144000-checks
        assert checks>=0
        assert b['phases']['sumo_verification']-snapshot['charges'].get('sumo_verification',0)>=remainder
        b['phases']['checks']-=checks
        b['phases']['sumo_verification']-=remainder
        b['phases']['train']+=144000
        value=reconcile(b,snapshot['charges'],snapshot['superseded'])
        value.update(authority='MAIN2026-11-03 07:06',snapshot=snapshot,
            parent_sha256=sha(ops.EVIDENCE/NAME),
            transfer=dict(checks_to_train=checks,sumo_verification_to_train=remainder,total=144000),
            learning_checks='B-REPLAY lesson checks replace old B/R learning checks; bank_verification_ext',
            no_phase_transfer=False,new_sumo_sim_s=0,live_guard_changed=False)
        assert value['funding_complete']
        write(path,value)
    h=sha(path)
    line=('\nS-COOPMWM MAIN2026-11-03 07:06 AUTHORIZED PHASE TRANSFER amendmentSHA'+h+
        f': checks->train{value["transfer"]["checks_to_train"]}; sumo_verification->train{value["transfer"]["sumo_verification_to_train"]}; '
        'total14500000 unchanged; train3008400/checks288000; all retained charges protected. '
        'B/Ru4-u24 PPO1209600 fully covered at snapshot; B-REPLAY lessons replace old '
        'B/R learning checks and use bank_verification_ext; interim uses interim_u8_test. '
        'Append-only record, original bindings retained; no reservation/physics/optimizer, '
        'restart still requires10 admitted entries and qualified runtime.\n')
    registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(line)
    binding=dict(value,amendment_sha256=h,line=line)
    print(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({ROOT!r});p=r/{name!r};v=json.loads({json.dumps(binding)!r})
if p.exists():assert json.loads(p.read_text())==v
else:
 assert r.is_dir()
 with p.open('x') as f:json.dump(v,f,sort_keys=True)
p=r/'AUTHORITY.md'
if {h!r} not in p.read_text():
 with p.open('a') as f:f.write({line!r})
print(json.dumps(dict(registered=True,amendment_sha256={h!r},new_reserved_sim_s=0)))''',timeout=40))
    print(json.dumps(dict(amendment_sha256=h,transfer=value['transfer'],
        train_phase=value['phases']['train'],funding_complete=value['funding_complete'])))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('register','transfer'),nargs='?',default='register')
    a=p.parse_args();globals()[a.action]()
