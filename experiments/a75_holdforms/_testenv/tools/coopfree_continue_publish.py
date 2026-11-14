"""Singleton OWNER15:11 publisher/control-plane observer; never dispatches SUMO."""
import argparse
import base64
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from tools.coopbandit.ops import remote
from tools.coopfree_continue_ops import ROOT, REG
from tools.coopfree_publish import immutable, DocumentationBusy
from tools.coopfree_hold_publish import transport_failure
from tools.coopfree_learning_checks import render

LOCAL=Path(__file__).resolve().parents[1]
PUBLIC=LOCAL/'tools/coopfree/publications/owner1511'
BEGIN='<!-- COOPFREE OWNER1511 BEGIN -->'
END='<!-- COOPFREE OWNER1511 END -->'


def collect(known=()):
    source='''import pathlib,json,hashlib,base64,time,os
r=pathlib.Path(ROOT)
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=read(r/'CONTINUE_DISPATCH.json');assert d['root']==ROOT and d['pid']==1579328 and d['start_ticks']=='53193103'
cert=read(r/'CONTINUE_CPU.json')
for name,expected in cert['sources'].items():assert sha(r/'pytools/tools'/name)==expected
assert sha(r/'CONTINUE_PRIOR.json')==cert['prior_sha256']
assert sha(r/'CONTINUE_DESIGN.md')==cert['design_sha256']
p=pathlib.Path('/proc/'+str(d['pid'])+'/stat')
alive=p.exists() and p.read_text().split()[21]==d['start_ticks'] and p.read_text().split()[2]!='Z'
files={};updates=[];panels=[];checks=[];plans=[]
def add(name,view=None):
 p=r/name
 if not p.exists():return
 data=p.read_bytes() if view is None else (json.dumps(view,sort_keys=True,separators=(',',':'))+'\\n').encode()
 expected=hashlib.sha256(data).hexdigest()
 if name not in KNOWN:files[name]=dict(sha256=expected,base64=base64.b64encode(data).decode())
 return expected
for name in ('CONTINUE_CPU.json','CONTINUE_AUTHORITY.json','CONTINUE_DISPATCH.json','CONTINUE_PRIOR.json','INHERITED.json','STOP.json','CONTINUE_BUDGET_STOP.json','CONTINUE_COMPLETE.json','CONTINUE_FIX_REQUIRED.json','CONTINUE_FIX_AUTHORITY.json','RECOVERED_HOLD_DISCOVERY_001.json'):
 add(name)
for p in sorted(r.glob('HOLD_UPDATE_*.json')):
 u=read(p);name='policy_hold_u%03d.pt'%u['update']
 assert sha(r/name)==u['checkpoint_sha256']
 assert u['windows']>=150 and u['worlds']>=12 and u['baseline'].startswith('zero-update policy-mean HOLD')
 add(name);add(p.name)
 updates.append({k:v for k,v in u.items() if k not in ('records','charged')})
for p in sorted(r.glob('CONTINUE_DEV_*.json')):
 full=read(p);view={k:v for k,v in full.items() if k!='learned_rows'}
 view['remote_full_receipt_sha256']=sha(p)
 view['paired_world_metrics']=[dict(world=v['world'],metrics=v['metrics']) for v in full['learned_rows']]
 add(p.name,view);panels.append(view)
for p in sorted(r.glob('CONTINUE_CHECK_*.json')):add(p.name);checks.append(read(p))
for p in sorted(r.glob('CONTINUE_DISCOVERY_*.json')):add(p.name)
for p in sorted(r.glob('CONTINUE_PLAN_*.json')):
 v=read(p);plans.append(dict(update=v['update'],tasks=len(v['tasks']),complete=sum((r/'jobs'/t['id']/'CLOSE.json').exists() for t in v['tasks']),charge=v['charge'],cached_holds=v['cached_holds']))
charges=[json.loads(s) for s in (r/'CHARGES.jsonl').read_text().splitlines()]
closed=[read(r/'jobs'/v['id']/'CLOSE.json') for v in charges if (r/'jobs'/v['id']/'CLOSE.json').exists()]
out=dict(unix=time.time(),dispatch=d,alive=alive,load=os.getloadavg(),charged=sum(v['sim_s'] for v in charges),actual_closed=sum(v['actual_sim_s'] for v in closed),updates=updates,panels=panels,checks=checks,plans=plans,files=files,stop=(r/'STOP.json').exists(),complete=(r/'CONTINUE_COMPLETE.json').exists(),fix=read(r/'CONTINUE_FIX_REQUIRED.json') if (r/'CONTINUE_FIX_REQUIRED.json').exists() else None,fix_authorized=(r/'CONTINUE_FIX_AUTHORITY.json').exists())
print(json.dumps(out))
'''
    return json.loads(remote('cpuQ','ROOT='+repr(ROOT)+'\nKNOWN='+repr(set(known))+'\n'+source,timeout=60))


def decoded(files):
    out={}
    for name,item in files.items():
        if not re.fullmatch(r'(?:[A-Z][A-Z0-9_]*\.json|policy_hold_u\d{3}\.pt)',name):
            raise ValueError('invalid public artifact name: '+name)
        data=base64.b64decode(item['base64'],validate=True)
        if hashlib.sha256(data).hexdigest()!=item['sha256']:
            raise ValueError('publication transport hash')
        out[name]=data
    return out


def report(snapshot):
    lines=['## OWNER15:11 live continuation',
        'Observed UTC '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(snapshot['unix']))+'.',
        f"cpuQ owned1579328/start53193103 alive={snapshot['alive']}; four nice19 workers, load admission12.",
        f"Budget: {snapshot['charged']:,} precharged / {snapshot['actual_closed']:,.0f} closed-actual sim-s; total cap6,000,000.",
        'Corrected u001 is inherited; every subsequent update uses fresh exploration. TRAIN effects are not DEV gains.']
    for p in snapshot['plans'][-2:]:
        lines.append(f"- u{p['update']:03}: {p['complete']}/{p['tasks']} new arms closed; charge{p['charge']:,}; cachedHOLD{p['cached_holds']}.")
    for u in snapshot['updates']:
        lines += [f"### Proposal checkpoint u{u['update']:03}",
            f"`tools/coopfree/publications/owner1511/policy_hold_u{u['update']:03}.pt`",
            f"SHA256 `{u['checkpoint_sha256']}`; receipt `HOLD_UPDATE_{u['update']:03}.json`.",
            f"- {u['windows']}windows/{u['worlds']}worlds; exactsoloCG {u['exact_credit_windows']}windows/{u['exact_credit_worlds']}worlds.",
            f"- Executed {u['executed_change_share']:.4f}; issued {u['issued_change_share']:.4f}; executed/issued {u['executed_vs_issued_share']}; overrides {u['safety_override_share']:.4f}."]
        for key in ('D_G','sum_D_i','C_G'):
            v=u[key]
            lines.append(f"- {key}: n={v['n']}, mean{v['mean']:+.3f}, SD{v['sd']:.3f}; quantiles{v['quantiles']}.")
        lines.append(f"- Share exactCG>0={u['positive_C_G_share']}; sigma{u['sigma_used']} -> {u['sigma_next']}.")
    for panel in snapshot['panels']:
        lines.append(f"### Paired DEV u{panel['update']:03} (24worlds; negative delay is better)")
        comparators=dict(panel['learned_minus_comparator'],hold=panel['paired_hold']['learned_minus_hold'])
        for comparator,values in comparators.items():
            for horizon in ('600','900'):
                v=values[horizon+'_team_delay_vs']
                lines.append(f"- vs {comparator} J{horizon}: {v['mean']:+.3f}, paired95%CI{v['ci95']} vehicle-s.")
                safety={k:values[horizon+'_'+k] for k in ('collisions','teleports','ttc_below_3_vs','hard_brake')}
                lines.append('- Paired safety: '+json.dumps(safety,sort_keys=True)+'.')
    for check in snapshot['checks']:
        lines.append(render(check))
    lines.append('Discovery: observational COOPBEHAV/clustering exports are separate from exact HOLD C_G; do not infer paired native-release mechanisms.')
    if snapshot['fix']:
        lines.append('Registered phase fix: '+json.dumps(snapshot['fix']['phase'],sort_keys=True)+'.')
    if snapshot['stop']:
        lines.append('**Durable STOP: inspect published receipt; no automatic physics retry.**')
    if snapshot['complete']:
        lines.append('Production complete within whole-batch accounting; inspect final DEV and limitations before any gain claim.')
    return '\n'.join(lines)


def replace(path,text):
    original=path.read_text(encoding='utf-8')
    block=BEGIN+'\n'+text+'\n'+END
    if BEGIN in original:
        result=original[:original.index(BEGIN)]+block+original[original.index(END)+len(END):]
    else:
        result=original.rstrip()+'\n\n'+block+'\n'
    path.write_text(result,encoding='utf-8')


def authorize_fix(snapshot):
    if not snapshot['fix'] or snapshot['fix_authorized']:
        return
    fix=snapshot['fix']
    marker='- S-COOPFREE OWNER15:11 CONDITIONAL PHASE FIX '
    line=marker+json.dumps(dict(after_update=fix['after_update'],phase=fix['phase'],
        owner='15:11 registered fix playbook; same6M cap/no completed physics replay'),sort_keys=True,separators=(',',':'))
    text=REG.read_text(encoding='utf-8')
    existing=[v for v in text.splitlines() if v.startswith(marker)]
    if existing:
        if existing!=[line]:raise ValueError('conflicting phase registration')
    else:
        with REG.open('a',encoding='utf-8') as f:
            f.write('\n'+line+'\n');f.flush();os.fsync(f.fileno())
    authority=dict(registration_line=line,phase=fix['phase'])
    # This releases an already-running supervisor at a predeclared phase boundary;
    # it neither creates a process nor retries a simulation.
    remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({ROOT!r});assert r.is_dir()
p=r/'CONTINUE_FIX_AUTHORITY.json';value=json.loads({json.dumps(authority)!r})
assert not (r/'STOP.json').exists()
fix=json.loads((r/'CONTINUE_FIX_REQUIRED.json').read_text());assert fix['phase']==value['phase']
if p.exists():assert json.loads(p.read_text())==value
else:
 with p.open('x') as f:f.write(json.dumps(value,sort_keys=True)+'\\n');f.flush()
print('registered phase authority accepted; no Popen/no retry')
''',timeout=20)
    immutable(PUBLIC/'PHASE_FIX_REGISTRATION.json',(json.dumps(authority,sort_keys=True)+'\n').encode())


def git(*args):
    return subprocess.check_output(['git',*args],cwd=LOCAL,text=True).strip()


def once(commit=False):
    manifest_path=PUBLIC/'MANIFEST.json'
    known=list(json.loads(manifest_path.read_text())) if manifest_path.exists() else []
    snapshot=collect(known)
    artifacts=decoded(snapshot['files'])
    milestone=bool(artifacts)
    if milestone:
        for name,data in artifacts.items():immutable(PUBLIC/name,data)
        # Never overwrite owner edits; dirty docs defer only documentation/publication commit.
        if git('status','--porcelain','--','HANDOFF.md','docs/paper_results/COOPFREE.md'):
            raise DocumentationBusy('publication docs have uncommitted changes')
        text=report(snapshot)
        for name in ('HANDOFF.md','docs/paper_results/COOPFREE.md'):replace(LOCAL/name,text)
        manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in PUBLIC.iterdir() if p.is_file() and p.name!='MANIFEST.json'}
        (PUBLIC/'MANIFEST.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf-8')
        if commit:
            if git('diff','--cached','--name-only'):raise DocumentationBusy('foreign staged changes')
            paths=['HANDOFF.md','docs/paper_results/COOPFREE.md',str(PUBLIC.relative_to(LOCAL))]
            git('status','--short');git('diff','--stat');git('log','--oneline','-10')
            git('add','--',*paths)
            if git('diff','--cached','--name-only'):git('commit','-m','coopfree: publish approved HOLD continuation milestone')
    authorize_fix(snapshot)
    print(json.dumps(dict(observed=snapshot['unix'],alive=snapshot['alive'],
        updates=[u['update'] for u in snapshot['updates']],plans=snapshot['plans'],
        charged=snapshot['charged'],actual_closed=snapshot['actual_closed'],new_artifacts=list(artifacts),
        stop=snapshot['stop'],complete=snapshot['complete'])),flush=True)
    return snapshot['stop'] or snapshot['complete']


@contextlib.contextmanager
def singleton():
    # Existing Windows worktree has a separate gitdir; use git, not .git/path assumptions.
    path=Path(git('rev-parse','--git-path','coopfree_owner1511_publisher.lock'))
    if not path.is_absolute():path=LOCAL/path
    with path.open('a+b') as lock:
        if os.name=='nt':
            import msvcrt
            lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--commit',action='store_true');a=p.parse_args()
    with singleton():
        deadline=time.monotonic()+48*3600
        while True:
            try:
                done=once(a.commit)
            except DocumentationBusy as e:
                print(json.dumps(dict(documentation_busy=str(e))),flush=True);done=False
            except (RuntimeError,subprocess.TimeoutExpired,ConnectionError) as e:
                if not transport_failure(e):raise
                print(json.dumps(dict(transport_error=str(e),no_physics_retry=True)),flush=True);done=False
            if done or not a.watch or time.monotonic()>=deadline:break
            time.sleep(120)


if __name__=='__main__':main()
