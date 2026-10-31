"""Read-only SUMO-run observer; publish verified proposals without dispatching physics.

Run from this worktree: python -m tools.coopfree_publish --watch --commit
The singleton lock and operational log live under .git, outside scientific sources.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from tools.coopbandit.ops import remote

LOCAL = Path(__file__).resolve().parents[1]
ROOT = '/home/ubuntu/coopfree20261101_build5'
PUBLIC = LOCAL / 'tools/coopfree/publications'
BEGIN = '<!-- COOPFREE PUBLICATIONS BEGIN -->'
END = '<!-- COOPFREE PUBLICATIONS END -->'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def collect(known=()):
    """Read only explicit public receipts/checkpoints and this run's charge/job receipts."""
    source = '''import base64,hashlib,json,pathlib,time
r=pathlib.Path(ROOT)
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
dispatch=read(r/'DISPATCH.json')
assert dispatch['pid']==339173 and dispatch['start_ticks']=='47835664'
assert dispatch['root']==ROOT and dispatch['host']=='cpuQ'
binding=read(r/'BINDING.json')
assert sha(r/'BINDING.json')=='987a7ae0bd0c056bd06f8c6b4a099f00852ca2ba98961659efd6bb5a1481e05f'
for name,expected in binding['sources'].items():
 assert sha(r/'pytools'/name)==expected,'bound scientific source changed'
proc=pathlib.Path('/proc/'+str(dispatch['pid'])+'/stat')
alive=proc.exists() and proc.read_text().split()[21]==dispatch['start_ticks']
charges=[json.loads(line) for line in (r/'CHARGES.jsonl').read_text().splitlines()]
closed=[];complete=[];started=[]
for row in charges:
 p=r/'jobs'/row['id']
 if (p/'START.json').exists(): started.append(row['id'])
 if (p/'CLOSE.json').exists(): closed.append(read(p/'CLOSE.json'))
 if (p/'RESULT.json').exists() and (p/'CLOSE.json').exists(): complete.append(row['id'])
files={};updates=[]
def add(name):
 p=r/name
 if p.is_file():
  data=p.read_bytes();digest=hashlib.sha256(data).hexdigest()
  if name not in KNOWN: files[name]=dict(sha256=digest,base64=base64.b64encode(data).decode())
  return digest
for name in ('BINDING.json','CPU_APPROVED.json','jobs/zero_update_canary/RESULT.json',
             'PRIOR.json','BANDIT.json','BUDGET_STOP.json','TRAIN_NOT_ADMITTED.json',
             'WAITING_PEER.json','STOP.json','COMPLETE.json'):
 add(name)
for p in sorted(r.glob('UPDATE_*.json')):
 u=read(p);name='policy_u%03d.pt'%u['update']
 assert sha(r/name)==u['checkpoint_sha256'],'checkpoint/update binding'
 assert u['windows']>=150 and u['worlds']>=12
 add(name);add(p.name);updates.append(dict(update=u['update'],file=name,sha256=u['checkpoint_sha256']))
for pattern in ('DEV_*.json','DISCOVERY_*.json'):
 for p in sorted(r.glob(pattern)):
  if p.stem.split('_')[-1].isdigit(): add(p.name)
repeats=[]
for ident in complete:
 if ident.endswith('_repeat'):
  original=ident.replace('_repeat','_screen')
  a=sha(r/'jobs'/ident/'RESULT.json');b=sha(r/'jobs'/original/'RESULT.json')
  assert a==b,'baseline repeat mismatch'
  repeats.append(dict(id=ident,sha256=a,bitwise=True))
plan=None
if (r/'PLAN_001.json').exists():
 p=read(r/'PLAN_001.json');sizes=[len(w['sample']['context']['members']) for w in p['windows']]
 plan=dict(windows=len(sizes),worlds=p['selection']['worlds'],max_members=max(sizes),
           multi_member_windows=sum(n>=2 for n in sizes),selection=p['selection'])
print(json.dumps(dict(unix=time.time(),dispatch=dispatch,owned_alive=alive,
 charged=sum(x['sim_s'] for x in charges),actual_closed=sum(x['actual_sim_s'] for x in closed),
 complete_jobs=len(complete),started_jobs=len(started),reserved_jobs=len(charges),
 updates=updates,repeats=repeats,plan=plan,files=files),allow_nan=False))
'''
    return json.loads(remote('cpuQ', 'ROOT=' + repr(ROOT) + '\nKNOWN=' + repr(set(known)) + '\n' + source,
                             timeout=60))


def immutable(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('publication changed: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def unpack(files):
    decoded = {}
    for name, item in files.items():
        # The remote read whitelist is reinforced locally before writing any artifact.
        if name == 'jobs/zero_update_canary/RESULT.json':
            local = 'ZERO_UPDATE_CANARY.json'
        elif '/' not in name and (name.endswith('.json') or name.startswith('policy_u') and name.endswith('.pt')):
            local = name
        else:
            raise ValueError('unapproved publication name: ' + name)
        data = base64.b64decode(item['base64'], validate=True)
        if sha(data) != item['sha256']:
            raise ValueError('transport hash: ' + name)
        decoded[local] = data
    return decoded


def replace_block(path, text):
    original = path.read_text(encoding='utf-8')
    block = BEGIN + '\n' + text.rstrip() + '\n' + END
    if BEGIN in original:
        before, tail = original.split(BEGIN, 1)
        _, after = tail.split(END, 1)
        updated = before + block + after
    else:
        updated = original.rstrip() + '\n\n' + block + '\n'
    if updated != original:
        path.write_text(updated, encoding='utf-8')


def manifest():
    files = {p.name: sha(p.read_bytes()) for p in PUBLIC.iterdir()
             if p.is_file() and p.name != 'MANIFEST.json'}
    return dict(format='coopfree/proposals/1',scope='TRAIN-only learner; DEV/discovery descriptive',
                host='cpuQ',root=ROOT,files=files)


def describe(snapshot):
    lines = ['## Automatically published proposal checkpoints',
             'Consumer guide: `docs/paper_results/COOPFREE_PROPOSALS.md`.',
             'Public artifacts: `tools/coopfree/publications/`; verify `MANIFEST.json` hashes.',
             'Frozen remote root: `' + ROOT + '` (cpuQ).']
    if not snapshot['updates']:
        lines.append('No learned checkpoint yet; the full156-window/all-member-solo first update is running.')
    for u in snapshot['updates']:
        report = json.loads((PUBLIC / ('UPDATE_%03d.json' % u['update'])).read_text())
        lines += ['- **u%03d**: `tools/coopfree/publications/%s`' % (u['update'],u['file']),
                  '  SHA256 `' + u['sha256'] + '`; companion `UPDATE_%03d.json`.' % u['update'],
                  '  Windows/worlds: %s/%s; executed-change share %.4f; override share %.4f;' %
                  (report['windows'],report['worlds'],report['executed_change_share'],report['safety_override_share']),
                  '  mean D_G %.3f, sum D_i %.3f, C_G %.3f vehicle-s; positive-C_G share %.4f.' %
                  (report['D_G']['mean'],report['sum_D_i']['mean'],report['C_G']['mean'],report['positive_C_G_share'])]
    for p in sorted(PUBLIC.glob('DEV_*.json')):
        lines.append('- Paired dev receipt: `tools/coopfree/publications/' + p.name + '`.')
    for p in sorted(PUBLIC.glob('DISCOVERY_*.json')):
        lines.append('- Learned-group C_G/detectors/clustering: `tools/coopfree/publications/' + p.name + '`.')
    if (PUBLIC / 'STOP.json').exists():
        lines.append('**Durable STOP observed.** Inspect published `STOP.json`; no automatic simulation restart.')
    elif (PUBLIC / 'COMPLETE.json').exists():
        lines.append('Pipeline COMPLETE; consult paired dev/discovery receipts before performance claims.')
    lines += ['Latest milestone: %d complete jobs; %d charged/%s closed-actual sim-s.' %
              (snapshot['complete_jobs'],snapshot['charged'],snapshot['actual_closed']),
              'Training-window credits describe exploration; they are not paired dev-policy performance.']
    return '\n'.join(lines)


def git(*args, check=True):
    return subprocess.run(['git',*args],cwd=LOCAL,text=True,capture_output=True,check=check)


def once(commit=False):
    previous = json.loads((PUBLIC/'MANIFEST.json').read_text()) if (PUBLIC/'MANIFEST.json').exists() else None
    if previous:
        for name,expected in previous['files'].items():
            if Path(name).name != name or sha((PUBLIC/name).read_bytes()) != expected:
                raise ValueError('existing local publication hash: ' + name)
    known = [] if previous is None else list(previous['files'])
    if 'ZERO_UPDATE_CANARY.json' in known:
        known.remove('ZERO_UPDATE_CANARY.json')
        known.append('jobs/zero_update_canary/RESULT.json')
    snapshot = collect(known)
    decoded = unpack(snapshot.pop('files'))
    # Do not edit/stage user work in the two shared documentation files.
    for name in ('HANDOFF.md','docs/paper_results/COOPFREE.md'):
        if git('status','--porcelain','--',name).stdout.strip():
            raise ValueError('uncommitted documentation; preserve and reconcile: ' + name)
    if not decoded:
        return snapshot, False
    for name,data in decoded.items():
        immutable(PUBLIC/name,data)
    content = describe(snapshot)
    replace_block(LOCAL/'HANDOFF.md',content)
    replace_block(LOCAL/'docs/paper_results/COOPFREE.md',content)
    record = json.dumps(manifest(),indent=2,sort_keys=True,allow_nan=False)+'\n'
    (PUBLIC/'MANIFEST.json').write_text(record,encoding='utf-8')
    ident = '%03d' % (len(list(PUBLIC.glob('MILESTONE_*.json')))+1)
    immutable(PUBLIC/('MILESTONE_'+ident+'.json'),
              (json.dumps(snapshot,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
    # Include this milestone in the final manifest too.
    (PUBLIC/'MANIFEST.json').write_text(json.dumps(manifest(),indent=2,sort_keys=True)+'\n',encoding='utf-8')
    if commit:
        # Inspect all required state before staging only the observer-owned publication paths.
        for args in (('status','--short'),('diff','--','HANDOFF.md','docs/paper_results/COOPFREE.md'),
                     ('log','--oneline','-10')):
            print(git(*args).stdout,flush=True)
        git('diff','--check')
        paths = ['tools/coopfree/publications/'+name for name in decoded]
        paths += ['tools/coopfree/publications/MANIFEST.json',
                  'tools/coopfree/publications/MILESTONE_'+ident+'.json']
        git('add','--','HANDOFF.md','docs/paper_results/COOPFREE.md',*paths)
        print(git('commit','-m','coopfree: publish verified proposal and run milestone '+ident).stdout,flush=True)
    return snapshot, True


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--commit',action='store_true')
    parser.add_argument('--interval',type=int,default=120)
    parser.add_argument('--hours',type=float,default=12)
    args = parser.parse_args()
    if args.interval < 30 or not 0 < args.hours <= 24:
        parser.error('interval>=30 and 0<hours<=24 required')
    gitdir = Path(git('rev-parse','--absolute-git-dir').stdout.strip())
    with (gitdir/'coopfree-publisher.lock').open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.monotonic()+args.hours*3600
        while True:
            try:
                snapshot,changed = once(args.commit)
            except (subprocess.TimeoutExpired,OSError,RuntimeError) as error:
                # SSH transport retries only. Source/hash/assertion failures remain terminal.
                transient = isinstance(error,(subprocess.TimeoutExpired,ConnectionError)) or \
                    any(s in str(error).lower() for s in
                        ('connection timed out','connection reset','connection refused','connection closed',
                         'could not resolve hostname','broken pipe'))
                if not args.watch or not transient or time.monotonic()>=deadline:
                    raise
                print(json.dumps(dict(transport_error=str(error),simulations_dispatched=0)),flush=True)
                time.sleep(args.interval)
                continue
            print(json.dumps(dict(unix=snapshot['unix'],owned_alive=snapshot['owned_alive'],
                                  complete_jobs=snapshot['complete_jobs'],charged=snapshot['charged'],
                                  actual_closed=snapshot['actual_closed'],updates=snapshot['updates'],
                                  publication_changed=changed)),flush=True)
            if not args.watch or time.monotonic()>=deadline or (PUBLIC/'STOP.json').exists() or \
                    (PUBLIC/'COMPLETE.json').exists() or not snapshot['owned_alive']:
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
