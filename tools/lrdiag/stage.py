"""Read-only SSH export into the task's own CPU directory; no source writes."""
import subprocess
import sys

EXPORT = r'''
import io,json,tarfile,sys,hashlib
from pathlib import Path
bad=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth')
def safe(p):
    assert not any(x in str(p).lower() for x in bad), p
    return p
rt=Path('/root/autodl-tmp/c4/wmcombined20261027/runtime')
root=Path('/root/autodl-tmp/c4/e3data')
m=json.loads(safe(root/'C4_EPISODES_MANIFEST.json').read_text())
s=json.loads(safe(root/'SPLIT.json').read_text())
eps=[e for e in m['episodes'] if s['worlds'][str(e['seed'])]=='validation'
     and e['arm'] in ('native','mappo','ours_c1')]
assert len(eps)==108 and len({e['world'] for e in eps})==36
assert not any(964000<=e['seed']<=964199 for e in eps)
receipt={'n_worlds':36,'n_episodes':108,'role':'validation','files':{}}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
    def add(p,name):
        p=safe(p); assert p.is_file() and not p.is_symlink(),p
        receipt['files'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
        t.add(p,arcname=name,recursive=False)
    def blob(name,data):
        b=json.dumps(data,indent=1).encode(); i=tarfile.TarInfo(name); i.size=len(b)
        t.addfile(i,io.BytesIO(b))
    for sub in ('src','configs','tools','traffic_shepherd'):
        for p in (rt/sub).rglob('*'):
            if any(x in str(p).lower() for x in bad) or '__pycache__' in str(p): continue
            if p.is_file() and p.suffix in ('.py','.json','.toml'):
                add(p,'runtime/'+str(p.relative_to(rt)))
    add(Path('/root/autodl-tmp/c4/wmcombined20261027/wm2/wm_s.pt'),'wm2/wm_s.pt')
    for e in eps:
        for name in ('control_time_s.npy','controller.npy','counter.npy','episode.json',
                     'is_cav.npy','issued_acceleration_mps2.npy','issued_target_speed_mps.npy',
                     'target_mask.npy','time_s.npy','veh_kin.npy','veh_mask.npy','vid.npy'):
            add(root/e['dir']/name,'e3data/'+e['dir']+'/'+name)
    m['episodes']=eps
    blob('e3data/C4_EPISODES_MANIFEST.json',m)
    blob('e3data/SPLIT.json',s)
    blob('STAGING.json',receipt)
'''

IMPORT = r'''
from pathlib import Path
import tarfile,sys
p=Path.home()/'lrdiag'; assert p.parent.is_dir(); p.mkdir(exist_ok=True)
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as t:
    for m in t:
        q=p/m.name
        assert q.resolve().is_relative_to(p.resolve()) and not m.issym() and not m.islnk()
        assert not any(x in str(q).lower() for x in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1'))
        t.extract(m,p,filter='data')
print('staged',p)
'''

PAIRS_EXPORT = r'''
import io,json,tarfile,sys,hashlib
from pathlib import Path
import numpy as np
bad=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth')
def safe(p):
    assert not any(x in str(p).lower() or x in str(p.resolve()).lower() for x in bad),p
    return p
root=Path('/root/autodl-tmp/c4/wmdeep20261027/score_pairs')
m=json.loads(safe(root/'C4_EPISODES_MANIFEST.json').read_text())
groups={}
for e in m['episodes']:
    assert not 964000<=e['seed']<=964199
    if not e.get('pair_key'):continue
    groups.setdefault(e['pair_key'],[]).append(e)
keys=sorted(k for k,g in groups.items() if any(e['branch']['kind']=='B' for e in g)
            and any(e['branch']['kind']=='SEED' for e in g))
order=np.random.default_rng(20261026).permutation(len(keys))
selected=[keys[i] for i in sorted(order[:48])]
eps=[e.copy() for k in selected for e in groups[k]]
receipt={'n_groups':len(selected),'keys':selected,'panel_seed':20261026,
         'source':str(root),'scope':'pairs300 diagnostic, not E3 verdict','files':{}}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
    def add(p,name):
        p=safe(p); assert p.is_file() and not p.is_symlink(),p
        receipt['files'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
        t.add(p,arcname=name,recursive=False)
    def blob(name,data):
        b=json.dumps(data,indent=1).encode(); i=tarfile.TarInfo(name); i.size=len(b)
        t.addfile(i,io.BytesIO(b))
    for e in eps:
        src=safe(Path(e['dir'])); rel='episodes/'+e['episode']
        for name in ('control_time_s.npy','controller.npy','counter.npy','episode.json',
                     'is_cav.npy','issued_acceleration_mps2.npy','issued_target_speed_mps.npy',
                     'target_mask.npy','time_s.npy','veh_kin.npy','veh_mask.npy','vid.npy',
                     'lane_command_target.npy','lane_request_target.npy','requested_acceleration_mps2.npy'):
            if (src/name).exists():add(src/name,'pairs300/'+rel+'/'+name)
        e['dir']=rel
    m['episodes']=eps
    blob('pairs300/C4_EPISODES_MANIFEST.json',m)
    blob('PAIRS_STAGING.json',receipt)
'''


def main():
    host = sys.argv[1] if len(sys.argv)>1 else 'cpuP'
    assert host in ('cpuP','cpuQ')
    import base64
    def encoded(s):
        # Single shell quotes survive Windows native argv quoting.
        b=base64.b64encode(s.encode()).decode()
        return "'import base64;exec(base64.b64decode(\""+b+"\"))'"
    code=PAIRS_EXPORT if '--pairs' in sys.argv else EXPORT
    src=subprocess.Popen(['ssh','-o','BatchMode=yes','gpuR',
                          '/root/miniconda3/bin/python -c '+encoded(code)],stdout=subprocess.PIPE)
    dst=subprocess.run(['ssh','-o','BatchMode=yes',host,
                        '/usr/bin/python3 -c '+encoded(IMPORT)],stdin=src.stdout)
    src.stdout.close()
    assert src.wait()==0 and dst.returncode==0


if __name__=='__main__': main()
