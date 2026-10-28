"""Merge disjoint task-owned CPUQ parity results into CPUP, via local SSH pipes."""
import base64
import subprocess
import sys

EXPORT=r'''
from pathlib import Path
import tarfile,sys,time,io,json,hashlib,re
p=Path.home()/'lrdiag';assert p.is_dir()
stamp=str(int(time.time()))
skip=json.load(sys.stdin)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
    upgrades={}
    for f in (p/'fullclass_upgrades').glob('*.json'):
        r=json.loads(f.read_text());target=p/'results'/r['file']
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==r['new_sha256']:upgrades[r['file']]=r
    b=json.dumps(upgrades).encode();i=tarfile.TarInfo('receipts/cpuQ/'+stamp+'/FULLCLASS_UPGRADES.json');i.size=len(b)
    t.addfile(i,io.BytesIO(b))
    for sub,pattern in (('results','*_s600.json'),('traces','*.npz'),('dense_reference/results','*.json'),('fullclass_upgrades','*.json')):
        for f in sorted((p/sub).glob(pattern)):
            assert f.is_file() and not f.is_symlink()
            if sub=='traces':
                match=re.fullmatch(r'g([0-9]{2})_k([0-7])',f.stem)
                assert match,f
                if not (p/'results'/('g'+match[1]+'_baseline_k'+match[2]+'_s600.json')).exists():continue
            rel=sub+'/'+f.name
            if sub=='results' and (p/'baseline_before_fullfield'/f.name).exists():
                d=json.loads(f.read_text())
                if d.get('fullfield_version',0)>=2 and f.name not in upgrades:continue
            if skip.get(rel)==hashlib.sha256(f.read_bytes()).hexdigest():continue
            t.add(f,arcname=rel,recursive=False)
    for name in ('STAGING.json','PAIRS_STAGING.json','COMPACT_CHECK.json','OWNER.json','PROGRESS.json',
                 'DONE.json','PAIRS_OWNER.json','PAIRS_PROGRESS.json','PAIRS_DONE.json','ALL_DONE.json'):
        f=p/name
        if f.is_file():
            b=f.read_bytes();i=tarfile.TarInfo('receipts/cpuQ/'+stamp+'/'+name);i.size=len(b)
            t.addfile(i,io.BytesIO(b))
'''

INDEX=r'''
from pathlib import Path
import json,hashlib,re
p=Path.home()/'lrdiag';assert p.is_dir()
d={}
for sub,pattern in (('results','*_s600.json'),('traces','*.npz'),('dense_reference/results','*.json'),('fullclass_upgrades','*.json')):
    for f in (p/sub).glob(pattern):
        m=re.search(r'g([0-9]{2})_',f.name)
        if m and int(m[1])%2==1:d[sub+'/'+f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
print(json.dumps(d))
'''

IMPORT=r'''
from pathlib import Path
import tarfile,sys,hashlib,re,os,time,json
p=Path.home()/'lrdiag';assert p.is_dir()
n=0;size=0;upgrades={}
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as t:
    for m in t:
        q=p/m.name
        assert m.isfile() and not m.issym() and not m.islnk()
        assert q.resolve().is_relative_to(p.resolve())
        assert not any(x in str(q).lower() for x in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1'))
        assert m.name.startswith(('results/','traces/','dense_reference/results/','receipts/cpuQ/','fullclass_upgrades/'))
        content=t.extractfile(m).read();assert len(content)==m.size
        if m.name.endswith('/FULLCLASS_UPGRADES.json'):upgrades=json.loads(content)
        if q.exists():
            newhash=hashlib.sha256(content).hexdigest();oldhash=hashlib.sha256(q.read_bytes()).hexdigest()
            if newhash==oldhash:continue
            upgrade=upgrades.get(q.name) if q.parent==p/'results' else None
            if upgrade and upgrade['old_sha256']==oldhash and upgrade['new_sha256']==newhash:
                archive=p/'imported_pre_fullclass'/q.name
                archive.parent.mkdir(exist_ok=True)
                if archive.exists():assert hashlib.sha256(archive.read_bytes()).hexdigest()==oldhash
                else:q.rename(archive)
            else:
                # Preserve a partial from the first interrupted owned transfer.
                match=re.search(r'g([0-9]{2})_',q.name)
                assert match and int(match[1])%2==1 and q.stat().st_size<m.size,q
                archive=p/'transfer_partial'/str(int(time.time()))/m.name
                archive.parent.mkdir(parents=True,exist_ok=True);q.rename(archive)
        q.parent.mkdir(parents=True,exist_ok=True)
        tmp=q.with_name(q.name+'.incoming-'+str(os.getpid()))
        with tmp.open('xb') as f:f.write(content)
        tmp.replace(q);n+=1;size+=m.size
print('merged files',n,'MiB',round(size/2**20,1))
'''


def command(code):
    b=base64.b64encode(code.encode()).decode()
    return "'import base64;exec(base64.b64decode(\""+b+"\"))'"


if __name__=='__main__':
    ssh=['ssh','-o','BatchMode=yes','-o','ServerAliveInterval=15']
    if '--jump' in sys.argv:ssh+=['-J','gpuR']
    index=subprocess.check_output([*ssh,'cpuP','python3 -c '+command(INDEX)])
    src=subprocess.Popen([*ssh,'cpuQ',
        '/home/ubuntu/miniconda3/bin/python -c '+command(EXPORT)],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    src.stdin.write(index);src.stdin.close()
    dst=subprocess.run([*ssh,'cpuP','python3 -c '+command(IMPORT)],stdin=src.stdout)
    src.stdout.close();assert src.wait()==0 and dst.returncode==0
