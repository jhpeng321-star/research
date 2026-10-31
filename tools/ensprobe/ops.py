"""Task-local, short CPU-only SSH operations; never inspect assignment/truth files."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'tools/ensprobe/evidence'
REG = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
BAD = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vehicle_params|driver_params|assignment|requests\.rou|driver_routes', re.I)
SEEDS = list(range(969040, 969045))
RNG_SEEDS = list(range(969086, 969094))


def write(name, obj):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE/name).write_text(json.dumps(obj, indent=2)+'\n', encoding='utf-8')


def ssh(code, timeout=90, host='cpuQ'):
    python = '/home/ubuntu/miniconda3/bin/python' if host == 'cpuQ' else '/root/miniconda3/bin/python'
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', host, python+' -'],
                       input=code, text=True, encoding='utf-8', capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(p.stderr+p.stdout)
    return json.loads(p.stdout)


def inspect():
    code = '''import json,os,hashlib
from pathlib import Path
r=Path('/home/ubuntu/e3final/runtime')
m=r/'SOURCE_MANIFEST.json'
out=dict(load=os.getloadavg(),cpus=os.cpu_count(),manifest_sha256=hashlib.sha256(m.read_bytes()).hexdigest())
names=['traffic_shepherd/scenarios.py','traffic_shepherd/c4/population.py','traffic_shepherd/simulation.py',
 'traffic_shepherd/c4/env.py','traffic_shepherd/c4/seeds.py','traffic_shepherd/seeds.py',
 'traffic_shepherd/metrics.py','traffic_shepherd/config.py','baselines/circles/adapter.json',
 'traffic_shepherd/c4/evalprep.py','src/wms/control/deploy.py']
manifest=json.loads(m.read_text());out['sources']={}
for name in names:
 p=r/name;raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
 if name in manifest:assert digest==manifest[name]
 out['sources'][name]=dict(sha256=digest,in_source_manifest=name in manifest,text=raw.decode())
w=Path('/home/ubuntu/wmdata20261031/tools')
out['wrappers']={}
for name in ['wmdata/worker.py','speedhead/observer.py','ctrlarm/zero.py']:
 p=w/name
 if p.is_file():out['wrappers'][name]=p.read_text()
out['wmdata_names']=[p.name for p in (w/'wmdata').iterdir()] if (w/'wmdata').exists() else []
import sys,xml.etree.ElementTree as ET,gzip
sys.path[:0]=[str(r),str(r/'src')]
from traffic_shepherd.config import load_config
c=load_config();np=c.network_path
with (gzip.open(np,'rb') if str(np).endswith('.gz') else np.open('rb')) as f:net=ET.parse(f).getroot()
out['network']=dict(path=str(np),junctions=[j.attrib for j in net.findall('junction')],params=[p.attrib for p in net.iter('param')])
print(json.dumps(out))
'''
    out = ssh(code)
    write('runtime_sources.json', out)
    for name,row in out['sources'].items():
        (EVIDENCE/(name.replace('/','__')+'.txt')).write_text(row['text'].replace('\r',''),encoding='utf-8')
    for name,text in out['wrappers'].items():
        (EVIDENCE/(name.replace('/','__')+'.txt')).write_text(text.replace('\r',''),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('sources','wrappers')}, indent=2))


def audit():
    candidates=SEEDS+RNG_SEEDS
    conflicts=[]
    umbrellas={(965184,969999),(966000,969999),(966000,969699)}
    text=REG.read_text(encoding='utf-8')
    for n,line in enumerate(text.splitlines(),1):
        if 'S-ENSPROBE' in line:continue
        tokens={int(x) for x in re.findall(r'(?<![\w.])(?:[sw])?(\d{6})(?![\w.])',line)}
        ranges=[]
        for a,b in re.findall(r'(\d{6})\s*(?:-|–|—|\.\.|锟\?)\s*(\d{3,6})',line):
            lo=int(a);hi=int(b) if len(b)==6 else int(a[:-len(b)]+b)
            if (lo,hi) not in umbrellas and any(lo<=s<=hi for s in candidates):ranges.append([lo,hi])
        if tokens.intersection(candidates) or ranges:conflicts.append(dict(line=n,ranges=ranges,text=line))
    hosts={}
    for host,base in [('cpuQ','/home/ubuntu'),('gpuR','/root'),('gpuA','/root')]:
        hosts[host]=ssh(fr'''import os,re,json,sys
from pathlib import Path
bad=re.compile({BAD.pattern!r},re.I);hits=[];count=0
for here,dirs,files in os.walk({base!r},followlinks=False):
 dirs[:]=[n for n in dirs if not bad.search(str(Path(here)/n)) and not n.startswith('.') and n not in ('miniconda3','miniconda','ensprobe20261031')]
 for name in dirs+files:
  p=str(Path(here)/name)
  if bad.search(p):continue
  count+=1
  if any(int(x) in {candidates!r} for x in re.findall(r'(?<!\d)(96\d{{4}})(?!\d)',name)):hits.append(p)
sys.path.insert(0,{base!r}+'/e3final/runtime')
guards=[]
if Path({base!r}+'/e3final/runtime').is_dir():
 from traffic_shepherd.seeds import require_development_case
 from traffic_shepherd.c4.seeds import check_case
 for s,d in zip({candidates!r},[2400,2700,2400,2700,2700]+[2400]*8):
  require_development_case(s,d,'exposed');check_case(s,d);guards.append(s)
print(json.dumps(dict(hits=hits,safe_names_checked=count,guards=guards)))
''',host=host,timeout=120)
    out=dict(seeds=SEEDS,rng_seeds=RNG_SEEDS,registry_sha256=hashlib.sha256(text.encode()).hexdigest(),registry_conflicts=conflicts,hosts=hosts)
    out['pass']=not conflicts and not any(v['hits'] for v in hosts.values()) and hosts['cpuQ']['guards']==candidates
    write('exposure_'+str(SEEDS[0])+'.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inspect','audit']);a=p.parse_args()
    globals()[a.action]()
