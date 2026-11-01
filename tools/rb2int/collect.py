"""Read-only incremental verified collection of complete K8 world jobs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

from tools.rb2int.audit import ROOT
from tools.rb2int.predictions import validate_identity
from tools.rbeval.core import load, digest, save_new

REMOTE = '/root/autodl-tmp/c4/rb2int20261101/final_v5_v1'


def collect_host(host,cache,receipt):
    cache,receipt = Path(cache),Path(receipt)
    if not cache.parent.is_dir() or not receipt.parent.is_dir() or receipt.exists():
        raise ValueError('existing cache/receipt parents and exclusive receipt required')
    cache.mkdir(exist_ok=True)
    known = {}
    for path in cache.glob('*/COLLECTED.json'):
        record = load(path)
        complete_path = path.parent/'tasks/COMPLETE.json'
        if digest(complete_path) != record['complete_sha256']:
            raise ValueError('local completed cache marker changed')
        completion = load(complete_path)
        if any(digest(path.parent/'tasks'/name) != value for name,value in completion['hashes'].items()):
            raise ValueError('local completed cache task changed')
        known[path.parent.name] = record['complete_sha256']
    code = f'''
from pathlib import Path
import hashlib,io,json,sys,tarfile,time
root=Path({REMOTE!r});known={known!r}
plan=json.loads((root/'RUN_PLAN.json').read_text());files={{}};complete=[];pending=[]
for job in plan['jobs']:
 if job['host']!={host!r}: continue
 directory=root/'jobs'/job['id'];out=directory/'tasks';marker=out/'COMPLETE.json'
 if not marker.is_file(): pending.append(job['id']);continue
 raw=marker.read_bytes();completion=json.loads(raw);expected={{f"w{{job['seed']}}_k{{k}}.json" for k in range(8)}}
 assert completion['tasks']==8 and set(completion['hashes'])==expected
 identity=(out/'IDENTITY.json').read_bytes();assert hashlib.sha256(identity).hexdigest()==completion['identity_sha256']
 checksum=hashlib.sha256(raw).hexdigest();complete.append(dict(job=job,complete_sha256=checksum,identity_sha256=completion['identity_sha256']))
 if job['id'] in known:
  assert known[job['id']]==checksum, 'completed world changed after collection'
  continue
 for name,expected_sha in completion['hashes'].items():
  data=(out/name).read_bytes();assert hashlib.sha256(data).hexdigest()==expected_sha
  files[job['id']+'/tasks/'+name]=data
 files[job['id']+'/tasks/IDENTITY.json']=identity;files[job['id']+'/tasks/COMPLETE.json']=raw
 for name in ('CLAIM_ONCE.json','PROCESS.json','EXIT.json'):
  if (directory/name).is_file(): files[job['id']+'/'+name]=(directory/name).read_bytes()
state=dict(host={host!r},complete=complete,pending=pending,read_only=True,unix=time.time(),
 hashes={{name:hashlib.sha256(data).hexdigest() for name,data in files.items()}})
for name in ('PANEL_COMPLETE.json','PANEL_TERMINAL_INCOMPLETE.json','SUPERVISOR_PROCESS.json'):
 if (root/name).is_file(): state[name]=json.loads((root/name).read_text())
files['COLLECTION.json']=json.dumps(state).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name,data in files.items():
  info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
'''
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,
        '/root/miniconda3/bin/python','-B','-c',shlex.quote(code)],capture_output=True,timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:gz') as stream:
        for member in stream.getmembers():
            if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:
                raise ValueError('invalid final snapshot archive member')
        state = json.loads(stream.extractfile('COLLECTION.json').read())
        for name,expected in state['hashes'].items():
            if hashlib.sha256(stream.extractfile(name).read()).hexdigest() != expected:
                raise ValueError('final snapshot transfer hash mismatch')
        for world in state['complete']:
            job = world['job']
            if job['id'] in known:
                continue
            out = cache/job['id']
            if out.exists():
                raise ValueError('preserve an incomplete transfer; explicit recovery required')
            members = [m for m in stream.getmembers() if m.name.startswith(job['id']+'/')]
            for member in members:
                member.name = member.name[len(job['id'])+1:]
            out.mkdir()
            stream.extractall(out,members=members,filter='data')
            complete = load(out/'tasks/COMPLETE.json')
            if any(digest(out/'tasks'/name) != value for name,value in complete['hashes'].items()):
                raise ValueError('local completed task checksum differs')
            save_new(out/'COLLECTED.json',dict(host=host,job=job,complete_sha256=world['complete_sha256'],
                identity_sha256=world['identity_sha256'],original_root=REMOTE+'/jobs/'+job['id']))
    save_new(receipt,state)
    return dict(host=host,complete_world_jobs=len(state['complete']),pending_world_jobs=len(state['pending']),
                newly_collected=len(state['complete'])-len(known))


def registry(cache_root,out):
    base = ROOT/'docs/rb2int'
    release_path = base/'FINAL_RELEASE_V2.json'
    release = load(release_path)
    models = {}
    for label,catalog in release['catalog'].items():
        binding = catalog['binding']
        proof = ROOT/catalog['baseline_bridge']['path']
        model = dict(identity_name=catalog['identity_name'],wm_sha256=binding['wm_sha256'],
            config_sha256=binding['config_sha256'],candidate_sha256=catalog['candidate_sha256'],
            export_sha256=catalog['export_sha256'],switches=catalog['switches'],
            frozen_model_sources=binding['sources'],development_bridge_passed=True,
            bridge_receipts={str(proof):digest(proof)},runs=[])
        if label in ('WM2','FULL-v4'):
            old = ROOT.parent/'c4-rb-int/runs/rbint_evidence'/(
                'noon_wm2_20261101T054124+0800' if label=='WM2' else 'noon_FULL-v4_20261101T102523+0800')
            for identity in sorted(old.glob('shard*/IDENTITY.json')):
                data = load(identity);validate_identity(data,model)
                model['runs'].append(dict(root=str(identity.parent),identity_sha256=digest(identity),
                    real_receipt_sha256=data['real_receipt_sha256'],original_reserved_reuse=True))
        for host in ('gpuA','gpuR'):
            for record_path in sorted((Path(cache_root)/host).glob('*/COLLECTED.json')):
                record = load(record_path)
                if record['job']['model'] != label:
                    continue
                folder = record_path.parent/'tasks';identity=folder/'IDENTITY.json';data=load(identity)
                validate_identity(data,model)
                if digest(identity) != record['identity_sha256']:
                    raise ValueError('collected world identity changed')
                model['runs'].append(dict(root=str(folder),identity_sha256=digest(identity),
                    real_receipt_sha256=data['real_receipt_sha256'],host=host,world=record['job']['seed']))
        models[label] = model
    save_new(out,dict(schema='rb2int-approved-runs/1',registration_path=str(base/'EVALUATION_REGISTRATION.json'),
        registration_sha256=release['registration_sha256'],release_path=str(release_path),
        release_sha256=digest(release_path),models=models,read_only_collection=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('host','registry'))
    parser.add_argument('--host',choices=('gpuA','gpuR'))
    parser.add_argument('--cache',required=True)
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    if args.action=='host':
        print(json.dumps(collect_host(args.host,args.cache,args.out)))
    else:
        registry(args.cache,args.out);print('APPROVED RUN REGISTRY',args.out)


if __name__ == '__main__':
    main()
