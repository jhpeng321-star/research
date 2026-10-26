"""Copy existing numeric queue arrays from read-only gpuR into owned gpuA overlays."""
import json

from remote import ROOT, REMOTE, ssh


def main():
    source = '''import pathlib,json,base64,zlib,hashlib
root=pathlib.Path('/root/autodl-tmp/c4/v5/data')
out={}
for name in ('v5_native','genrand_plain'):
 p=root/(name+'_q'); manifest=json.loads((p/'C4_EPISODES_MANIFEST.json').read_text())
 arrays={}; entries=[]
 eligible=[e for e in manifest['episodes'] if e.get('split_role')=='train']
 for pen in sorted(set(e['penetration'] for e in eligible)):
  candidates=[e for e in eligible if e['penetration']==pen]
  entries+=sorted(candidates,key=lambda e:hashlib.sha256(('965140:'+str(e['seed'])).encode()).hexdigest())[:6]
 for e in entries:
  rel=e['dir']
  if any(x in rel.lower() for x in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1')): continue
  for key in ('entry_backlog','request_counter','entry_cav_counter'):
   f=p/rel/(key+'.npy')
   if f.is_file(): arrays[rel+'/'+key+'.npy']=base64.b64encode(zlib.compress(f.read_bytes())).decode()
 out[name]={'arrays':arrays,'entries':entries}
print(json.dumps(out))
'''
    result = ssh('gpuR', source)
    if result.returncode:
        raise RuntimeError(result.stderr)
    payload = json.loads(result.stdout)
    assert all(v['arrays'] for v in payload.values())
    target = '''import pathlib,json,base64,hashlib,zlib
owned=pathlib.Path('/root/autodl-tmp/c4/longcritic'); assert owned.is_dir()
root=pathlib.Path('/root/autodl-tmp/c4/v5/data'); receipts={}
payload=json.loads(PAYLOAD)
for name,block in payload.items():
 arrays=block['arrays']
 src=root/name; dst=owned/'data'/(name+'_q'); dst.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((src/'C4_EPISODES_MANIFEST.json').read_text()); manifest['episodes']=block['entries']
 manifest['n_episodes']=len(block['entries']); manifest['n_train']=len(block['entries']); manifest['n_val']=0
 mf=dst/'C4_EPISODES_MANIFEST.json'
 assert not mf.is_symlink()
 mf.write_text(json.dumps(manifest))
 for e in manifest['episodes']:
  rel=e['dir']
  if any(x in rel.lower() for x in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1')): continue
  target=dst/rel; target.mkdir(parents=True,exist_ok=True)
  for f in (src/rel).iterdir():
   if f.suffix!='.npy' or any(x in f.name.lower() for x in ('truth','vtype','assignment')): continue
   t=target/f.name
   if rel+'/'+f.name not in arrays and not t.exists(): t.symlink_to(f)
 for rel,data in arrays.items():
  t=dst/rel; raw=zlib.decompress(base64.b64decode(data))
  if t.exists(): assert t.read_bytes()==raw
  else: t.write_bytes(raw)
 receipts[name]={'numeric_queue_arrays':len(arrays),'worlds':[e['seed'] for e in manifest['episodes']],'queue_sha256':hashlib.sha256(json.dumps(arrays,sort_keys=True).encode()).hexdigest()}
(owned/'INPUT_OVERLAYS.json').write_text(json.dumps(receipts,indent=2))
print(json.dumps(receipts))
'''.replace('PAYLOAD', repr(json.dumps(payload)))
    done = ssh('gpuA', target)
    print(done.stdout, done.stderr)
    if done.returncode:
        raise RuntimeError('overlay staging failed')
    (ROOT / 'tools/longcritic/INPUT_OVERLAYS.json').write_text(done.stdout, encoding='utf-8')
    # Current-policy real returns exported by the main loop, no simulation/reconstruction.
    r = ssh('gpuR', "import pathlib,json,base64,hashlib\n"
            "root=pathlib.Path('/root/main14_loop'); state=json.loads((root/'LOOP14_STATE.json').read_text())\n"
            "p=root/'returns/returns_c00.a1.npz'; assert p.is_file()\n"
            "print(json.dumps({'data':base64.b64encode(p.read_bytes()).decode(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'path':str(p),'policy':state['policy']}))\n")
    if r.returncode:
        raise RuntimeError(r.stderr)
    data = json.loads(r.stdout)
    raw = data.pop('data')
    r = ssh('gpuA', "import pathlib,base64,json,hashlib\n"
            f"root=pathlib.Path('{REMOTE}'); p=root/'current_real_returns.npz'\n"
            f"raw=base64.b64decode({raw!r}); assert hashlib.sha256(raw).hexdigest()=={data['sha256']!r}\n"
            "assert not p.exists() or p.read_bytes()==raw\np.write_bytes(raw)\n"
            f"(root/'CURRENT_REAL_SOURCE.json').write_text({json.dumps(data)!r})\nprint('current real returns copied')\n")
    print(r.stdout, r.stderr)
    if r.returncode:
        raise RuntimeError('current return staging failed')
    (ROOT / 'tools/longcritic/CURRENT_REAL_SOURCE.json').write_text(json.dumps(data, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
