"""At most four independently SHA-verified chunks per invocation; no retries."""
import base64,hashlib,json
from planfix_ops import ARCHIVE,REMOTE_ARCHIVE,remote,note
data=ARCHIVE.read_bytes();sha=hashlib.sha256(data).hexdigest()
SIZE=4096
parts=[data[i:i+SIZE] for i in range(0,len(data),SIZE)]
hashes=[hashlib.sha256(p).hexdigest() for p in parts]
root='/root/source_chunks_'+sha[:16]+'_4k'
preflight='''from pathlib import Path
import hashlib,json
r=Path(__ROOT__);r.mkdir(exist_ok=True);source=Path(__PACKAGE__)
prefix=source.read_bytes()[:294912] if source.exists() else b'';present=[]
if len(prefix)==294912 and hashlib.sha256(prefix).hexdigest()==__PREFIX_SHA__:
 for i in range(72):
  fragment=prefix[i*4096:(i+1)*4096];sha=hashlib.sha256(fragment).hexdigest();p=r/('%04d_%s.part'%(i,sha))
  if not p.exists():
   with p.open('xb') as out:out.write(fragment)
for p in r.glob('*.part'):
 i,sha=p.stem.split('_');assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 present.append(int(i))
print('PRESENT '+json.dumps(present),flush=True)
'''.replace('__ROOT__',repr(root)).replace('__PREFIX_SHA__',repr(hashlib.sha256(data[:294912]).hexdigest())).replace('__PACKAGE__',repr(REMOTE_ARCHIVE))
out=remote('gpuA',preflight,timeout=20)
present=json.loads(next(s.split(' ',1)[1] for s in out.splitlines() if s.startswith('PRESENT ')))
todo=[i for i in range(len(parts)) if i not in present][:4]
for i in todo:
 code='''from pathlib import Path
import base64,hashlib,os
p=Path(__PATH__);data=base64.b64decode(__DATA__);assert hashlib.sha256(data).hexdigest()==__SHA__
if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==__SHA__
else:
 q=p.with_suffix('.partial')
 with q.open('wb') as out:out.write(data);out.flush();os.fsync(out.fileno())
 assert hashlib.sha256(q.read_bytes()).hexdigest()==__SHA__;os.rename(q,p)
print('CHUNK_SHA_OK',__INDEX__,__SHA__,flush=True)
'''.replace('__PATH__',repr(root+'/%04d_%s.part'%(i,hashes[i]))).replace(
   '__DATA__',repr(base64.b64encode(parts[i]).decode())).replace('__SHA__',repr(hashes[i])).replace('__INDEX__',str(i))
 remote('gpuA',code,timeout=20)
note('gpuA','chunk coverage observed '+str(len(present))+'/'+str(len(parts))+'; this batch confirmed '+str(todo))
if not todo:
 code='''from pathlib import Path
import hashlib,os
r=Path(__ROOT__);chunks=[];paths=sorted(r.glob('*.part'));assert len(paths)==__COUNT__
for i,p in enumerate(paths):
 index,sha=p.stem.split('_');assert int(index)==i
 data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==sha;chunks.append(data)
data=b''.join(chunks);assert hashlib.sha256(data).hexdigest()==__SHA__
p=Path(__PACKAGE__+'.verified')
if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==__SHA__
else:
 with p.open('xb') as out:out.write(data);out.flush();os.fsync(out.fileno())
print('PACKAGE_ALL_CHUNKS_VERIFIED',str(p),__SHA__,flush=True)
'''.replace('__ROOT__',repr(root)).replace('__COUNT__',str(len(parts))).replace('__SHA__',repr(sha)).replace('__PACKAGE__',repr(REMOTE_ARCHIVE))
 remote('gpuA',code,timeout=20)
