"""Verify all source chunks on gpuA, then reuse the preserved fresh-stage payload."""
import ast
import hashlib
from pathlib import Path
from planfix_ops import ARCHIVE, HOME, REMOTE_ARCHIVE, TEMP, remote

data=ARCHIVE.read_bytes();sha=hashlib.sha256(data).hexdigest()
REMOTE_ARCHIVE=REMOTE_ARCHIVE+'.verified'
hashes=[hashlib.sha256(data[i:i+16384]).hexdigest() for i in range(0,len(data),16384)]
public=(TEMP/'id_ed25519_planfix_20261105.pub').read_text().strip()
assert public.startswith('ssh-ed25519 ') and public.endswith('planfix-gpua-20261105')
tree=ast.parse((HOME/'planfix_deploy_gpua.py').read_text())
template=next(node.value for node in ast.walk(tree) if isinstance(node,ast.Constant) and
              isinstance(node.value,str) and node.value.startswith('\nimport base64, hashlib, json, os, subprocess, sys, time, zlib'))
stage=template.replace('__ARCHIVE__',repr(REMOTE_ARCHIVE)).replace('__ARCHIVE_SHA__',repr(sha)).replace(
    '__COMMIT__',repr('f631ce90cfd2f23213f7c57a828c848e2db10ecd')).replace('__PUBLIC__',repr(public))
stage=stage.replace("added_lines=1,restriction=", "added_lines=1,removed_lines=0,added_line=line.strip(),restriction=")
ast.parse(stage)
verify='''import hashlib,json,os
from pathlib import Path
p=Path(__PATH__);source=p if p.exists() else p.with_name(p.name+'.partial')
data=source.read_bytes();assert len(data)==__LENGTH__ and hashlib.sha256(data).hexdigest()==__SHA__
for i,sha in enumerate(__HASHES__):assert hashlib.sha256(data[i*16384:(i+1)*16384]).hexdigest()==sha
if not p.exists():
 with p.open('xb') as out:out.write(data);out.flush();os.fsync(out.fileno())
print('SOURCE_ALL_CHUNKS_VERIFIED',__COUNT__,__SHA__,flush=True)
'''.replace('__PATH__',repr(REMOTE_ARCHIVE)).replace('__LENGTH__',str(len(data))).replace(
    '__SHA__',repr(sha)).replace('__HASHES__',repr(hashes)).replace('__COUNT__',str(len(hashes)))
# Unknown outcomes remain inspect-only: this helper has no command replay loop.
remote('gpuA',verify+stage,timeout=30)
