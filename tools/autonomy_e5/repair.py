"""Stage the one-time coordinator JSON transport repair; never rerun preparation."""
import base64
from pathlib import Path

from tools.loopresume.operate import remote


if __name__ == '__main__':
    payload = base64.b64encode(Path(__file__).with_name('remote.py').read_bytes()).decode()
    remote('gpuA', f'''
import pathlib,json,base64,hashlib
r=pathlib.Path('/root/autodl-tmp/c4/restart20261026/e5'); assert r.is_dir()
assert json.loads((r/'STATUS.json').read_text())['status']=='WAIT_OWNER'
receipt=json.loads((r/'PID.json').read_text()); proc=pathlib.Path('/proc')/str(receipt['pid'])
if proc.exists():
 stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
 assert stat[19]!=receipt['start_ticks'] or stat[0]=='Z'
old=r/'remote.py.before_transport_repair'; assert not old.exists()
old.write_bytes((r/'remote.py').read_bytes())
(r/'remote.py').write_bytes(base64.b64decode({payload!r}))
print(json.dumps(dict(staged=True,sha256=hashlib.sha256((r/'remote.py').read_bytes()).hexdigest(),old_pid=receipt)))
''')
