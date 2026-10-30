"""Stage only named scoring code in a new owned gpuA directory."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
from remote import remote

ROOT = Path(__file__).resolve().parents[2]
OUT = '/root/autodl-tmp/c4/wmrequal20261030'


def main():
    names = ['src/wms/model/wms.py', 'src/wms/control/entrant_mix.py',
             'tools/errsrc/inference.py', 'tools/wmcombined/e3_score.py',
             'tools/wmrequal/labels.py', 'tools/wmrequal/score.py',
             'tools/wmrequal/real_panel.py']
    # One self-consistent WM source tree, rather than mixing the modern rollout
    # implementation with an older runtime lacking its fastpath dependencies.
    tracked=subprocess.check_output(['git','ls-files','--','src/wms','configs'],cwd=ROOT,text=True).splitlines()
    bad=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth')
    names=sorted(set(names+[n for n in tracked if n.endswith(('.py','.json')) and
                           not any(x in n.lower() for x in bad)]))
    payload = {n:base64.b64encode((ROOT/n).read_bytes()).decode() for n in names}
    data = base64.b64encode(json.dumps(payload).encode()).decode()
    sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote(f'''import base64,hashlib,json,pathlib
p=pathlib.Path({OUT!r}); assert p.parent.is_dir()
p.mkdir(exist_ok=True)
assert not (p/'LAUNCH.json').exists(), 'running source immutable'
hashes={{}}
for name,b64 in json.loads(base64.b64decode({data!r})).items():
 q=p/name;q.parent.mkdir(parents=True,exist_ok=True)
 raw=base64.b64decode(b64);q.write_bytes(raw);hashes[name]=hashlib.sha256(raw).hexdigest()
(p/'SOURCE.json').write_text(json.dumps(dict(base={sha!r},sha256=hashes),indent=2))
print('STAGED',len(hashes))
''')


if __name__ == '__main__':
    main()
