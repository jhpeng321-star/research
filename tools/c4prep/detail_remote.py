"""Read observable summary schemas only; no simulator state/type inputs."""
exec(compile(open('/dev/stdin').read(), '<unused>', 'exec')) if False else None
from pathlib import Path
import json
import re

R = Path('/root/autodl-tmp/c4/restart20261026')
bad = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype', re.I)


def ls(p):
    p = Path(p)
    assert not bad.search(str(p)) and not bad.search(str(p.resolve()))
    if p.is_dir():
        print(str(p), [q.name for q in p.iterdir() if not bad.search(q.name)])


p = R / 'arm/loop/worlds/c03/w968183.a1/artifact'
for q in (p/'summaries', p/'summaries/0', p/'pairs'):
    ls(q)
if (p/'summaries/0').is_file():
    d = json.loads((p/'summaries/0').read_text())
    print('summary', {k: (v if isinstance(v, (str, int, float, bool)) else
                         list(v)[:20] if isinstance(v, dict) else type(v).__name__)
                      for k, v in d.items()})
base = Path('/root/autodl-tmp/c4/wmlossonly20261028')
for rel in ('fallback', 'lossonly'):
    ls(base/rel)
    p = base/rel/'RESULTS.json'
    if p.is_file():
        d = json.loads(p.read_text())
        print('loss_results', {k: v for k, v in d.items() if 'checkpoint' in k or 'sha' in k or 'path' in k})
ls(R/'arm/loop/checks/c03.a2')
print('gate_log', (R/'arm/loop/checks/c03.a2/HORIZON_GATE.log').read_text()[-1800:])
