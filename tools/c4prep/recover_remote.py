"""Bounded read-only recovery of C4 preparation, gate and observable inputs."""
from pathlib import Path
import json
import re

R = Path('/root/autodl-tmp/c4/restart20261026')
BAD = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype', re.I)


def safe(p):
    p = Path(p)
    assert not BAD.search(str(p)) and not BAD.search(str(p.resolve()))
    return p


for rel in ('AUTONOMY_STATUS.json', 'arm/loop/longint/c3wmft/check_repair1/STATUS.json',
            'arm/loop/longint/c3wmft/check_repair1/PID.json'):
    p = safe(R / rel)
    if p.is_file():
        print(rel, p.read_text())
for rel in ('arm/loop/c4prep', 'runtime_c4prep', 'arm/loop/checks/c03.a2',
            'arm/loop/longint/c3wmft/check_repair1'):
    p = safe(R / rel)
    print('directory', rel, [q.name for q in p.iterdir() if not BAD.search(q.name)] if p.is_dir() else None)
p = safe(R / 'arm/loop/checks/c03.a2/HORIZON_GATE.log')
if p.is_file():
    print('gate_log_tail', p.read_text()[-2600:])
p = safe(R / 'arm/loop/worlds/c03/w968183.a1/artifact/summaries/0')
if p.is_file():
    d = json.loads(p.read_text())
    print('summary_schema', {k: (list(v)[:30] if isinstance(v, dict) else
          v[:2] if isinstance(v, list) else v) for k, v in d.items()})
for rel in ('arm/loop/worlds/c03/w968183.a1/artifact/pairs',):
    p = safe(R / rel)
    print('pairs', [q.name for q in p.iterdir() if not BAD.search(q.name)])
