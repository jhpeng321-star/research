"""Bounded explicit-path inspection; never opens simulator assignments."""
import json
from pathlib import Path
import re

BAD = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype', re.I)


def safe(p):
    p = Path(p)
    assert not BAD.search(str(p)) and not BAD.search(str(p.resolve()))
    return p


def show(p, tail=None):
    p = safe(p)
    if p.is_file():
        text = p.read_text()
        print(str(p), text[-tail:] if tail else text)


def listing(p):
    p = safe(p)
    if p.is_dir():
        print('files', str(p), [q.name for q in p.iterdir()
                              if not BAD.search(q.name) and q.name != 'scenario'])


if Path.home() == Path('/root'):
    r = Path('/root/autodl-tmp/c4/restart20261026')
    for rel in ('AUTONOMY_STATUS.json', 'arm/loop/longint/c3wmft/check_repair1/RESULTS.json',
                'arm/loop/checks/c03.a2/HORIZON_GATE.json',
                'arm/loop/sumocredit/REAL_CREDIT_c03.a1.json'):
        show(r / rel)
    show(r / 'arm/loop/checks/c03.a2/HORIZON_GATE.log', tail=4500)
    listing(r / 'arm/loop/longint/c3wmft/check_repair1')
    for rel in ('arm/loop/c4prep', 'backlog_stage', 'arm/loop/returns'):
        listing(r / rel)
    show(r / 'arm/loop/c4prep/backlog_import_v1/IMPORT.json', tail=2500)
    listing(r / 'arm/loop/c4prep/backlog_import_v1')
else:
    home = Path.home()
    for seed in (968183, 968187):
        root = home / f'c4/main14/restart20261026/work/w{seed}'
        if not root.is_dir():
            continue
        for cycle in root.glob('*/c03*'):
            for rel in ('collect', 'collect/episodes/policy/s968183_m1900_p0050',
                        'collect/branches/policy/s968183_m1900_p0050/t0360_policy'):
                listing(cycle / rel)
            for p in cycle.glob('collect/episodes/policy/*/summary.json'):
                d = json.loads(safe(p).read_text())
                print('base_summary_schema', str(p), {k: list(v) if isinstance(v, dict) else
                      type(v).__name__ for k, v in d.items()})
    listing(home / 'backlog')
