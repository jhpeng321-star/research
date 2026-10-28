"""Inspect observable summary/credit schemas without opening referenced inputs."""
from pathlib import Path
import json

R = Path('/root/autodl-tmp/c4/restart20261026')
p = R / 'arm/loop/worlds/c03/w968183.a1/artifact'
d = json.loads((p / 'summaries/0/summary.json').read_text())
print('summary', json.dumps(d)[:20000])
for name in ('SUMOCREDIT_POINTS.jsonl', 'GATE_CALIBRATION.jsonl'):
    with (p / name).open() as stream:
        print(name, stream.readline()[:10000])
for rel in ('arm/loop/policy/c01.a3/RUN_SUMMARY.json', 'arm/loop/policy/c03.a6/RUN_SUMMARY.json'):
    d = json.loads((R / rel).read_text())
    print('policy_config', rel, json.dumps({k: d[k] for k in ('config', 'resumed_from', 'wm_members') if k in d})[:18000])
print('gate_pid_stat', Path('/proc/323434/stat').read_text() if Path('/proc/323434/stat').exists() else None)
