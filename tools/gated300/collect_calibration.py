"""One read-only SSH transaction for the three owner-requested calibration JSONs."""
import json
from pathlib import Path
import subprocess
import sys

from tools.gated300.cct1 import compute


def main():
    paths = {
        'macroloss_0000': '/root/autodl-tmp/c4/macroloss20261028/LONG/selection_0000.json',
        'macroloss_0030': '/root/autodl-tmp/c4/macroloss20261028/LONG/selection_0030.json',
        'w1_paired_0010': '/root/autodl-tmp/c4/overnight20261028/W1_PAIRED/LONG/selection_0010.json',
    }
    code = ('import json,hashlib; paths=' + repr(paths) + '; out={}; '
            '\nfor name,p in paths.items():\n raw=open(p,"rb").read(); '
            'out[name]={"path":p,"sha256":hashlib.sha256(raw).hexdigest(),"document":json.loads(raw)}'
            '\nprint(json.dumps(out))')
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             'gpuR', '/root/miniconda3/bin/python -B -'], input=code, capture_output=True,
                            text=True, timeout=40,
                            creationflags=0x08000000 if sys.platform == 'win32' else 0)
    if result.returncode:
        raise RuntimeError(result.stderr)
    panels = json.loads(result.stdout)
    output = Path(__file__).parent / 'evidence'
    output.mkdir(exist_ok=True)
    for name, item in panels.items():
        print(name, json.dumps({k: v for k, v in item['document'].items()
                                if k not in ('rows', 'horizons')}))
        print('first_row', json.dumps(item['document']['rows'][0]))
        with (output / (name + '_source.json')).open('x', encoding='utf-8') as stream:
            json.dump(item, stream, indent=2)
        report = compute(item['document'], panel='calibration', source_sha256=item['sha256'])
        with (output / (name + '_CCT1.json')).open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(json.dumps(report))


if __name__ == '__main__':
    main()
