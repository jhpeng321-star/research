"""Run reduce_remote.py read-only over ssh (nice 19, single thread); save receipts locally."""
import hashlib, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'evidence'
BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
       'hetdata_truth', 'vehicle_params', 'driver_params', 'vtype')
ARMS = ['native', 'ctrl0', 'mappo', 'pi', 'circles']
JOBS = {
    'WMDATA': ('cpuQ', '/home/ubuntu/wmdata20261031/views', '/home/ubuntu/miniconda3/bin/python', 969400, 969440),
    'WMDATA2': ('gpuR', '/root/wmdata2/views', '/root/miniconda3/bin/python', 969550, 969600),
}


def run(corpus, lo=None, hi=None, target=None):
    host, root, py, LO, HI = JOBS[corpus]
    lo = LO if lo is None else lo; hi = HI if hi is None else hi
    body = (HERE / 'reduce_remote.py').read_text()
    header = ''.join(f'{k}={v!r}\n' for k, v in dict(ROOT=root, LO=lo, HI=hi, CORPUS=corpus, ARMS=ARMS, BAD=BAD).items())
    cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
           'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ' + py + ' -B -']
    p = subprocess.run(cmd, input=header + body, text=True, capture_output=True, timeout=5400)
    if p.returncode:
        raise RuntimeError(f'{host}: {p.stderr[-3000:]}')
    data = json.loads(p.stdout)
    data['reducer_sha256'] = hashlib.sha256(body.encode()).hexdigest()
    data['host'] = host
    target = target or OUT / f'{corpus.lower()}_vehicle_metrics.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=0))
    print('wrote', target, len(data['rows']))


if __name__ == '__main__':
    c = sys.argv[1]
    if len(sys.argv) > 3:
        run(c, int(sys.argv[2]), int(sys.argv[3]), OUT / f'_smoke_{c.lower()}.json')
    else:
        run(c)
