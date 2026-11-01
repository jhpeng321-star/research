"""Stage only committed INT inference code in a NEW immutable remote directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from tools.rbint.remote import ssh


def main():
    p = argparse.ArgumentParser()
    p.add_argument('host', choices=('gpuA', 'gpuR', 'cpuQ'))
    p.add_argument('--endpoint-only', action='store_true', help='analysis helpers only; reuse original WM2 model/runtime')
    p.add_argument('--parent', choices=('/root/rbeval20261031', '/root/autodl-tmp/c4/rb2int20261101'))
    p.add_argument('--receipt', help='exclusive local staging receipt after verified transfer')
    p.add_argument('--source-commit', help='explicit frozen source; defaults to current HEAD')
    args = p.parse_args()
    local_receipt = Path(args.receipt) if args.receipt else None
    if local_receipt is not None and (not local_receipt.parent.is_dir() or local_receipt.exists()):
        raise ValueError('new local staging receipt in existing parent required')
    root = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(['git', 'rev-parse', '--verify', (args.source_commit or 'HEAD')+'^{commit}'], cwd=root, text=True).strip()
    paths = ['src/wms', 'tools/rbint', 'tools/rbeval',
             'configs/execparity_cav_idm_v1.json', 'configs/native_cav_head_v5.json',
              'configs/wm_phys.toml', 'configs/wm_phys_cpu.toml',
              'artifacts/rbfit/pooled_full_v5/idm_pooled.json',
              'artifacts/rbfit/shrinkage_v1/idm_population.json',
              'artifacts/rbfit/shrinkage_v1/idm_posteriors.json.gz',
              'artifacts/rbfit/shrinkage_v1/idm_shrinkage.json',
              'docs/paper_results/rbunc/observed_arrivals.json',
              'tools/rbfit', 'tests/unit/test_rbfit.py', 'tests/unit/test_rbfit_directional.py',
              'tests/test_rbint_merge_fast.py', 'tests/test_rbint_lane_fast.py',
               'tests/test_rbint_motion_posterior.py', 'tests/test_rbint_endpoints.py']
    paths += ['tools/rb2int', 'tools/rbcal', 'tools/rb2lphys',
              'configs/wm_phys_v5.toml', 'configs/wm_phys_v5_lp.toml',
              'artifacts/rb2lphys/learned_v1.json',
              'docs/paper_results/rb2cal/v1/fit/calibration.json',
              'docs/rb2int/EVALUATION_DESIGN.md', 'docs/rb2int/EVALUATION_REGISTRATION.json',
              'docs/rb2int/PANEL_METADATA.json',
              'tests/test_rb2int_candidate_recipes.py', 'tests/test_rb2int_lane_cal_composition.py',
              'tests/test_rb2lphys.py', 'tests/test_rb2_spread_calibration.py']
    # Explicit round-2 LC continuation dependency closure; old recipe remains OFF.
    paths += ['tools/rblane', 'configs/wm_phys_rb2_lane.toml',
              'artifacts/rbfit/lane_raw_c5602102891e']
    if args.endpoint_only:
        paths = ['tools/rbint', 'tools/rbeval']
    archive = subprocess.check_output(['git', 'archive', '--format=tar', sha, *paths], cwd=root)
    archive_sha = hashlib.sha256(archive).hexdigest()
    parent = '/home/ubuntu/rbint20261101' if args.host == 'cpuQ' else '/root/rbeval20261031'
    if args.parent:
        if args.host == 'cpuQ':
            raise ValueError('explicit alternate parent is for GPU hosts only')
        parent = args.parent
    python = '/home/ubuntu/miniconda3/bin/python' if args.host == 'cpuQ' else '/root/miniconda3/bin/python'
    dest = f'{parent}/phys_code_{sha[:12]}'
    # This archive consists solely of this repository's committed code/data receipts.
    code = f'''
import io, json, hashlib, shutil, tarfile
from pathlib import Path
parent=Path({parent!r})
assert parent.parent.is_dir()
assert shutil.disk_usage(parent if parent.exists() else parent.parent).free > max(512*2**20, 3*{len(archive)}), 'wait for free staging disk; preserve existing files'
parent.mkdir(exist_ok=True)
dest=Path({dest!r})
if dest.exists():
 receipt=json.loads((dest/'INT_STAGE.json').read_text())
 assert receipt['commit']=={sha!r} and receipt['archive_sha256']=={archive_sha!r}
 print(json.dumps(receipt)); raise SystemExit(0)
dest.mkdir()
'''
    prepared = ssh(args.host, code + '\nprint("NEW_STAGE",str(dest))')
    print(prepared)
    if 'NEW_STAGE' not in prepared:
        if local_receipt is not None:
            with local_receipt.open('x', encoding='utf-8') as stream:
                json.dump(json.loads(prepared), stream, indent=1)
        return
    # Send trusted local git bytes, never a remote download or peer runtime edit.
    unpack = f'''import sys,io,tarfile,hashlib,json
from pathlib import Path
data=sys.stdin.buffer.read();assert hashlib.sha256(data).hexdigest()=={archive_sha!r}
dest=Path({dest!r});assert dest.is_dir() and not any(dest.iterdir())
with tarfile.open(fileobj=io.BytesIO(data)) as tar:
 for m in tar.getmembers():
  assert not m.issym() and not m.islnk() and '..' not in Path(m.name).parts and not Path(m.name).is_absolute()
 tar.extractall(dest,filter='data')
receipt=dict(host={args.host!r},commit={sha!r},archive_sha256={archive_sha!r},runtime=str(dest),endpoint_only={args.endpoint_only!r})
with (dest/'INT_STAGE.json').open('x') as f: json.dump(receipt,f,indent=2)
print(json.dumps(receipt))'''
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host,
        python, '-B', '-c', __import__('shlex').quote(unpack)],
        input=archive, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr.decode() + result.stdout.decode())
    print(result.stdout.decode())
    if local_receipt is not None:
        receipt = json.loads(result.stdout.decode())
        with local_receipt.open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, indent=1)


if __name__ == '__main__':
    main()
