"""AMENDMENT 77: build the two self-contained ship bundles on the laptop (zero SUMO; copies only small files).

    python -I ops/a77/build_ships.py --out E:/agent-work/a77-ship
writes <out>/train/ + a77_train_ship.tgz and <out>/eval/ + a77_eval_ship.tgz (each dir with SHA256SUMS), then prints
the tgz sha256. Write-once: refuses an existing out dir. Optional extras (copied only when the path exists):
Python-3.12.16.tgz (--python-src; else setup downloads it from python.org) and the MAPPO_CT policy (--mappo-ct).
"""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

OPS = Path(__file__).resolve().parents[1]
A77 = OPS/'a77'
RUN = Path('E:/research-dense/runs/active/C4-V7FT-20261017')
TRAIN_INPUTS = {
    'a51_code_4c2abb14.tgz': (RUN/'mappo_p60/a51_code_4c2abb14.tgz',
                              '800f210c8f11913ee73d376d35010a7a42043ac71669b002690d69db3a2df1e0'),
    'mappo_u0_policy_00169.pt': (RUN/'mappo_wz/inputs/mappo_u0_policy_00169.pt',
                                 'd9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca'),
    'a65_sanity_pre4_rows.jsonl': (RUN/'mappo_p60c/sanity_pre4/rows.jsonl',
                                   '3fe97fb70d3b8804570b41bbc04eb5a8bbb9e4e1c6f7eaea73b07e916bae9acb')}
TRAIN_FILES = {'setup_a77_train.sh': A77/'train/host/setup_a77_train.sh', 'go_train.sh': A77/'train/host/go_train.sh',
               'compare_rows.py': A77/'train/host/compare_rows.py', 'charge.py': A77/'train/host/charge.py',
               'requirements_a65.txt': A77/'train/host/requirements_a65.txt',
               'test_a77_train.py': A77/'train/test_a77_train.py',
               'EXPECTED_PLAN_a77_run84.json': A77/'train/EXPECTED_PLAN_a77_run84.json',
               'overlay/configs/c4_mappo_speed_wz3to2_d2800p20_a77.toml':
                   A77/'train/overlay/configs/c4_mappo_speed_wz3to2_d2800p20_a77.toml',
               'overlay/tools/nm_mappo/n1_adapter_a77.py': A77/'train/overlay/tools/nm_mappo/n1_adapter_a77.py'}
EVAL_FILES = {'setup_a77e.sh': A77/'host/setup_a77e.sh', 'run_a77e.sh': A77/'host/run_a77e.sh',
              'patch_tables.py': A77/'host/patch_tables.py', 'a77_mappo20c.py': OPS/'a77_mappo20c.py',
              'test_a77_mappo20c.py': OPS/'test_a77_mappo20c.py',
              'nm_eval_panel_wz3to2_a77c_D2800_p20.json': A77/'nm_eval_panel_wz3to2_a77c_D2800_p20.json',
              'SEED_AUDIT.json': A77/'SEED_AUDIT.json'}
TOP = dict(train='a77ship', eval='a77eship')         # unpack dirs: /home/ubuntu/a77ship, /home/ubuntu/a77eship
EXTRA = {'python_src': ('Python-3.12.16.tgz', '710c3e75dac8b2f258d9cd0f76e5c4c2e9175fac9e249c98901552f7cf33b7dd'),
         'mappo_ct': ('mappo_ct_policy_00012.pt', '069a4f85ab4ac7bdc5e01404dbe903f9b21957d5e1478c34d0929e5dcb2020c3')}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def put(src, dst, text):
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = Path(src).read_bytes()
    if text:
        if b'\r\n' in data:
            raise SystemExit(f'{src}: CRLF (ship files must be LF)')
    dst.write_bytes(data)


def build(d, files, inputs, extra):
    d.mkdir(parents=True)
    for name, src in files.items():
        put(src, d/name, text=True)
    for name, (src, want) in inputs.items():
        if sha(src) != want:
            raise SystemExit(f'{src}: sha {sha(src)[:12]} != {want[:12]}')
        shutil.copyfile(src, d/name)
    sums = sorted(p for p in d.rglob('*') if p.is_file())
    (d/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(d).as_posix()}\n' for p in sums), newline='\n')
    for key, (name, want) in extra.items():                       # optional: pinned by the setup, not in SHA256SUMS
        if key and Path(key).is_file():
            if sha(key) != want:
                raise SystemExit(f'{key}: sha != {want[:12]}')
            shutil.copyfile(key, d/name)
    tgz = d.parent/f'a77_{d.name}_ship.tgz'
    with tarfile.open(tgz, 'w:gz', format=tarfile.PAX_FORMAT) as t:
        for p in sorted(d.rglob('*')):
            info = t.gettarinfo(str(p), arcname=f'{TOP[d.name]}/{p.relative_to(d).as_posix()}')
            info.uid = info.gid = 1000
            info.uname = info.gname = 'ubuntu'
            info.mtime = 0
            if p.is_file():
                info.mode = 0o755 if p.suffix == '.sh' else 0o644
                with open(p, 'rb') as f:
                    t.addfile(info, f)
            else:
                info.mode = 0o755
                t.addfile(info)
    return tgz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--python-src', default='')
    ap.add_argument('--mappo-ct', default='')
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'{out} exists (write-once)')
    t = build(out/'train', TRAIN_FILES, TRAIN_INPUTS, {a.python_src: EXTRA['python_src']})
    e = build(out/'eval', EVAL_FILES, {}, {a.mappo_ct: EXTRA['mappo_ct']})
    doc = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in (t, e)}
    (out/'SHIPS.json').write_text(json.dumps(doc, indent=1)+'\n', newline='\n')
    print(json.dumps(doc, indent=1))


if __name__ == '__main__':
    main()
