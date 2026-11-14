"""Build a75_search_box.tgz (complete: code tree, A75 search code, checkpoint, cfgs, scripts) + SHIP_MEMBERS.sha256.

Layout under the tgz root ``a75/`` (extract with ``tar xzf a75_search_box.tgz -C /root/autodl-tmp``):
  code_split/code_v12r7b/   code_v12r7b.tar 320e19f3 (every file checked against code_v12r7b.files.sha256) with
                            tools/wzprop/wm_backend.py = a47-relay wm_backend_split.py (728fc917, the box version)
  ckpt/ckpt/                ckpt.tar f9b3a854 (wm_s.pt 6da9d333)
  a75ship/                  a75ss.py, a75_wmforms.py, wm_fix_crn.py (E1-FIX copy)
  cfg/                      A75_SEARCH_CFG.json (stage 1), A75B_SEARCH_CFG.json (stage 2)
  run_search_a75.sh, setup_a75.sh, MANIFEST.sha256
"""
import hashlib
import io
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE_TAR = Path('E:/agent-work/a47-relay/code_v12r7b.tar')
CODE_SUMS = Path('E:/agent-work/a47-relay/code_v12r7b.files.sha256')
CKPT_TAR = Path('E:/agent-work/a47-relay/ckpt.tar')
SPLIT = ROOT / 'ship' / 'tree_overlay' / 'tools' / 'wzprop' / 'wm_backend.py'
PINS = {CODE_TAR: '320e19f3de8bddf4ea789b9c911e68b8dadafdda8a09600cd42e27728bd25de6',
        CKPT_TAR: 'f9b3a854dbba7adc646a13b0308f40628fef799f6cdbf5900057e084707baa5e',
        SPLIT: '728fc917cc1f8f9ca7f78f835871642b5143dbc5d4edf5fb4907ea25c07793a4'}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    for p, want in PINS.items():
        if sha(p.read_bytes()) != want:
            raise SystemExit(f'pin mismatch {p}')
    sums = {}
    for line in CODE_SUMS.read_text().splitlines():
        h, name = line.split(' *', 1)
        sums[name] = h
    members = {}                                   # tgz path -> bytes
    with tarfile.open(CODE_TAR) as t:
        for m in t.getmembers():
            if not m.isfile() or '__pycache__' in m.name:
                continue
            b = t.extractfile(m).read()
            if sums.get(m.name) != sha(b):
                raise SystemExit(f'code tar member mismatch {m.name}')
            members['a75/code_split/' + m.name] = b
    if len(members) != len(sums):
        raise SystemExit(f'code members {len(members)} != listed {len(sums)}')
    members['a75/code_split/code_v12r7b/tools/wzprop/wm_backend.py'] = SPLIT.read_bytes()
    with tarfile.open(CKPT_TAR) as t:
        for m in t.getmembers():
            if m.isfile():
                members['a75/ckpt/' + m.name] = t.extractfile(m).read()
    for n in ('a75ss.py', 'a75_wmforms.py', 'wm_fix_crn.py'):
        members['a75/a75ship/' + n] = (ROOT / 'ship' / n).read_bytes()
    for n in ('A75_SEARCH_CFG.json', 'A75B_SEARCH_CFG.json'):
        members['a75/cfg/' + n] = (ROOT / 'cfg' / n).read_bytes()
    for n in ('run_search_a75.sh', 'setup_a75.sh'):
        b = (ROOT / 'box' / n).read_bytes()
        if b'\r' in b:
            raise SystemExit(f'{n} has CR line endings')
        members['a75/' + n] = b
    manifest = ''.join(f'{sha(b)}  {k[len("a75/"):]}\n' for k, b in sorted(members.items()))
    members['a75/MANIFEST.sha256'] = manifest.encode()
    out = ROOT / 'a75_search_box.tgz'
    if out.exists():
        raise SystemExit(f'REFUSE: {out} exists (write-once)')
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz', format=tarfile.PAX_FORMAT) as t:
        dirs = set()
        for k in sorted(members):
            parts = k.split('/')
            for i in range(1, len(parts)):
                d = '/'.join(parts[:i])
                if d not in dirs:
                    ti = tarfile.TarInfo(d)
                    ti.type, ti.mode, ti.mtime = tarfile.DIRTYPE, 0o755, 1794600000
                    t.addfile(ti)
                    dirs.add(d)
            ti = tarfile.TarInfo(k)
            ti.size, ti.mtime = len(members[k]), 1794600000
            ti.mode = 0o755 if k.endswith('.sh') else 0o644
            t.addfile(ti, io.BytesIO(members[k]))
    out.write_bytes(buf.getvalue())
    sel = [k for k in sorted(members) if not k.startswith('a75/code_split/')] + \
          ['a75/code_split/code_v12r7b/tools/wzprop/wm_backend.py',
           'a75/code_split/code_v12r7b/tools/wzprop/strategy_search.py',
           'a75/code_split/code_v12r7b/src/wms/planapi.py', 'a75/code_split/code_v12r7b/data/regime_STATES.json']
    lines = [f'{sha(out.read_bytes())}  a75_search_box.tgz  ({len(members)} members, {out.stat().st_size} bytes)']
    lines += [f'{sha(members[k])}  {k}' for k in sel]
    (ROOT / 'SHIP_MEMBERS.sha256').write_text('\n'.join(lines) + '\n', newline='\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    sys.exit(main())
