"""Package only explicit task Python files for the existing remote environments."""
import argparse
import hashlib
import re
import zipfile
from pathlib import Path

from .audit import safe


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('--configs', default='E:/agent-work/c4-speedhead/configs')
    a = p.parse_args()
    out = safe(a.out)
    if out.exists() or not out.parent.is_dir():
        raise ValueError('new output with existing parent required')
    source = Path(__file__).parent
    with zipfile.ZipFile(out, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for path in source.glob('*.py'):
            z.write(safe(path), 'tools/wmvalue/'+path.name)
        pending = ['c4_wm_loop14_r16_ft.toml']
        seen = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            raw = safe(Path(a.configs)/name).read_bytes().replace(b'\r\n', b'\n')
            if name == 'c4_wm_loop14_r16_ft.toml':
                assert hashlib.sha256(raw).hexdigest() == 'f95f7c4cbb501ccb85bb3968e29f3d6ca53db4ecef5f586225f96df1f48193e4'
            z.writestr('configs/'+name, raw)
            pending.extend(re.findall(r'^extends\s*=\s*"([^"]+)"', raw.decode(), re.M))
    print(out)


if __name__ == '__main__':
    main()
