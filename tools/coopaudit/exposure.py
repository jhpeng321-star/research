"""Pruned metadata/register exposure audit; no truth/route/assignment file reads."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from tools.coopaudit.design import SEEDS, RESTRICTED, checked


def audit(roots, content=False):
    hits, count, index = [], 0, hashlib.sha256()
    for root in roots:
        checked(root)
        for folder, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if not RESTRICTED.search(str(Path(folder)/d))
                       and d not in ('.git', '.venv', 'node_modules', '__pycache__')
                       and not (Path(folder)/d).is_symlink()]
            for name in names:
                p = Path(folder)/name
                if RESTRICTED.search(str(p)) or p.is_symlink():
                    continue
                count += 1
                index.update(str(p).encode())
                found = SEEDS & {int(x) for x in re.findall(r'(?<!\d)969\d{3}(?!\d)', name)}
                if found:
                    hits.append(dict(path=str(p), kind='filename', seeds=sorted(found)))
                if content and name in ('REGISTRATION.md', 'DESIGN.md', 'EXPERIMENT_HISTORY.md',
                                        'experiment_history.json', 'registration.json'):
                    if p.stat().st_size > 10_000_000:
                        continue
                    for i, line in enumerate(p.read_text(errors='replace').splitlines(), 1):
                        found = SEEDS & {int(x) for x in re.findall(r'(?<!\d)969\d{3}(?!\d)', line)}
                        if found:
                            hits.append(dict(path=str(p), line=i, kind='text', seeds=sorted(found),
                                             excerpt=line[:300]))
    return dict(roots=roots, files=count, path_index_sha256=index.hexdigest(), hits=hits)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('roots', nargs='+')
    p.add_argument('--content', action='store_true')
    p.add_argument('--out')
    a = p.parse_args()
    text = json.dumps(audit(a.roots, a.content), indent=1)
    if a.out:
        dest = checked(a.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text+'\n')
    print(text)
