"""Read only allowed metadata/text surfaces; never descend into excluded paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

FORBIDDEN = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
             'hetdata_truth', 'vtype')
SEEDS = set(range(969200, 969392)) | set(range(969600, 969625))


def allowed(path):
    return not any(word in str(path).lower() for word in FORBIDDEN)


def audit(roots, content=False):
    hits, files, hashed = [], 0, hashlib.sha256()
    for root in roots:
        for folder, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if allowed(Path(folder) / d)
                       and d not in ('.git', 'node_modules', '__pycache__', '.venv')]
            for name in names:
                p = Path(folder) / name
                if not allowed(p):
                    continue
                files += 1
                hashed.update(str(p).encode())
                identities = sorted(SEEDS & {int(s) for s in re.findall(r'(?<!\d)969\d{3}(?!\d)', name)})
                if identities:
                    hits.append(dict(path=str(p), kind='filename', seeds=identities))
                if content and p.suffix.lower() in ('.md', '.json', '.jsonl', '.toml', '.txt'):
                    if p.stat().st_size > 10_000_000:
                        continue
                    for i, line in enumerate(p.read_text(errors='replace').splitlines(), 1):
                        identities = sorted(SEEDS & {int(s) for s in re.findall(r'(?<!\d)969\d{3}(?!\d)', line)})
                        if identities:
                            hits.append(dict(path=str(p), line=i, kind='text', seeds=identities,
                                             excerpt=line[:500]))
    return dict(roots=roots, content=content, files=files, path_index_sha256=hashed.hexdigest(), hits=hits)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('roots', nargs='+')
    p.add_argument('--content', action='store_true')
    args = p.parse_args()
    print(json.dumps(audit(args.roots, args.content), indent=1))
