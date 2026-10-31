"""Pruned filename-only exposure census; never opens dataset/truth files."""
import argparse
import json
import os
import re
from datetime import datetime, timezone

from .protocol import BLOCKS, FORBIDDEN, audit_registration, safe_path


def inventory(roots, *, variant='fleet'):
    count, hits, errors = 0, [], []
    for root in roots:
        root = safe_path(root)
        if not root.is_dir():
            raise ValueError('audit root absent')
        def error(exc):
            errors.append(str(exc))
        for parent, dirs, files in os.walk(root, followlinks=False, onerror=error):
            dirs[:] = [d for d in dirs if d not in ('.git', 'node_modules', '.venv', '__pycache__',
                                                    'miniconda3', 'miniconda', '.cache', 'site-packages')
                       and not FORBIDDEN.search(str(parent)+'/'+d)
                       and not os.path.islink(os.path.join(parent, d))]
            for name in files:
                if FORBIDDEN.search(str(parent)+'/'+name):
                    continue
                count += 1
                numbers = (int(m[0]) for m in re.finditer(r'(?<!\d)\d{6,10}(?!\d)', name))
                lo, hi = BLOCKS[variant]
                if any(lo <= n <= hi for n in numbers):
                    hits.append(str(parent)+'/'+name)
    return dict(roots=[str(r) for r in roots], safe_filenames=count, conflicts=hits,
                traversal_errors=errors, passed=not hits and not errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration')
    parser.add_argument('--root', action='append', default=[])
    parser.add_argument('--variant', choices=BLOCKS, default='fleet')
    parser.add_argument('--design-sha')
    args = parser.parse_args()
    result = dict(timestamp=datetime.now(timezone.utc).isoformat(), inventory=inventory(args.root, variant=args.variant))
    if args.registration:
        result['registration'] = audit_registration(args.registration, variant=args.variant, design_sha=args.design_sha)
    print(json.dumps(result, indent=2))
    if not all(r.get('passed', True) for r in result.values() if isinstance(r, dict)):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
