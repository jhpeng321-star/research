#!/usr/bin/env python3
"""Maintain protocols/offloaded_paths.json from clean-up journals.

Every row a clean-up journal marks ``DELETED`` was removed only after its Drive copy
passed a live size+MD5 check; its bytes still exist on Google Drive. This tool records
each such local path with its Drive location so the experiment ledger can tell
"offloaded to Drive" apart from "missing" (tools/experiment_history.py ``offloaded``).

    python tools/offload_map.py update tmp/morning/journal.log [more journals ...]
    python tools/offload_map.py check            # entries are well-formed, no duplicates

The map is append-only: existing entries are kept, new local paths are added.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP = os.path.join(ROOT, 'protocols', 'offloaded_paths.json')
SCHEMA = 'offloaded_paths/1'


def load(path=MAP):
    if not os.path.exists(path):
        return {'schema': SCHEMA,
                'description': 'Local paths removed by owner-run clean-ups after a live Drive '
                               'size+MD5 check; bytes remain on Google Drive at "drive". '
                               'Restore with tools/fetch_data.py (python -m wms fetch).',
                'entries': []}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def journal_rows(path):
    """DELETED rows of an EXEC run: (utc, local path, bytes, drive location)."""
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            col = line.rstrip('\n').split('\t')
            if len(col) < 7 or col[2] != 'EXEC' or col[3] != 'DELETED':
                continue
            drive = col[6].split(';')[0].strip()
            if not drive.startswith('gdrive:'):
                continue
            yield col[0], col[4], int(col[5] or 0), drive


def update(journals, path=MAP):
    m = load(path)
    seen = {os.path.normcase(e['local']) for e in m['entries']}
    added = 0
    for j in journals:
        for utc, local, size, drive in journal_rows(j):
            key = os.path.normcase(local)
            if key in seen:
                continue
            seen.add(key)
            m['entries'].append({'local': local, 'drive': drive, 'bytes': size, 'removed_utc': utc,
                                 'journal': os.path.relpath(os.path.abspath(j), ROOT).replace(os.sep, '/')})
            added += 1
    m['entries'].sort(key=lambda e: e['local'].lower())
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(m, f, indent=1, ensure_ascii=True)
        f.write('\n')
    return added, len(m['entries'])


def check(path=MAP):
    m = load(path)
    problems = []
    if m.get('schema') != SCHEMA:
        problems.append('schema %r != %s' % (m.get('schema'), SCHEMA))
    seen = set()
    for e in m['entries']:
        key = os.path.normcase(e.get('local', ''))
        if not key or not os.path.isabs(e['local']):
            problems.append('entry without an absolute local path: %r' % e)
        if not str(e.get('drive', '')).startswith('gdrive:'):
            problems.append('entry without a gdrive: location: %r' % e)
        if key in seen:
            problems.append('duplicate local path: %s' % e['local'])
        seen.add(key)
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    u = sub.add_parser('update', help='add DELETED rows of clean-up journals')
    u.add_argument('journals', nargs='+')
    sub.add_parser('check', help='validate the map')
    a = ap.parse_args(argv)
    if a.cmd == 'update':
        added, total = update(a.journals)
        print('added %d, total %d entries -> %s' % (added, total, os.path.relpath(MAP, ROOT)))
        return 0
    problems = check()
    for p in problems:
        print(p)
    print('%d problem(s)' % len(problems))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
