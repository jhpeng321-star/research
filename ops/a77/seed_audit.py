"""AMENDMENT 77 seed audit (S-MAPPO20C), zero SUMO, no outcome read: are the proposed seed blocks unused?
  a77t 987200-987599  NEW TRAIN block for the MAPPO20C trainer (400 seeds, = the size of the A65 block 989200-989599;
                      sanity panel = its last 12 seeds 987588-987599)
  a77c 997600-997699  NEW evaluation block; confirmation panel 997600-997695 (96 worlds, D2800_p20)
File scan + git scan: the code of ops/a73/seed_audit.py (copied unchanged; only BLOCKS / KNOWN / schema / docstring
changed). Hits inside the A77 files written for this amendment are expected and are classified separately.

Usage: python -I ops/a77/seed_audit.py --part files|git --out <json> [--repo E:/agent-work/a54-mappo60b]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

RD = Path('E:/research-dense')
RUN = RD/'runs/active/C4-V7FT-20261017'
AW = Path('E:/agent-work')
BLOCKS = dict(a77t=(987200, 987599), a77c=(997600, 997699))
FORBIDDEN = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth', 'driver_params')
SKIP_DIRS = {'evaluation', '.git', '__pycache__', 'node_modules', 'venv', '.venv'}
SKIP_FILE = re.compile(r'^(WAVE_|RESULT|A\d+B?_RESULTS|vehicles|observable)', re.I)
TEXT_EXT = {'.json', '.jsonl', '.md', '.txt', '.py', '.sh', '.csv', '.log', '.out', '.yaml', '.yml', '.toml'}
MAX_BYTES = 40 << 20
KNOWN = dict(A70=((996300, 996399),), A68=((992500, 992599), (996200, 996299)), A42=((992000, 992099), (998600, 998699)), A50=((992800, 992899), (996800, 996899)),
             A44_proposed=((992400, 992499),), A73=((996400, 996499),), A60A=((995200, 995299),), A41=((991724, 991771), (998400, 998591)),
             A33=((999500, 999699), (991600, 991699)), A35=((998000, 998399),), A37=((991700, 991899),),
             p20x_p20c=((993000, 993299),), finaleval_trainblock=((991000, 991999),), DISTILLRL=((998700, 998999),),
             N1_train_heldout=((989000, 989099), (989200, 989699)),
             FORBIDDEN=((964000, 964199), (962000, 962999), (977000, 977999), (989600, 989699)))
NL = chr(10)
NUL = bytes(1)


def forbidden(p):
    s = str(p).lower()
    return any(t in s for t in FORBIDDEN)


def pattern(lo, hi):
    return re.compile(r'(?<![\da-fA-F.])(\d{%d})(?![\da-fA-F.])' % len(str(lo)))


def scan_text(text, lo, hi, rx):
    return [(m.start(), m.group(1)) for m in rx.finditer(text) if lo <= int(m.group(1)) <= hi]


def roots():
    out = [d for d in sorted(RUN.iterdir()) if d.is_dir()]
    out += [d for d in sorted(AW.iterdir()) if d.is_dir() and re.match(r'^a\d', d.name)]
    return out


def files():
    yield RUN/'REGISTRATION.md'
    yield RD/'protocols/experiment_history.json'
    for root in roots():
        if forbidden(root):
            continue
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in SKIP_DIRS and not forbidden(Path(dp)/d)]
            for f in fns:
                p = Path(dp)/f
                if forbidden(p) or SKIP_FILE.match(f) or p.suffix.lower() not in TEXT_EXT:
                    continue
                try:
                    if p.stat().st_size > MAX_BYTES:
                        continue
                except OSError:
                    continue
                yield p


def scan_files(out):
    n = 0
    for p in files():
        try:
            text = p.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        n += 1
        for name, (lo, hi) in BLOCKS.items():
            for pos, tok in scan_text(text, lo, hi, pattern(lo, hi)):
                line = text.count(NL, 0, pos)+1
                out['file_hits'][name].append(dict(path=str(p), line=line, token=tok,
                                                   context=text[max(0, pos-60):pos+40].replace(NL, ' ')))
    out['n_files'] = n


def scan_git(out, repo):
    tips = sorted(set(subprocess.run(['git', '-C', repo, 'for-each-ref', '--format=%(objectname)', 'refs/heads/c4',
                                      'refs/heads/main'], capture_output=True, text=True).stdout.split()))
    out['n_tips'] = len(tips)
    r = subprocess.run(['git', '-C', repo, 'rev-list', '--objects', '--no-walk', '--stdin'], input=NL.join(tips)+NL,
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode:
        raise SystemExit('git rev-list failed: '+r.stderr[:300])
    paths = {}
    for line in r.stdout.splitlines():
        sha, _, path = line.partition(' ')
        if path and not forbidden(path):
            paths.setdefault(sha, path)
    rx = re.compile(rb'(?<![\da-fA-F.])(\d{6})(?![\da-fA-F.])')
    hits = {k: {} for k in BLOCKS}
    proc = subprocess.Popen(['git', '-C', repo, 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    order = list(paths)

    def feed():
        proc.stdin.write((NL.join(order)+NL).encode())
        proc.stdin.close()
    threading.Thread(target=feed, daemon=True).start()
    n_blob = n_bin = 0
    for sha in order:
        head = proc.stdout.readline().split()
        if len(head) != 3 or head[0].decode() != sha:
            raise SystemExit(f'cat-file protocol error at {sha}: {head}')
        data = proc.stdout.read(int(head[2]))
        proc.stdout.read(1)
        if head[1] != b'blob':
            continue
        n_blob += 1
        if NUL in data[:8000]:
            n_bin += 1
            continue
        for m in rx.finditer(data):
            v = int(m.group(1))
            for name, (lo, hi) in BLOCKS.items():
                if lo <= v <= hi:
                    hits[name].setdefault(f'{paths[sha]}:{v}', set()).add(sha[:12])
    proc.wait()
    out.update(n_blobs=n_blob, n_binary_skipped=n_bin, git_method='unique blobs of the tip trees (cat-file --batch)')
    for name in BLOCKS:
        out['git_hits'][name] = [dict(path_token=k, n_blobs=len(v), example_blob=sorted(v)[0])
                                 for k, v in sorted(hits[name].items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--part', choices=('files', 'git'), required=True)
    ap.add_argument('--repo', default='E:/agent-work/a54-mappo60b')
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f'{a.out} exists (write-once)')
    overlap = [dict(block=b, known=k, range=list(r)) for b, (lo, hi) in BLOCKS.items() for k, rs in KNOWN.items()
               for r in rs if not (hi < r[0] or lo > r[1])]
    out = dict(schema='a77.seed_audit_raw.v1', part=a.part, blocks={k: list(v) for k, v in BLOCKS.items()},
               known_overlap=overlap, roots=[str(r) for r in roots()], forbidden_skipped=list(FORBIDDEN),
               skipped_files=SKIP_FILE.pattern, file_hits={k: [] for k in BLOCKS}, git_hits={k: [] for k in BLOCKS})
    if a.part == 'files':
        scan_files(out)
    else:
        scan_git(out, a.repo)
    out['audited'] = time.strftime('%Y-%m-%d %H:%M:%S %z')
    Path(a.out).write_text(json.dumps(out, indent=1)+NL, encoding='utf-8')
    print(json.dumps({k: {b: len(v) for b, v in out[k].items()} for k in ('file_hits', 'git_hits')}),
          {k: out.get(k) for k in ('n_files', 'n_tips', 'n_blobs', 'n_binary_skipped')}, 'overlap', overlap)


if __name__ == '__main__':
    main()
