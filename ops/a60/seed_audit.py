"""AMENDMENT 60 seed audit for 60A (S-WMPROP96), zero SUMO, no outcome read: is the candidate block 995200-995299
(96 used: 995200-995295) unused?

Same method as ops/a50/seed_audit.py (verbatim scan code): standalone integers of the block (hex / decimal fragments
excluded) in REGISTRATION.md, protocols/experiment_history.json, text files of EVERY C4 run dir and EVERY amendment
agent dir on E: (a32 .. a60), and ``git grep`` over every c4/* + main branch tip. Paths containing a forbidden token are
never opened; per-episode evaluation dirs and outcome files (WAVE_* / RESULT* / *_RESULTS* / vehicles* / observable*)
are skipped. Host run roots are audited separately (ops/a60/host/seed_audit_host.sh). Known taken / forbidden blocks
are checked for overlap explicitly (KNOWN).

Usage: python -I ops/a60/seed_audit.py --out <SEED_AUDIT_RAW.json> [--repo E:/agent-work/a60-wmvalue]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

RD = Path('E:/research-dense')
RUN = RD/'runs/active/C4-V7FT-20261017'
AW = Path('E:/agent-work')
BLOCKS = dict(a60a=(995200, 995299))
FORBIDDEN = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth', 'driver_params')
SKIP_DIRS = {'evaluation', '.git', '__pycache__', 'node_modules', 'venv', '.venv'}
SKIP_FILE = re.compile(r'^(WAVE_|RESULT|A\d+B?_RESULTS|vehicles|observable)', re.I)
TEXT_EXT = {'.json', '.jsonl', '.md', '.txt', '.py', '.sh', '.csv', '.log', '.out', '.yaml', '.yml', '.toml'}
MAX_BYTES = 40 << 20
KNOWN = dict(A42=((992000, 992099), (998600, 998699)), A50=((992800, 992899), (996800, 996899)),
             A44_proposed=((992400, 992499),), A41=((991724, 991771), (998400, 998591)),
             A33=((999500, 999699), (991600, 991699)), A35=((998000, 998399),), A37=((991700, 991899),),
             p20x_p20c=((993000, 993299),), finaleval_trainblock=((991000, 991999),), DISTILLRL=((998700, 998999),),
             N1_train_heldout=((989000, 989099), (989200, 989699)),
             FORBIDDEN=((964000, 964199), (962000, 962999), (977000, 977999), (989600, 989699)))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--repo', default='E:/agent-work/a60-wmvalue')
    a = ap.parse_args()
    overlap = [dict(block=b, known=k, range=list(r)) for b, (lo, hi) in BLOCKS.items() for k, rs in KNOWN.items()
               for r in rs if not (hi < r[0] or lo > r[1])]
    out = dict(schema='a60.seed_audit_raw.v1', blocks={k: list(v) for k, v in BLOCKS.items()}, known_overlap=overlap,
               roots=[str(r) for r in roots()], forbidden_skipped=list(FORBIDDEN), skipped_files=SKIP_FILE.pattern,
               file_hits={k: [] for k in BLOCKS}, git_hits={k: [] for k in BLOCKS})
    n = 0
    for p in files():
        try:
            text = p.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        n += 1
        for name, (lo, hi) in BLOCKS.items():
            for pos, tok in scan_text(text, lo, hi, pattern(lo, hi)):
                line = text.count('\n', 0, pos)+1
                out['file_hits'][name].append(dict(path=str(p), line=line, token=tok,
                                                   context=text[max(0, pos-60):pos+40].replace('\n', ' ')))
    out['n_files'] = n
    tips = sorted(set(subprocess.run(['git', '-C', a.repo, 'for-each-ref', '--format=%(objectname)', 'refs/heads/c4',
                                      'refs/heads/main'], capture_output=True, text=True).stdout.split()))
    out['n_tips'] = len(tips)
    for name, (lo, hi) in BLOCKS.items():
        pre = str(lo)[:3]
        rx = r'(^|[^0-9a-fA-F.])%s[0-9]{3}([^0-9a-fA-F.]|$)' % pre
        hits = set()
        for i in range(0, len(tips), 40):
            r = subprocess.run(['git', '-C', a.repo, 'grep', '-I', '-o', '-E', rx]+tips[i:i+40],
                               capture_output=True, text=True, errors='replace')
            for line in r.stdout.splitlines():
                parts = line.split(':', 2)
                if len(parts) < 3:
                    continue
                for m in re.finditer(r'\d{6}', parts[2]):
                    if lo <= int(m.group()) <= hi:
                        hits.add(f'{parts[0][:12]}:{parts[1]}:{m.group()}')
        coll = {}
        for h in hits:
            c, path, tok = h.split(':', 2)
            coll.setdefault(f'{path}:{tok}', []).append(c)
        out['git_hits'][name] = [dict(path_token=k, n_tips=len(v), example_tip=sorted(v)[0]) for k, v in sorted(coll.items())]
    out['audited'] = time.strftime('%Y-%m-%d %H:%M:%S %z')
    Path(a.out).write_text(json.dumps(out, indent=1)+'\n', encoding='utf-8')
    print(json.dumps({k: {b: len(v) for b, v in out[k].items()} for k in ('file_hits', 'git_hits')}),
          'files', n, 'tips', len(tips), 'overlap', overlap)


if __name__ == '__main__':
    main()
